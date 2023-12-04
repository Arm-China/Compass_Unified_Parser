# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import os
import re
from collections import OrderedDict

from ..logger import ERROR, FATAL, INFO, WARN
from .common import (get_model_type, match_node_name,
                     save_data_to_file)


def caffe_forward_impl(output_dict, proto_path, model_path, feed_dict, output_names):
    import caffe

    net = caffe.Net(proto_path, model_path, caffe.TEST)
    for k, v in feed_dict.items():
        net.blobs[k].data[...] = v
    _ = net.forward()

    if output_names is None:
        output_names = net.outputs

    for out_name in output_names:
        out_data = net.blobs[out_name].data[...]
        output_dict[out_name] = out_data
    return


def caffe_forward(proto_path, model_path, feed_dict, output_names=None, save_output=True):
    import multiprocessing as mp
    from multiprocessing import Process, Manager

    output_dict = {}

    original_start_method = mp.get_start_method()
    with Manager() as manager:
        # Force start method to spawn so that the environment is clean(avoid duplicate define issue of caffe.proto)
        mp.set_start_method('spawn', force=True)
        mgr_dict = manager.dict()
        process = Process(target=caffe_forward_impl, args=(mgr_dict, proto_path, model_path, feed_dict, output_names))
        process.start()
        process.join()
        # exit_code = process.exitcode
        # Copy the result returned from caffe_forward_impl to output_dict
        output_dict = {key: value for key, value in mgr_dict.items()}
        try:
            process.close()
        except Exception as e:
            DEBUG('[Parser]: Fail to close process because %s' % str(e))
        # Reset to previous start method
        mp.set_start_method(original_start_method, force=True)

    if save_output:
        save_data_to_file('caffe_outputs.npy', output_dict)
    return output_dict


def keras_forward(model_path, feed_dict, output_names=None, save_output=True):
    import tensorflow as tf
    from tensorflow.keras import backend as K

    load_options = tf.saved_model.LoadOptions(allow_partial_checkpoint=True)
    model = tf.keras.models.load_model(model_path, compile=False, options=load_options)

    output_dict = {}
    if output_names is None:
        output_names = [out.name for out in model.outputs]
        if not output_names:
            FATAL('Please provide output names!')
            return output_dict

    feed_model_inputs = []
    for inp_name in model.input_names:
        if ':' not in inp_name:
            trim_tensor_name = inp_name
            tensor_name = trim_tensor_name + ':0'
        else:
            tensor_name = inp_name
            trim_tensor_name = tensor_name.split(':')[0]
        if tensor_name in feed_dict.keys():
            key = tensor_name
        elif trim_tensor_name in feed_dict.keys():
            key = trim_tensor_name
        else:
            FATAL('Cannot find value for input (%s) in feed_dict!' % tensor_name)
            return output_dict
        feed_model_inputs.append(feed_dict[key])

    layer_out_tensors = []
    for layer in model.layers:
        if isinstance(layer.output, (list, tuple)):
            layer_out_tensors.extend(layer.output)
        else:
            layer_out_tensors.append(layer.output)
    outputs = [out for out in layer_out_tensors if out.name in output_names or out.name.split(':')[0] in output_names]
    functors = K.function(model.inputs, outputs)
    layer_outputs = functors(feed_model_inputs)
    for out, out_value in zip(outputs, layer_outputs):
        output_dict.update({out.name: out_value})

    if save_output:
        save_data_to_file('tf_outputs.npy', output_dict)

    return output_dict


def onnx_forward(model_path, feed_dict, output_names=None, save_output=True):
    import copy
    import onnxruntime as rt
    import onnx

    # input_name = sess.get_inputs()[0].name
    if output_names is None:
        sess = rt.InferenceSession(model_path)
        output_names = [o.name for o in sess.get_outputs()]
    else:
        assert isinstance(output_names, list), 'Argument output_names should be a list!'
        model = onnx.load(model_path)
        # clear outputs first
        for output in model.graph.output:
            model.graph.output.pop()
        # convert node name to tensor name
        original_output_names = copy.deepcopy(output_names)
        for node in model.graph.node:
            if node.name in original_output_names:
                index = output_names.index(node.name)
                output_names.pop(index)
                for idx, output in enumerate(node.output):
                    output_names.insert(index+idx, output)
        for output in output_names:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
            print("Model add output %s" % output)
        sess = rt.InferenceSession(model.SerializeToString())

    model_inputs = sess.get_inputs()
    input_names_from_feed_dict = list(feed_dict.keys())
    updated_feed_dict = {}
    for model_input, default_name in zip(model_inputs, input_names_from_feed_dict):
        input_name = model_input.name
        if input_name in feed_dict:
            key_name = input_name
        else:
            WARN('Cannot find input name (%s) from feed_dict! Will try to use input (%s) from feed_dict!' %
                 (input_name, default_name))
            key_name = default_name
        updated_feed_dict.update({input_name: feed_dict[key_name]})

    try:
        o_names = [o.name for o in sess.get_outputs()]
        output_data = sess.run(o_names, updated_feed_dict)
    except Exception as e:
        ERROR("Fail to run because %s" % str(e))

    output_dict = dict()
    for out_name, out_data in zip(o_names, output_data):
        if isinstance(out_data, list):
            assert len(out_data) == 1, 'out_data is a list of more than 1 element!'
            out_data = out_data[0]
        output_dict[out_name] = out_data

    if save_output:
        save_data_to_file('onnx_outputs.npy', output_dict)

    return output_dict


def tf_forward(model_path, feed_dict, output_names=None, save_output=True):
    def get_default_output_names(input_tensors, output_tensors):
        output_names = []
        input_tensor_name = []
        for input_tensor in input_tensors:
            input_tensor_name.extend([t.name for t in input_tensor])
        for tensors in output_tensors:
            for tensor in tensors:
                if tensor.name not in input_tensor_name:
                    output_names.append(tensor.name)
        return output_names

    import tensorflow.compat.v1 as tf

    tf.reset_default_graph()
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    valid_feed_dict = {}
    for name, value in feed_dict.items():
        tensor_name = (name + ':0') if ':' not in name else name
        valid_feed_dict.update({tensor_name: value})

    with tf.Session() as session:
        if output_names is None:
            all_output_tensors = [op.outputs for op in tf.get_default_graph().get_operations()]
            all_input_tensors = [op.inputs for op in tf.get_default_graph().get_operations()]
            output_names = get_default_output_names(all_input_tensors, all_output_tensors)
        else:
            output_names = [(name + ':0') if ':' not in name else name for name in output_names]
        output_tensors = [tf.get_default_graph().get_tensor_by_name(out_name)
                          for out_name in output_names]
        output_data = session.run(output_tensors, feed_dict=valid_feed_dict)

    output_dict = dict()
    for out_name, out_data in zip(output_names, output_data):
        # print(out_name, out_data.shape)
        # if out_data.shape:
        #     print(out_data.flatten()[:50])
        output_dict[out_name] = out_data

    if save_output:
        save_data_to_file('tf_outputs.npy', output_dict)

    return output_dict


def tflite_forward(model_path, feed_dict, output_names=None, save_output=True):
    import tensorflow.compat.v1 as tf

    interpreter = tf.lite.Interpreter(model_path)
    interpreter.reset_all_variables()
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for inp in input_details:
        if inp['name'] not in feed_dict:
            ERROR('[Parser]: Cannot find input %s from feed_dict!' % inp['name'])
            continue
        interpreter.set_tensor(inp['index'], feed_dict[inp['name']])
    interpreter.invoke()
    output_dict = dict()
    if output_names:
        for detail in interpreter.get_tensor_details():
            tensor_name = detail['name']
            if tensor_name and tensor_name in output_names:
                output_data = interpreter.get_tensor(detail['index'])
                output_dict[tensor_name] = output_data
    else:
        for out in output_details:
            out_name = out['name']
            output_data = interpreter.get_tensor(out['index'])
            output_dict[out_name] = output_data

    if save_output:
        save_data_to_file('tflite_outputs.npy', output_dict)

    return output_dict


def torch_forward(model_path, ordered_feed_dict, output_names=None, save_output=True):
    import torch
    from ..front_end.torch.utils import get_tuple_from_tensor_type

    def _flat_outputs(data):
        outputs = []
        if isinstance(data, (list, tuple)):
            for nested_data in data:
                outputs.extend(_flat_outputs(nested_data))
        elif isinstance(data, dict):
            outputs.extend(list(data.values()))
        else:
            outputs.append(data)
        return outputs

    try:
        model = torch.jit.load(model_path)
        jit_model = model
    except:
        model = torch.load(model_path)
        model.eval()
        jit_model = torch.jit.freeze(torch.jit.script(model))
    inputs = [torch.tensor(inp) for inp in ordered_feed_dict.values()]
    input_tensors = ()
    input_index = 0
    for inp in jit_model.graph.inputs():
        tensors, input_index = get_tuple_from_tensor_type(inp.type(), inputs, input_index)
        if len(tensors) > 0:
            input_tensors += tensors
    out_tensors = model(*input_tensors)
    out_tensors = _flat_outputs(out_tensors)

    if output_names is None:
        # graph = model.graph
        # output_names = [out.debugName() for out in graph.outputs()]
        output_names = ['out_' + str(idx) for idx in range(len(out_tensors))]
    assert len(out_tensors) == len(output_names), 'The length of out_tensors is different with output_names!'

    output_dict = OrderedDict()
    for out_name, out_tensor in zip(output_names, out_tensors):
        if torch.cuda.is_available():
            out_tensor = out_tensor.cpu()
        elif str(out_tensor.dtype).startswith('torch.q'):
            # origin_out_tensor = torch.int_repr(out_tensor).numpy()
            out_tensor = torch.dequantize(out_tensor)
        try:
            out_tensor = out_tensor.detach().numpy()
        except:
            out_tensor = np.array(out_tensor)
        output_dict[out_name] = out_tensor

    if save_output:
        save_data_to_file('torch_outputs.npy', output_dict)

    return output_dict


def rt_forward(model_path, feed_dict, output_names=None, save_output=True, proto_path=None):
    model_type = get_model_type(model_path)
    if model_type == 'onnx':
        return onnx_forward(model_path, feed_dict, output_names, save_output)
    if model_type in ('tensorflow', 'tf'):
        if model_path.endswith('.pb'):
            return tf_forward(model_path, feed_dict, output_names, save_output)
        return keras_forward(model_path, feed_dict, output_names, save_output)
    if model_type == 'caffe':
        if proto_path is None:
            proto_path = model_path.rsplit('.', 1)[0] + '.prototxt'
        return caffe_forward(proto_path, model_path, feed_dict, output_names, save_output)
    if model_type == 'tflite':
        return tflite_forward(model_path, feed_dict, output_names, save_output)
    if model_type == 'torch':
        return torch_forward(model_path, feed_dict, output_names, save_output)

    # TODO: Support other model types
    ERROR('Runtime forward for %s is not yet supported!' % model_type)
    return {}


def opt_forward(txt_path, bin_path, feed_dict, output_names=None, save_output=True,
                forward_type='float', transfer_to_float=None):
    from .run_ir_forward import run_ir_forward

    is_quantized_model = (forward_type != 'float')
    input_is_quantized = False

    with open(txt_path, 'r') as f:
        txt_content = f.read()
        in_match = re.search(r'input_tensors=\[(\S+)\]', txt_content)
        out_match = re.search(r'output_tensors=\[(\S+)\]', txt_content)
    input_names = []
    if in_match and in_match.group(0) and in_match.group(1):
        input_names = in_match.group(1).rsplit(',')
    input_names_from_feed_dict = list(feed_dict.keys())
    # some inputs can be removed during parsing because they are not needed
    assert len(input_names) <= len(input_names_from_feed_dict), \
        'Expects %d inputs but got %d in opt_forward!' % (len(input_names), len(input_names_from_feed_dict))
    # The order of inputs matters for opt forward. Update feed_dict to match input order in IR.
    ordered_feed_dict = {}
    for name, default_name in zip(input_names, input_names_from_feed_dict):
        key_name = None
        if name in feed_dict.keys():
            key_name = name
        elif (name + ':0') in feed_dict.keys():
            key_name = name + ':0'  # tensor_name
        elif (name + '_0') in feed_dict.keys():
            key_name = name + '_0'
        else:
            name_without_postfix = name.rsplit('_', 1)[0]
            if name_without_postfix in feed_dict.keys():
                key_name = name_without_postfix
            elif (name_without_postfix + ':0') in feed_dict.keys():
                key_name = name_without_postfix + ':0'
            else:
                WARN('Cannot find input name (%s) from feed_dict! Will try to use input (%s) from feed_dict!' %
                     (name, default_name))
                key_name = default_name
        input_data = feed_dict[key_name]
        ordered_feed_dict.update({name: input_data})
        if not input_is_quantized and is_quantized_model and 'int' in input_data.dtype.name:
            input_is_quantized = True

    # Get default output names from txt_path if it's not set
    if output_names is None:
        if out_match and out_match.group(0) and out_match.group(1):
            output_names = out_match.group(1).rsplit(',')
            INFO('Find output names: %s' % str(output_names))
        else:
            ERROR('Cannot find output names from IR!')

    outputs = run_ir_forward(txt_path, bin_path, ordered_feed_dict, forward_type, transfer_to_float, input_is_quantized)

    if len(output_names) != len(outputs):
        WARN('Outputs name len != outputs data len. Will save parts of outputs.')

    output_dict = {}
    for name, value in zip(output_names, outputs):
        output_dict[name] = value

    if save_output:
        save_data_to_file('opt_outputs.npy', output_dict)

    return output_dict
