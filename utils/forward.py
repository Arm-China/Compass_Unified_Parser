# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import re

from UnifiedParser.logger import ERROR, FATAL, INFO, WARN
from .common import (get_model_type, match_node_name,
                     save_data_to_file)


def caffe_forward(proto_path, model_path, feed_dict, output_names=None, save_output=True):
    import caffe

    net = caffe.Net(proto_path, model_path, caffe.TEST)
    for k, v in feed_dict.items():
        net.blobs[k].data[...] = v
    _ = net.forward()

    if output_names is None:
        output_names = net.outputs

    output_dict = dict()
    for out_name in output_names:
        out_data = net.blobs[out_name].data[...]
        output_dict[out_name] = out_data

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
        if isinstance(layer.output, list):
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
    import onnxruntime as rt

    sess = rt.InferenceSession(model_path)
    # input_name = sess.get_inputs()[0].name
    if output_names is None:
        output_names = [o.name for o in sess.get_outputs()]
    elif not isinstance(output_names, list):
        ERROR('Argument output_names should be a list!')

    output_data = sess.run(output_names, feed_dict)
    output_dict = dict()
    for out_name, out_data in zip(output_names, output_data):
        # print(out_name, out_data.shape)
        # if out_data.shape:
        #     print(out_data.flatten()[:50])
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

    with tf.Session() as session:
        if output_names is None:
            all_output_tensors = [op.outputs for op in tf.get_default_graph().get_operations()]
            all_input_tensors = [op.inputs for op in tf.get_default_graph().get_operations()]
            output_names = get_default_output_names(all_input_tensors, all_output_tensors)
        output_tensors = [tf.get_default_graph().get_tensor_by_name(out_name)
                          for out_name in output_names]
        output_data = session.run(output_tensors, feed_dict=feed_dict)

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
        interpreter.set_tensor(inp['index'], feed_dict[inp['name']])
    interpreter.invoke()
    output_dict = dict()
    for out in output_details:
        out_name = out['name']
        output_data = interpreter.get_tensor(out['index'])
        output_dict[out_name] = output_data

    if save_output:
        save_data_to_file('tflite_outputs.npy', output_dict)

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

    # TODO: Support other model types
    ERROR('Runtime forward for %s is not yet supported!' % model_type)
    return {}


def opt_forward(txt_path, bin_path, feed_dict, output_names=None, save_output=True, forward_type='float'):
    from .run_ir_forward import run_ir_forward

    with open(txt_path, 'r') as f:
        txt_content = f.read()
        in_match = re.search(r'input_tensors=\[(\S+)\]', txt_content)
        out_match = re.search(r'output_tensors=\[(\S+)\]', txt_content)
    if in_match and in_match.group(0) and in_match.group(1):
        input_names = in_match.group(1).rsplit(',')
    # The order of inputs matters for opt forward. Update feed_dict to match input order in IR.
    ordered_feed_dict = {}
    for name in input_names:
        if name in feed_dict.keys():
            ordered_feed_dict.update({name: feed_dict[name]})
        elif (name + ':0') in feed_dict.keys():
            tensor_name = name + ':0'
            ordered_feed_dict.update({tensor_name: feed_dict[tensor_name]})
        else:
            ERROR('Cannot find input name (%s) from feed_dict!' % name)

    # Get default output names from txt_path if it's not set
    if output_names is None:
        if out_match and out_match.group(0) and out_match.group(1):
            output_names = out_match.group(1).rsplit(',')
            INFO('Find output names: %s' % str(output_names))
        else:
            ERROR('Cannot find output names from IR!')

    outputs = run_ir_forward(txt_path, bin_path, ordered_feed_dict, forward_type)

    if len(output_names) != len(outputs):
        WARN('Outputs name len != outputs data len. Will save parts of outputs.')

    output_dict = {}
    for name, value in zip(output_names, outputs):
        output_dict[name] = value

    if save_output:
        save_data_to_file('opt_outputs.npy', output_dict)

    return output_dict
