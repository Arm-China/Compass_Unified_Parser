# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os

from UnifiedParser.logger import ERROR, WARN
from .common import get_model_type


def generate_cfg(model_path, model_type, inputs=None, inputs_shape=None, output_folder_name='output_dir', proto_path=None):
    test_name = model_path.split('/')[-1]
    test_name = test_name.rsplit('.', maxsplit=1)[0]

    model_path_fullpath = os.path.abspath(model_path)

    shape_str = ''
    if inputs_shape is not None:
        for shape in inputs_shape:
            shape_str = shape_str + ',' + \
                str(shape) if shape_str != '' else str(shape)

    output_dir = os.path.dirname(
        model_path_fullpath) + '/' + output_folder_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cfg_file = test_name + '.cfg'
    if os.path.exists(cfg_file):
        WARN('File %s will be overwriten!' % cfg_file)

    with open(cfg_file, 'w') as f:
        f.write('[Common]\n')
        f.write('model_type = ' + model_type + '\n')
        f.write('model_name = ' + test_name + '\n')
        f.write('detection_postprocess = \n')
        f.write('model_domain = image_classification\n')
        f.write('input_model = ' + model_path_fullpath + '\n')
        if inputs is not None:
            f.write('input = ' + inputs + '\n')
            if inputs_shape is not None:
                f.write('input_shape = ' + shape_str + '\n')
        if model_type == 'caffe':
            f.write('caffe_prototxt = ' + proto_path + '\n')
        f.write('output_dir = ' + output_dir + '\n')
    return cfg_file


def read_caffe_model(model_path, proto_path, save_cfg=False):
    # Make sure caffe is installed(Try: conda activate caffe)
    import caffe

    assert os.path.exists(proto_path), 'File %s does not exist!' % proto_path
    caffe.set_mode_cpu()
    net = caffe.Net(proto_path, caffe.TEST)

    input_names = []
    input_shapes = []
    for idx, layer in enumerate(net.layers):
        if layer.type != 'Input':
            continue
        for b_idx in list(net._top_ids(idx)):
            inp_name = net._blob_names[b_idx]
            if inp_name in input_names:
                continue
            input_names.append(inp_name)
            input_shapes.append(list(net.blobs[inp_name].data.shape))

    input_names_str = ''
    for input_name in input_names:
        input_names_str = (input_names_str + ',' + input_name) if input_names_str else input_name

    model_content = proto_path
    model_content += str([(k, layer.type) for k, layer in net.layer_dict.items()])
    model_content += '\nINPUT_NAME:' + input_names_str
    model_content += '\nINPUT_SHAPE:' + str(input_shapes)

    if save_cfg:
        cfg_path = generate_cfg(model_path, 'caffe', input_names_str, input_shapes, proto_path=proto_path)
    else:
        cfg_path = None

    return model_content, cfg_path


def read_keras_model(model_path, save_cfg=False):
    import tensorflow as tf
    load_options = tf.saved_model.LoadOptions(allow_partial_checkpoint=True)
    model = tf.keras.models.load_model(model_path, compile=False, options=load_options)

    model_content = model_path
    model_content += '\n--------------- summary ---------------\n'
    model_content += str(model.summary())

    inputs = [inp.name for inp in model.inputs]
    inputs_shape = [inp.shape.as_list() for inp in model.inputs]

    inputs = ','.join(inputs)
    # inputs_shape = ','.join(inputs_shape)

    model_content += '\nINPUT_NAME:' + inputs
    model_content += '\nINPUT_SHAPE:' + str(inputs_shape)

    if save_cfg:
        cfg_path = generate_cfg(model_path, 'tensorflow', inputs, inputs_shape)
    else:
        cfg_path = None

    return model_content, cfg_path


def read_onnx_model(model_path, save_cfg=False):
    import onnx

    def get_tensor_info(graph_info):
        tensors = ''
        tensors_shapes = []
        for inp in graph_info:
            tensors = tensors + ',' + inp.name if tensors != '' else inp.name
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                shape.append(dim.dim_value)
            tensors_shapes.append(shape)
        return tensors, tensors_shapes

    model_content = model_path
    model = onnx.load(model_path)
    try:
        onnx.checker.check_model(model)
    except:
        WARN('Error in checking model, but will still proceed...')

    model_content += str(model)

    graph = model.graph
    inputs, inputs_shape = get_tensor_info(graph.input)
    model_content += '\nINPUT_NAME:' + inputs
    model_content += '\nINPUT_SHAPE:' + str(inputs_shape)

    outputs, outputs_shape = get_tensor_info(graph.output)
    model_content += '\nOUTPUT_NAME:' + outputs
    model_content += '\nOUTPUT_SHAPE:' + str(outputs_shape)

    if save_cfg:
        cfg_path = generate_cfg(model_path, 'onnx', inputs, inputs_shape)
    else:
        cfg_path = None

    return model_content, cfg_path


def read_tf_model(frozen_pb, save_cfg=False, model_type='tensorflow'):
    import tensorflow.compat.v1 as tf
    from tensorflow.python.framework import tensor_util

    g_def = tf.GraphDef()
    with open(frozen_pb, 'rb') as f:
        g_def.ParseFromString(f.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(g_def, name='')

    inputs = ''
    inputs_shape = []
    model_content = frozen_pb
    for op in graph.get_operations():
        node = op.node_def
        model_content += '\n--------------- node ---------------\n'
        model_content += str(node)
        if node.op.lower() == 'placeholder':
            inputs = inputs + ',' + node.name if inputs != '' else node.name
            shape = tf.TensorShape(node.attr['shape'].shape)
            inputs_shape.append(shape.as_list())
        try:
            tensor_content = tensor_util.MakeNdarray(node.attr['value'].tensor)
            model_content += '\n------ tensor content of ' + node.name + ' ------\n'
            model_content += str(tensor_content)
        except:
            pass

    model_content += '\nINPUT_NAME:' + inputs
    model_content += '\nINPUT_SHAPE:' + str(inputs_shape)

    if save_cfg:
        cfg_path = generate_cfg(frozen_pb, model_type, inputs, inputs_shape)
    else:
        cfg_path = None

    return model_content, cfg_path


def read_tflite_model(model_path, save_cfg=False):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path)
    all_tensor_details = interpreter.get_tensor_details()
    interpreter.allocate_tensors()

    model_content = model_path
    model_content += '\nTensor Details:'
    for tensor_item in all_tensor_details:
        model_content += '\n' + str(tensor_item)
        model_content += '\nWeight ' + tensor_item['name'] + ':'
        model_content += '\n' + str(interpreter.tensor(tensor_item['index'])())

    model_content += '\nInput Details:'
    input_details = interpreter.get_input_details()
    for input_detail in input_details:
        model_content += '\n' + str(input_detail)

    model_content += '\nOutput Details:'
    output_details = interpreter.get_output_details()
    for output_detail in output_details:
        model_content += '\n' + str(output_detail)

    if save_cfg:
        cfg_path = generate_cfg(model_path, 'tflite')
    else:
        cfg_path = None

    return model_content, cfg_path


def read_model(model_path, save_cfg=False, model_type=None, proto_path=None):
    ''' Read model and save it to log file.
    Basing on the suffix of model path, decide model type if it's not set.
    '''
    if not os.path.exists(model_path):
        ERROR('File %s does not exist!' % model_path)
        return None

    if model_type is None:
        model_type = get_model_type(model_path)

    model_content = None
    cfg_path = None
    if model_type == 'caffe':
        model_content, cfg_path = read_caffe_model(model_path, proto_path, save_cfg)
    elif model_type == 'onnx':
        model_content, cfg_path = read_onnx_model(model_path, save_cfg)
    elif model_type in ('tf', 'tensorflow'):
        if model_path.endswith('.pb'):
            model_content, cfg_path = read_tf_model(
                model_path, save_cfg, model_type)
        else:
            model_content, cfg_path = read_keras_model(
                model_path, save_cfg)
    elif model_type == 'tflite':
        model_content, cfg_path = read_tflite_model(model_path, save_cfg)
    else:
        # TODO: Support other models
        ERROR('Unsupported model type!')
    return model_content, cfg_path
