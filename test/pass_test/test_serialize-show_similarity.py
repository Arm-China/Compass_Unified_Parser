# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import os
import torch
import numpy as np

from utils.run import generate_ir


def create_caffe_model(model_path, prototxt_path):
    ''' Create caffe model.
    '''
    import caffe
    prototxt_content = '''
name: 'test'
input: 'X'
input_shape { dim: 20  dim: 10}

layer {
    name: 'ip1'
    type: 'InnerProduct'
    bottom: 'X'
    top: 'ip1'
    inner_product_param {
        num_output: 64
        weight_filler {
        type: 'gaussian'
        std: 0.1
        }
        bias_filler {
        type: 'constant'
        }
    }
}'''
    with open(prototxt_path, 'w') as txt_file:
        txt_file.write(prototxt_content)
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_path, caffe.TEST)
    net.save(model_path)
    return model_path


def create_tf_model(model_path, input_shape, is_tf_model):
    ''' Create tf/tflite model.
    '''
    import tensorflow.compat.v1 as tf
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_shape, name='X')
        y = tf.add(x, 10.02, name='Y')

        sess.run(tf.global_variables_initializer())
        if is_tf_model:
            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ['Y'])

            # save to pb file
            with tf.gfile.GFile(model_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
        else:
            converter = tf.lite.TFLiteConverter.from_session(sess,
                                                             input_tensors=[x], output_tensors=[y])
            tflite_model = converter.convert()
            open(model_path, 'wb').write(tflite_model)


def create_tf2_model(model_path, input_shape):
    ''' Create tf2 model.
    '''
    import tensorflow as tf
    from tensorflow import keras
    x = keras.Input(shape=input_shape[1:], batch_size=input_shape[0], name='X')
    y = tf.math.add(x, 12.2, name='Y')

    model = keras.models.Model([x], y)
    model.save(model_path)


def create_onnx_model(onnx_path, input_size, version=13):
    ''' Create onnx model.
    '''
    import onnx
    from onnx import TensorProto, helper
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_size)
    flatten = helper.make_node(
        'Flatten', ['X'],
        ['Y']
    )
    graph_def = helper.make_graph(
        [flatten],  # nodes
        'flatten_model',  # name
        [X],  # inputs
        [Y],  # outputs

    )
    model_def = helper.make_model(graph_def, producer_name='flatten_model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


class flatten_model(torch.nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super(flatten_model, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, self.start_dim, self.end_dim)


def create_torch_model(model_path):
    try:
        model = flatten_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'similarity'
input_shape = [20, 10]
# Generate input data
feed_dict = dict()
feed_dict['X'] = (np.random.ranf(input_shape) * 10).astype(np.float32)
np.save('input', feed_dict)

output_dir = './output_dir'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_name = TEST_NAME
cfg_path = model_name + '.cfg'
for model_type, format_type in zip(['caffe', 'tf', 'tflite', 'tf', 'onnx', 'torch'], ['caffemodel', 'pb', 'tflite', 'h5', 'onnx', 'pt']):
    model_path = model_name + '.' + format_type
    prototxt_path = (model_name + '.prototxt') if model_type == 'caffe' else ''
    # Generate cfg
    cfg_content = '''[Common]
    model_type = {0}
    model_name = similarity
    input_model = {1}
    caffe_prototxt = {2}
    input = X
    input_shape = {3}
    output_dir = ./output_dir
    similarity_input_npy = input.npy
    '''.format(model_type, model_path, prototxt_path, str(input_shape))
    with open(cfg_path, 'w') as txt_file:
        txt_file.write(cfg_content)
    # Create model
    if format_type == 'caffemodel':
        create_caffe_model(model_path, prototxt_path)
    elif format_type in ('pb', 'tflite'):
        create_tf_model(model_path, input_shape, (format_type == 'pb'))
    elif format_type == 'h5':
        create_tf2_model(model_path, input_shape)
    elif format_type == 'onnx':
        create_onnx_model(model_path, input_shape)
    else:  # torch
        create_torch_model(model_path)
    # Run tests with parser and compare result with runtime
    exit_status = generate_ir(cfg_path, verbose=True)
    assert exit_status
