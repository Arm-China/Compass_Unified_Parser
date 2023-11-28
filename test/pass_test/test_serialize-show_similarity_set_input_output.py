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
    top: 'Y0'
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
}

layer {
  name: 'prob'
  type: 'Softmax'
  bottom: 'Y0'
  top: 'Y'
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
        x0 = tf.math.tanh(x, name='X0')
        y0 = tf.math.sqrt(x0, name='Y0')
        y = tf.add(y0, 10.02, name='Y')

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
    x0 = tf.math.tanh(x, name='X0')
    y0 = tf.math.sqrt(x0, name='Y0')
    y = tf.math.add(y0, 12.2, name='Y')

    model = keras.models.Model([x], y)
    model.save(model_path)


def create_onnx_model(onnx_path, input_size, version=13):
    ''' Create onnx model.
    '''
    import onnx
    from onnx import TensorProto, helper
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    X0 = helper.make_tensor_value_info('X0', TensorProto.FLOAT, input_size)
    Y0 = helper.make_tensor_value_info('Y0', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_size)
    flatten = helper.make_node(
        'Flatten', ['X'],
        ['X0']
    )
    sinh = helper.make_node(
        'Sinh', ['X0'],
        ['Y0']
    )
    add = helper.make_node(
        'Add', ['X0', 'Y0'],
        ['Y']
    )
    graph_def = helper.make_graph(
        [flatten, sinh, add],  # nodes
        'flatten_model',  # name
        [X],  # inputs
        [Y],  # outputs

    )
    model_def = helper.make_model(graph_def, producer_name='flatten_model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


TEST_NAME = 'similarity_set_in_out'
# Generate input data
input_shape = [20, 10]
feed_dict = dict()

output_dir = './output_dir'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_name = TEST_NAME
cfg_path = model_name + '.cfg'
for input_name in ('X', 'X0'):
    feed_dict.clear()
    feed_dict[input_name] = (np.random.ranf(input_shape) * 10).astype(np.float32)
    np.save('input', feed_dict)
    for output_name in ('Y', 'Y0', ''):
        for model_type, format_type in zip(['caffe', 'tf', 'tflite', 'onnx', ], ['caffemodel', 'pb', 'tflite', 'onnx', ]):
            if input_name == 'X0' \
                    and model_type in ('caffe', 'tflite', 'onnx'):
                continue
            model_path = model_name + '.' + format_type
            prototxt_path = (model_name + '.prototxt') if model_type == 'caffe' else ''
            # Generate cfg
            cfg_content = '''[Common]
model_type = {0}
model_name = similarity
input_model = {1}
caffe_prototxt = {2}
input = {3}
input_shape = [20, 10]
output = {4}
output_dir = ./output_dir
similarity_input_npy = input.npy'''.format(model_type, model_path, prototxt_path, input_name, output_name)
            with open(cfg_path, 'w') as txt_file:
                txt_file.write(cfg_content)
            # Create model
            if format_type == 'caffemodel':
                create_caffe_model(model_path, prototxt_path)
            elif format_type in ('pb', 'tflite'):
                create_tf_model(model_path, input_shape, (format_type == 'pb'))
            # elif format_type == 'h5':
            #     create_tf2_model(model_path, input_shape)
            elif format_type == 'onnx':
                create_onnx_model(model_path, input_shape)
            else:  # torch
                continue
            # Run tests with parser and compare result with runtime
            exit_status = generate_ir(cfg_path, verbose=True)
            assert exit_status
