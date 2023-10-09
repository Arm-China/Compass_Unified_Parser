# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_conv_transpose_model(pb_file_path, input_size, filters_shape, output_shape, strides, padding):
    ''' Create tensorflow model for conv_transpose op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        filters = np.random.ranf(filters_shape).astype(np.float32)
        if len(input_size) == 4:
            conv_trans = tf.nn.conv2d_transpose(x, filters, output_shape, strides, padding)
        else:
            conv_trans = tf.nn.conv3d_transpose(x, filters, output_shape, strides, padding)
        y = tf.math.add(conv_trans, 11.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'conv2d_transpose'
input_shapes = [[1, 200, 100, 4, 5], [1, 200, 100, 5], ]
filters_shapes = [[2, 45, 1, 48, 5], [2, 45, 48, 5], ]
output_shapes = [[1, 1000, 400, 16, 48], [1, 1000, 400, 48], ]
feed_dict = dict()

for input_shape, filters_shape, output_shape in zip(input_shapes, filters_shapes, output_shapes):
    for padding in ('SAME', ):  # 'VALID' requests size of out_backprop to match computed
        # Generate input data
        feed_dict.clear()
        feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

        model_name = '-'.join([TEST_NAME, str(len(input_shape)), padding])
        model_path = model_name + '.pb'
        strides = [5, 4] if len(input_shape) == 4 else [5, 4, 4]
        # Create model
        create_conv_transpose_model(model_path, input_shape, filters_shape, output_shape, strides, padding)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', save_output=True, verify=True)
        assert exit_status
