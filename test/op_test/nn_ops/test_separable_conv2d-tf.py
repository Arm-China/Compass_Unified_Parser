# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_separable_conv2d_model(pb_file_path, input_shape, depthwise_filter_shape, pointwise_filter_shape, strides, padding):
    ''' Create tensorflow model for separable_conv2d op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_shape, name='X')
        depthwise_filter = np.random.ranf(depthwise_filter_shape).astype(np.float32)
        pointwise_filter = np.random.ranf(pointwise_filter_shape).astype(np.float32)
        conv_trans = tf.nn.separable_conv2d(x, depthwise_filter, pointwise_filter, strides, padding)
        y = tf.math.add(conv_trans, 11.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'separable_conv2d'
input_shapes = [[1, 200, 100, 2], ]
depthwise_filter_shapes = [[4, 2, 2, 3], ]
pointwise_filter_shapes = [[1, 1, 6, 5], ]
feed_dict = dict()

for input_shape, depthwise_filter_shape, pointwise_filter_shape in zip(input_shapes, depthwise_filter_shapes, pointwise_filter_shapes):
    for padding in ('SAME', 'VALID'):
        # Generate input data
        feed_dict.clear()
        feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

        model_path = '-'.join([TEST_NAME, str(len(input_shape)), padding]) + '.pb'
        # Current tf implementation only supports equal length strides in the row and column dimensions. [Op:DepthwiseConv2dNative]
        strides = [1, 5, 5, 1]
        # Create model
        create_separable_conv2d_model(model_path, input_shape, depthwise_filter_shape,
                                      pointwise_filter_shape, strides, padding)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(model_path, feed_dict, model_type='tf', verify=True)
        assert exit_status
