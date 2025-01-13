# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_transpose_reshape_model(pb_file_path, input_shape, output_shape):
    ''' Create tensorflow model for testing sink_transpose_from_reshape pass.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_shape, name='X')
        trans = tf.transpose(x, [0, 3, 1, 2])
        y = tf.reshape(trans, output_shape, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'sink_transpose_from_reshape'
feed_dict = dict()
input_shapes = ([1, 1, 1, 256], [1, 10, 20, 30], [10, 1, 1, 3])
output_shapes = ([1, 1, 256], [30, 10, 20, 1, 1], [10, 1, 3])
for idx, (input_shape, output_shape) in enumerate(zip(input_shapes, output_shapes)):
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

    model_path = '-'.join([TEST_NAME, str(idx)]) + '.pb'
    # Create model
    create_transpose_reshape_model(model_path, input_shape, output_shape)

    if np.array_equal(np.unique(np.cumprod(input_shape)), np.unique(np.cumprod(output_shape))):
        unexpected_keywords = ['layer_type=Transpose']
        expected_keywords = ['layer_type=Reshape']
    else:
        unexpected_keywords = []
        expected_keywords = ['layer_type=Reshape', 'layer_type=Transpose']
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True,
        expected_keywords=expected_keywords,
        unexpected_keywords=unexpected_keywords)
    assert exit_status
