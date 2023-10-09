# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_onehot_model(pb_file_path, input_size, axis):
    ''' Create tensorflow model for onehot op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, shape=input_size, name='X')

        one_hot = tf.raw_ops.OneHot(
            indices=x, depth=6, on_value=5.0, off_value=0.0, axis=axis, name='onehot')
        y = tf.add(one_hot, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'one_hot'
input_shapes = [[], [3, 4], [3, 4, 5]]
axes = [0, -1]

for input_shape in input_shapes:
    for axis in axes:

        # Generate input data
        feed_dict = dict()
        feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.int32)

        model_path = TEST_NAME + str(input_shape) + str(axis) + '.pb'
        # Create model
        create_onehot_model(model_path, input_shape, axis)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', save_output=False, verify=True)
        assert exit_status
