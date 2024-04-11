# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_onehot_model(pb_file_path, input_size, axis, dtype):
    ''' Create tensorflow model for onehot op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(dtype, shape=input_size, name='X')

        y = tf.raw_ops.OneHot(
            indices=x, depth=6, on_value=5.0, off_value=0.0, axis=axis, name='Y')

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
    for dtype in ('int32', 'uint8'):
        # Generate input data
        feed_dict = dict()
        feed_dict['X:0'] = np.random.ranf(input_shape).astype(dtype)
        for axis in axes:
            model_path = '-'.join([TEST_NAME, str(len(input_shape)), str(axis)]) + '.pb'
            # Create model
            tf_dtype = tf.int32 if dtype == 'int32' else tf.uint8
            create_onehot_model(model_path, input_shape, axis, tf_dtype)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(
                model_path, feed_dict, model_type='tf', save_output=False, verify=True,
                unexpected_keywords=['layer_type=Cast'])
            assert exit_status
