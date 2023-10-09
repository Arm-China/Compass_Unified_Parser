# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_reshape_cast_model(pb_file_path, input_size):
    ''' Create tensorflow model for reshape_cast op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int64, shape=input_size, name='X')
        cast_1 = tf.cast(x, tf.float64)
        reshape = tf.reshape(cast_1, [1] + input_size)
        cast_2 = tf.cast(reshape, tf.float32)
        y = tf.add(cast_2, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'reshape_cast'
input_shape = [1, 3, 10]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.int64)

model_path = TEST_NAME + '.pb'
# Create model
create_reshape_cast_model(model_path, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', save_output=True, verify=True,
    expected_keywords=['int32'],
    unexpected_keywords=['int64'])
assert exit_status
