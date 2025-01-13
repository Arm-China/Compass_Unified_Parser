# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_multi_transpose_model(pb_file_path, input_size):
    ''' Create tensorflow model for multi_transpose op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        mul = tf.math.multiply(x, 1.23, name='mul')
        trans1 = tf.transpose(mul, [1, 2, 3, 0])
        trans2 = tf.transpose(trans1, [1, 0, 2, 3])
        trans3 = tf.transpose(trans1, [0, 2, 3, 1])
        y = tf.add(tf.math.reduce_sum(trans2), tf.math.reduce_sum(trans3), name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'multi_transpose'
input_shape = [1, 3, 4, 10]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

model_path = TEST_NAME + '.pb'
# Create model
create_multi_transpose_model(model_path, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', save_output=True, verify=True,
    expected_keywords=['perm=[1,3,0,2]'],
    unexpected_keywords=['perm=[1,2,3,0]'])
assert exit_status
