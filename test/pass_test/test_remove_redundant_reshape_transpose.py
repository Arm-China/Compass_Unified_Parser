# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_reshape_transpose_model2(pb_file_path, input_size):
    ''' Create tensorflow model for reshape_transpose op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        mul = tf.math.multiply(x, 1.23, name='mul')
        reshape1 = tf.reshape(mul, [2, 3])
        trans = tf.transpose(reshape1, [1, 0])
        reshape2 = tf.reshape(trans, [6])
        y = tf.add(reshape2, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


def create_reshape_transpose_model(pb_file_path, input_size):
    ''' Create tensorflow model for reshape_transpose op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        mul = tf.math.multiply(x, 1.23, name='mul')
        reshape1 = tf.reshape(mul, [1] + input_size)
        trans = tf.transpose(reshape1, [1, 0, 2, 3])
        reshape2 = tf.reshape(trans, input_size)
        y = tf.add(reshape2, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'reshape_transpose'
input_shape = [1, 3, 10]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

model_path = TEST_NAME + '.pb'
# Create model
create_reshape_transpose_model(model_path, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', save_output=True, verify=True,
    expected_keywords=['BatchNorm'],
    unexpected_keywords=['Reshape', 'Transpose'])
assert exit_status

TEST_NAME = 'reshape_transpose2'
input_shape = [6]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

model_path = TEST_NAME + '.pb'
# Create model
create_reshape_transpose_model2(model_path, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', save_output=True, verify=True,
    expected_keywords=['BatchNorm', 'Reshape', 'Transpose'])
assert exit_status
