# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_normalize_moments_model(pb_file_path, input_size):
    ''' Create tensorflow model for normalize_moments op.
        should not be merged.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.float32, shape=input_size, name='X1')
        x2 = tf.placeholder(tf.float32, shape=input_size, name='X2')
        x3 = tf.placeholder(tf.float32, shape=input_size, name='X3')

        mul1 = tf.math.multiply(x1, 0.043478261679410934, name='mul1')
        mul2 = tf.math.multiply(x2, 0.043478261679410934, name='mul2')
        square = tf.math.square(mul1)

        extra_add = tf.add(mul1, 10.0, name='extra_add')

        sub = tf.math.subtract(mul2, square)
        relu6 = tf.nn.relu6(sub, name='relu6')  # out
        add = tf.add(mul1, x3)
        relu = tf.nn.relu(add, name='relu')  # out

        #y = tf.add(mul, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['relu', 'relu6', 'extra_add'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


def create_normalize_moments_model1(pb_file_path, input_size):
    ''' Create tensorflow model for normalize_moments op.
        should be merge.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.float32, shape=input_size, name='X1')
        x2 = tf.placeholder(tf.float32, shape=input_size, name='X2')

        mul1 = tf.math.multiply(x1, 0.043478261679410934, name='mul1')
        mul2 = tf.math.multiply(x2, 0.043478261679410934, name='mul2')
        square = tf.math.square(mul1)

        extra_add = tf.add(mul1, 10.0, name='extra_add')

        sub = tf.math.subtract(mul2, square)
        relu6 = tf.nn.relu6(sub, name='relu6')  # out

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['relu6', 'mul1', 'extra_add'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'normalized_moments'
input_shape = [2, 3]


# Generate input data
feed_dict = dict()
feed_dict['X1:0'] = np.random.ranf(input_shape).astype(np.float32)
feed_dict['X2:0'] = np.random.ranf(input_shape).astype(np.float32)
feed_dict['X3:0'] = np.random.ranf(input_shape).astype(np.float32)

model_path = TEST_NAME + '.pb'
# Create model
create_normalize_moments_model(model_path, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tensorflow', save_output=True, verify=False)
assert exit_status


# Generate input data
feed_dict = dict()
feed_dict['X1:0'] = np.random.ranf(input_shape).astype(np.float32)
feed_dict['X2:0'] = np.random.ranf(input_shape).astype(np.float32)

model_path = TEST_NAME + '1.pb'
# Create model
create_normalize_moments_model1(model_path, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tensorflow', save_output=True, verify=False)
assert exit_status
