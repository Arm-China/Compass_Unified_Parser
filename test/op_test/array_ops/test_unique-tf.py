# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
import os

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_unique_model(pb_file_path, input_size):
    ''' Create tensorflow model for tile op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        y = tf.unique(x, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


def create_unique_with_counts_model(pb_file_path, input_size):
    ''' Create tensorflow model for tile op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        y = tf.unique_with_counts(x, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


cases = [
    'unique',
    'unique_with_counts']
input_shape = [10]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for test_name in cases:
    model_path = os.path.dirname(os.path.realpath(__file__)) + '/' + test_name + '.pb'
    # Create model
    if test_name == 'unique':
        create_unique_model(model_path, input_shape)
    else:
        create_unique_with_counts_model(model_path, input_shape)

# Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=False, verify=False)
    assert exit_status
