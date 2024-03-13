# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_pack_model(pb_file_path, input_size, axis_num):
    ''' Create tensorflow model for pack op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.float32, shape=input_size, name='X1')
        x2 = tf.placeholder(tf.float32, shape=input_size, name='X2')
        x3 = tf.placeholder(tf.float32, shape=input_size, name='X3')
        op1 = tf.raw_ops.Pack(values=[x1, x2, x3], axis=axis_num, name='pack')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'pack'
input_shape = [[2, 3, 4, 5], [2, 3, 4], [2, 3], [2, 3], [2]]
axis_num = [-5, -2, -1, 0, -1]

for i in range(0, len(axis_num)):
    print(i)
    # Generate input data
    feed_dict = dict()
    feed_dict['X1:0'] = np.random.ranf(input_shape[i]).astype(np.float32)
    feed_dict['X2:0'] = np.random.ranf(input_shape[i]).astype(np.float32)
    feed_dict['X3:0'] = np.random.ranf(input_shape[i]).astype(np.float32)

    model_path = TEST_NAME + '.pb'
    # Create model
    create_pack_model(model_path, input_shape[i], axis_num[i])

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=False, verify=True)
    assert exit_status
