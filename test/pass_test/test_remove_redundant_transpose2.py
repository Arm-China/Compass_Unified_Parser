# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_transpose_model(pb_file_path, input_size, input_is_nchw=True):
    ''' Create tensorflow model for pass remove_redundant_transpose2.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        mul = tf.math.multiply(x, 1.23, name='mul')
        if input_is_nchw:
            trans1_perm = [0, 2, 3, 1]
            trans2_perm = [0, 3, 1, 2]
            reshape_dim = [input_size[0]] + [-1, 1] + [input_size[1]]  # NHWC
        else:
            trans1_perm = [0, 3, 1, 2]
            trans2_perm = [0, 2, 3, 1]
            reshape_dim = [input_size[0], input_size[-1]] + [-1, 1]  # NCHW
        trans1 = tf.transpose(mul, trans1_perm)
        reshape = tf.reshape(trans1, reshape_dim)
        trans2 = tf.transpose(reshape, trans2_perm, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'remove_redundant_transpose2'
input_shape = [2, 3, 4, 10]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for input_is_nchw in (True, False):
    model_path = '-'.join([TEST_NAME, str(input_is_nchw)]) + '.pb'
    # Create model
    create_transpose_model(model_path, input_shape, input_is_nchw)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True,
        expected_keywords=['layer_type=Reshape'],
        unexpected_keywords=['layer_type=Transpose'])
    assert exit_status
