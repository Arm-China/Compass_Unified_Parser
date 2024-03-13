# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_invert_permutation_model(pb_file_path, input_size):
    ''' Create tensorflow model for invert_permutation op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, shape=input_size, name='X')
        op1 = tf.math.invert_permutation(x, name='invert_permutation')
        y = tf.math.add(op1, x, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'invert_permutation'
input_shape = [10]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.arange(input_shape[0]).astype(np.int32)

model_path = TEST_NAME + '.pb'
# Create model
create_invert_permutation_model(model_path, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', save_output=False, verify=True)
assert exit_status
