# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_reverse_model(pb_file_path, input_size, axis, is_const_input=False):
    ''' Create tensorflow model for reverse op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        if is_const_input:
            inp = tf.constant(np.random.ranf(input_size), tf.float32)
        else:
            inp = x
        op1 = tf.raw_ops.ReverseV2(tensor=inp, axis=axis, name='reverse')
        y = tf.math.add(op1, x, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'reverse'
input_shapes = [[10], [4, 5], [4, 2, 3, 5]]
axes = [[0], [-1], [3]]  # TODO: Consider multiple axis, like [1, 3]

# Generate input data
feed_dict = dict()
for input_shape, axis in zip(input_shapes, axes):
    feed_dict.clear()
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)
    for is_const_input in (True, False):
        model_path = '-'.join([TEST_NAME, str(len(input_shape)), str(is_const_input)]) + '.pb'
        # Create model
        create_reverse_model(model_path, input_shape, axis, is_const_input)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', verify=True)
        assert exit_status
