# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_mul_add_model(pb_file_path, input_size, const_is_scalar=True):
    ''' Create tensorflow model for mul_add op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        mul_const = 1.23 if const_is_scalar else np.random.ranf([input_size[1], 1, 1]).astype(np.float32)
        mul = tf.math.multiply(x, mul_const, name='mul')
        add_const = 10.0 if const_is_scalar else np.random.ranf([input_size[1], 1, 1]).astype(np.float32)
        y = tf.add(mul, add_const, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'mul_add'
input_shape = [1, 3, 10, 16]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for const_is_scalar in (True, False):
    model_path = '-'.join([TEST_NAME, str(const_is_scalar)]) + '.pb'
    # Create model
    create_mul_add_model(model_path, input_shape, const_is_scalar)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tensorflow', verify=True)
    assert exit_status
