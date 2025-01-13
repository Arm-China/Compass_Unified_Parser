# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_roll_model(pb_file_path, inp_size, roll_num, axis_value):
    ''' Create tensorflow model for where op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=inp_size, name='X')
        op1 = tf.raw_ops.Roll(input=x, shift=[roll_num], axis=[
                              axis_value], name='Roll')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'Roll'
input_shape = [2, 3, 4, 5, 6]

# Generate input data
num1 = np.random.ranf(input_shape)
feed_dict = dict()
feed_dict['X:0'] = (num1).astype(np.float32)

for axis_value in range(-len(num1.shape), len(num1.shape)):
    for roll_num in range(-10, 10):
        model_name = '-'.join([TEST_NAME, str(axis_value), str(roll_num)])
        model_path = model_name + '.pb'

        # Create model
        create_roll_model(model_path, input_shape, roll_num, axis_value)
        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', save_output=True, verify=True)
        assert exit_status
