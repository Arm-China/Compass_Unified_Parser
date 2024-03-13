# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_fakequant_model(pb_file_path, input_size, min_num, max_num):
    ''' Create tensorflow model for fake_quant_with_min_max_vars op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        op1 = tf.quantization.fake_quant_with_min_max_vars(x, min_num, max_num, name='fake_quant')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'fake_quant_with_min_max_vars'
input_shapes = [[1, 3, 10, 7], ]

# Generate input data
feed_dict = dict()
for idx, input_shape in enumerate(input_shapes):
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)
    min_num = -0.01
    max_num = 100.0

    model_name = '-'.join([TEST_NAME, str(idx)])
    model_path = model_name + '.pb'
    # Create model
    create_fakequant_model(
        model_path, input_shape, min_num, max_num)
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=False, verify=True)
    assert exit_status
