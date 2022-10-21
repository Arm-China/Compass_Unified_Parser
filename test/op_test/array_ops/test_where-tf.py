# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_where_model(pb_file_path, cond_shape, input_size):
    ''' Create tensorflow model for where op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        cond = tf.placeholder(tf.bool, shape=cond_shape, name='X1')
        x_true = tf.placeholder(tf.float32, shape=input_size, name='X2')
        x_false = tf.placeholder(tf.float32, shape=input_size, name='X3')
        op1 = tf.where(cond, x_true, x_false, name='where')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'where'
cond_shapes = [[20, 20, 30], ]
input_shape = [20, 20, 30]

# Generate input data
feed_dict = dict()
feed_dict['X2:0'] = (np.random.ranf(input_shape) * 100).astype(np.float32)
feed_dict['X3:0'] = (np.random.ranf(input_shape) * 10).astype(np.float32)

for idx, cond_shape in enumerate(cond_shapes):
    model_name = '-'.join([TEST_NAME, str(idx)])
    model_path = model_name + '.pb'

    # Create model
    create_where_model(model_path, cond_shape, input_shape)

    feed_dict['X1:0'] = np.random.randint(0, 2, cond_shape).astype(bool)
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=True, verify=True)
    assert exit_status
