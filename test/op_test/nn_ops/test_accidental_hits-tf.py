# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_accidental_hits_model(pb_file_path, input_size):
    ''' Create tensorflow model for accidental_hits op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int64, shape=input_size, name='X_0')
        candidate = tf.placeholder(tf.int64, shape=[3], name='X_1')
        op1 = tf.nn.compute_accidental_hits(x, candidate, x.shape[-1], name='accidental_hits')
        op2 = tf.math.add(op1[0], tf.cast(op1[1], tf.int32), name='add_1')
        op3 = tf.cast(op2, tf.float32, name='add_2')
        y = tf.math.add(op3, op1[2], name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'accidental_hits'
input_shapes = [[10, 20], ]
feed_dict = dict()

for input_shape in input_shapes:
    # Generate input data
    feed_dict.clear()
    feed_dict['X_0:0'] = np.random.randint(0, 3, input_shape).astype(np.int64)
    feed_dict['X_1:0'] = np.random.randint(0, 3, [3]).astype(np.int64)

    model_name = TEST_NAME + '-' + str(len(input_shape))
    model_path = model_name + '.pb'
    # Create model
    create_accidental_hits_model(model_path, input_shape)

    # Run tests with parser and compare result with runtime
    # Set verify to False as this op's output shape is not fixed.
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=True, verify=False)
    assert exit_status
