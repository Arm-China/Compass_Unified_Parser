# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_clip_model(pb_file_path, x1_size, x2_size):
    ''' Create tensorflow model for clip op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.float32, shape=x1_size, name='X1')
        x2 = tf.placeholder(tf.float32, shape=x2_size, name='X2')
        op1 = tf.raw_ops.ClipByValue(t=x1, clip_value_min=-1.5, clip_value_max=2.3, name='clip')
        y = tf.math.add(op1, x2, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'clip'
input_shape1 = [2, 1, 1, 1, 2]
input_shape2 = [2, 1, 2]

# Generate input data
feed_dict = dict()
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
feed_dict['X2:0'] = np.random.ranf(input_shape2).astype(np.float32)

model_path = TEST_NAME + '.pb'
# Create model
create_clip_model(
    model_path, input_shape1, input_shape2)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
