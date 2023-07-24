# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_cast_model(pb_file_path, input_size, to_dtype):
    ''' Create tensorflow model for cast op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.float32, shape=input_size, name='X1')
        x2 = tf.placeholder(tf.int32, shape=input_size, name='X2')
        # Truncate doesn't really work in tf.
        op1 = tf.raw_ops.Cast(x=x1, DstT=to_dtype, name='cast1')
        op2 = tf.raw_ops.Cast(x=x2, DstT=to_dtype, name='cast2')
        y = tf.bitwise.bitwise_or(op1, op2, name='Y')  # add may overflow

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'cast'
input_shape = [2, 3, 4, 5]
# Generate input data
feed_dict = dict()
feed_dict['X1:0'] = np.random.ranf(input_shape).astype(np.float32) * 10
feed_dict['X2:0'] = np.random.randint(-10000, 1000, input_shape).astype(np.int32)
np.save('input', feed_dict)

model_path = TEST_NAME + '.pb'
# Create model
create_cast_model(model_path, input_shape, to_dtype='int8')

expected_keywords = ['clip_mode=TRUNCATION', 'ignore_scale_zp=true']
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True,
    expected_keywords=expected_keywords)
assert exit_status
