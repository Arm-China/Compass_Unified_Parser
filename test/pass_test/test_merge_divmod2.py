# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_model(pb_file_path, input_size, to_dtype):
    ''' Create tensorflow model for merge_divmod2 pass.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, shape=input_size, name='X')
        op1 = tf.cast(x, tf.float32, name='cast1')
        op2 = tf.math.divide(op1, tf.constant(50.))
        y = tf.cast(op2, to_dtype, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'merge_divmod2'
input_shape = [2, 3, 4, 5]
# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.randint(-10000, 1000, input_shape).astype(np.int32)
np.save('input', feed_dict)

model_path = TEST_NAME + '.pb'
# Create model
create_model(model_path, input_shape, to_dtype=tf.int8)

expected_keywords = ['layer_type=DivMod', 'clip_mode=TRUNCATION', 'ignore_scale_zp=true']
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True,
    expected_keywords=expected_keywords)
assert exit_status
