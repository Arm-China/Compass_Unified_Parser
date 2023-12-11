# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import os
import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_reduce_model(model_path, x1_size, x2_size, axes):
    ''' Create tflite model for reduce op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.float32, shape=x1_size, name='X1')
        x2 = tf.placeholder(tf.float32, shape=x2_size, name='X2')
        reduce_sum = tf.math.reduce_sum(x1)
        reduce_prod = tf.math.reduce_prod(x2, axis=axes, keepdims=True)
        y = tf.add(reduce_sum, reduce_prod, name='Y')

        sess.run(tf.global_variables_initializer())
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x1, x2], output_tensors=[y])
        tflite_model = converter.convert()
        open(model_path, 'wb').write(tflite_model)


TEST_NAME = 'reduce_sum'
input_shape1 = [2, 3, 4, 4, 2]
input_shape2 = [2, 5]

# Generate input data
feed_dict = {}
feed_dict['X1'] = np.random.ranf(input_shape1).astype(np.float32)
feed_dict['X2'] = np.random.ranf(input_shape2).astype(np.float32)

for idx, axes in enumerate([1, [-2, -1], None]):
    model_path = TEST_NAME + '-' + str(idx) + '.tflite'
    # Create model
    create_reduce_model(
        model_path, input_shape1, input_shape2, axes)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, verify=True)
    assert exit_status
