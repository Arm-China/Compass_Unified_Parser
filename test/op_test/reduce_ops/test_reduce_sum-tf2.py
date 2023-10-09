# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_reduce_sum_model(model_path, x1_size, x2_size, axes):
    ''' Create tensorflow model for reduce_sum op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], name='X2')
    reduce_sum1 = tf.math.reduce_sum(x1)
    reduce_sum2 = tf.math.reduce_sum(x2, axis=axes, keepdims=True)
    y = tf.add(reduce_sum1, reduce_sum2, name='Y')

    model = keras.models.Model([x1, x2], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'reduce_sum'
input_shape1 = [2, 3, 4, 4, 2]
input_shape2 = [2, 5]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
feed_dict['X2:0'] = np.random.ranf(input_shape2).astype(np.float32)

for idx, axes in enumerate([1, [-2, -1], None]):
    model_path = TEST_NAME + '-' + str(idx) + '.h5'
    # Create model
    create_reduce_sum_model(
        model_path, input_shape1, input_shape2, axes)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True)
    assert exit_status
