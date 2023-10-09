# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_reduce_sum_model(model_path, x1_size, x2_size, axes):
    ''' Create tensorflow model for reduce_mean op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], name='X2')
    reduce_sum1 = tf.math.reduce_mean(x1)
    reduce_sum2 = tf.math.reduce_mean(x2, axis=axes, keepdims=True)
    y = tf.add(reduce_sum1, reduce_sum2, name='Y')

    model = keras.models.Model([x1, x2], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'reduce_mean'
input_shape1 = [2, 3, 40]
input_shape2 = [2, 1, 2, 40, 20]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
feed_dict['X2:0'] = np.random.ranf(input_shape2).astype(np.float32)
type_is_pooling3d_str = 'layer_type=Pooling3D'

for idx, axes in enumerate([[1, 2, 3], [-2, -1, -3], ]):
    model_path = TEST_NAME + '-' + str(idx) + '.h5'
    # Create model
    create_reduce_sum_model(
        model_path, input_shape1, input_shape2, axes)
    kernel_size = int(np.prod([input_shape2[axis] for axis in axes]))
    if kernel_size <= 256:
        expected_keywords = [type_is_pooling3d_str]
        unexpected_keywords = []
    else:
        expected_keywords = []
        unexpected_keywords = [type_is_pooling3d_str]
    # Check whether reduce mean is converted to Pooling if kernel size <= 256
    exit_status = run_parser(
        model_path, feed_dict, expected_keywords=expected_keywords, unexpected_keywords=unexpected_keywords, verify=True)
    assert exit_status
