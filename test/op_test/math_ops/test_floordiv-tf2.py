# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_floordiv_model(model_path, x_shape, y_shape, dtype='float32'):
    ''' Create tensorflow model for floordiv op.
    '''
    x1 = keras.Input(shape=x_shape[1:], batch_size=x_shape[0], dtype=dtype, name='X1')
    x2 = keras.Input(shape=y_shape[1:], batch_size=y_shape[0], dtype=dtype, name='X2')
    y0 = tf.math.floordiv(x1, x2)
    y = tf.add(y0, x1)

    model = keras.models.Model([x1, x2], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'floordiv'
input_shape1 = [2, 2, 3]
input_shape2 = [2, 1]

# Generate input data
feed_dict = {}

for dtype in ('float32', 'uint8', 'int32'):
    if dtype == 'uint8':
        feed_dict['X1:0'] = np.random.randint(10, 20, input_shape1).astype(dtype)
        feed_dict['X2:0'] = np.array([[4], [3]]).astype(dtype)
    else:
        feed_dict['X1:0'] = np.random.randint(-10, 10, input_shape1).astype(dtype)
        feed_dict['X2:0'] = np.array([[-2.3], [4.5]]).astype(dtype)
    model_path = '-'.join([TEST_NAME, dtype]) + '.h5'
    # Create model
    create_floordiv_model(model_path, input_shape1, input_shape2, dtype)

    if dtype == 'float32':
        unexpected_keywords = ['layer_type=DivMod']
        expected_keywords = ['layer_type=Div', 'layer_type=Floor']
    else:
        unexpected_keywords = []
        expected_keywords = ['layer_type=DivMod']
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, model_type='tf',
                             expected_keywords=expected_keywords,
                             unexpected_keywords=unexpected_keywords,
                             verify=True)
    assert exit_status
