# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_separable_conv2d_model(model_path, x_size, use_bias=True):
    ''' Create tensorflow model for separable_conv2d op.
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')
    separable_conv2d_func = tf.keras.layers.SeparableConv2D(2, 2, padding='valid',
                                                            use_bias=use_bias,
                                                            bias_initializer='glorot_uniform')
    separable_conv2d = separable_conv2d_func(x)
    y = tf.math.minimum(separable_conv2d, 10.0)

    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'separable_conv2d'
input_shape = [2, 5, 5, 6]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for use_bias in (True, False, ):
    model_path = '-'.join([TEST_NAME, str(use_bias)]) + '.h5'
    # Create model
    create_separable_conv2d_model(model_path, input_shape, use_bias)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, verify=True)
    assert exit_status
