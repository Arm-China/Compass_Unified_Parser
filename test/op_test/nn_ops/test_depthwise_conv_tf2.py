# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_depthwise_conv_model(model_path, x_size, use_bias=True):
    ''' Create tensorflow model for depthwise_conv op.
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')
    conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=[25, 27],
        strides=[6, 6],
        padding='valid',
        depth_multiplier=6,
        data_format='channels_last',
        dilation_rate=[1, 1],
        activation=None,
        use_bias=use_bias,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros')
    conv_res = conv(x)
    y = tf.math.minimum(conv_res, 10.0)
    model = keras.models.Model(x, y)
    # model.summary()
    # save to h5 file
    model.save(model_path)


TEST_NAME = 'conv2d'
input_shape = [10, 77, 89, 5]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for use_bias in (False, True,):
    model_path = '-'.join([TEST_NAME, str(use_bias)]) + '.h5'
    # Create model
    create_depthwise_conv_model(model_path, input_shape, use_bias)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True)
    assert exit_status
