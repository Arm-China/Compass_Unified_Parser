# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_lambda_model(model_path, input_shape):
    ''' Create tensorflow model for lambda op.
    '''
    x = keras.Input(shape=input_shape[1:], batch_size=input_shape[0], name='X')
    lambda_func = tf.keras.layers.Lambda(lambda a: a * 1.2 + 2.3)
    y = tf.math.add(lambda_func(x), 10.0)
    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'lambda'
input_shapes = [[4, 5], ]


for input_shape in input_shapes:
    # Generate input data
    feed_dict = {}
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.h5'
    # Create model
    create_lambda_model(model_path, input_shape)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, verify=True)
    assert exit_status
