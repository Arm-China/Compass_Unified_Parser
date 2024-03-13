# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_dropout_model(model_path, x_size, rate, training):
    ''' Create tensorflow model for dropout op.
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')

    dropout_func = tf.keras.layers.Dropout(rate, seed=10)
    dropout = dropout_func(x, training=training)
    y = tf.math.add(dropout, 10.0)
    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'dropout'
input_shapes = [[4, 5], ]


for input_shape in input_shapes:
    # Generate input data
    feed_dict = {}
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)
    for training in (None, True, False, ):
        for rate in (0.0, 0.2):
            model_path = '-'.join([TEST_NAME,
                                   str(len(input_shape)), str(training), str(rate)]) + '.h5'
            # Create model
            create_dropout_model(model_path, input_shape, rate, training)

            # Run tests with parser and compare result with runtime
            verify = True if training is None or np.isclose(rate, 0.0) else not training
            exit_status = run_parser(model_path, feed_dict, verify=verify)
            assert exit_status
