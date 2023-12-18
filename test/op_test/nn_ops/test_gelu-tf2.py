# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_gelu_model(model_path, x_size, approximate):
    ''' Create tensorflow model for gelu op.
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')

    gelu = tf.nn.gelu(x, approximate=approximate)
    y = tf.math.add(gelu, 10.0)
    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'gelu'
input_shapes = [[4, 5], ]
feed_dict = {}
for input_shape in input_shapes:
    # Generate input data
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)
    for approximate in (True, False, ):
        model_path = '-'.join([TEST_NAME,
                               str(len(input_shape)), str(approximate)]) + '.h5'
        create_gelu_model(model_path, input_shape, approximate)
        exit_status = run_parser(model_path, feed_dict)
        assert exit_status
