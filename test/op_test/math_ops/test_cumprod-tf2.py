# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_cumprod_model(model_path, x_size, use_keras_op=False):
    ''' Create tensorflow model for cumprod op.
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')
    x1 = tf.math.cumprod(x, axis=0, exclusive=True, reverse=True, name='cumprod')
    y = tf.math.minimum(x1, 10.0)

    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'cumprod'
input_shape = [1, 2, 3, 4]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)
use_keras_op = True

model_path = '-'.join([TEST_NAME, str(use_keras_op)]) + '.h5'
# Create model
create_cumprod_model(model_path, input_shape, use_keras_op)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
