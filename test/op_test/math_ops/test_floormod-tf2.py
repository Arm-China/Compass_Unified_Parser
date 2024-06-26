# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_floormod_model(model_path, x_shape, y_shape):
    ''' Create tensorflow model for floormod op.
    '''
    x1 = keras.Input(shape=x_shape[1:], batch_size=x_shape[0], dtype='float32', name='X1')
    x2 = keras.Input(shape=y_shape[1:], batch_size=y_shape[0], dtype='float32', name='X2')
    floormod = tf.math.floormod(x1, x2)
    y = tf.math.add(10.1, floormod, name='Y')

    model = keras.models.Model([x1, x2], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'floormod'
input_shape1 = [2, 2, 3]
input_shape2 = [2, 1]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.randint(10, 20, input_shape1).astype(np.float32)
feed_dict['X2:0'] = np.random.randint(30, 40, input_shape2).astype(np.float32)

model_path = TEST_NAME + '.h5'
# Create model
create_floormod_model(model_path, input_shape1, input_shape2)

# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
