# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_centercrop_model(model_path, input_size, height, width):
    ''' Create tensorflow model for centercrop op.
    '''
    x = keras.Input(shape=input_size[1:], batch_size=input_size[0], name='X')
    centercrop = tf.keras.layers.CenterCrop(height, width)
    y0 = centercrop(x)
    y = tf.math.add(y0, 10.1, name='Y')

    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'CenterCrop'
input_shape = [12, 30, 10]
target_shapes = [[6, 25], [12, 29]]  # invalid target shape: [10, 43]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for idx, target_shape in enumerate(target_shapes):
    model_path = '-'.join([TEST_NAME, str(idx)]) + '.h5'
    # Create model
    create_centercrop_model(
        model_path, input_shape, target_shape[0], target_shape[1])

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True)
    assert exit_status
