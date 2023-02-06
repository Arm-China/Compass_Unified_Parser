# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_norm_model(model_path, x1_size, x2_size, axes):
    ''' Create tensorflow model for norm op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], name='X2')
    norm1 = tf.norm(x1)
    norm2 = tf.norm(x2, ord='euclidean', axis=tuple(axes))
    y = tf.add(norm1, norm2, name='Y')

    model = keras.models.Model([x1, x2], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'norm'
input_shape1 = [2, 3, 4, 4, 2]
input_shape2 = [2, 5]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
feed_dict['X2:0'] = np.random.ranf(input_shape2).astype(np.float32)

for idx, axes in enumerate([[0, 1], [-2, -1], [-1, -2]]):
    model_path = TEST_NAME + '-' + str(idx) + '.h5'
    # Create model
    create_norm_model(
        model_path, input_shape1, input_shape2, axes)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True)
    assert exit_status
