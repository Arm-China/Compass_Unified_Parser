# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_squeeze_model(model_path, x1_size, axis):
    ''' Create tensorflow model for squeeze op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    squeeze = tf.squeeze(x1, axis=axis)
    y = tf.math.add(squeeze, 12.2, name='Y')

    model = keras.models.Model([x1], y)
    # model.summary()
    # save to h5 file
    model.save(model_path)


def create_squeeze_model_with_squeezedims(model_path, x1_size, squeeze_dims):
    ''' Create tensorflow model for squeeze op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    squeeze = tf.compat.v1.squeeze(
        x1, axis=None, squeeze_dims=squeeze_dims, name='squeeze')
    y = tf.math.add(squeeze, 12.2, name='Y')

    model = keras.models.Model([x1], y)
    # model.summary()
    # save to h5 file
    model.save(model_path)


TEST_NAME = 'squeeze'
input_shapes = [[1, 3, 1, 2, 1], [1, 1, 5],
                [3, 1], [3, 1], [5], [3, 1, 1, 2, 1]]

for index, input_shape in enumerate(input_shapes):
    # Generate input data
    feed_dict = {}
    feed_dict['X1:0'] = np.random.ranf(input_shape).astype(np.float32)
    squeeze_dims = None
    if index == 1:
        squeeze_dims = [0, 1]
    elif index == 2:
        squeeze_dims = [1]
    elif index == 3:
        squeeze_dims = 1
    elif index == 5:
        squeeze_dims = [1, 2]

    model_path = '-'.join([TEST_NAME, str(input_shape),
                          str(squeeze_dims)]) + 'squeeze_dim.h5'
    # Create model
    create_squeeze_model_with_squeezedims(
        model_path, input_shape, squeeze_dims)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True)
    assert exit_status


for index, input_shape in enumerate(input_shapes):
    # Generate input data
    feed_dict = {}
    feed_dict['X1:0'] = np.random.ranf(input_shape).astype(np.float32)
    axis = None
    if index == 1:
        axis = [0, 1]
    elif index == 2:
        axis = [1]
    elif index == 3:
        axis = 1

    model_path = '-'.join([TEST_NAME, str(input_shape), str(axis)]) + '.h5'
    # Create model
    create_squeeze_model(model_path, input_shape, axis)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True)
    assert exit_status
