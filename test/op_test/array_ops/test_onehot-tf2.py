# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_one_hot_model(model_path, x_size, axis):
    ''' Create tensorflow model for one_hot op.
    '''
    x = keras.Input(shape=x_size[1:],
                    batch_size=x_size[0], dtype='int32', name='X')
    one_hot = tf.one_hot(x, 6, axis=axis)
    y = tf.math.add(one_hot, 1, name='Y')
    model = keras.models.Model([x], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


def create_one_hot_scalar_model(model_path, x_size, axis):
    ''' Create tensorflow model for one_hot scalar op.
    '''
    x = keras.Input(shape=x_size[1:],
                    batch_size=x_size[0], dtype='int32', name='X')
    squeeze = tf.squeeze(x, name='squeeze')
    one_hot = tf.one_hot(squeeze, 6, axis=axis)
    y = tf.math.add(one_hot, 1, name='Y')
    model = keras.models.Model([x], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'one_hot'
input_shape = [1]

# Generate input data
for axis in (0, -1):
    feed_dict = {}
    feed_dict['X:0'] = np.array(3).astype(np.int32)

    model_path = '-'.join([TEST_NAME, str(input_shape) +
                          'scalar' + str(axis)]) + '.h5'
    # Create model
    create_one_hot_scalar_model(model_path, input_shape, axis)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True)
    assert exit_status


TEST_NAME = 'one_hot'
input_shapes = [[3], [3, 4, 5]]

# Generate input data
for axis in (0, -1):
    for input_shape in input_shapes:
        feed_dict = {}
        feed_dict['X:0'] = np.random.randint(-1,
                                             6, input_shape).astype(np.int32)

        model_path = '-'.join([TEST_NAME,
                              str(input_shape) + str(axis)]) + '.h5'
        # Create model
        create_one_hot_model(
            model_path, input_shape, axis)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', verify=True)
        assert exit_status
