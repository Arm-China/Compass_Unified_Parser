# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
import os

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_unique_model(model_path, input_size):
    ''' Create tensorflow model for unique op.
    '''
    x1 = keras.Input(shape=input_size[1:], batch_size=input_size[0], name='X1')
    y = tf.unique(x1, name='Y')

    model = keras.models.Model([x1], y)
    # model.summary()
    # save to h5 file
    model.save(model_path)


def create_unique_with_counts_model(model_path, input_size):
    ''' Create tensorflow model for unique_with_counts op.
    '''
    x1 = keras.Input(shape=input_size[1:], batch_size=input_size[0], name='X1')
    y = tf.unique_with_counts(x1, name='Y')

    model = keras.models.Model([x1], y)
    # model.summary()
    # save to h5 file
    model.save(model_path)


cases = [
    'unique',
    'unique_with_counts']
input_shape = [10]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for test_name in cases:
    model_path = os.path.dirname(os.path.realpath(__file__)) + '/' + test_name + '.h5'
    # Create model
    if test_name == 'unique':
        create_unique_model(model_path, input_shape)
    else:
        create_unique_with_counts_model(model_path, input_shape)

# Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=False, verify=False)
    assert exit_status
