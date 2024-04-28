# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_add_model(model_path, x1_size, x2_size):
    ''' Create tensorflow model for bitwise op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], dtype=tf.int32, name='X1')
    x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], dtype=tf.int32, name='X2')
    and1 = tf.bitwise.bitwise_and(x1, x2)
    xor2 = tf.bitwise.bitwise_xor(and1, x1)
    y = tf.bitwise.bitwise_or(and1, xor2, name='Y')

    model = keras.models.Model([x1, x2], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'bitwise'
input_shape1 = [2, 16, 5]
input_shape2 = [2, 1, 5]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.randint(-10, 20, input_shape1).astype(np.int32)
feed_dict['X2:0'] = np.random.randint(-20, 30, input_shape2).astype(np.int32)

model_path = TEST_NAME + '.h5'
# Create model
create_add_model(
    model_path, input_shape1, input_shape2)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf',
    expected_keywords=['layer_type=Bitwise'],
    unexpected_keywords=['layer_type=Logical'])
assert exit_status
