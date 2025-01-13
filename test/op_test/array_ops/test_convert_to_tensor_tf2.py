# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_convert_to_tensor_model(model_path, x1_size):
    ''' Create tensorflow model for convert_to_tensor op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    y1 = tf.convert_to_tensor(value=x1, dtype=tf.float32)
    y = tf.math.add(y1, 10, name='Y')

    model = keras.models.Model([x1], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'convert_to_tensor'
input_shape1 = [2, 1, 3]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)

model_path = TEST_NAME + '.h5'
# Create model
create_convert_to_tensor_model(model_path, input_shape1)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
