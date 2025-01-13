# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_gather_model(model_path, x1_size, x2_size, axis, batch_dims):
    ''' Create tensorflow model for gather op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], dtype='int32', name='X2')
    gather = tf.gather(x1, x2, axis=axis, batch_dims=batch_dims)
    y = tf.math.add(gather, 12.2, name='Y')

    model = keras.models.Model([x1, x2], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'gather'
input_shape1 = [3, 4, 5]
input_shape2 = [3]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
feed_dict['X2:0'] = np.random.randint(0, 3, input_shape2).astype(np.int32)

for axis in (1, -1):
    for batch_dims in (0, 1):
        model_path = '-'.join([TEST_NAME, str(axis), str(batch_dims)]) + '.h5'
        # Create model
        create_gather_model(
            model_path, input_shape1, input_shape2, axis, batch_dims)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', verify=True)
        assert exit_status
