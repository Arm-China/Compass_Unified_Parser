# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_softmax_model(model_path, x_size, set_mask):
    ''' Create tensorflow model for softmax op.
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')
    softmax_func = tf.keras.layers.Softmax([0, 1])
    mask = None
    if set_mask:
        mask = np.random.randint(0, 2, x_size).astype(np.bool)
    softmax = softmax_func(x, mask)
    y = tf.math.add(softmax, 10.0)

    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'softmax'
input_shape = [4, 5, 5, 3]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for set_mask in (True, False):
    model_path = '-'.join([TEST_NAME, str(set_mask)]) + '.h5'
    # Create model
    create_softmax_model(model_path, input_shape, set_mask)

    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True)
    assert exit_status
