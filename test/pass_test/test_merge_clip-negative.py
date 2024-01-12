# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_model(model_path, x_size):
    ''' Create tensorflow model for testing merge_clip(negative test).
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')
    conv = keras.layers.Conv2D(2, 3, input_shape=x_size[1:])(x)
    thresholdedrelu = keras.layers.ThresholdedReLU(theta=0.0)(conv)
    y = tf.math.minimum(thresholdedrelu, tf.constant([-6.0]))  # all the outputs will be -6.0

    model = keras.models.Model([x, ], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'merge_clip'
input_shape = [4, 28, 28, 3]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

model_path = TEST_NAME + '.h5'
# Create model
create_model(model_path, input_shape)

# min won't be merged to clip because its const input is less than 0
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True,
    expected_keywords=['with_activation=RELU', 'layer_type=Eltwise'],
    unexpected_keywords=['layer_type=Activation'])
assert exit_status
