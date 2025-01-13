# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_model(model_path, x_size):
    ''' Create tensorflow model for converting thresholdedrelu(alpha=0)+min(x/y=const) to relu/relu6/clip.
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')
    conv = keras.layers.Conv2D(2, 3, input_shape=x_size[1:])(x)
    thresholdedrelu = keras.layers.ThresholdedReLU(theta=0.0)(conv)
    y = tf.math.minimum(thresholdedrelu, tf.constant([6.0]))

    model = keras.models.Model([x, ], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'convert_special_thresholdedrelu_to_relu'
input_shape = [4, 28, 28, 3]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

model_path = TEST_NAME + '.h5'
# Create model
create_model(model_path, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True,
    expected_keywords=['with_activation=RELU6'],
    unexpected_keywords=['layer_type=Activation', 'layer_type=Eltwise'])
assert exit_status
