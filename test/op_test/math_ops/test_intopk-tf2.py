# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_intopk_model(model_path, predict_shape, target_shape):
    ''' Create tensorflow model for intopk op.
    '''
    predictions = keras.Input(shape=predict_shape[1:], batch_size=predict_shape[0], name='X1')
    targets = keras.Input(shape=target_shape[1:], batch_size=target_shape[0], dtype='int32', name='X2')
    intopk1 = tf.math.in_top_k(targets, predictions, k=5)
    intopk2 = tf.compat.v1.math.in_top_k(predictions, targets, k=5)
    y = tf.math.equal(intopk1, intopk2, name='Y')

    model = keras.models.Model([predictions, targets], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'intopk'
predict_shape = [10, 6]
target_shape = [10]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(predict_shape).astype(np.float32)
feed_dict['X2:0'] = np.random.randint(0, 6, target_shape).astype(np.int32)

model_path = TEST_NAME + '.h5'
# Create model
create_intopk_model(
    model_path, predict_shape, target_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
