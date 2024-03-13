# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_matmul_model(model_path, x1_size, x2_size, transpose_a, adjoint_b):
    ''' Create tensorflow model for matmul op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], dtype='int32', name='X1')
    x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], dtype='int32', name='X2')
    matmul = tf.linalg.matmul(x1, x2, transpose_a=transpose_a, adjoint_b=adjoint_b)
    y = tf.math.add(10, matmul, name='Y')

    model = keras.models.Model([x1, x2], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'matmul'
input_shape1 = [3, 3]
input_shape2 = input_shape1

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.randint(10, 20, input_shape1).astype(np.int32)
feed_dict['X2:0'] = np.random.randint(30, 40, input_shape2).astype(np.int32)

for transpose_a in (True, False):
    for adjoint_b in (True, False):
        model_path = '-'.join([TEST_NAME, str(transpose_a), str(adjoint_b)]) + '.h5'
        # Create model
        create_matmul_model(
            model_path, input_shape1, input_shape2, transpose_a, adjoint_b)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', verify=True)
        assert exit_status
