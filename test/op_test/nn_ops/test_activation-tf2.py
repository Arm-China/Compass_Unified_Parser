# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_activation_model(model_path, x_size, function, use_function=True):
    ''' Create tensorflow model for keras activation op.
    '''
    model_created = False
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')
    function_name = function.__name__
    print('model %s with function %s' % (model_path, function_name))
    try:
        activation = function if use_function else function_name
        activation_func = tf.keras.layers.Activation(activation)
        activation = activation_func(x)
        y = tf.math.minimum(activation, 10.0)

        model = keras.models.Model(x, y)
        # model.summary()

        # save to h5 file
        model.save(model_path)
        model_created = True
    except Exception as e:
        print('Fail to create model %s because %s' % (model_path, str(e)))
    return model_created


TEST_NAME = 'activation'
input_shape = [1, 5, 5, 6, 10]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for use_function in (True, False, ):
    for function in (tf.math.abs, tf.math.exp, tf.nn.softmax, tf.compat.v1.nn.softmax, tf.compat.v1.nn.log_softmax):  # tf.math.reduce_logsumexp
        # Some functions cannot be activation functions if model is saved to h5; while they are supported in saved model.
        # model_path = '-'.join([TEST_NAME, str(use_function), function.__name__]) + '.h5'
        model_path = '-'.join([TEST_NAME, str(use_function), function.__name__])
        print('------------%s-------------' % model_path)
        if not use_function \
                and (function.__name__.endswith('_v2')
                     or function.__name__ in ('abs', 'exp', 'log_softmax')):  # reduce_logsumexp
            # tf raises error for _v2 function and log_softmax function if use function name, for example:
            # ValueError: Unknown activation function: log_softmax_v2
            continue
        # Create model
        assert create_activation_model(model_path, input_shape, function, use_function)
        # Run tests with parser and compare result with runtime
        exit_status = run_parser(model_path, feed_dict, verify=True)
        assert exit_status
