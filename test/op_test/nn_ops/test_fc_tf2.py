import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_fc_model(model_path, x_size, activation_type, use_bias):
    ''' Create tensorflow model for fc op.
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')

    dense_x = tf.keras.layers.Dense(
        units=7,
        activation=activation_type,
        use_bias=use_bias,
        # kernel_initializer='glorot_uniform',
        kernel_initializer='random_normal',
        # bias_initializer='zeros',
        bias_initializer='glorot_uniform',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    )(x)
    y = tf.math.add(dense_x, 10.0)
    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'fc'
input_shapes = [[4, 5], [4, 5, 6]]


for input_shape in input_shapes:
    for activation_type in ('LeakyReLU', 'leaky_relu'):
        for use_bias in (True, False):
            # Generate input data
            feed_dict = {}
            feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

            model_path = '-'.join([TEST_NAME,
                                   str(activation_type), str(use_bias)]) + '.h5'
            # Create model
            create_fc_model(model_path, input_shape, activation_type, use_bias)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(
                model_path, feed_dict, verify=True)
            assert exit_status
