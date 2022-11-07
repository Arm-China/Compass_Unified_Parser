import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_conv2d_model(model_path, x_size, use_bias=True, use_keras_op=False):
    ''' Create tensorflow model for conv2d op.
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')
    if use_keras_op:
        conv2d_func = tf.keras.layers.Conv2D(2, 2, padding='valid',
                                             use_bias=use_bias,
                                             bias_initializer='glorot_uniform')
        conv2d = conv2d_func(x)
    else:
        kernel_in = np.random.ranf([2, 2, x_size[-1], 2]) * 5
        kernel = tf.constant(kernel_in, dtype=tf.float32)
        conv2d = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    y = tf.math.minimum(conv2d, 10.0)

    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'conv2d'
input_shape = [1, 5, 5, 1]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for use_bias in (False, True,):
    for use_keras_op in (False, True,):
        model_path = '-'.join([TEST_NAME, str(use_bias), str(use_keras_op)]) + '.h5'
        # Create model
        create_conv2d_model(model_path, input_shape, use_bias, use_keras_op)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', verify=True)
        assert exit_status
