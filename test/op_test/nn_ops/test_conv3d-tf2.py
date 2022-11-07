import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_conv2d_model(model_path, x_size, use_bias=True):
    ''' Create tensorflow model for conv3d op.
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')
    conv3d_func = tf.keras.layers.Conv3D(2, 2, padding='valid',
                                         use_bias=use_bias,
                                         bias_initializer='glorot_uniform')
    conv3d = conv3d_func(x)
    y = tf.math.minimum(conv3d, 10.0)

    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'conv3d'
input_shape = [1, 5, 5, 6, 1]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for use_bias in (False, True,):
    model_path = '-'.join([TEST_NAME, str(use_bias)]) + '.h5'
    # Create model
    create_conv2d_model(model_path, input_shape, use_bias)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, verify=True)
    assert exit_status
