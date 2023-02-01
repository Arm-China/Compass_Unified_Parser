import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_batchnorm_model(model_path, x_size, scale, center, training=True):
    ''' Create tensorflow model for batchnorm op.
    '''
    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')
    batchnorm_func = tf.keras.layers.BatchNormalization(scale=scale,
                                                        center=center,
                                                        gamma_initializer='GlorotUniform',
                                                        beta_initializer='GlorotUniform')
    batchnorm = batchnorm_func(x, training=training)
    y = tf.math.minimum(batchnorm, 10.0)

    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'batchnorm'
input_shape = [1, 5, 5, 6, 10]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for training in (False, True):
    for scale in (False, True):
        for center in (False, True):
            model_path = '-'.join([TEST_NAME, str(training), str(scale), str(center)]) + '.h5'
            # Create model
            create_batchnorm_model(model_path, input_shape, scale, center, training)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(
                model_path, feed_dict, verify=True)
            assert exit_status
