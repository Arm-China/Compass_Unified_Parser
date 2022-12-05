import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_model(model_path, x1_size, x2_size, x3_size):
    ''' Create tensorflow model for tf2 multidirectional_broadcasting pass.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], name='X2')
    x3 = keras.Input(shape=x3_size[1:], batch_size=x3_size[0], name='X3')
    avg = tf.keras.layers.Average()([x1, x2, x3])
    mul = tf.keras.layers.Multiply()([x1, x2, x3])
    maximum = tf.keras.layers.Maximum()([x1, x2, x3])
    y = tf.keras.layers.Add()([avg, maximum, mul])

    model = keras.models.Model([x1, x2, x3], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'broadcasting'
input_shape1 = [3, 1, 1, 1, 2]
input_shape2 = [3, 4, 2]
input_shape3 = [3, 5, 1, 2]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
feed_dict['X2:0'] = np.random.ranf(input_shape2).astype(np.float32)
feed_dict['X3:0'] = np.random.ranf(input_shape3).astype(np.float32)

model_path = TEST_NAME + '.h5'
# Create model
create_model(
    model_path, input_shape1, input_shape2, input_shape3)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
