import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_multiply_model(model_path, x1_size, x2_size, x3_size):
    ''' Create tensorflow model for multiply op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], name='X2')
    x3 = keras.Input(shape=x3_size[1:], batch_size=x3_size[0], name='X3')
    mul = tf.keras.layers.Multiply()([x1, x2, x3])
    y = tf.math.add(10.0, mul, name='Y')

    model = keras.models.Model([x1, x2, x3], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'multiply'
input_shape1 = [3, 1, 1, 1, 2]
input_shape2 = input_shape1
input_shape3 = input_shape1

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
feed_dict['X2:0'] = np.random.ranf(input_shape2).astype(np.float32)
feed_dict['X3:0'] = np.random.ranf(input_shape3).astype(np.float32)

model_path = TEST_NAME + '.h5'
# Create model
create_multiply_model(
    model_path, input_shape1, input_shape2, input_shape3)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
