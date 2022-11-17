import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_cumsum_model(model_path, x1_size, x2_size):
    ''' Create tensorflow model for cumsum op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], name='X2')
    cumsum1 = tf.cumsum(x1, axis=2, exclusive=True, reverse=True)
    cumsum2 = tf.cumsum(x2)
    y = tf.math.add(cumsum1, cumsum2, name='Y')

    model = keras.models.Model([x1, x2], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'cumsum'
input_shape1 = [1, 1, 6]
input_shape2 = [1]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
feed_dict['X2:0'] = np.random.ranf(input_shape2).astype(np.float32)

model_path = TEST_NAME + '.h5'
# Create model
create_cumsum_model(
    model_path, input_shape1, input_shape2)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
