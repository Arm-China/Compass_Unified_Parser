import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_flatten_model(model_path, x1_size, x2_size):
    ''' Create tensorflow model for flatten op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], name='X2')
    flatten1 = tf.keras.layers.Flatten()
    flatten2 = tf.keras.layers.Flatten('channels_first')
    y1 = flatten1(x1)
    y2 = flatten2(x2)
    y = tf.math.add(y1, y2, name='Y')

    model = keras.models.Model([x1, x2], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'flatten'
input_shape1 = [2, 1, 3]
input_shape2 = [2, 3]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
feed_dict['X2:0'] = np.random.ranf(input_shape2).astype(np.float32)

model_path = TEST_NAME + '.h5'
# Create model
create_flatten_model(
    model_path, input_shape1, input_shape2)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
