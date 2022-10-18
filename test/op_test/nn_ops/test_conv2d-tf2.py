import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_add_model(model_path, x_size):
    ''' Create tensorflow model for conv2d op.
    '''
    if os.path.exists(model_path):
        print('Model %s already exists! Reuse it!' % model_path)
        return

    x = keras.Input(shape=x_size[1:], batch_size=x_size[0], name='X')
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

model_path = TEST_NAME + '.h5'
# Create model
create_add_model(model_path, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
