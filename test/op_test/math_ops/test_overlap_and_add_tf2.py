import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_overlap_add_model(model_path, x1_size):
    ''' Create tensorflow model for overlapadd op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    #x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], name='X2')
    add1 = tf.signal.overlap_and_add(signal=x1, frame_step=1)
    add2 = tf.keras.layers.Add()([add1, x1])
    add3 = tf.keras.layers.add([add1, add2])
    y = tf.math.add(10.0, add3, name='Y')

    model = keras.models.Model([x1], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'overlap_add'
input_shape1 = [2, 1, 3]
#input_shape2 = [1]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
#feed_dict['X2:0'] = np.random.ranf(input_shape2).astype(np.float32)

model_path = TEST_NAME + '.h5'
# Create model
create_overlap_add_model(
    model_path, input_shape1)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=False)
assert exit_status
