import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_repeat_model(model_path, input_shape, repeats, axis):
    ''' Create tensorflow model for repeat op.
    '''
    x = keras.Input(shape=input_shape[1:], batch_size=input_shape[0], name='X')
    repeat = tf.repeat(x, repeats, axis=axis)
    y = tf.math.add(repeat, 12.2, name='Y')

    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'repeat'
input_shape = [2, 3, 3]
repeats = [3, 2, 2]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for axis in (1, -1, ):
    model_path = '-'.join([TEST_NAME, str(axis)]) + '.h5'
    # FIXME: opt doesn't support axis=None now. Add cases of axis=None after it's supported.
    # repeats = repeats[:1] if axis is None else repeats
    # Create model
    create_repeat_model(
        model_path, input_shape, repeats, axis)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True)
    assert exit_status
