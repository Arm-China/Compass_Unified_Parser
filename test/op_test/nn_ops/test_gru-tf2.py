import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_gru_model(model_path, x1_size, initial_state, time_major, reset_after, go_backwards):
    ''' Create tensorflow model for gru op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    gru_layer = keras.layers.GRU(4, bias_initializer='glorot_uniform', time_major=time_major, activation='sigmoid',
                                 reset_after=reset_after, return_sequences=True, return_state=False,
                                 go_backwards=go_backwards)
    #initial_state = tf.constant(initial_state)
    gru_y = gru_layer(x1, initial_state=initial_state, training=False)

    model = keras.models.Model(x1, gru_y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'gru'
x1_shape = [224, 224, 64]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(x1_shape).astype(np.float32)

for time_major in (True, False):
    for reset_after in (True, False):  # gruv1: reset_after=True; gruv3: reset_after=False
        for go_backwards in (True, False):
            initial_state = None  # if reset_after else np.random.ranf([224, 4]).astype(np.float32)
            model_name = '-'.join([TEST_NAME, str(time_major), str(reset_after), str(go_backwards)])
            model_path = model_name + '.h5'
            # Create model
            create_gru_model(model_path, x1_shape, initial_state, time_major, reset_after, go_backwards)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(
                model_path, feed_dict, model_type='tf', verify=True)
            assert exit_status
