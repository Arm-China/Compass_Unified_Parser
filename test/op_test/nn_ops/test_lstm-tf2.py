import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_lstm_model(model_path, x1_size, time_major, return_sequences):
    ''' Create tensorflow model for lstm op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    lstm_layer = keras.layers.LSTM(4, bias_initializer='glorot_uniform', time_major=time_major,
                                   activation='sigmoid', recurrent_activation='tanh',
                                   return_sequences=return_sequences, return_state=True)
    lstm_y, lstm_h, lstm_c = lstm_layer(x1, training=False)
    out1 = tf.add(lstm_y, 10.0)
    out2 = tf.add(lstm_h, lstm_c)

    model = keras.models.Model(x1, [out1, out2])
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'lstm'
x1_shape = [24, 24, 64]

# Generate input data
feed_dict = {}
feed_dict['X1:0'] = np.random.ranf(x1_shape).astype(np.float32)

for time_major in (True, False):
    for return_sequences in (False, True):
        model_name = '-'.join([TEST_NAME, str(time_major), str(return_sequences)])
        model_path = model_name + '.h5'
        # Create model
        create_lstm_model(model_path, x1_shape, time_major, return_sequences)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', verify=True)
        assert exit_status
