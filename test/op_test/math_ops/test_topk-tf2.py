import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_topk_model(model_path, input_shape, largest, input_dtype):
    ''' Create tensorflow model for topk op.
    '''
    input_data = keras.Input(shape=input_shape[1:], batch_size=input_shape[0], dtype=input_dtype, name='X')
    values1, indices1 = tf.math.top_k(input_data, k=largest, sorted=True)
    # FIXME: Check the similarity of sorted=False after opt fixes the sorted issue
    # values2, indices2 = tf.math.top_k(input_data, k=largest, sorted=False)
    values2, indices2 = tf.math.top_k(input_data, k=largest, sorted=True)
    y1 = tf.math.add(values1, values2, name='Y1')
    y2 = tf.math.add(indices1, indices2, name='Y2')

    model = keras.models.Model([input_data], [y1, y2])
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'topk'
input_shape = [3, 5, 6]

# Generate input data
feed_dict = {}
data = np.random.randint(0, 3, input_shape) * 100
for input_dtype in ('float16', 'float32'):
    feed_dict['X:0'] = data.astype(input_dtype)

    model_path = '-'.join([TEST_NAME, input_dtype]) + '.h5'
    # Create model
    create_topk_model(model_path, input_shape, 4, input_dtype)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, verify=True)
    assert exit_status
