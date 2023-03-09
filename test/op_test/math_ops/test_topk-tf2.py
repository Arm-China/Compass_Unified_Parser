import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_topk_model(model_path, input_shape, largest):
    ''' Create tensorflow model for topk op.
    '''
    input_data = keras.Input(shape=input_shape[1:], batch_size=input_shape[0], name='X')
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
data = np.random.randint(0, 3, input_shape).astype(np.float32) * 100
# print(data)
feed_dict['X:0'] = data

model_path = TEST_NAME + '.h5'
# Create model
create_topk_model(model_path, input_shape, largest=4)

# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict, verify=True)
assert exit_status
