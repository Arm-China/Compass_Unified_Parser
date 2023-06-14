import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser


def create_leaky_relu_model(pb_file_path, input_shape):
    ''' Create tf2 model for LeakyRelu op.
    '''
    x = keras.Input(shape=input_shape[1:], batch_size=input_shape[0], name='X')
    op1 = tf.nn.leaky_relu(features=x, alpha=0.33, name='leaky_relu')
    y = tf.nn.relu(op1, name='relu')

    model = keras.models.Model(x, y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'leakyrelu'
input_shape = [2, 3, 4]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

model_path = TEST_NAME + '.h5'
# Create model
create_leaky_relu_model(model_path, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict, verify=True)
assert exit_status
