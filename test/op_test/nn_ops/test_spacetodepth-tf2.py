import numpy as np

import tensorflow as tf
from tensorflow import keras
from os.path import exists

from utils.run import run_parser


def create_space_to_depth_model(model_path, input_size, input_dtype, data_format):
    ''' Create tensorflow model for space_to_depth op.
    '''
    if exists(model_path):
        return
    dtype = tf.uint8 if input_dtype == 'uint8' else tf.float32
    x = keras.Input(shape=input_shape[1:], batch_size=input_shape[0], dtype=dtype, name='X')
    d2s = tf.nn.space_to_depth(x, 2, data_format=data_format)
    cast = tf.cast(d2s, tf.float32)
    y = tf.math.add(cast, 10.1, name='Y')

    model = keras.models.Model(x, y)
    # model.summary()
    model.save(model_path)


TEST_NAME = 'space_to_depth'
common_input_shape = [2, 4, 8, 12]
feed_dict = dict()

for data_format in ['NCHW_VECT_C', 'NCHW', 'NHWC', ]:
    if data_format == 'NCHW_VECT_C':
        input_shape = common_input_shape + [4]
        input_dtype = 'uint8'
    else:
        input_shape = common_input_shape
        input_dtype = 'float32'
    # Generate input data
    feed_dict.clear()
    feed_dict['X:0'] = np.random.randint(0, 120, input_shape).astype(input_dtype)

    model_name = TEST_NAME + '-' + data_format
    model_path = model_name + '.h5'
    # Create model
    model_created = create_space_to_depth_model(model_path, input_shape, input_dtype, data_format)

    # Model cannot be forwarded for NCHW and NCHW_VECT_C if running on a host without GPU
    verify = False if data_format.startswith('NC') else True

    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=True, verify=verify)
    assert exit_status
