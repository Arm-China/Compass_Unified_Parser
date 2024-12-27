# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
import os

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_unique_model(model_path, input_size):
    ''' Create tensorflow model for cast op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        y, index = tf.unique(x, name='Y')

        sess.run(tf.global_variables_initializer())
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x], output_tensors=[y, index])
        tflite_model = converter.convert()
        open(model_path, 'wb').write(tflite_model)


TEST_NAME = 'unique'
input_shape = [5]
# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

model_path = os.path.dirname(os.path.realpath(__file__)) + '/' + TEST_NAME + '.tflite'
# Create model
# tflite only supports float32, int32 and uint8
create_unique_model(model_path, input_shape)

exit_status = run_parser(model_path, feed_dict, model_type='tflite', save_output=False, verify=False)
assert exit_status
