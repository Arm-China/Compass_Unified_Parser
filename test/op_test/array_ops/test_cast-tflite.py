# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_cast_model(model_path, input_size, to_dtype):
    ''' Create tensorflow model for cast op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, shape=input_size, name='X')
        y = tf.cast(x, to_dtype, name='Y')

        sess.run(tf.global_variables_initializer())
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x], output_tensors=[y])
        tflite_model = converter.convert()
        open(model_path, 'wb').write(tflite_model)


TEST_NAME = 'cast'
input_shape = [5]
# Generate input data
feed_dict = dict()
feed_dict['X'] = np.array([256, 129, 300, 1, 128], np.int32)
np.save('input.npy', feed_dict)

model_path = TEST_NAME + '.tflite'
# Create model
# tflite only supports float32, int32 and uint8
create_cast_model(model_path, input_shape, to_dtype='uint8')

expected_keywords = ['clip_mode=TRUNCATION', 'ignore_scale_zp=true']
exit_status = run_parser(
    model_path, feed_dict, verify=True,
    expected_keywords=expected_keywords)
assert exit_status
