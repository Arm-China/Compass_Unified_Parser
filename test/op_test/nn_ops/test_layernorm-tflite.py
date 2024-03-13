# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_layernorm_model(tflite_file_path, input_size, axis):
    ''' Create tensorflow model for layernorm op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        ln_layer = tf.keras.layers.LayerNormalization(axis=axis, epsilon=0.3,
                                                      beta_initializer='ones',
                                                      gamma_initializer='glorot_uniform')
        ln = ln_layer(x, training=False)
        out = tf.math.add(ln, 10.1, name='output')

        sess.run(tf.global_variables_initializer())

        # save to tflite file
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x], output_tensors=[out])
        tflite_model = converter.convert()
        open(tflite_file_path, "wb").write(tflite_model)


TEST_NAME = 'layernorm'

input_shapes = [[3, 4, 5], [4, 5, 244, 100], [4, 5, 244, 100]]
axes = [[0, 1], [-3, -2, -1], [1, 3]]
feed_dict = dict()
for input_shape, axis in zip(input_shapes, axes):
    feed_dict.clear()
    feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32)

    model_name = '-'.join([TEST_NAME, str(len(input_shape)), str(len(axis))])
    model_path = model_name + '.tflite'
    # Create model
    create_layernorm_model(model_path, input_shape, axis)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, verify=True, expected_keywords=['LayerNorm'])
    assert exit_status
