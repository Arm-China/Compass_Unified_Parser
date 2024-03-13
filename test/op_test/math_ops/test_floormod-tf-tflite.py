# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_floormod_model(model_path, x_shape, y_shape, input_dtype=tf.float32, is_tflite_model=False):
    ''' Create tensorflow/tflite model for floormod op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(input_dtype, shape=x_shape, name='X1')
        y = tf.placeholder(input_dtype, shape=y_shape, name='X2')
        out = tf.math.floormod(x=x, y=y, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        if is_tflite_model:
            # save to tflite file
            converter = tf.lite.TFLiteConverter.from_session(sess,
                                                             input_tensors=[x, y], output_tensors=[out])
            tflite_model = converter.convert()
            open(model_path, 'wb').write(tflite_model)
        else:
            # save to pb file
            with tf.gfile.GFile(model_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())


TEST_NAME = 'floormod'
x_shape = [1, 3, 10, 6, 2]
y_shape = [1, 6, 1]

# Generate input data
feed_dict = dict()

for input_dtype in (tf.int32, tf.float32):
    if input_dtype == tf.int32:
        feed_dict['X1'] = np.random.randint(-20, 20, x_shape).astype(np.int32)
        feed_dict['X2'] = np.random.randint(1, 10, y_shape).astype(np.int32)
    else:
        feed_dict['X1'] = np.random.ranf(x_shape).astype(np.float32) * 20
        feed_dict['X2'] = np.random.ranf(y_shape).astype(np.float32) * 15
    np.save('input', feed_dict)
    for model_type in ('tflite', 'tf'):
        model_name = '-'.join([TEST_NAME, input_dtype.name, model_type])
        model_path = model_name + ('.pb' if model_type == 'tf' else '.tflite')
        # Create model
        create_floormod_model(
            model_path, x_shape, y_shape, input_dtype, model_type == 'tflite')
        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, verify=True)
        assert exit_status
