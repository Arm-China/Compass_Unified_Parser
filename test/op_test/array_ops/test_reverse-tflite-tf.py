# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_reverse_model(model_path, input_size, axis, is_const_input=False, is_tflite=False):
    ''' Create tensorflow/tflite model for reverse op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        if is_const_input:
            inp = tf.constant(np.random.ranf(input_size), tf.float32)
        else:
            inp = x
        op1 = tf.raw_ops.ReverseV2(tensor=inp, axis=axis, name='reverse')
        y = tf.math.add(op1, x, name='Y')

        sess.run(tf.global_variables_initializer())
        if is_tflite:
            # save to tflite file
            converter = tf.lite.TFLiteConverter.from_session(sess,
                                                             input_tensors=[x], output_tensors=[y])
            tflite_model = converter.convert()
            open(model_path, 'wb').write(tflite_model)
        else:
            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ['Y'])

            # save to pb file
            with tf.gfile.GFile(model_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())


TEST_NAME = 'reverse'
input_shapes = [[10], [4, 5, 7], [4, 2, 3, 5]]
axes = [[0], [-1], [3, 1]]

# Generate input data
feed_dict = dict()
for input_shape, axis in zip(input_shapes, axes):
    feed_dict.clear()
    feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32)
    for is_const_input in (True, False):
        for is_tflite in (True, False):
            if is_tflite and len(axis) > 1:
                # tflite runtime doesn't support multiple axes for now
                # refer to: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/reverse.cc#L61
                continue
            model_format = '.tflite' if is_tflite else '.pb'
            model_path = '-'.join([TEST_NAME, str(len(input_shape)), str(is_const_input)]) + model_format
            # Create model
            create_reverse_model(model_path, input_shape, axis, is_const_input, is_tflite)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(model_path, feed_dict, verify=True)
            assert exit_status
