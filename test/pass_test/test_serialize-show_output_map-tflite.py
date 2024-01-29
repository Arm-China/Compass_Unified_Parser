# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_where_model(tflite_file_path, cond_shape, input_size):
    ''' Create tensorflow model for where op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        cond = tf.placeholder(tf.bool, shape=cond_shape, name='X1')
        x_true = tf.placeholder(tf.float32, shape=input_size, name='X2')
        x_false = tf.placeholder(tf.float32, shape=input_size, name='X3')
        op1 = tf.where(cond, x_true, x_false, name='where')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())

        # save to tflite file
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[cond, x_true, x_false],
                                                         output_tensors=[y])
        tflite_model = converter.convert()
        open(tflite_file_path, 'wb').write(tflite_model)
    return


TEST_NAME = 'where'
cond_shape = [20]
input_shape = [20, 20, 30]

# Generate input data
feed_dict = dict()
feed_dict['X1'] = np.random.randint(0, 2, cond_shape).astype(bool)
feed_dict['X2'] = (np.random.ranf(input_shape) * 100).astype(np.float32)
feed_dict['X3'] = (np.random.ranf(input_shape) * 10).astype(np.float32)

model_path = 'where.tflite'
create_where_model(model_path, cond_shape, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, output_names=['where', 'Y'],
    expected_logs=['Output where from cfg is shown as tensor where_0 in IR',
                   'Output Y from cfg is shown as tensor Y_0 in IR'])
assert exit_status
