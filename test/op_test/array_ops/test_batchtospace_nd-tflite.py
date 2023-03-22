# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_batchtospacend_model(pb_file_path, input_size, crop_is_zero=True):
    ''' Create tflite model for batchtospace_nd op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        if crop_is_zero:
            crops = np.zeros([2, 2], dtype=np.int32).tolist()
        else:
            crops = [[5, 0], [2, 3]]
        op1 = tf.raw_ops.BatchToSpaceND(input=x, block_shape=[2, 2], crops=crops, name='batchtospace_nd')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x], output_tensors=[y])
        tflite_model = converter.convert()
        open(pb_file_path, 'wb').write(tflite_model)


TEST_NAME = 'batchtospace_nd'
input_shape = [28, 16, 18, 1]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32)

for crop_is_zero in (False, True, ):
    model_path = '-'.join([TEST_NAME, str(crop_is_zero)]) + '.tflite'
    # Create model
    create_batchtospacend_model(model_path, input_shape, crop_is_zero)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tflite', verify=True)
    assert exit_status
