# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_batchtospacend_model(pb_file_path, input_size, crop_is_zero=True):
    ''' Create tensorflow model for batchtospace_nd op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        spatial_len = len(input_size) - 2
        if crop_is_zero:
            crops = np.zeros([spatial_len, 2], dtype=np.int32)
        else:
            crops = np.ones([spatial_len, 2], dtype=np.int32)
        op1 = tf.raw_ops.BatchToSpaceND(input=x, block_shape=[2] * spatial_len, crops=crops, name='batchtospace_nd')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'batchtospace_nd'
input_shapes = [[32, 12, 14, 16, 20], [10, 3, 20], [32, 12, 14, 16]]

# Generate input data
feed_dict = dict()
for input_shape in input_shapes:
    feed_dict.clear()
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

    for crop_is_zero in (False, True, ):
        model_path = '-'.join([TEST_NAME, str(crop_is_zero)]) + '.pb'
        # Create model
        create_batchtospacend_model(model_path, input_shape, crop_is_zero)

        expected_keywords = []
        if len(input_shape) == 5:
            expected_keywords.append('layer_type=BatchToSpaceND')
        elif len(input_shape) == 4:
            expected_keywords.append('layer_type=BatchToSpace')
        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', verify=True,
            expected_keywords=expected_keywords)
        assert exit_status
