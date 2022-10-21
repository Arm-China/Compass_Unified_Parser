# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_stridedslice_model(pb_file_path, input_size, shrink_axis_mask, new_axis_mask):
    ''' Create tensorflow model for stridedslice op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        op1 = tf.strided_slice(x, [-1, 5, 3, 4], [10, 6, 30, 30], [1, 1, 2, 3],
                               shrink_axis_mask=shrink_axis_mask,
                               new_axis_mask=new_axis_mask,
                               ellipsis_mask=0,
                               name='stridedslice')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'stridedslice'
input_shape = [10, 12, 20, 30]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for shrink_axis_mask in (0, 2, 6):
    for new_axis_mask in (0, 1, 3):
        model_name = '-'.join([TEST_NAME, str(shrink_axis_mask), str(new_axis_mask)])
        model_path = model_name + '.pb'
        # Create model
        create_stridedslice_model(
            model_path, input_shape, shrink_axis_mask, new_axis_mask)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tensorflow', save_output=False, verify=True)
        assert exit_status
