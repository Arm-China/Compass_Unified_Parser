# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import tensorflow.compat.v1 as tf
from utils.run import run_parser
from collections import OrderedDict


def create_overlapadd_model(pb_file_path, input_size, step):
    ''' Create tensorflow model for overlap_add op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        y = tf.signal.overlap_and_add(signal=x, frame_step=step, name='y')
        relu6 = tf.nn.relu6(y, name='relu6')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['relu6'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'overlap_add'
input_shapes = [[2, 3, 4], [3, 4], [2, 3, 4, 5]]


for frame_step in range(2, 5):
    for input_shape in input_shapes:

        # Generate input data
        feed_dict = dict()
        feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

        model_path = TEST_NAME + str(frame_step)+'.pb'
        # Create model
        create_overlapadd_model(model_path, input_shape, frame_step)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, model_type='tensorflow', save_output=True, verify=True)
        assert exit_status
