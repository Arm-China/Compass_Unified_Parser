# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import os
import numpy as np
import tensorflow.compat.v1 as tf
from utils.run import run_parser


def create_model(pb_file_path, input_size):
    ''' Create tensorflow model for testing subgraph plugin.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, shape=input_size, name='INPUT')
        constant_y = np.random.randint(10, 20, [1] * len(input_size)).astype(np.int32)
        op1 = tf.math.divide(x, constant_y, name='op_div')
        op2 = tf.math.multiply(op1, constant_y, name='op_mul')
        op3 = tf.math.subtract(op2, 15, name='op_sub')
        y = tf.add(op3, 10, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'preprocess_name'
input_shape = [1, 4, 30, 70]

# Generate input data
feed_dict = dict()
feed_dict['INPUT:0'] = np.random.randint(0, 100, input_shape).astype(np.int32)

model_path = TEST_NAME + '.pb'
# Create model
create_model(model_path, input_shape)

# Set environment variable AIPUPLUGIN_PATH
os.environ['AIPUPLUGIN_PATH'] = os.path.join(os.path.dirname(__file__), 'plugins')

exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=False,
    expected_keywords=['layer_type=TfPreprocess', 'bias', 'axis=3', 'layer_top_shape=[[1,4,200,200]]'],
    unexpected_keywords=['layer_top_shape=[[1,4,30,70]]'])
assert exit_status
