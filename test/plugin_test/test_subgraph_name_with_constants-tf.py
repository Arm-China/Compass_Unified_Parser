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
        x = tf.placeholder(tf.int32, shape=input_size, name='X')
        constant_y = np.random.randint(10, 20, input_size).astype(np.int32)
        op1 = tf.math.divide(x, constant_y, name='div')
        op2 = tf.math.multiply(op1, constant_y, name='mul')
        op3 = tf.math.subtract(op2, 15, name='sub')
        y = tf.add(op3, 10, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'subgraph_name'
input_shape = [4, 3, 7]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.randint(0, 100, input_shape).astype(np.int32)

model_path = TEST_NAME + '.pb'
# Create model
create_model(model_path, input_shape)

# Set environment variable AIPUPLUGIN_PATH
os.environ['AIPUPLUGIN_PATH'] = os.path.join(os.path.dirname(__file__), 'plugins')

exit_status = run_parser(
    model_path, feed_dict, model_type='tf', save_output=False, verify=False,
    expected_keywords=['MySubGraph', 'div_y_offset', 'bias_offset'],
    unexpected_keywords=['Constant'])
assert exit_status
