# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import tensorflow.compat.v1 as tf
from utils.run import run_parser


def create_invert_model(pb_file_path, input_size):
    ''' Create tensorflow model for invert op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, shape=input_size, name='X')
        op1 = tf.bitwise.invert(x=x, name='invert')
        y = tf.add(op1, 10, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'invert'
input_shape = [1, 3, 10, 16, 32]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.randint(0, 100, input_shape).astype(np.int32)

model_path = TEST_NAME + '.pb'
# Create model
if not os.path.exists(model_path):
    create_invert_model(model_path, input_shape)

# Set environment variable AIPUPLUGIN_PATH
os.environ["AIPUPLUGIN_PATH"] = os.path.join(os.path.dirname(__file__), 'plugins')

exit_status = run_parser(
    model_path, feed_dict, model_type='tf', save_output=False, verify=False)
assert exit_status
