# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import generate_ir
from utils.forward import opt_forward, tf_forward
from utils.compare import compare_data_dict


def create_model(pb_file_path, input_shape):
    ''' Create tf model with unknown dim.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_shape, name='X')
        y = tf.add(x, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'unknown_dim'
input_shape = [5, 20, 10, 3]
# Generate input data
feed_dict = dict()
feed_dict['X'] = (np.random.ranf(input_shape) * 10).astype(np.float32)
np.save('input', feed_dict)

output_dir = './output_dir'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for idx, model_input_shape in enumerate([None, [None, None, None, 3], ]):
    model_name = '-'.join([TEST_NAME, str(idx)])
    model_path = model_name + '.pb'
    # Generate cfg
    cfg_path = model_name + '.cfg'
    cfg_content = '''[Common]
    model_type = tf
    model_name = {0}
    input_model = {1}
    input = X
    input_shape = {2}
    output_dir = {3}
    '''.format(TEST_NAME, model_path, str(input_shape), output_dir)
    with open(cfg_path, 'w') as txt_file:
        txt_file.write(cfg_content)
    # Create model
    create_model(model_path, model_input_shape)
    # Run tests with parser and compare result with runtime
    exit_status = generate_ir(cfg_path, verbose=True)
    assert exit_status

    # opt forward
    opt_outputs_dict = opt_forward(os.path.join(output_dir, TEST_NAME + '.txt'),
                                   os.path.join(output_dir, TEST_NAME + '.bin'),
                                   feed_dict)

    # tf forward
    tf_outputs_dict = tf_forward(model_path, feed_dict)

    # compare results
    same_outputs = compare_data_dict(tf_outputs_dict, opt_outputs_dict)
    assert same_outputs, 'Expect tf forward and opt forward getting same outputs!'
