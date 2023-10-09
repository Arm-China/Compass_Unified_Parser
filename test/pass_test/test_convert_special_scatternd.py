# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from utils.run import run_parser


def create_scatternd_model(pb_file_path, input_size, indices_start_from, update_size):
    ''' Create tensorflow model for convert_special_scatternd pass.
    '''
    with tfv1.Session(graph=tfv1.Graph()) as sess:
        x = tfv1.placeholder(tf.float32, shape=input_size, name='X0')
        updates = tfv1.placeholder(tf.float32, shape=update_size, name='X1')
        indices = np.ndindex(*update_size)
        add_to_indices = np.array([0] * (len(update_size)-1) + [indices_start_from])
        indices_data = np.reshape(np.array(list(indices)), update_size + [-1]) + add_to_indices
        scatternd = tfv1.tensor_scatter_nd_update(x, indices_data, updates)
        y = tf.add(scatternd, 10.0, name='Y')

        sess.run(tfv1.global_variables_initializer())
        constant_graph = tfv1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tfv1.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'scatternd'
input_shape = [2, 3, 10, 16]
update_shapes = [[1, 3, 10, 3], [2, 3, 10, 3], ]

# Generate input data
feed_dict = dict()
feed_dict['X0:0'] = np.random.ranf(input_shape).astype(np.float32)
for update_shape in update_shapes:
    feed_dict['X1:0'] = (np.random.ranf(update_shape).astype(np.float32) * 100)
    update_shape_expected = (input_shape[:-1] == update_shape[:-1])
    expected_keywords = ['Split', 'Concat'] if update_shape_expected else ['ScatterND']
    unexpected_keywords = ['ScatterND'] if update_shape_expected else ['Split', 'Concat']
    for indices_start_from in (13, 7, 0):
        model_path = '-'.join([TEST_NAME, str(indices_start_from)]) + '.pb'
        # Create model
        create_scatternd_model(model_path, input_shape, indices_start_from, update_shape)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(model_path, feed_dict, verify=True,
                                 expected_keywords=expected_keywords, unexpected_keywords=unexpected_keywords)
        assert exit_status
