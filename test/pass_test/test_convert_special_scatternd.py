# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from utils.run import run_parser


def create_scatternd_model(pb_file_path, input_size, indices_start_from, update_size, axis):
    ''' Create tensorflow model for convert_special_scatternd pass.
    '''
    with tfv1.Session(graph=tfv1.Graph()) as sess:
        x = tfv1.placeholder(tf.float32, shape=input_size, name='X0')
        updates = tfv1.placeholder(tf.float32, shape=update_size, name='X1')
        indices = np.ndindex(*update_size)
        add_to_indices = np.array([0] * len(update_size))
        add_to_indices[axis] = indices_start_from
        indices_data = np.reshape(np.array(list(indices)) + add_to_indices, update_size + [len(update_size)])
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
axis_and_update_shapes = [(1, [2, 1, 10, 16]), (2, [2, 3, 3, 16]),
                          (1, [2, 2, 10, 16]), (2, [2, 3, 7, 16]),
                          (3, [1, 3, 10, 3]), (3, [2, 3, 10, 3])]

# Generate input data
feed_dict = dict()
feed_dict['X0:0'] = np.random.ranf(input_shape).astype(np.float32)
for axis, update_shape in axis_and_update_shapes:
    feed_dict['X1:0'] = (np.random.ranf(update_shape).astype(np.float32) * 100)
    different_axes = [idx for idx, (i_shape, u_shape) in enumerate(
        zip(input_shape, update_shape)) if i_shape != u_shape]
    update_shape_expected = (len(different_axes) == 0 or (len(different_axes) == 1 and different_axes[0] == axis))
    expected_keywords = ['Split', 'Concat'] if update_shape_expected else ['ScatterND']
    unexpected_keywords = ['ScatterND'] if update_shape_expected else ['Split', 'Concat']
    for indices_start_from in (3, 7, 0):  # tf doesn't allow negative indices value for scatternd
        if input_shape[axis] < indices_start_from \
                or (update_shape[axis] + indices_start_from) > input_shape[axis]:
            continue
        model_path = '-'.join([TEST_NAME, str(axis), str(indices_start_from)]) + '.pb'
        # Create model
        create_scatternd_model(model_path, input_shape, indices_start_from, update_shape, axis)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(model_path, feed_dict, verify=True,
                                 expected_keywords=expected_keywords, unexpected_keywords=unexpected_keywords)
        assert exit_status
