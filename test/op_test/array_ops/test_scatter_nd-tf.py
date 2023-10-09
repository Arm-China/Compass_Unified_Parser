# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_scatter_nd_model(model_path, indices_shape, updates_shape, shape, is_tflite=False):
    ''' Create tensorflow model for scatter_nd op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.int32, shape=indices_shape, name='X1')
        x2 = tf.placeholder(tf.float32, shape=updates_shape, name='X2')
        x3 = tf.constant(shape)
        op1 = tf.scatter_nd(x1, x2, x3, name='scatter_nd')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        if is_tflite:
            # save to tflite file
            converter = tf.lite.TFLiteConverter.from_session(sess,
                                                             input_tensors=[x1, x2], output_tensors=[y])
            tflite_model = converter.convert()
            open(model_path, 'wb').write(tflite_model)
        else:
            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ['Y'])

            # save to pb file
            with tf.gfile.GFile(model_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())


TEST_NAME = 'scatter_nd'
indices_shape = [1728, 4]
updates_shape = [1728]
shape = [4, 18, 24, 4]
# Generate input data
feed_dict = dict()
# feed_dict['X1'] = np.ones(indices_shape).astype(np.int32)
feed_dict['X1'] = np.tile(np.array([[0, 1, 2, 3], [2, 10, 20, 3]], np.int32), [int(1728/2), 1])
feed_dict['X2'] = np.random.ranf(updates_shape).astype(np.float32)

for model_type in ('tf', 'tflite'):
    if model_type == 'tf':
        model_path = TEST_NAME + '.pb'
        is_tflite = False
    else:
        model_path = TEST_NAME + '.tflite'
        is_tflite = True
    # Create model
    create_scatter_nd_model(model_path, indices_shape, updates_shape, shape, is_tflite)

    # ScatterND with non-constant indices could be converted to ScatterND with reduction=add
    exit_status = run_parser(
        model_path, feed_dict, model_type=model_type, expected_keywords=['ScatterND'], verify=True)
    assert exit_status
