# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_SparseToDense_model(pb_file_path, sparse_shape, indices_shape, output_shape):
    ''' Create tflite model for SparseToDense op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x0 = tf.placeholder(tf.int32, shape=indices_shape, name='X0')
        x1 = tf.placeholder(tf.float32, shape=sparse_shape, name='X1')
        x2 = tf.placeholder(tf.float32, shape=[], name='X2')
        out = tf.raw_ops.SparseToDense(sparse_indices=x0,
                                       # [15],#[10, 20],
                                       output_shape=output_shape,
                                       sparse_values=x1,
                                       default_value=x2,
                                       validate_indices=False,
                                       name='SparseToDense')
        y = tf.math.add(out, 11.2, name='Y')

        sess.run(tf.global_variables_initializer())

        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x0, x1, x2], output_tensors=[y])
        tflite_model = converter.convert()
        open(pb_file_path, "wb").write(tflite_model)


TEST_NAME = 'SparseToDense'
sparse_shapes = [[3], [4], ]  # []]  # [3, 5, 10]
indices_shapes = [[3, 2], [4], ]  # []]  # [1, 3]
output_shapes = [[10, 20], [15], ]  # [15]]
feed_dict = dict()

for sparse_shape, indices_shape, output_shape in zip(sparse_shapes, indices_shapes, output_shapes):
    # Generate input data
    feed_dict.clear()
    feed_dict['X1'] = (np.random.ranf(sparse_shape) * 100).astype(np.float32)
    if indices_shape == [3, 2]:
        feed_dict['X0'] = np.array([[9, 10], [4, 18], [0, 3]]).astype(np.int32)
    elif indices_shape == []:
        feed_dict['X0'] = np.array(2).astype(np.int32)
    else:
        feed_dict['X0'] = np.array([2, 3, 4, 5]).astype(np.int32)

    feed_dict['X2'] = np.array(0.54).astype(np.float32)
    model_path = '-'.join([TEST_NAME, str(len(sparse_shape)),
                          str(indices_shape)]) + '.tflite'
    # Create model
    create_SparseToDense_model(
        model_path, sparse_shape, indices_shape, output_shape)

    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
