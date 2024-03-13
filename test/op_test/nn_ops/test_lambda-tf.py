# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.compat.v1 as tfv1

from utils.run import run_parser


def create_lambda_model(model_path, input_shape):
    ''' Create tensorflow model for lambda op.
    '''
    with tfv1.Session(graph=tfv1.Graph()) as sess:
        x = tfv1.placeholder(tf.float32, shape=input_shape, name='X')
        lambda_func = tf.keras.layers.Lambda(lambda a: a * 1.2 + 2.3)
        y = tf.math.add(lambda_func(x), 10.0, name='Y')
        # model = keras.models.Model(x, y)
        # # model.summary()

        # # save to h5 file
        # model.save(model_path)
        sess.run(tfv1.global_variables_initializer())
        constant_graph = tfv1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tfv1.gfile.GFile(model_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'lambda'
input_shapes = [[4, 5], ]


for input_shape in input_shapes:
    # Generate input data
    feed_dict = {}
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.pb'
    # Create model
    create_lambda_model(model_path, input_shape)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, verify=True)
    assert exit_status
