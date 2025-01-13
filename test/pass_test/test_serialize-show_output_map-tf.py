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
    out_tensor_name = []
    with tfv1.Session(graph=tfv1.Graph()) as sess:
        x = tfv1.placeholder(tf.float32, shape=input_shape, name='X')
        lambda_func = tf.keras.layers.Lambda(lambda a: a * 1.2 + 2.3)
        y1 = tf.math.add(lambda_func(x), 10.0, name='Y1')
        y2 = tf.math.minimum(y1, 10.0, name='Y2')
        out_tensor_name = [y1.name, y2.name]

        sess.run(tfv1.global_variables_initializer())
        constant_graph = tfv1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y2'])

        # save to pb file
        with tfv1.gfile.GFile(model_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
    return out_tensor_name


TEST_NAME = 'lambda'
input_shapes = [[4, 5], ]


for input_shape in input_shapes:
    # Generate input data
    feed_dict = {}
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.pb'
    # Create model
    out_tensor_names = create_lambda_model(model_path, input_shape)

    for output_names in (out_tensor_names, ['Y1', 'Y2']):
        exit_status = run_parser(model_path, feed_dict, output_names=output_names,
                                 expected_logs=['Output ' + output_names[0] + ' from cfg is shown as tensor Y1_0 in IR',
                                                'Output ' + output_names[1] + ' from cfg is shown as tensor Y2_0 in IR'])
        assert exit_status
