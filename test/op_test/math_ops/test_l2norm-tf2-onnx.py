# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
import onnx
from onnx import TensorProto, helper
from utils.run import run_parser
from utils.common import get_feed_dict
from utils.compare import compare_data_dict


def create_lp_normalization_model(onnx_path, input_size, output_size, axis=-1, version=1):
    ''' Create onnx model for lp_normalization op.
    '''
    X = helper.make_tensor_value_info('X:0', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    lp_normalization = helper.make_node(
        'LpNormalization', ['X:0'],
        ['Y'],
        axis=axis,
        p=2,
    )
    graph_def = helper.make_graph(
        [lp_normalization],  # nodes
        'lp_norm-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name='lp_norm-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


def create_l2_normalize_model(model_path, input_size, axis, epsilon):
    ''' Create tensorflow model for l2_normalize op.
    '''
    x = keras.Input(shape=input_size[1:], batch_size=input_size[0], name='X')
    y = tf.math.l2_normalize(x, axis=axis, epsilon=epsilon, name='Y')

    model = keras.models.Model([x], y)
    # model.summary()

    # save to h5 file
    model.save(model_path)


TEST_NAME = 'l2_normalize'
input_shape = [2, 3, 4, 4, 2]

# Generate input data
feed_dict = {}
feed_dict['X:0'] = (np.random.randint(-1, 2, input_shape).astype(np.float32) * 1e-3)
# feed_dict['X:0'] = np.zeros(input_shape).astype(np.float32) # passed

for axis in (None, 0, -2, ):
    model_path = TEST_NAME + '-' + str(axis)

    if axis is not None:
        onnx_model_path = model_path + '.onnx'
        # Create onnx model
        create_lp_normalization_model(onnx_model_path, input_shape, input_shape, axis)

        # Run tests with parser and compare result with runtime
        # FIXME: Enable verify after opt fixes the issue
        exit_status = run_parser(
            onnx_model_path, feed_dict, model_type='onnx', verify=False)
        assert exit_status

    for idx, epsilon in enumerate([0.0, 1e-4, 1e-8, 1e-12]):
        tf_model_path = model_path + '-' + str(idx) + '.h5'
        # Create tf model
        create_l2_normalize_model(
            tf_model_path, input_shape, axis, epsilon)

        # Run tests with parser and compare result with runtime
        # FIXME: Enable verify after opt fixes the issue
        exit_status = run_parser(
            tf_model_path, feed_dict, model_type='tf', verify=False)
        assert exit_status
