# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import onnx
from onnx import TensorProto, helper
from utils.run import run_parser


def create_softmax_model(onnx_path, input_shape, output_shape, axis, version):
    ''' Create onnx model for softmax op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)

    softmax = helper.make_node(
        'Softmax', ['X'],
        ['Y'],
        axis=axis,
    )
    graph_def = helper.make_graph(
        [softmax],  # nodes
        'softmax-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name='softmax-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


TEST_NAME = 'softmax'
input_shapes = [[4], [3, 5], [2, 3, 6]]

feed_dict = {}
for input_shape in input_shapes:
    # Generate input data
    feed_dict['X'] = (np.random.randint(-100, 200, input_shape).astype(np.float32) * 1e-3)

    for axis in (0, 1, -1, ):
        if axis >= len(input_shape):
            continue
        for opset_version in (1, 11, 13, ):
            model_path = '-'.join([TEST_NAME, str(len(input_shape)), str(axis), str(opset_version)])

            onnx_model_path = model_path + '.onnx'
            # Create onnx model
            create_softmax_model(onnx_model_path, input_shape, input_shape, axis, opset_version)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(
                onnx_model_path, feed_dict, model_type='onnx', verify=True)
            assert exit_status
