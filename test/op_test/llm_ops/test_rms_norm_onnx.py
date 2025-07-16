# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_rms_model(onnx_path, input_size, output_size, axis, version=23):
    ''' Create onnx model for RMSNorm op.
    '''

    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size[0])
    X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, input_size[1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)
    if axis is None:
        rms_norm = helper.make_node(
            OP_NAME, ['X1', 'X2'],
            ['Y']
        )
    else:
        rms_norm = helper.make_node(
            OP_NAME, ['X1', 'X2'],
            ['Y'],
            axis=axis
        )
    graph_def = helper.make_graph(
        [rms_norm],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs

    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME + '-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'RMSNormalization'
model_path = OP_NAME + '.onnx'

input_shape = [[[3], [1]], [[3, 4], [4]], [[3, 4, 5], [3, 4, 5]], [[3, 4, 5, 6], [6]]]
output_shape = [[3], [3, 4], [3, 4, 5], [3, 4, 5, 6]]
axis = [0, None, 1, -2]
for i in range(len(input_shape)):
    feed_dict = {
        'X1': np.random.ranf(input_shape[i][0]).astype(np.float32),
        'X2': np.random.ranf(input_shape[i][1]).astype(np.float32)
    }
    create_rms_model(model_path, input_shape[i], output_shape[i], axis[i])
    exit_status = run_parser(model_path, feed_dict,
                             model_type=None, save_output=True, verify=True)
    assert exit_status
