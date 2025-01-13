# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_flatten_model(onnx_path, input_size, output_size, axis, version=13):
    ''' Create onnx model for faltten op.
    '''

    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)
    if axis is None:
        flatten = helper.make_node(
            OP_NAME, ['X1'],
            ['Y']
        )
    else:
        flatten = helper.make_node(
            OP_NAME, ['X1'],
            ['Y'],
            axis=axis
        )
    graph_def = helper.make_graph(
        [flatten],  # nodes
        OP_NAME + '-model',  # name
        [X1],  # inputs
        [Y],  # outputs

    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Flatten'
model_path = OP_NAME + '.onnx'

input_shape = [[2, 3, 4, 5, 6], [4, 3], [4, 3],
               [4, 3], [4, 3], [2, 3, 4], [2, 3, 4, 5]]
output_shape = [[1, 720], [1, 12], [4, 3], [4, 3], [1, 12], [1, 24], [1, 120]]
axis = [-5, 0, None, -1, -2, 0, -4]
for i in range(len(input_shape)):
    feed_dict = {'X1': np.random.ranf(input_shape[i]).astype(np.float32)}
    create_flatten_model(model_path, input_shape[i], output_shape[i], axis[i])
    exit_status = run_parser(model_path, feed_dict,
                             model_type=None, save_output=True, verify=True)
    assert exit_status
