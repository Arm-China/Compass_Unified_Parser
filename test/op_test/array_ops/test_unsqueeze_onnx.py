# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_unsqueeze_model(onnx_path, input_size, output_size, axis, version=11):
    ''' Create onnx model for unsqueeze op.
    '''

    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    unsqueeze = helper.make_node(
        OP_NAME, ['X1'],
        ['Y'],
        axes=axis
    )
    graph_def = helper.make_graph(
        [unsqueeze],  # nodes
        OP_NAME + '-model',  # name
        [X1],  # inputs
        [Y],  # outputs

    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Unsqueeze'
model_path = OP_NAME + '.onnx'

input_shape = [[3, 4, 5, 6], [3, 4], [3, 4, 5], [3]]
output_shape = [[3, 1, 4, 1, 5, 6, 1], [
    1, 3, 4, 1], [3, 4, 5, 1, 1, 1], [1, 3]]
axis = [[1, -1, 3], [0, -1], [-1, -2, -3], [0]]
for i in range(len(input_shape)):
    feed_dict = {'X1': np.random.ranf(input_shape[i]).astype(np.float32)}
    create_unsqueeze_model(
        model_path, input_shape[i], output_shape[i], axis[i])
    exit_status = run_parser(model_path, feed_dict,
                             model_type=None, save_output=True, verify=True)
    assert exit_status
