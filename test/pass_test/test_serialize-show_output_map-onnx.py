# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_unsqueeze_model(onnx_path, input_size, output_size, axis, version=11):
    ''' Create onnx model for unsqueeze op.
    '''

    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
    Y1 = helper.make_tensor_value_info('Y1', TensorProto.FLOAT, output_size)
    Y2 = helper.make_tensor_value_info('Y2', TensorProto.FLOAT, output_size)

    unsqueeze = helper.make_node(
        OP_NAME, ['X1'],
        ['Y1'],
        axes=axis,
        name='unsqueeze',
    )
    softmax = helper.make_node(
        'Softmax', ['Y1'],
        ['Y2'],
        axis=-1,
        name='softmax',
    )
    graph_def = helper.make_graph(
        [unsqueeze, softmax],  # nodes
        OP_NAME + '-model',  # name
        [X1],  # inputs
        [Y2],  # outputs

    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Unsqueeze'
model_path = OP_NAME + '.onnx'

input_shape = [3, 4, 5, 6]
output_shape = [3, 4, 5, 1, 1, 1]
axis = [-1, -2, -3]
feed_dict = {'X1': np.random.ranf(input_shape).astype(np.float32)}
create_unsqueeze_model(
    model_path, input_shape, output_shape, axis)
for output_names in (['Y1', 'Y2'], ['unsqueeze', 'softmax']):
    exit_status = run_parser(model_path, feed_dict, output_names=output_names,
                             expected_logs=['Output ' + output_names[0] + ' from cfg is shown as tensor unsqueeze_0 in IR',
                                            'Output ' + output_names[1] + ' from cfg is shown as tensor softmax_0 in IR'])
    assert exit_status
