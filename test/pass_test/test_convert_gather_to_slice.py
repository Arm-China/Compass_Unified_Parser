# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_gather_model(onnx_path, input_size, output_size, indices, axis, version=14):
    ''' Create onnx model for gather op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    if isinstance(indices, list):
        const_node = helper.make_node('Constant', [], ['Indices'], value_ints=indices)
    else:
        const_node = helper.make_node('Constant', [], ['Indices'], value_int=indices)
    gather_tensor = helper.make_tensor_value_info('Gather', TensorProto.INT64, output_size)
    gather_node = helper.make_node(
        OP_NAME,
        inputs=['X', 'Indices'],
        outputs=['Y'],
        axis=axis
    )
    graph_def = helper.make_graph(
        [const_node, gather_node],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Gather'
input_shape = [3, 3, 2, 4, 5]

# Generate input data
feed_dict = dict()
input_data = np.random.ranf(input_shape).astype(np.float32) * 100
feed_dict['X'] = input_data

for idx, indices in enumerate([[-1], [0], -1, 0, 2]):
    for axis in (-1, 0, 1, 3):
        model_name = '-'.join([OP_NAME, str(idx), str(axis)])
        model_path = model_name + '.onnx'
        output_shape = list(np.take(input_data, np.array(indices), axis).shape)
        # Create model
        create_gather_model(
            model_path, input_shape, output_shape, indices, axis, 14)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, verify=True,
            expected_keywords=['layer_type=Slice'],
            unexpected_keywords=['layer_type=Gather'])
        assert exit_status
