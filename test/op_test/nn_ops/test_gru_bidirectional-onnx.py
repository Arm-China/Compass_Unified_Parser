# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_gru_model(onnx_path, input_size, output_size, linear_before_reset, version):
    X = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)
    Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, output_size[1:])
    Y1 = helper.make_tensor_value_info('Y1', TensorProto.FLOAT, output_size)
    Y2 = helper.make_tensor_value_info('Y2', TensorProto.FLOAT, output_size[1:])

    const_nodes = []
    tensors = []
    wrb_tensor_names = ['W', 'R', 'B']
    wrb_shapes = [[2, 180, 60], [2, 180, 60], [2, 360]]
    for tensor_name, tensor_shape in zip(wrb_tensor_names, wrb_shapes):
        tensor = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, tensor_shape)
        tensors.append(tensor)
        const_value = np.random.ranf(tensor_shape).astype(np.float32)
        const_tensor = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, const_value.shape)
        const_node = helper.make_node(
            'Constant',
            [],
            [tensor_name],
            value=helper.make_tensor(
                name=tensor_name + '_value',
                data_type=TensorProto.FLOAT,
                dims=const_value.shape,
                vals=const_value,
            )
        )
        const_nodes.append(const_node)
    gru = helper.make_node(
        'GRU', ['X1'] + wrb_tensor_names,
        ['Y', 'Y_h'],
        direction='bidirectional',
        hidden_size=60,
        linear_before_reset=linear_before_reset,
    )
    add = helper.make_node(
        'Add', ['Y', 'Y'],
        ['Y1'],
    )
    add_h = helper.make_node(
        'Add', ['Y_h', 'Y_h'],
        ['Y2'],
    )
    nodes_list = const_nodes + [gru, add, add_h]
    outputs_list = [Y1, Y2]

    graph_def = helper.make_graph(
        nodes_list,  # nodes
        'gru-model',  # name
        [X],  # inputs
        outputs_list,  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name='gru-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


input_shape = [10, 30, 60]
output_shape = [10, 2, 30, 60]
feed_dict = {'X1': np.random.ranf(input_shape).astype(np.float32)}

for version in [7, 14]:
    for linear_before_reset in [0, 1]:
        model_path = '-'.join(['gru', str(version), str(linear_before_reset)]) + '.onnx'
        create_gru_model(model_path, input_shape, output_shape, linear_before_reset, version)
        # Run tests with parser and compare result with runtime
        exit_status = run_parser(model_path, feed_dict, verify=True)
        assert exit_status
