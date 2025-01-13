# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_div_model(onnx_path, input_size, output_size, onnx_dtype, version=13):
    ''' Create onnx model for div op.
    '''
    X1 = helper.make_tensor_value_info('X1', onnx_dtype, input_size)
    X2 = helper.make_tensor_value_info('X2', onnx_dtype, input_size)
    Y0 = helper.make_tensor_value_info('Y0', onnx_dtype, output_size)
    Y = helper.make_tensor_value_info('Y', onnx_dtype, output_size)

    div = helper.make_node(
        OP_NAME, ['X1', 'X2'],
        ['Y0'],
    )
    add = helper.make_node(
        'Add', ['Y0', 'Y0'],
        ['Y']
    )
    graph_def = helper.make_graph(
        [div, add],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Div'

input_shapes = [[3, 2], [1, 2, 3, 4, 5], [3, 4, 5, 6, 7, 8]]

for input_shape in input_shapes:
    X1_data = np.random.randint(-10, 20, input_shape)
    X2_data = np.random.randint(-20, -10, input_shape)
    for dtype in ('float32', 'int32'):
        model_path = '-'.join([OP_NAME, str(len(input_shape)), dtype]) + '.onnx'
        onnx_dtype = TensorProto.FLOAT if dtype == 'float32' else TensorProto.INT32
        create_div_model(model_path, input_shape, input_shape, onnx_dtype)
        feed_dict = {'X1': X1_data.astype(dtype), 'X2': X2_data.astype(dtype)}
        expected_keywords = []
        if len(input_shape) > 5 and dtype == 'float32':
            expected_keywords = ['Reshape']
        elif dtype == 'int32':
            expected_keywords = ['layer_type=DivMod', 'mode=TRUNC']
        exit_status = run_parser(model_path, feed_dict, expected_keywords=expected_keywords, verify=True)
        assert exit_status
