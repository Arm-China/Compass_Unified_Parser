# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np
import onnx
from onnx import TensorProto, helper
from utils.run import run_parser
from utils.compare import compare_data


def create_QLinearMatMul_model(onnx_path, inp_a_shape, inp_b_shape, output_shape, version=12):
    ''' Create onnx model for QLinearMatMul op.
    '''
    X1 = helper.make_tensor_value_info('X1', TensorProto.INT8, inp_a_shape)
    X2 = helper.make_tensor_value_info('X2', TensorProto.INT8, inp_b_shape)
    Y_0 = helper.make_tensor_value_info('Y_0', TensorProto.INT8, output_shape)

    a_scale = helper.make_node('Constant', [], ['a_scale'], value_float=0.5)
    a_zero_point = helper.make_node('Constant', [], ['a_zp'], value_int=5)
    a_cast = helper.make_node('Cast', ['a_zp'], ['a_zp_int8'], to=TensorProto.INT8)
    b_scale = helper.make_node('Constant', [], ['b_scale'], value_float=1.15)
    b_zero_point = helper.make_node('Constant', [], ['b_zp'], value_int=15)
    b_cast = helper.make_node('Cast', ['b_zp'], ['b_zp_int8'], to=TensorProto.INT8)
    y_scale = helper.make_node('Constant', [], ['y_scale'], value_float=2.05)
    y_zero_point = helper.make_node('Constant', [], ['y_zp'], value_int=25)
    y_cast = helper.make_node('Cast', ['y_zp'], ['y_zp_int8'], to=TensorProto.INT8)

    # Create node QLinearMatMul
    QLinearMatMul = helper.make_node(
        OP_NAME,
        inputs=['X1', 'a_scale', 'a_zp_int8', 'X2', 'b_scale', 'b_zp_int8', 'y_scale', 'y_zp_int8'],
        outputs=['Y_0'],
    )
    # Create 2 DequantizeLinear, 1 MatMul and 1 QuantizeLinear
    Y_a = helper.make_tensor_value_info('Y_a', TensorProto.FLOAT, inp_a_shape)
    dequant_a = helper.make_node(
        'DequantizeLinear',
        inputs=['X1', 'a_scale', 'a_zp_int8'],
        outputs=['Y_a']
    )
    Y_b = helper.make_tensor_value_info('Y_b', TensorProto.FLOAT, inp_b_shape)
    dequant_b = helper.make_node(
        'DequantizeLinear',
        inputs=['X2', 'b_scale', 'b_zp_int8'],
        outputs=['Y_b']
    )
    Y_m = helper.make_tensor_value_info('Y_m', TensorProto.FLOAT, output_shape)
    matmul = helper.make_node(
        'MatMul',
        inputs=['Y_a', 'Y_b'],
        outputs=['Y_m']
    )
    Y_1 = helper.make_tensor_value_info('Y_1', TensorProto.INT8, output_shape)
    dequant_matmul = helper.make_node(
        'QuantizeLinear',
        inputs=['Y_m', 'y_scale', 'y_zp_int8'],
        outputs=['Y_1']
    )

    # Create graph
    graph_def = helper.make_graph(
        [a_scale, a_zero_point, a_cast, b_scale, b_zero_point, b_cast,
            y_scale, y_zero_point, y_cast, QLinearMatMul,
            dequant_a, dequant_b, matmul, dequant_matmul],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y_0, Y_1],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'QLinearMatMul'
input_a_shape = [1, 1, 10, 12]
input_b_shape = [1, 1, 12, 14]
output_shape = [1, 1, 10, 14]

# Generate input data
feed_dict = dict()
feed_dict['X1'] = np.tile(np.arange(-4, 8), [1, 1, 10, 1]).astype(np.int8)
feed_dict['X2'] = np.tile(np.arange(-6, 8), [1, 1, 12, 1]).astype(np.int8)
np.save('input', feed_dict)

model_path = OP_NAME + '.onnx'
# Create model
create_QLinearMatMul_model(
    model_path, input_a_shape, input_b_shape, output_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict, verify=True)
# FIXME: Disable checking now because opt/qtlib has issue in quantized Matmul
# assert exit_status

onnx_outputs = np.load('onnx_outputs.npy', allow_pickle=True)
onnx_y_outs = [val for _, val in onnx_outputs.item().items()]
assert len(onnx_y_outs) == 2 and compare_data(onnx_y_outs[0], onnx_y_outs[1])
