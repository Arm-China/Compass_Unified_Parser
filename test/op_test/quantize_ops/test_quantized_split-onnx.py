# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np
import onnx
from onnx import TensorProto, helper
from utils.run import run_parser
from utils.compare import compare_data


def create_q_split_model(onnx_path, input_shape, output_shape, axis=2, version=13):
    ''' Create onnx model for q_split op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.INT8, input_shape)
    Y_0 = helper.make_tensor_value_info('Y_0', TensorProto.INT8, output_shape)
    Y_1 = helper.make_tensor_value_info('Y_1', TensorProto.INT8, output_shape)

    scale_len = input_shape[axis]
    a_scale = helper.make_node('Constant', [], ['a_scale'], value_floats=[0.5] * scale_len)
    a_zero_point = helper.make_node('Constant', [], ['a_zp'], value_ints=[5] * scale_len)
    a_cast = helper.make_node('Cast', ['a_zp'], ['a_zp_int8'], to=TensorProto.INT8)
    y_0_scale = helper.make_node('Constant', [], ['y_0_scale'], value_floats=[2.05] * scale_len)
    y_0_zero_point = helper.make_node('Constant', [], ['y_0_zp'], value_ints=[25] * scale_len)
    y_0_cast = helper.make_node('Cast', ['y_0_zp'], ['y_0_zp_int8'], to=TensorProto.INT8)
    y_1_scale = helper.make_node('Constant', [], ['y_1_scale'], value_floats=[1.15] * scale_len)
    y_1_zero_point = helper.make_node('Constant', [], ['y_1_zp'], value_ints=[15] * scale_len)
    y_1_cast = helper.make_node('Cast', ['y_1_zp'], ['y_1_zp_int8'], to=TensorProto.INT8)
    split_len = helper.make_node('Constant', [], ['split'], value_ints=[output_shape[-1]] * 2)

    # Create 1 DequantizeLinear, 1 Split and 2 QuantizeLinear
    Y_a = helper.make_tensor_value_info('Y_a', TensorProto.FLOAT, input_shape)
    dequant_a = helper.make_node(
        'DequantizeLinear',
        inputs=['X', 'a_scale', 'a_zp_int8'],
        outputs=['Y_a'],
        axis=axis,
    )
    Y_s_0 = helper.make_tensor_value_info('Y_s_0', TensorProto.FLOAT, output_shape)
    Y_s_1 = helper.make_tensor_value_info('Y_s_1', TensorProto.FLOAT, output_shape)
    split = helper.make_node(
        'Split',
        inputs=['Y_a', 'split'],
        outputs=['Y_s_0', 'Y_s_1'],
        axis=-1,
    )
    quant_split_0 = helper.make_node(
        'QuantizeLinear',
        inputs=['Y_s_0', 'y_0_scale', 'y_0_zp_int8'],
        outputs=['Y_0'],
        axis=axis,
    )
    quant_split_1 = helper.make_node(
        'QuantizeLinear',
        inputs=['Y_s_1', 'y_1_scale', 'y_1_zp_int8'],
        outputs=['Y_1'],
        axis=axis,
    )

    # Create graph
    graph_def = helper.make_graph(
        [a_scale, a_zero_point, a_cast,
            y_0_scale, y_0_zero_point, y_0_cast,
            y_1_scale, y_1_zero_point, y_1_cast,
            split_len,
            dequant_a, split, quant_split_0, quant_split_1],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y_0, Y_1],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME + '-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'q_split'
input_shape = [2, 3, 10, 22]
output_shape = [1, 1, 10, 11]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.tile(np.arange(-4, 8), [1, 1, 10, 1]).astype(np.int8)
feed_dict['X2'] = np.tile(np.arange(-6, 8), [1, 1, 10, 1]).astype(np.int8)
np.save('input', feed_dict)

model_path = OP_NAME + '.onnx'
# Create model
create_q_split_model(
    model_path, input_shape, output_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict, verify=False,
                         expected_keywords=['activation_quantization_axis=[2,2]'],
                         unexpected_keywords=['layer_type=Quantize', 'layer_type=DeQuantize'])
assert exit_status
