# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np
import onnx
from onnx import TensorProto, helper
from utils.run import run_parser
from utils.compare import compare_data


def create_q_concat_model(onnx_path, inp_a_shape, inp_b_shape, output_shape, axis=2, version=13):
    ''' Create onnx model for q_concat op.
    '''
    X1 = helper.make_tensor_value_info('X1', TensorProto.INT8, inp_a_shape)
    X2 = helper.make_tensor_value_info('X2', TensorProto.INT8, inp_b_shape)
    Y_0 = helper.make_tensor_value_info('Y_0', TensorProto.INT8, output_shape)

    scale_len = inp_a_shape[axis]
    a_scale = helper.make_node('Constant', [], ['a_scale'], value_floats=[0.5] * scale_len)
    a_zero_point = helper.make_node('Constant', [], ['a_zp'], value_ints=[5] * scale_len)
    a_cast = helper.make_node('Cast', ['a_zp'], ['a_zp_int8'], to=TensorProto.INT8)
    b_scale = helper.make_node('Constant', [], ['b_scale'], value_floats=[1.15] * scale_len)
    b_zero_point = helper.make_node('Constant', [], ['b_zp'], value_ints=[15] * scale_len)
    b_cast = helper.make_node('Cast', ['b_zp'], ['b_zp_int8'], to=TensorProto.INT8)
    y_scale = helper.make_node('Constant', [], ['y_scale'], value_floats=[2.05] * scale_len)
    y_zero_point = helper.make_node('Constant', [], ['y_zp'], value_ints=[25] * scale_len)
    y_cast = helper.make_node('Cast', ['y_zp'], ['y_zp_int8'], to=TensorProto.INT8)

    # Create 2 DequantizeLinear, 1 Concat and 1 QuantizeLinear
    Y_a = helper.make_tensor_value_info('Y_a', TensorProto.FLOAT, inp_a_shape)
    dequant_a = helper.make_node(
        'DequantizeLinear',
        inputs=['X1', 'a_scale', 'a_zp_int8'],
        outputs=['Y_a'],
        axis=axis,
    )
    Y_b = helper.make_tensor_value_info('Y_b', TensorProto.FLOAT, inp_b_shape)
    dequant_b = helper.make_node(
        'DequantizeLinear',
        inputs=['X2', 'b_scale', 'b_zp_int8'],
        outputs=['Y_b'],
        axis=axis,
    )
    Y_c = helper.make_tensor_value_info('Y_c', TensorProto.FLOAT, output_shape)
    concat = helper.make_node(
        'Concat',
        inputs=['Y_a', 'Y_b'],
        outputs=['Y_c'],
        axis=-1,
    )
    Y = helper.make_tensor_value_info('Y', TensorProto.INT8, output_shape)
    quant_concat = helper.make_node(
        'QuantizeLinear',
        inputs=['Y_c', 'y_scale', 'y_zp_int8'],
        outputs=['Y'],
        axis=axis,
    )

    # Create graph
    graph_def = helper.make_graph(
        [a_scale, a_zero_point, a_cast, b_scale, b_zero_point, b_cast,
            y_scale, y_zero_point, y_cast,
            dequant_a, dequant_b, concat, quant_concat],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'q_concat'
input_a_shape = [1, 1, 10, 12]
input_b_shape = [1, 1, 10, 14]
output_shape = [1, 1, 10, 26]

# Generate input data
feed_dict = dict()
feed_dict['X1'] = np.tile(np.arange(-4, 8), [1, 1, 10, 1]).astype(np.int8)
feed_dict['X2'] = np.tile(np.arange(-6, 8), [1, 1, 10, 1]).astype(np.int8)
np.save('input', feed_dict)

model_path = OP_NAME + '.onnx'
# Create model
create_q_concat_model(
    model_path, input_a_shape, input_b_shape, output_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict, verify=False,
                         expected_keywords=['activation_quantization_axis=[2]'],
                         unexpected_keywords=['layer_type=Quantize', 'layer_type=DeQuantize'])
assert exit_status
