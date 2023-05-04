import os
import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_QLinearMatMul_model(onnx_path, inp_a_shape, inp_b_shape, output_shape, version=12):
    ''' Create onnx model for QLinearMatMul op.
    '''
    X1 = helper.make_tensor_value_info('X1', TensorProto.INT8, inp_a_shape)
    X2 = helper.make_tensor_value_info('X2', TensorProto.INT8, inp_b_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.INT8, output_shape)

    a_scale = helper.make_node('Constant', [], ['a_scale'], value_float=0.5)
    a_zero_point = helper.make_node('Constant', [], ['a_zp'], value_int=5)
    a_cast = helper.make_node('Cast', ['a_zp'], ['a_zp_int8'], to=TensorProto.INT8)
    b_scale = helper.make_node('Constant', [], ['b_scale'], value_float=1.15)
    b_zero_point = helper.make_node('Constant', [], ['b_zp'], value_int=15)
    b_cast = helper.make_node('Cast', ['b_zp'], ['b_zp_int8'], to=TensorProto.INT8)
    y_scale = helper.make_node('Constant', [], ['y_scale'], value_float=2.05)
    y_zero_point = helper.make_node('Constant', [], ['y_zp'], value_int=25)
    y_cast = helper.make_node('Cast', ['y_zp'], ['y_zp_int8'], to=TensorProto.INT8)

    QLinearMatMul = helper.make_node(
        OP_NAME,
        inputs=['X1', 'a_scale', 'a_zp_int8', 'X2', 'b_scale', 'b_zp_int8', 'y_scale', 'y_zp_int8'],
        outputs=['Y'],
    )
    graph_def = helper.make_graph(
        [a_scale, a_zero_point, a_cast, b_scale, b_zero_point, b_cast,
            y_scale, y_zero_point, y_cast, QLinearMatMul],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs
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
exit_status = run_parser(model_path, feed_dict, force_float_ir=True,
                         expected_keywords=['MatMul', 'Round', 'compat_quantized_model=false'],
                         unexpected_keywords=['layer_top_scale', 'layer_top_zp'], verify=True)
assert exit_status

# Set verify to False because need qtlib to generate IR for opt firstly
exit_status = run_parser(model_path, feed_dict, force_float_ir=False,
                         expected_keywords=['MatMul', 'compat_quantized_model=true', 'layer_top_scale', 'layer_top_zp'],
                         unexpected_keywords=['Round'], verify=False)
assert exit_status

# force_float_ir=False has the same effect as not to set force_float_ir
exit_status = run_parser(model_path, feed_dict,
                         expected_keywords=['MatMul', 'compat_quantized_model=true', 'layer_top_scale', 'layer_top_zp'],
                         unexpected_keywords=['Round'], verify=False)
assert exit_status
