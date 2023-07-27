import os
import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_QLinearAdd_model(onnx_path, inp_a_shape, inp_b_shape, output_shape, version=12):
    ''' Create onnx model for QLinearAdd op.
    '''
    X1 = helper.make_tensor_value_info('X1', TensorProto.INT8, inp_a_shape)
    X2 = helper.make_tensor_value_info('X2', TensorProto.INT8, inp_b_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.INT8, output_shape)

    a_scale = helper.make_node('Constant', [], ['a_scale'], value_float=2.5)
    a_zero_point = helper.make_node('Constant', [], ['a_zp'], value_int=10)
    a_cast = helper.make_node('Cast', ['a_zp'], ['a_zp_int8'], to=TensorProto.INT8)
    b_scale = helper.make_node('Constant', [], ['b_scale'], value_float=4.2)
    b_zero_point = helper.make_node('Constant', [], ['b_zp'], value_int=-5)
    b_cast = helper.make_node('Cast', ['b_zp'], ['b_zp_int8'], to=TensorProto.INT8)
    y_scale = helper.make_node('Constant', [], ['y_scale'], value_float=3.4)

    QLinearAdd = helper.make_node(
        OP_NAME,
        inputs=['X1', 'a_scale', 'a_zp_int8', 'X2', 'b_scale', 'b_zp_int8', 'y_scale'],
        outputs=['Y'],
    )
    QLinearAdd.domain = 'com.microsoft'
    graph_def = helper.make_graph(
        [a_scale, a_zero_point, a_cast, b_scale, b_zero_point, b_cast,
            y_scale, QLinearAdd],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs
    )
    # add ms domain
    onnxdomain = onnx.OperatorSetIdProto()
    onnxdomain.version = version
    onnxdomain.domain = ''
    msdomain = onnx.OperatorSetIdProto()
    msdomain.version = 1
    msdomain.domain = 'com.microsoft'
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model', opset_imports=[onnxdomain, msdomain])
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'QLinearAdd'
input_a_shape = [1, 1, 3, 4]
input_b_shape = input_a_shape
output_shape = input_a_shape

# Generate input data
feed_dict = dict()
feed_dict['X1'] = np.random.randint(-20, 30, input_a_shape).astype(np.int8)
feed_dict['X2'] = np.random.randint(-20, 30, input_b_shape).astype(np.int8)
np.save('input', feed_dict)

model_path = OP_NAME + '.onnx'
# Create model
create_QLinearAdd_model(
    model_path, input_a_shape, input_b_shape, output_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict, force_float_ir=True,
                         expected_keywords=['Add', 'Round', 'compat_quantized_model=false'],
                         unexpected_keywords=['layer_top_scale', 'layer_top_zp'], verify=True)
assert exit_status

# Set verify to False because need qtlib to generate IR for opt firstly
exit_status = run_parser(model_path, feed_dict, force_float_ir=False,
                         expected_keywords=['Add', 'compat_quantized_model=true', 'layer_top_scale', 'layer_top_zp'],
                         unexpected_keywords=['Round'], verify=False)
assert exit_status

# force_float_ir=False has the same effect as not to set force_float_ir
exit_status = run_parser(model_path, feed_dict,
                         expected_keywords=['Add', 'compat_quantized_model=true', 'layer_top_scale', 'layer_top_zp'],
                         unexpected_keywords=['Round'], verify=False)
assert exit_status
