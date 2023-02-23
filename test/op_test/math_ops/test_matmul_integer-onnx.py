import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_matmul_integer_model(onnx_path, input1_size, input2_size, output_size, version=10):
    ''' Create onnx model for matmul_integer op.
    '''
    X1 = helper.make_tensor_value_info('X1', TensorProto.INT8, input1_size)
    X2 = helper.make_tensor_value_info('X2', TensorProto.UINT8, input2_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.INT32, output_size)

    matmul_integer = helper.make_node(
        OP_NAME, ['X1', 'X2'],
        ['Y'],
    )
    graph_def = helper.make_graph(
        [matmul_integer],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'MatMulInteger'
model_path = OP_NAME + '.onnx'

input1_shape = [2, 3, 4, 5]
input2_shape = [2, 3, 5, 6]
output_shape = [2, 3, 4, 6]
feed_dict = {'X1': np.random.randint(10, 20, input1_shape).astype(np.int8),
             'X2': np.random.randint(20, 30, input2_shape).astype(np.uint8)}
create_matmul_integer_model(model_path, input1_shape, input2_shape, output_shape)
# Run tests with parser and compare result with runtime
# FIXME: Enable verify after onnx runtime supports int8+uint8 as inputs.
# Currently onnx runtime raises error for int8+uint8:
# onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 :
#  NOT_IMPLEMENTED : Could not find an implementation for MatMulInteger(10) node with name ''
exit_status = run_parser(model_path, feed_dict, save_output=True, verify=False)
assert exit_status
