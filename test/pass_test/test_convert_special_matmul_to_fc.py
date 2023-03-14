import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_matmul_model(onnx_path, input1_size, input2_size, output_size, version=13):
    ''' Create onnx model for matmul op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input1_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    const_value = np.random.ranf(input2_size).astype(np.float32)
    const_tensor = helper.make_tensor_value_info('Const', TensorProto.FLOAT, const_value.shape)
    const_node = helper.make_node(
        'Constant',
        [],
        ['Const'],
        value=helper.make_tensor(
            name='const_value',
            data_type=TensorProto.FLOAT,
            dims=const_value.shape,
            vals=const_value,
        )
    )
    matmul = helper.make_node(
        OP_NAME, ['X', 'Const'],
        ['Y'],
    )
    graph_def = helper.make_graph(
        [const_node, matmul],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'MatMul'
model_path = OP_NAME + '.onnx'

input1_shape = [1, 1, 4, 5]
input2_shape = [5, 6]
output_shape = [1, 1, 4, 6]
feed_dict = {'X': np.random.randint(10, 20, input1_shape).astype(np.float32)}
create_matmul_model(model_path, input1_shape, input2_shape, output_shape)
# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict, expected_keywords=['FullyConnected'])
assert exit_status
