import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_div_model(onnx_path, input_size, output_size, version=13):
    ''' Create onnx model for div op.
    '''
    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
    X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    div = helper.make_node(
        OP_NAME, ['X1', 'X2'],
        ['Y'],
    )
    graph_def = helper.make_graph(
        [div],  # nodes
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
    model_path = '-'.join([OP_NAME, str(len(input_shape))]) + '.onnx'
    feed_dict = {'X1': np.random.ranf(input_shape).astype(np.float32),
                 'X2': np.random.ranf(input_shape).astype(np.float32)}
    create_div_model(model_path, input_shape, input_shape)
    exit_status = run_parser(model_path, feed_dict, expected_keywords=(
        ['Reshape'] if len(input_shape) > 5 else []), verify=True)
    assert exit_status