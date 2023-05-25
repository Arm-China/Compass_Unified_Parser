import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_mish_model(onnx_path, input_size, output_size, inptype, version=18):
    ''' Create onnx model for mish op.
    '''
    if inptype == 'float32':
        X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    mish = helper.make_node(
        OP_NAME, ['X1'],
        ['Y'],
    )
    graph_def = helper.make_graph(
        [mish],  # nodes
        OP_NAME + '-model',  # name
        [X1],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Mish'
model_path = OP_NAME + '.onnx'

input_shapes = [[3, 2], [3], [], [1, 2, 3, 4, 5]]

for input_shape in input_shapes:

    feed_dict = {'X1': np.random.ranf(input_shape).astype(np.float32)}
    create_mish_model(model_path, input_shape, input_shape, inptype='float32')
    exit_status = run_parser(model_path, feed_dict,
                             model_type=None, save_output=True, verify=True)
    assert exit_status
