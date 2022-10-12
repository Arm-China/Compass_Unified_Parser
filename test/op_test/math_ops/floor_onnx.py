import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_floor_model(onnx_path, input_size, output_size, inptype, version=13):
    ''' Create onnx model for Hardmax op.
    '''
    if inptype == 'float32':
        X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    floor = helper.make_node(
        OP_NAME, ['X1'],
        ['Y'],
    )
    graph_def = helper.make_graph(
        [floor],  # nodes
        OP_NAME + '-model',  # name
        [X1],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Floor'
model_path = OP_NAME + '.onnx'

input_shape = [3, 2]
output_shape = [3, 2]

feed_dict = {'X1': np.random.ranf(input_shape).astype(np.float32)}
create_floor_model(model_path, input_shape, output_shape, inptype='float32')
exit_status = run_parser(model_path, feed_dict,
                         model_type=None, save_output=True, verify=True)
assert exit_status
