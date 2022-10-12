import numpy as np
import onnx
from AIPUBuilder.Parser.tool_utils.run import run_parser
from onnx import TensorProto, helper


def create_hardmax_model(onnx_path, input_size, output_size, axis=-1, version=13):
    ''' Create onnx model for Hardmax op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    hardmax = helper.make_node(
        OP_NAME, ['X'],
        ['Y'],
        axis=axis
    )
    graph_def = helper.make_graph(
        [hardmax],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Hardmax'
model_path = OP_NAME + '.onnx'

input_shape = [1, 3, 2]
output_shape = [1, 3, 2]
feed_dict = {'X': np.random.ranf(input_shape).astype(np.float32)}
# TODO: Add loop to set axis, version, different input/output shape
create_hardmax_model(model_path, input_shape, output_shape, axis=0)
# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict,
                         model_type=None, save_output=True, verify=True)
assert exit_status
