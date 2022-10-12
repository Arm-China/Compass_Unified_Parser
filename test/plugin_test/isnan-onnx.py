import os
import numpy as np
import onnx
from AIPUBuilder.Parser.tool_utils.run import run_parser
from onnx import TensorProto, helper


def create_isnan_model(onnx_path, input_size, output_size, version=13):
    ''' Create onnx model for isnan op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    isnan = helper.make_node(
        OP_NAME, ['X'],
        ['Y'],
    )
    graph_def = helper.make_graph(
        [isnan],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'IsNaN'
model_path = OP_NAME + '.onnx'

input_shape = [1, 3, 2]
output_shape = [1, 3, 2]
feed_dict = {'X': np.random.ranf(input_shape).astype(np.float32)}
create_isnan_model(model_path, input_shape, output_shape)
# Set environment variable AIPUPLUGIN_PATH
os.environ["AIPUPLUGIN_PATH"] = os.path.join(os.path.dirname(__file__), 'plugins')
exit_status = run_parser(model_path, feed_dict, save_output=False,
                         expected_keywords=['IsNaN'], verify=False)
assert exit_status
