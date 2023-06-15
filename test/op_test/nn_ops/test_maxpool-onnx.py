import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_maxpool_model(onnx_path, input_size, output_size, ceil_mode, dilations, kernel_shape, pads, version=12):
    ''' Create onnx model for maxpool op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y0 = helper.make_tensor_value_info('Y0', TensorProto.FLOAT, output_size)
    Y1 = helper.make_tensor_value_info('Y1', TensorProto.INT64, output_size)

    maxpool = helper.make_node(
        'MaxPool',
        inputs=['X'],
        outputs=['Y0', 'Y1'],
        kernel_shape=kernel_shape,
        pads=pads,
        ceil_mode=ceil_mode,
        dilations=dilations,
    )
    graph_def = helper.make_graph(
        [maxpool],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y0, Y1],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'maxpool'
input_shape = [2, 6, 51, 52]
output_shape = [2, 6, 41, 42]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32) * 100
# input_data_path = 'input.npy'
# np.save(input_data_path, feed_dict)

model_path = OP_NAME + '.onnx'
create_maxpool_model(
    model_path, input_shape, output_shape,
    True, [1, 1], kernel_shape=[8, 8], pads=[0, 0, 0, 0])

# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict, verify=True)
assert exit_status
