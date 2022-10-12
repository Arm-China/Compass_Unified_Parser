import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_gridsample_model(onnx_path, input_size, output_size, grid_size, mode, padding_mode, align_corners, version=16):
    ''' Create onnx model for gridsample op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Grid = helper.make_tensor_value_info('Grid', TensorProto.FLOAT, grid_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    gridsample = helper.make_node(
        OP_NAME,
        inputs=['X', 'Grid'],
        outputs=['Y'],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )
    graph_def = helper.make_graph(
        [gridsample],  # nodes
        OP_NAME + '-model',  # name
        [X, Grid],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'GridSample'
input_shape = [1, 3, 16, 32]
output_shape = [1, 3, 64, 128]
grid_shape = [1, 64, 128, 2]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32) * 100
feed_dict['Grid'] = 2.0 * np.random.ranf(grid_shape).astype(np.float32) - 1.0
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)

# TODO: OPT doesn't support 'bicubic' for now. Ignore it for now.
for mode in ('bilinear', 'nearest', ):
    for padding_mode in ('zeros', 'border', 'reflection'):
        for align_corners in (0, 1):
            model_name = '-'.join([OP_NAME, mode,
                                   padding_mode, str(align_corners)])
            model_path = model_name + '.onnx'
            # Create model
            create_gridsample_model(
                model_path, input_shape, output_shape,
                grid_shape, mode, padding_mode, align_corners)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(
                model_path, feed_dict, model_type=None, save_output=True, verify=True)
            assert exit_status
