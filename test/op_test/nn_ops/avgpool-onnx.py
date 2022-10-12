import os
import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_AveragePool_model(onnx_path, input_size, output_size, count_include_pad, version=11):
    ''' Create onnx model for AveragePool op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    AveragePool = helper.make_node(
        OP_NAME,
        inputs=['X'],
        outputs=['Y'],
        strides=[2, 2],
        kernel_shape=[3, 3],
        ceil_mode=1,
        count_include_pad=count_include_pad
    )
    graph_def = helper.make_graph(
        [AveragePool],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'AveragePool'
input_shape = [1, 56, 56, 24]
output_shape = [1, 56, 28, 12]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32) * 10
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)

for count_include_pad in (0, 1):
    model_name = OP_NAME + '-' + str(count_include_pad)
    model_path = model_name + '.onnx'
    # Create model
    create_AveragePool_model(
        model_path, input_shape, output_shape, count_include_pad)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type=None, save_output=True, verify=True)
    assert exit_status
    # NOTE: The outputs are different for different count_include_pad in onnx even all the pads are 0.
    # And opt forward aligns with onnx runtime.
    os.rename('opt_outputs.npy', 'opt_' + model_name + '.npy')
    os.rename('onnx_outputs.npy', 'onnx_' + model_name + '.npy')
