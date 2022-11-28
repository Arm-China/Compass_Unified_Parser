import os
import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_DepthToSpace_model(onnx_path, input_size, output_size, mode='DCR', version=11):
    ''' Create onnx model for DepthToSpace op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    DepthToSpace = helper.make_node(
        OP_NAME,
        inputs=['X'],
        outputs=['Y'],
        blocksize=3,
        mode=mode
    )
    graph_def = helper.make_graph(
        [DepthToSpace],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'DepthToSpace'
input_shape = [1, 54, 10, 12]
output_shape = [1, 6, 30, 36]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32) * 10

for mode in ('CRD', 'DCR'):
    model_name = OP_NAME + '-' + mode
    model_path = model_name + '.onnx'
    # Create model
    create_DepthToSpace_model(
        model_path, input_shape, output_shape, mode)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, expected_keywords=['mode=' + mode], verify=True)
    assert exit_status
