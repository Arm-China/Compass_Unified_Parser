import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_conv_transpose_model(onnx_path, input_size, output_size, weight_shape, version=1):
    ''' Create onnx model for conv_transpose op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    W = np.ones(weight_shape).astype(np.float32)
    W_tensor = helper.make_tensor_value_info('W', TensorProto.FLOAT, W.shape)
    const_W = helper.make_node(
        'Constant',
        [],
        ['W'],
        value=helper.make_tensor(
            name='weight_value',
            data_type=TensorProto.FLOAT,
            dims=W.shape,
            vals=W,
        )
    )
    conv_transpose = helper.make_node(
        OP_NAME,
        inputs=['X', 'W'],
        outputs=['Y'],
        # auto_pad='SAME_UPPER', # failed for now, should pass with latest onnx rt
        # pads=[2,2,3,3], # passed
        pads=[3, 3, 2, 2],  # passed
        kernel_shape=[7, 8],
        output_padding=[1, 1],
        strides=[3, 4],
    )
    graph_def = helper.make_graph(
        [const_W, conv_transpose],  # nodes, sequences matters
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'ConvTranspose'
input_shape = [2, 6, 57, 58]
output_shape = [2, 10, 171, 232]
weight_shape = [6, 10, 7, 8]

# Generate input data
feed_dict = dict()
#feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32) * 100
feed_dict['X'] = np.ones(input_shape).astype(np.float32) * 100
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)

for version in (11, ):  # 1,
    model_name = '-'.join([OP_NAME, str(version)])
    model_path = model_name + '.onnx'
    # Create model
    create_conv_transpose_model(
        model_path, input_shape, output_shape, weight_shape, version)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, save_output=True, verify=True)
    assert exit_status
