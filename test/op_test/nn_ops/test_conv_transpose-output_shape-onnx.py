import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_conv_transpose_model(onnx_path, input_size, output_size, weight_shape, version=1):
    ''' Create onnx model for conv_transpose op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    W = np.random.ranf(weight_shape).astype(np.float32)
    # W = np.ones(weight_shape).astype(np.float32)
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
        kernel_shape=[2, 45],
        output_shape=[1000, 400],
        strides=[5, 4],
        auto_pad='NOTSET',
        pads=[1, 1, 1, 1],  # should be ignored because output_shape is set
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
input_shape = [1, 5, 200, 100]
output_shape = [1, 48, 1000, 400]
weight_shape = [5, 48, 2, 45]

# Generate input data
feed_dict = dict()
input_data = np.random.ranf(input_shape).astype(np.float32) * 100
feed_dict['X'] = input_data
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)

for version in (11, ):
    model_name = '-'.join([OP_NAME, str(version)])
    model_path = model_name + '.onnx'
    # Create model
    create_conv_transpose_model(
        model_path, input_shape, output_shape, weight_shape, version)

    # FIXME: Enable verify after onnx runtime is upgraded to >= 1.13.1
    # The similarity issue is caused by a bug of auto_pad in onnx runtime, which
    # is fixed in onnx runtime 1.13.1. See this commit:
    # https://github.com/microsoft/onnxruntime/commit/f96f2225262ed9aaa17604aeb3185b98c5dc71d2
    exit_status = run_parser(
        model_path, feed_dict, save_output=True, verify=False)
    assert exit_status
