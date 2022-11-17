import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_batchnorm_model(onnx_path, input_size, train=False, version=14):
    ''' Create onnx model for batchnorm op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_size)
    mean_var_shape = [input_size[1]]
    if train:
        running_mean = helper.make_tensor_value_info('running_mean', TensorProto.FLOAT, mean_var_shape)
        running_var = helper.make_tensor_value_info('running_var', TensorProto.FLOAT, mean_var_shape)

    # Use the same const node as the inputs(2-5)
    const_value = np.random.ranf(mean_var_shape).astype(np.float32)
    const_tensor = helper.make_tensor_value_info('Const', TensorProto.FLOAT, const_value.shape)
    const_node = helper.make_node(
        'Constant',
        [],
        ['Const'],
        value=helper.make_tensor(
            name='const_value',
            data_type=TensorProto.FLOAT,
            dims=const_value.shape,
            vals=const_value,
        )
    )
    batchnorm = helper.make_node(
        OP_NAME,
        inputs=['X', 'Const', 'Const', 'Const', 'Const'],
        outputs=['Y'] if not train else ['Y', 'running_mean', 'running_var'],
        training_mode=int(train)
    )
    graph_def = helper.make_graph(
        [const_node, batchnorm],  # nodes, sequences matters
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y] if not train else [Y, running_mean, running_var],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'BatchNormalization'
input_shape = [2, 10, 3, 4]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32)

for version in (14, ):
    for train in (True, False):
        model_name = '-'.join([OP_NAME, str(train), str(version)])
        model_path = model_name + '.onnx'
        # Create model
        create_batchnorm_model(model_path, input_shape, train, version)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(model_path, feed_dict, verify=True)
        assert exit_status
