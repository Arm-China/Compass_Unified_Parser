import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_groupnorm_model(onnx_path, input_size, groups, version=18):
    ''' Create onnx model for groupnorm op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_size)
    scale_tensor = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [groups])
    bias_tensor = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [groups])
    const_scale = helper.make_node(
        'Constant',
        [],
        ['scale'],
        value=helper.make_tensor(
            name='scale_value',
            data_type=TensorProto.FLOAT,
            dims=[groups],
            vals=np.random.ranf(groups).astype(np.float32),
        )
    )
    const_bias = helper.make_node(
        'Constant',
        [],
        ['bias'],
        value=helper.make_tensor(
            name='bias_value',
            data_type=TensorProto.FLOAT,
            dims=[groups],
            vals=np.random.ranf(groups).astype(np.float32),
        )
    )
    groupnorm = helper.make_node(
        'GroupNormalization',
        inputs=['X', 'scale', 'bias'],
        outputs=['Y'],
        num_groups=groups
    )
    graph_def = helper.make_graph(
        [const_scale, const_bias, groupnorm],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'groupnorm'
input_shape = [1, 12, 6, 8]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32) * 100

for groups in (3, 6):
    model_path = '-'.join([OP_NAME, str(groups)]) + '.onnx'
    # TODO: Seems not supported now: No Op registered for GroupNormalization with domain_version of 18
    create_groupnorm_model(model_path, input_shape, groups)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, verify=True,
                             unexpected_keywords=['layer_type=Transpose'])
    assert exit_status
