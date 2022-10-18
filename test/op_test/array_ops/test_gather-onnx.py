import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_gather_model(onnx_path, input_size, output_size, indices, version=14):
    ''' Create onnx model for gather op.
    '''
    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
    X2 = helper.make_tensor_value_info('X2', TensorProto.INT64, [1])
    Y = helper.make_tensor_value_info('Y', TensorProto.INT64, output_size)

    shape_tensor = helper.make_tensor_value_info('Shape', TensorProto.INT64, [1])
    shape_node = helper.make_node(
        'Shape',
        ['X1'],
        ['Shape']
    )
    indices_tensor = helper.make_tensor_value_info('Indices', TensorProto.INT64, [1])
    const_node = helper.make_node('Constant', [], ['Indices'],
                                  value=helper.make_tensor(name='const_value',
                                                           data_type=onnx.TensorProto.INT64,
                                                           dims=[1],
                                                           vals=np.array([indices]).astype(np.int64),
                                                           ))
    gather_tensor = helper.make_tensor_value_info('Gather', TensorProto.INT64, [1])
    gather_node = helper.make_node(
        OP_NAME,
        inputs=['Shape', 'Indices'],
        outputs=['Gather']
    )
    pow_node = helper.make_node(
        'Pow',
        inputs=['X2', 'Gather'],
        outputs=['Y']
    )
    graph_def = helper.make_graph(
        [shape_node, const_node, gather_node, pow_node],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Gather'
input_shape = [1, 3, 2, 4, 5]
output_shape = [1]

# Generate input data
feed_dict = dict()
feed_dict['X1'] = np.random.ranf(input_shape).astype(np.float32) * 100
feed_dict['X2'] = np.array([10])
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)

for indices in (-5, -3, 0, 2):
    model_name = '-'.join([OP_NAME, str(indices)])
    model_path = model_name + '.onnx'
    # Create model
    create_gather_model(
        model_path, input_shape, output_shape, indices, 14)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type=None, save_output=True, verify=True)
    assert exit_status
