import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_gru(onnx_path, input_size, output_size, version):
    """ Create onnx model. Note: For input, use X as its name if it has only 1 input;
    use X1, X2... as name if it has more than 1 inputs.
    """
    X = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)
    Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, output_size[1:])
    Y1 = helper.make_tensor_value_info('Y1', TensorProto.FLOAT, output_size)
    Y2 = helper.make_tensor_value_info('Y2', TensorProto.FLOAT, output_size[1:])

    const_nodes = []
    tensors = []
    wrb_tensor_names = ['W', 'R', 'B']
    wrb_shapes = [[1, 180, 60], [1, 180, 60], [1, 360]]
    for tensor_name, tensor_shape in zip(wrb_tensor_names, wrb_shapes):
        tensor = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, tensor_shape)
        tensors.append(tensor)
        const_value = np.random.ranf(tensor_shape).astype(np.float32)
        const_tensor = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, const_value.shape)
        const_node = helper.make_node(
            'Constant',
            [],
            [tensor_name],
            value=helper.make_tensor(
                name=tensor_name + '_value',
                data_type=TensorProto.FLOAT,
                dims=const_value.shape,
                vals=const_value,
            )
        )
        const_nodes.append(const_node)
    gru = helper.make_node(
        'GRU', ['X1'] + wrb_tensor_names,
        ['Y', 'Y_h'],
        direction='reverse',
        hidden_size=60
    )
    add = helper.make_node(
        'Add', ['Y', 'Y'],
        ['Y1'],
    )
    add_h = helper.make_node(
        'Add', ['Y_h', 'Y_h'],
        ['Y2'],
    )
    nodes_list = const_nodes + [gru, add, add_h]
    outputs_list = [Y1, Y2]

    graph_def = helper.make_graph(
        nodes_list,  # nodes
        'gru-model',  # name
        [X],  # inputs
        outputs_list,  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name='gru-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


input_shape = [10, 30, 60]
output_shape = [10, 1, 30, 60]
feed_dict = {'X1': np.random.ranf(input_shape).astype(np.float32)}

for version in [7, 14]:
    OP_NAME = 'gru-' + str(version)
    model_path = OP_NAME + '.onnx'
    create_gru(model_path, input_shape, output_shape, version)
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, verify=True)
    assert exit_status
