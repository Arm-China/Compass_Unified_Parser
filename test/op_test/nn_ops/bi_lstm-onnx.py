import numpy as np
import onnx
from AIPUBuilder.Parser.tool_utils.run import run_parser
from onnx import TensorProto, helper


def create_lstm(onnx_path, input_size, output_size, version, output_num=2):
    """ Create onnx model. Note: For input, use X as its name if it has only 1 input;
    use X1, X2... as name if it has more than 1 inputs.
    """
    X = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
    W = helper.make_tensor_value_info('X2', TensorProto.FLOAT, [2, 240, 60])
    R = helper.make_tensor_value_info('X3', TensorProto.FLOAT, [2, 240, 60])
    B = helper.make_tensor_value_info('X4', TensorProto.FLOAT, [2, 480])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)
    Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, output_size[1:])
    Y_c = helper.make_tensor_value_info('Y_c', TensorProto.FLOAT, output_size[1:])
    Y1 = helper.make_tensor_value_info('Y1', TensorProto.FLOAT, output_size)
    Y2 = helper.make_tensor_value_info('Y2', TensorProto.FLOAT, output_size[1:])

    lstm = helper.make_node(
        'LSTM', ['X1', 'X2', 'X3', 'X4'],
        ['Y', 'Y_h', 'Y_c'],
        direction='bidirectional',
        hidden_size=60
    )
    add = helper.make_node(
        'Add', ['Y', 'Y'],
        ['Y1'],
    )
    if output_num == 2:
        add_h_c = helper.make_node(
            'Add', ['Y_h', 'Y_c'],
            ['Y2'],
        )
        nodes_list = [lstm, add, add_h_c]
        outputs_list = [Y1, Y2]
    else:
        nodes_list = [lstm, add]
        outputs_list = [Y1]
    graph_def = helper.make_graph(
        nodes_list,  # nodes
        'lstm-model',  # name
        [X, W, R, B],  # inputs
        outputs_list,  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name='lstm-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


input_shape = [10, 30, 60]
output_shape = [10, 2, 30, 60]
feed_dict = {'X1': np.random.ranf(input_shape).astype(np.float32),
             'X2': np.random.ranf([2, 240, 60]).astype(np.float32),
             'X3': np.random.ranf([2, 240, 60]).astype(np.float32),
             'X4': np.random.ranf([2, 480]).astype(np.float32)}

for output_num in [2, 1]:
    for version in [7, 14]:
        OP_NAME = 'lstm-' + str(version) + '-' + str(output_num)
        model_path = OP_NAME + '.onnx'
        create_lstm(model_path, input_shape, output_shape, version, output_num)
        # Run tests with parser and compare result with runtime
        exit_status = run_parser(model_path, feed_dict,
                                 model_type=None, save_output=True, verify=True)
        assert exit_status
