import os
import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_QLinearAdd_model(onnx_path, input_shape, output_shape, channels_last, version=16):
    ''' Create onnx model for QLinearGlobalAveragePool op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.UINT8, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.UINT8, output_shape)

    x_scale = helper.make_node('Constant', [], ['x_scale'], value_float=2.5)
    x_zero_point = helper.make_node('Constant', [], ['x_zp'], value_int=10)
    x_cast = helper.make_node('Cast', ['x_zp'], ['x_zp_uint8'], to=TensorProto.UINT8)
    y_scale = helper.make_node('Constant', [], ['y_scale'], value_float=3.4)
    y_zero_point = helper.make_node('Constant', [], ['y_zp'], value_int=15)
    y_cast = helper.make_node('Cast', ['y_zp'], ['y_zp_uint8'], to=TensorProto.UINT8)

    QLinearGlobalAveragePool = helper.make_node(
        OP_NAME,
        inputs=['X', 'x_scale', 'x_zp_uint8', 'y_scale', 'y_zp_uint8'],
        outputs=['Y'],
        channels_last=channels_last,
    )
    QLinearGlobalAveragePool.domain = 'com.microsoft'
    graph_def = helper.make_graph(
        [x_scale, x_zero_point, x_cast, y_scale, y_zero_point, y_cast, QLinearGlobalAveragePool],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    # add ms domain
    onnxdomain = onnx.OperatorSetIdProto()
    onnxdomain.version = version
    onnxdomain.domain = ''
    msdomain = onnx.OperatorSetIdProto()
    msdomain.version = 1
    msdomain.domain = 'com.microsoft'
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model', opset_imports=[onnxdomain, msdomain])
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'QLinearGlobalAveragePool'
input_shape = [1, 10, 30, 40]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.randint(0, 250, input_shape).astype(np.uint8)
np.save('input', feed_dict)

for channels_last in (0, 1):
    model_path = '-'.join([OP_NAME, str(channels_last)]) + '.onnx'
    output_shape = [1, 10, 1, 1] if channels_last == 0 else [1, 1, 1, 40]
    # Create model
    create_QLinearAdd_model(
        model_path, input_shape, output_shape, channels_last)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, force_float_ir=True,
                             expected_keywords=['Pooling', 'Round', 'compat_quantized_model=false'],
                             unexpected_keywords=['layer_top_scale', 'layer_top_zp'], verify=True)
    assert exit_status

    exit_status = run_parser(model_path, feed_dict,
                             expected_keywords=['Pooling', 'compat_quantized_model=true',
                                                'layer_top_scale', 'layer_top_zp'],
                             unexpected_keywords=['Round'], verify=False)
    assert exit_status
