import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_LpPool_model(onnx_path, input_size, output_size, ceil_mode, dilations, kernel_shape, p, pads, version=18):
    ''' Create onnx model for LpPool op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    if version == 18:
        other_args = {'ceil_mode': ceil_mode, 'dilations': dilations}
    else:
        other_args = {}

    LpPool = helper.make_node(
        OP_NAME,
        inputs=['X'],
        outputs=['Y'],
        kernel_shape=kernel_shape,
        p=p,
        pads=pads,
        **other_args,
    )
    graph_def = helper.make_graph(
        [LpPool],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'LpPool'
input_shape = [1, 3, 16, 25]
output_shape = [1, 3, 14, 22]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32) * 100
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)

for ceil_mode in (1, 0, ):
    for dilations in ([1, 2], [4, 3], ):
        for p in (2, 1, ):
            model_name = '-'.join([OP_NAME, str(ceil_mode),
                                   str(dilations[0]), str(p)])
            model_path = model_name + '.onnx'
            # FIXME: onnx runtime has issue when auto_pad is set, so ignore them for now.
            # Add tests of auto_pad after the issue is fixed in onnx runtime.
            create_LpPool_model(
                model_path, input_shape, output_shape,
                ceil_mode, dilations, kernel_shape=[4, 5], p=p, pads=[1, 2, 0, 3], version=18)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(
                model_path, feed_dict, save_output=True, verify=True)
            assert exit_status
