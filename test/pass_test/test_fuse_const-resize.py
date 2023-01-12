import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_fuse_const_model(onnx_path, resize_input_size, pow_input_shape, mode, version=11):
    ''' Create onnx model for fuse_const op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, pow_input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, pow_input_shape)
    input_nodes = []
    for input_name in ('const_x', 'roi', 'scales', 'sizes'):
        if input_name == 'const_x':
            tensor_value_shape = resize_input_size
            data_type = TensorProto.FLOAT
            tensor_value = np.random.ranf(tensor_value_shape).astype(np.float32)
        elif input_name == 'sizes':
            tensor_value_shape = [len(pow_input_shape)]
            data_type = TensorProto.INT64
            tensor_value = pow_input_shape
        else:
            tensor_value_shape = [0]
            data_type = TensorProto.FLOAT
            tensor_value = []
        const_tensor = helper.make_tensor_value_info(input_name, data_type, tensor_value_shape)
        const_node = helper.make_node(
            'Constant',
            [],
            [input_name],
            value=helper.make_tensor(
                name=input_name + '_value',
                data_type=data_type,
                dims=tensor_value_shape,
                vals=tensor_value,
            )
        )
        input_nodes.append(const_node)
    resize_node = helper.make_node(
        'Resize',
        inputs=['const_x', 'roi', 'scales', 'sizes'],
        outputs=['Y0'],
        coordinate_transformation_mode='pytorch_half_pixel',
        mode=mode,
        nearest_mode='round_prefer_ceil'
    )
    pow_node = helper.make_node(
        'Pow',
        inputs=['X', 'Y0'],
        outputs=['Y']
    )
    graph_def = helper.make_graph(
        input_nodes + [resize_node, pow_node],  # nodes, sequences matters
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y]  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'fuse_const'
resize_input_shapes = [[3, 10, 15], [1, 7, 24, 24], [2, 5, 16, 17, 18]]
pow_input_shapes = [[3, 10, 19], [1, 7, 32, 32], [2, 5, 26, 27, 28]]

# Generate input data
feed_dict = dict()

for version in (11, 13):
    for resize_input_shape, pow_input_shape in zip(resize_input_shapes, pow_input_shapes):
        for mode in ('cubic', 'linear', 'nearest'):
            if mode == 'cubic' and len(resize_input_shape) != 4:
                # onnx Resize only supports 4d cubic
                continue
            model_name = '-'.join([OP_NAME, str(version), str(len(pow_input_shape)), mode])
            model_path = model_name + '.onnx'
            # Set feed_dict
            feed_dict.clear()
            feed_dict['X'] = np.random.ranf(pow_input_shape).astype(np.float32)
            # Create model
            create_fuse_const_model(model_path, resize_input_shape, pow_input_shape, mode, version)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(model_path, feed_dict, verify=True)
            assert exit_status
