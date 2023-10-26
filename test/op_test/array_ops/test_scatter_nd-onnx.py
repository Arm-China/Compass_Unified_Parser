import numpy as np
import onnx
import time
from utils.run import run_parser
from onnx import TensorProto, helper


def create_scatternd_model(onnx_path, scatter_inputs, add_input_shape, reduction,
                           const_scatter_inputs=True, version=18):
    ''' Create onnx model for scatternd op with constant inputs.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.INT32, add_input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.INT32, add_input_shape)
    input_nodes = []
    if const_scatter_inputs:
        const_input_names = ['data', 'index', 'updates']
        scatter_input_names = const_input_names
    else:
        const_input_names = ['index', 'updates']
        scatter_input_names = ['X'] + const_input_names
    for idx, input_name in enumerate(const_input_names):
        tensor_value_shape = scatter_inputs[idx].shape
        data_type = TensorProto.INT64 if input_name == 'index' else TensorProto.INT32
        tensor_value = scatter_inputs[idx]
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
    scatternd_node = helper.make_node(
        'ScatterND',
        inputs=scatter_input_names,
        outputs=['Y0'],
        reduction=reduction,
    )
    add_node = helper.make_node(
        'Add',
        inputs=['X', 'Y0'],
        outputs=['Y']
    )
    graph_def = helper.make_graph(
        input_nodes + [scatternd_node, add_node],  # nodes, sequences matters
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y]  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'scatternd'
add_input_shapes = [[300, 400, 50]]

# Generate input data
feed_dict = dict()

for version in (16, 18):
    for add_input_shape in add_input_shapes:
        for reduction in ('none', 'min', 'max', 'add', 'mul'):
            # Set feed_dict
            feed_dict.clear()
            feed_dict['X'] = np.random.randint(-3, 5, add_input_shape).astype(np.int32)
            for const_scatter_inputs in (False, True):
                if version < 18 and reduction in ('min', 'max'):
                    continue
                model_path = '-'.join([OP_NAME, str(version), str(len(add_input_shape)),
                                       reduction, str(const_scatter_inputs)]) + '.onnx'
                scatter_index = np.random.randint(0, 50, [300, 2]).astype(np.int32)
                scatter_update = np.random.randint(-1, 4, [300, 50]).astype(np.int32)
                if const_scatter_inputs:
                    scatter_data = np.random.randint(-4, 5, add_input_shape).astype(np.int32)
                    scatter_inputs = (scatter_data, scatter_index, scatter_update, )
                else:
                    scatter_inputs = (scatter_index, scatter_update, )
                # Create model
                create_scatternd_model(model_path, scatter_inputs, add_input_shape,
                                       reduction, const_scatter_inputs, version)

                # Run tests with parser and compare result with runtime
                start = time.time()
                exit_status = run_parser(model_path, feed_dict, verify=True)
                assert exit_status
                end = time.time()
                print(end - start)
