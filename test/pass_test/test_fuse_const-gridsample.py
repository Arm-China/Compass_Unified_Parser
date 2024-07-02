# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_gridsample_model(onnx_path, input_size, output_size, grid_size, mode, padding_mode, align_corners, version=20):
    ''' Create onnx model for gridsample op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, output_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)
    const_nodes = []
    for input_name in ('const_x', 'const_grid', ):
        tensor_value_shape = input_size if input_name == 'const_x' else grid_size
        data_type = TensorProto.FLOAT
        tensor_value = np.random.ranf(tensor_value_shape).astype(np.float32)
        if input_name == 'const_grid':
            tensor_value = 2.0 * tensor_value - 1.0
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
        const_nodes.append(const_node)

    helper.make_tensor_value_info('GS_Y', TensorProto.FLOAT, output_size)
    gridsample = helper.make_node(
        OP_NAME,
        inputs=['const_x', 'const_grid'],
        outputs=['GS_Y'],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )
    add = helper.make_node('Add', inputs=['GS_Y', 'X'], outputs=['Y'])
    graph_def = helper.make_graph(
        const_nodes + [gridsample, add],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME + '-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'GridSample'
input_shapes = [[1, 3, 16, 32], [2, 4, 6, 7, 8], [2, 3, 30], ]
add_input_shapes = [[1, 3, 64, 63], [2, 4, 10, 11, 12], [2, 3, 33], ]
grid_shapes = [[1, 64, 63, 2], [2, 10, 11, 12, 3], [2, 33, 1], ]

# Generate input data
feed_dict = dict()
for input_shape, add_input_shape, grid_shape in zip(input_shapes, add_input_shapes, grid_shapes):
    feed_dict['X'] = np.random.ranf(add_input_shape).astype(np.float32) * 100
    # input_data_path = 'input.npy'
    # np.save(input_data_path, feed_dict)
    for version in (16, 20):
        if len(input_shape) != 4 and version == 16:
            # opset 16 only supports 4d
            continue
        for mode in ('linear', 'nearest', 'cubic'):
            if version == 16:
                if mode == 'linear':
                    mode = 'bilinear'
                elif mode == 'cubic':
                    mode = 'bicubic'
            for padding_mode in ('border', 'reflection', 'zeros', ):
                for align_corners in (0, 1):
                    model_name = '-'.join([OP_NAME, str(len(input_shape)), mode,
                                           padding_mode, str(align_corners)])
                    model_path = model_name + '.onnx'
                    # Create model
                    create_gridsample_model(
                        model_path, input_shape, add_input_shape,
                        grid_shape, mode, padding_mode, align_corners, version)

                    # TODO: Enable verify after GridSample-20 is supported in onnxruntime.
                    # Currently Opset 20 is under development and support for this is limited.
                    verify = False if version == 20 else True
                    exit_status = run_parser(model_path, feed_dict, verify=verify)
                    assert exit_status
