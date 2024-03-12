# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from onnx import TensorProto, helper
from utils.run import run_parser


def create_resize_model(onnx_path, input_shape, output_shape, mode, coordinate_transformation_mode,
                        exclude_outside, antialias, provide_sizes=False, version=18):
    ''' Create onnx model for resize op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)

    input_nodes = []
    tensor_names = []
    for input_name in ('roi', 'scales', 'sizes'):
        if provide_sizes:
            if input_name == 'sizes':
                tensor_value_shape = [len(output_shape)]
                data_type = TensorProto.INT64
                tensor_value = output_shape
            else:
                tensor_value_shape = [0]
                data_type = TensorProto.FLOAT
                tensor_value = []
        else:
            if input_name == 'scales':
                tensor_value_shape = [4]
                data_type = TensorProto.FLOAT
                tensor_value = (np.array(output_shape) / np.array(input_shape)).tolist()
            elif input_name == 'sizes':
                continue
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
        tensor_names.append(input_name)
    resize = helper.make_node(
        'Resize', ['X'] + tensor_names,
        ['Y'],
        mode=mode,
        coordinate_transformation_mode=coordinate_transformation_mode,
        exclude_outside=exclude_outside,
        antialias=antialias,
    )
    graph_def = helper.make_graph(
        input_nodes + [resize],  # nodes
        'resize-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name='resize-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


TEST_NAME = 'resize'
input_shapes = [[3, 4, 11, 12], [3, 4, 5, 6], ]
output_shapes = [[3, 4, 5, 6], [3, 4, 11, 12], ]

feed_dict = {}
for input_shape, output_shape in zip(input_shapes, output_shapes):
    # Generate input data
    feed_dict['X'] = (np.random.randint(-100, 200, input_shape).astype(np.float32) * 1e-2)

    for exclude_outside in (False, True, ):
        for antialias in (True, False, ):
            for mode in ('linear', 'nearest', ):  # 'cubic'
                if mode == 'nearest' and antialias:
                    continue
                if mode != 'cubic' and exclude_outside:
                    continue
                for coordinate_transformation_mode in ('align_corners', 'half_pixel', ):
                    # if mode == 'cubic' and coordinate_transformation_mode == 'align_corners':
                    #     continue
                    model_path = '-'.join([TEST_NAME, str(len(input_shape)), mode, coordinate_transformation_mode,
                                           str(exclude_outside), str(antialias)])

                    onnx_model_path = model_path + '.onnx'
                    # Create onnx model
                    create_resize_model(onnx_model_path, input_shape, output_shape,
                                        mode, coordinate_transformation_mode,
                                        exclude_outside, antialias)

                    # Run tests with parser and compare result with runtime
                    exit_status = run_parser(
                        onnx_model_path, feed_dict, model_type='onnx', verify=True)
                    assert exit_status
