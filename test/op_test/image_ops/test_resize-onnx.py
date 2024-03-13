# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_resize_model(onnx_path, input_size, target_size, keep_aspect_ratio_policy, version=11):
    ''' Create onnx model for resize op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, target_size)
    input_nodes = []
    for input_name in ('roi', 'scales', 'sizes'):
        if input_name == 'sizes':
            tensor_value_shape = [4]
            tensor_vals = target_size
        else:
            tensor_value_shape = [0]
            tensor_vals = []
        data_type = TensorProto.INT64 if input_name == 'sizes' else TensorProto.FLOAT
        const_tensor = helper.make_tensor_value_info(input_name, data_type, tensor_value_shape)
        const_node = helper.make_node(
            'Constant',
            [],
            [input_name],
            value=helper.make_tensor(
                name=input_name + '_value',
                data_type=data_type,
                dims=tensor_value_shape,
                vals=tensor_vals,
            )
        )
        input_nodes.append(const_node)
    if version == 11:
        extra_args = {}
    else:
        extra_args = {'keep_aspect_ratio_policy': keep_aspect_ratio_policy}
    resize = helper.make_node(
        OP_NAME,
        inputs=['X', 'roi', 'scales', 'sizes'],
        outputs=['Y'],
        coordinate_transformation_mode='half_pixel',
        mode='nearest',
        nearest_mode='round_prefer_ceil',
        **extra_args
    )
    graph_def = helper.make_graph(
        input_nodes + [resize],  # nodes, sequences matters
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y]  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Resize'
input_shape = [2, 3, 20, 40]
# target_size = [2, 3, 19, 34] failed because scales will be [1., 0.6666667, 0.85, 0.825] if keep_aspect_ratio_policy is not_larger.
# In this case, the output shape will become [2, 2, 17, 33], which is not supported now because the scale at channel dim is not 1.
target_sizes = [[2, 3, 19, 44], [2, 3, 21, 39], ]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32)

for version in (18, 11, ):
    for keep_aspect_ratio_policy in ('not_smaller', 'not_larger', 'stretch'):
        if version == 11 and keep_aspect_ratio_policy != 'stretch':
            continue
        for idx, target_size in enumerate(target_sizes):
            model_name = '-'.join([OP_NAME, keep_aspect_ratio_policy, str(idx), str(version)])
            model_path = model_name + '.onnx'
            # Create model
            create_resize_model(model_path, input_shape, target_size, keep_aspect_ratio_policy, version)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(model_path, feed_dict, verify=True)
            assert exit_status
