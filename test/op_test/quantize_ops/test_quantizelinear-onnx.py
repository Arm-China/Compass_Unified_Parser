# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np
import onnx
from onnx import TensorProto, helper
from utils.run import run_parser
from utils.compare import compare_data


def create_quantize_model(onnx_path, inp_shape, scale_shape, output_shape, axis=2, version=13):
    ''' Create onnx model for quantize op.
    '''
    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, inp_shape)
    X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, scale_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.UINT8, output_shape)

    other_args = {}
    if version >= 13:
        other_args.update({'axis': axis})

    quantize = helper.make_node(
        'QuantizeLinear',
        inputs=['X1', 'X2'],
        outputs=['Y'],
        **other_args,
    )

    # Create graph
    graph_def = helper.make_graph(
        [quantize],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'quantize'
input_shapes = [[1, 1, 10, 12], [1, 1, 10, 12], [], [1], ]
scale_shapes = [[], [12], [], [1]]

# Generate input data
feed_dict = dict()
for input_shape, scale_shape in zip(input_shapes, scale_shapes):
    feed_dict['X1'] = np.random.ranf(input_shape).astype(np.float32)
    feed_dict['X2'] = np.random.ranf(scale_shape).astype(np.float32)
    np.save('input', feed_dict)

    for version in (10, 13):
        if version == 10 and len(scale_shape) > 0 and scale_shape[0] > 1:
            continue
        model_path = '-'.join([OP_NAME, str(version)]) + '.onnx'
        # Create model
        create_quantize_model(
            model_path, input_shape, scale_shape, input_shape, axis=-1, version=version)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(model_path, feed_dict)
        assert exit_status
