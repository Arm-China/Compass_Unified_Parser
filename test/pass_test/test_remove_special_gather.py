# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_gather_model(onnx_path, input_size, output_size, axis, version=13):
    ''' Create onnx model for gather op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    indices_shape = [input_size[axis]]
    indices_value = np.array(list(range(indices_shape[0])), dtype=np.int64)
    indices_tensor = helper.make_tensor_value_info('Indices', TensorProto.INT64, indices_shape)
    const_node = helper.make_node('Constant', [], ['Indices'],
                                  value=helper.make_tensor(name='const_value',
                                                           data_type=onnx.TensorProto.INT64,
                                                           dims=indices_shape,
                                                           vals=indices_value,
                                                           ))
    Y0 = helper.make_tensor_value_info('Y0', TensorProto.FLOAT, output_size)
    gather_node = helper.make_node(
        OP_NAME,
        inputs=['X', 'Indices'],
        outputs=['Y0'],
        axis=axis,
    )
    pow_node = helper.make_node(
        'Pow',
        inputs=['X', 'Y0'],
        outputs=['Y']
    )
    graph_def = helper.make_graph(
        [const_node, gather_node, pow_node],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Gather'
input_shape = [3, 2, 4, 5]
output_shape = input_shape

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32)

for axis in (-1, 0, 1):
    model_path = '-'.join([OP_NAME, str(axis)]) + '.onnx'
    # Create model
    create_gather_model(model_path, input_shape, output_shape, axis)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, unexpected_keywords=['layer_type=Gather'], verify=True)
    assert exit_status
