# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_reducesum_model(onnx_path, input_size, output_size, noop_with_empty_axes, keepdims=0, version=18):
    ''' Create onnx model for ReduceSum op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)
    # axis_input = helper.make_tensor_value_info('B', TensorProto.FLOATS, [0])
    axis_node = helper.make_node('Constant', [], ['B'],
                                 value=helper.make_tensor(name='const_value',
                                                          data_type=onnx.TensorProto.INT64,
                                                          dims=[0],
                                                          vals=np.array([]).astype(np.int64),
                                                          ))

    reduce_sum = helper.make_node(
        OP_NAME, ['X', 'B'],
        ['Y0'],
        noop_with_empty_axes=noop_with_empty_axes,
        keepdims=keepdims
    )
    add_node = helper.make_node(
        'Add',
        inputs=['X', 'Y0'],
        outputs=['Y']
    )
    graph_def = helper.make_graph(
        [axis_node, reduce_sum, add_node],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'ReduceSum'

input_shape = [1, 3, 2]
feed_dict = {'X': np.random.ranf(input_shape).astype(np.float32)}
np.save('input.npy', feed_dict)
for noop_with_empty_axes in (0, 1):
    output_shape = input_shape
    model_path = '-'.join([OP_NAME, str(noop_with_empty_axes)]) + '.onnx'
    create_reducesum_model(model_path, input_shape, output_shape, noop_with_empty_axes)
    # FIXME: onnxruntime has issues in Reduce ops with noop_with_empty_axes=1.
    # The output tensor is not same as input but from the definition they should be the same.
    verify = False if noop_with_empty_axes == 1 else True
    exit_status = run_parser(model_path, feed_dict, verify=verify)
    assert exit_status
