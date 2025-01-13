# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_nonzero_model(onnx_path, input_size, output_size, is_const_input=False, version=13):
    ''' Create onnx model for nonzero op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    if is_const_input:
        helper.make_tensor_value_info('Const', TensorProto.INT64, [2, 2])
        const_node = helper.make_node('Constant', [], ['Const'],
                                      value=helper.make_tensor(name='const_value',
                                                               data_type=onnx.TensorProto.INT64,
                                                               dims=[2, 2],
                                                               vals=np.array([[0, 3], [-10, 0]]).astype(np.int64),
                                                               ))
        helper.make_tensor_value_info('X0', TensorProto.INT64, [2, 0])
        nonzero = helper.make_node(OP_NAME, ['Const'], ['X0'])
        helper.make_tensor_value_info('X1', TensorProto.INT64, [])
        sum_node = helper.make_node('ReduceSum', ['X0'], ['X1'], keepdims=0)
        pow_node = helper.make_node('Pow', ['X', 'X1'], ['Y'])
        nodes = [const_node, nonzero, sum_node, pow_node]
    else:
        nonzero = helper.make_node(OP_NAME, ['X'], ['Y'])
        nodes = [nonzero]
    graph_def = helper.make_graph(
        nodes,  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'NonZero'

input_shape = [1, 3, 2]
output_shape = [4, 2]  # output shape is not fixed if input is not constant
feed_dict = {'X': np.random.randint(-5, 5, input_shape).astype(np.float32)}
for is_const_input in (True, False):
    model_path = '-'.join([OP_NAME, str(is_const_input)]) + '.onnx'
    create_nonzero_model(model_path, input_shape, output_shape, is_const_input)

    # FIXME: Enable assert and change verify to True for non const input after NonZero is supported
    exit_status = run_parser(model_path, feed_dict, verify=is_const_input)
    if is_const_input:
        assert exit_status
