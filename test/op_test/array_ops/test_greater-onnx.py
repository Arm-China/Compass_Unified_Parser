# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_greater_model(onnx_path, input_size, output_size, version=14):
    ''' Create onnx model for greater op.
    '''
    X1 = helper.make_tensor_value_info('X1', TensorProto.INT64, input_size)
    X2 = helper.make_tensor_value_info('X2', TensorProto.INT64, [1])
    Y = helper.make_tensor_value_info('Y', TensorProto.INT64, output_size)

    helper.make_tensor_value_info('Sub1', TensorProto.INT64, input_size)
    sub_node1 = helper.make_node('Sub', inputs=['X1', 'X2'], outputs=['Sub1'])
    shape_tensor = helper.make_tensor_value_info('Shape', TensorProto.INT64, [1])
    shape_node = helper.make_node(
        'Shape',
        ['Sub1'],
        ['Shape']
    )
    helper.make_tensor_value_info('One', TensorProto.INT64, [1])
    const_node = helper.make_node('Constant', [], ['One'],
                                  value=helper.make_tensor(name='const_value',
                                                           data_type=onnx.TensorProto.INT64,
                                                           dims=[1],
                                                           vals=np.array([1]).astype(np.int64),
                                                           ))
    helper.make_tensor_value_info('Mul', TensorProto.INT64, input_size)
    mul_node = helper.make_node('Mul', inputs=['Shape', 'One'], outputs=['Mul'])
    helper.make_tensor_value_info('Sub2', TensorProto.INT64, input_size)
    sub_node2 = helper.make_node('Sub', inputs=['X2', 'One'], outputs=['Sub2'])
    greater_tensor = helper.make_tensor_value_info('Greater', TensorProto.INT64, [1])
    greater_node = helper.make_node(OP_NAME, inputs=['Sub2', 'Mul'], outputs=['Greater'])
    where_node = helper.make_node(
        'Where',
        inputs=['Greater', 'Mul', 'Sub2'],
        outputs=['Y']
    )
    graph_def = helper.make_graph(
        [sub_node1, shape_node, const_node, sub_node2, mul_node, greater_node, where_node],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Greater'
input_shape = [1, 3, 2, 4, 5]
output_shape = [1]

# Generate input data
feed_dict = dict()
feed_dict['X1'] = np.random.randint(-20, 30, input_shape)
feed_dict['X2'] = np.array([10])
input_data_path = 'input.npy'
# np.save(input_data_path, feed_dict)

model_path = OP_NAME + '.onnx'
# Create model
create_greater_model(model_path, input_shape, output_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict, verify=True)
assert exit_status
