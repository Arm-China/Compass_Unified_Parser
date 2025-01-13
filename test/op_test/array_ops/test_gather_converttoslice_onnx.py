# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_gather_model(onnx_path, input_size, output_size, indices, version=14):
    ''' Create onnx model for gather op.
    '''
    def create_initializer_tensor(
            name: str,
            tensor_array: np.ndarray,
            data_type: onnx.TensorProto = onnx.TensorProto.FLOAT) -> onnx.TensorProto:

        # (TensorProto)
        initializer_tensor = onnx.helper.make_tensor(
            name=name,
            data_type=data_type,
            dims=tensor_array.shape,
            vals=tensor_array.flatten().tolist())

        return initializer_tensor

    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
    X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, [1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)
    Y4 = np.array(-1)
    Y4_tensor_name = 'Y4'

    indices_tensor = create_initializer_tensor(
        name=Y4_tensor_name,
        tensor_array=Y4,
        data_type=onnx.TensorProto.INT32)

    gather_tensor = helper.make_tensor_value_info('Gather', TensorProto.FLOAT, [1])
    gather_node = helper.make_node(
        OP_NAME,
        inputs=['X1', 'Y4'],
        outputs=['Gather'],
    )
    pow_node = helper.make_node(
        'Pow',
        inputs=['X2', 'Gather'],
        outputs=['Y']
    )
    graph_def = helper.make_graph(
        [gather_node, pow_node],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs
        initializer=[indices_tensor],
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Gather'
input_shape = [10, 20]
output_shape = [20]

# Generate input data
feed_dict = dict()
feed_dict['X1'] = np.random.ranf(input_shape).astype(np.float32)
feed_dict['X2'] = np.array([10]).astype(np.float32)
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)

for indices in (-1, 0):
    model_name = '-'.join([OP_NAME, str(indices)])
    model_path = model_name + '.onnx'
    # Create model
    create_gather_model(
        model_path, input_shape, output_shape, indices, 13)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type=None, save_output=True, verify=True)
    assert exit_status
