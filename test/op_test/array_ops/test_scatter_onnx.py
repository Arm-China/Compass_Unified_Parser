# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_scatter_model(onnx_path, input_size1, input_size2, output_size, axis, version=14):
    ''' Create onnx model for scatter op.
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

    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size1)
    X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, input_size2)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)
    #Y4 = np.array([[-7,-9,-10,-4,-8,4,-5],[1,9,-1,9,5,-8,-2],[-9,-8,2,2,1,4,-2],[-1,-1,4,7,6,6,4],[-4,7,4,6,-4,0,-3]])
    Y4 = np.array([[-2, -4, -1, -4, -3, 4, -2], [1, 3, -1, 3, 2, -2, -2],
                  [-4, -4, 2, 2, 1, 4, -2], [-1, -1, 4, 2, 3, 3, 1], [-4, 2, 4, 2, -4, 0, -3]])
    Y4_tensor_name = 'Y4'

    indices_tensor = create_initializer_tensor(
        name=Y4_tensor_name,
        tensor_array=Y4,
        data_type=onnx.TensorProto.INT32)

    gather_node = helper.make_node(
        OP_NAME,
        inputs=['X1', 'Y4', 'X2'],
        outputs=['Y'],
        axis=axis
    )
    graph_def = helper.make_graph(
        [gather_node],  # nodes
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


OP_NAME = 'Scatter'
input_shape = [[5, 10], [5, 7]]
output_shape = [5, 10]

# Generate input data
feed_dict = dict()
feed_dict['X1'] = np.random.ranf(input_shape[0]).astype(np.float32)
feed_dict['X2'] = np.random.ranf(input_shape[1]).astype(np.float32)
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)

for axis in (1, 0, -1, -2):
    model_name = '-'.join([OP_NAME, str(axis)])
    model_path = model_name + '.onnx'
    # Create model
    create_scatter_model(
        model_path, input_shape[0], input_shape[1], output_shape, axis, 9)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type=None, save_output=True, verify=True)
    assert exit_status
