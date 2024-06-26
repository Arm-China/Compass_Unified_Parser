# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_scatter_model(onnx_path, input_size, update_size, output_size, version):
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

    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
    X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, update_size)
    X3 = helper.make_tensor_value_info('X3', TensorProto.FLOAT, update_size)
    Y0 = helper.make_tensor_value_info('Y0', TensorProto.FLOAT, output_size)
    Y1 = helper.make_tensor_value_info('Y1', TensorProto.FLOAT, output_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)
    indices = np.expand_dims(np.array(list(np.ndindex(*input_size[:-1]))), list(range(len(input_size)-2)))
    indices1, indices2 = np.split(indices, 2, -2)

    indices1_tensor = create_initializer_tensor(
        name='indice1',
        tensor_array=indices1,
        data_type=onnx.TensorProto.INT64)
    indices2_tensor = create_initializer_tensor(
        name='indice2',
        tensor_array=indices2,
        data_type=onnx.TensorProto.INT64)

    scatternd1 = helper.make_node(
        OP_NAME,
        inputs=['X1', 'indice1', 'X2'],
        outputs=['Y0'],
    )
    scatternd2 = helper.make_node(
        OP_NAME,
        inputs=['Y0', 'indice2', 'X3'],
        outputs=['Y1'],
    )
    add = helper.make_node(
        'Add',
        inputs=['X1', 'Y1'],
        outputs=['Y'],
    )
    graph_def = helper.make_graph(
        [scatternd1, scatternd2, add],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2, X3],  # inputs
        [Y],  # outputs
        initializer=[indices1_tensor, indices2_tensor],
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'ScatterND'
input_shapes = [[1, 2, 250, 5], ]
update_shapes = [[1, 1, 250, 5], ]
output_shapes = input_shapes

# Generate input data
feed_dict = dict()

for input_shape, update_shape, output_shape in zip(input_shapes, update_shapes, output_shapes):
    model_name = '-'.join([OP_NAME, str(len(input_shape))])
    model_path = model_name + '.onnx'

    # Create model
    create_scatter_model(
        model_path, input_shape, update_shape, output_shape, 11)

    feed_dict['X1'] = np.random.ranf(input_shape).astype(np.float32)
    feed_dict['X2'] = np.random.ranf(update_shape).astype(np.float32)
    feed_dict['X3'] = np.random.ranf(update_shape).astype(np.float32) * 100
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, expected_keywords=['layer_type=Concat'],
        unexpected_keywords=['layer_type=ScatterND'], verify=True)
    assert exit_status
