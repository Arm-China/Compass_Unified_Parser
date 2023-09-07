# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)
    indices = np.reshape(np.array(list(np.ndindex(*update_size))), update_size + [len(update_size)])

    indices_tensor = create_initializer_tensor(
        name='indice',
        tensor_array=indices,
        data_type=onnx.TensorProto.INT64)

    scatternd = helper.make_node(
        OP_NAME,
        inputs=['X1', 'indice', 'X2'],
        outputs=['Y'],
    )
    graph_def = helper.make_graph(
        [scatternd],  # nodes
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


OP_NAME = 'ScatterND'
input_shapes = [[3, 4, 5, 6], [1, 1, 3, 4], [1, 2500, 256], ]
update_shapes = input_shapes
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
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, unexpected_keywords=['layer_type=ScatterND'], verify=True)
    assert exit_status
