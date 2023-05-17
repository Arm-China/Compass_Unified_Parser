# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_gathernd_model(onnx_path, input_shape, indices_shape, output_shape, version=11):
    ''' Create onnx model for gathernd op.
    '''
    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_shape)
    X2 = helper.make_tensor_value_info('X2', TensorProto.INT64, indices_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)

    gathernd_node = helper.make_node(
        OP_NAME,
        inputs=['X1', 'X2'],
        outputs=['Y']
    )
    graph_def = helper.make_graph(
        [gathernd_node],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'GatherND'
input_shape = [1, 32, 32, 17]
output_shape = [1, 17]

# Generate input data
feed_dict = dict()
feed_dict['X1'] = np.random.ranf(input_shape).astype(np.float32) * 100
indices = np.array([[0, 30, 31]])
feed_dict['X2'] = indices
np.save('input.npy', feed_dict)

indices_shape = list(indices.shape)
model_path = OP_NAME + '.onnx'
# Create model
for opset_version in (11, 12, 13):
    create_gathernd_model(
        model_path, input_shape, indices_shape, output_shape, opset_version)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, verify=True)
    assert exit_status
