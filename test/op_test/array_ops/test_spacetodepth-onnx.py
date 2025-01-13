# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_space_to_depth_model(onnx_path, input_shape, output_shape, blocksize, version=13):
    ''' Create onnx model for space_to_depth op.
    '''

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)

    space_to_depth = helper.make_node(
        OP_NAME, ['X'],
        ['Y'],
        blocksize=blocksize
    )
    graph_def = helper.make_graph(
        [space_to_depth],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs

    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'SpaceToDepth'

input_shape = [2, 3, 4, 6]
feed_dict = {'X': np.random.ranf(input_shape).astype(np.float32)}
blocksizes = [1, 2]
for blocksize in blocksizes:
    model_path = '-'.join([OP_NAME, str(blocksize)]) + '.onnx'
    output_shape = [input_shape[0],
                    int(input_shape[1]*blocksize*blocksize),
                    int(input_shape[2]/blocksize),
                    int(input_shape[3]/blocksize)]
    create_space_to_depth_model(model_path, input_shape, output_shape, blocksize)
    exit_status = run_parser(model_path, feed_dict, verify=True)
    assert exit_status
