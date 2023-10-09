# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from utils.common import get_feed_dict
from onnx import TensorProto, helper


def create_cast_model(onnx_path, input_shape, to_type, version=13):
    ''' Create onnx model for cast op.
    '''

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', to_dtype, input_shape)

    cast = helper.make_node(
        OP_NAME, ['X'],
        ['Y'],
        to=to_dtype
    )
    graph_def = helper.make_graph(
        [cast],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs

    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Cast'

input_data = np.array([-256.0, 256.0, 0.0, 174.32, -10.0, 1024.0], np.float32)
feed_dict = {'X': input_data}
input_shape = list(input_data.shape)
for to_dtype in (TensorProto.UINT8, TensorProto.BOOL):
    model_path = '-'.join([OP_NAME, str(int(to_dtype))]) + '.onnx'
    create_cast_model(model_path, input_shape, to_dtype)
    exit_status = run_parser(model_path, feed_dict, verify=True)
    assert exit_status
    print(get_feed_dict('onnx_outputs.npy'))
    # uint8 output: [0, 0, 0, 174, 246, 0]
    # bool output: [True, True, False, True, True, True]
