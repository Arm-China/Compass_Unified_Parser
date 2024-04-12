# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_atanh_model(onnx_path, input_size, output_size, version=9):
    ''' Create onnx model for atanh op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT16, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT16, output_size)

    atanh = helper.make_node(
        OP_NAME, ['X'],
        ['Y'],
    )
    graph_def = helper.make_graph(
        [atanh],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Atanh'

input_shapes = [[3, 2], [1, 2, 3, 4, 5], ]

for input_shape in input_shapes:
    X_data = np.random.ranf(input_shape).astype(np.float16)
    feed_dict = {'X': X_data}
    model_path = '-'.join([OP_NAME, str(len(input_shape))]) + '.onnx'
    create_atanh_model(model_path, input_shape, input_shape)
    # FIXME: Enable verify after opt supports this op
    exit_status = run_parser(model_path, feed_dict,
                             expected_keywords=['layer_type=Atanh'],
                             unexpected_keywords=['layer_top_type=[float32]'],
                             verify=False)
    assert exit_status
