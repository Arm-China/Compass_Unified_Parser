# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_col2im_model(onnx_path, input_shape, output_shape, version=18):
    ''' Create onnx model for Col2Im op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)

    image_shape = helper.make_node('Constant', [], ['image_shape'], value_ints=[8, 9, 6])
    block_shape = helper.make_node('Constant', [], ['block_shape'], value_ints=[2, 3, 2])

    Col2Im = helper.make_node(
        OP_NAME,
        inputs=['X', 'image_shape', 'block_shape'],
        outputs=['Y'],
        pads=[0, 1, 2, 1, 0, 1],
        strides=[1, 2, 1],
    )
    graph_def = helper.make_graph(
        [image_shape, block_shape, Col2Im],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Col2Im'
input_shape = [2, 36, 256]
output_shape = [2, 3, 8, 9, 6]

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32)
# np.save('input', feed_dict)

model_path = '-'.join([OP_NAME, ]) + '.onnx'
# Create model
create_col2im_model(model_path, input_shape, output_shape)

# FIXME: Enable verify after Col2Im is supported by opt
exit_status = run_parser(model_path, feed_dict, expected_keywords=['layer_type=Col2Im'], verify=False)
assert exit_status
