# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from utils.common import get_feed_dict
from onnx import TensorProto, helper


def create_const_model(onnx_path, input_shape, version=13):
    ''' Create onnx model for const op.
    '''

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y0 = helper.make_tensor_value_info('Y0', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_shape)

    attrs = {}
    if input_shape[-1] == 1:
        attrs.update({'value_float': 11.22})
    else:
        value_floats = np.random.ranf([input_shape[-1]]) * 11.22
        attrs.update({'value_floats': value_floats.tolist()})

    const = helper.make_node(
        OP_NAME, [],
        ['Y0'],
        **attrs
    )
    add = helper.make_node(
        'Add', ['X', 'Y0'],
        ['Y'],
    )
    graph_def = helper.make_graph(
        [const, add],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs

    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Constant'

model_path = OP_NAME + '.onnx'
for input_shape in ([4, 5, 1], [3, 6]):
    input_data = np.random.ranf(input_shape).astype(np.float32)
    feed_dict = {'X': input_data}
    create_const_model(model_path, input_shape)
    exit_status = run_parser(model_path, feed_dict, verify=True)
    assert exit_status
