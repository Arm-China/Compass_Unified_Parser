# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import os
import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_model(onnx_path, input0_size, input1_size, output_size, version=13):
    ''' Create onnx model for testing preprocess plugin.
    '''
    X0 = helper.make_tensor_value_info('ONNX_INPUT0', TensorProto.FLOAT, input0_size)
    X1 = helper.make_tensor_value_info('ONNX_INPUT1', TensorProto.FLOAT, input1_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    add = helper.make_node(
        'Add', ['ONNX_INPUT0', 'ONNX_INPUT1'],
        ['Y'],
    )
    graph_def = helper.make_graph(
        [add],  # nodes
        'model',  # name
        [X0, X1],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name='model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


model_path = 'preprocess.onnx'

input0_shape = [1, 3, 2]
input1_shape = [1, 4, 3, 2]
output_shape = [1, 4, 3, 2]
feed_dict = {'ONNX_INPUT0': np.random.ranf(input0_shape).astype(np.float32),
             'ONNX_INPUT1': np.random.ranf(input1_shape).astype(np.float32)}
create_model(model_path, input0_shape, input1_shape, output_shape)
# Set environment variable AIPUPLUGIN_PATH
os.environ["AIPUPLUGIN_PATH"] = os.path.join(os.path.dirname(__file__), 'plugins')
exit_status = run_parser(model_path, feed_dict, verify=False,
                         expected_keywords=['layer_type=Preprocess', 'bias',
                                            'axis=1', 'layer_top_shape=[[1,300,2]]', 'layer_top_shape=[[1,400,300,2]]'])
assert exit_status
