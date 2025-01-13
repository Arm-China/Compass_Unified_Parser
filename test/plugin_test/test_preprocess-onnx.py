# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import os
import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_sinh_model(onnx_path, input_size, output_size, version=13):
    ''' Create onnx model for sinh op.
    '''
    X = helper.make_tensor_value_info('ONNX_INPUT', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    sinh = helper.make_node(
        'Sinh', ['ONNX_INPUT'],
        ['Y'],
    )
    graph_def = helper.make_graph(
        [sinh],  # nodes
        'model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name='model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


model_path = 'preprocess.onnx'

input_shape = [1, 3, 2]
output_shape = [1, 3, 2]
feed_dict = {'ONNX_INPUT': np.random.ranf(input_shape).astype(np.float32)}
create_sinh_model(model_path, input_shape, output_shape)
# Set environment variable AIPUPLUGIN_PATH
os.environ["AIPUPLUGIN_PATH"] = os.path.join(os.path.dirname(__file__), 'plugins')
exit_status = run_parser(model_path, feed_dict, verify=False,
                         expected_keywords=['layer_type=MyPreprocess', 'bias',
                                            'axis=3', 'layer_top_shape=[[1,4,300,300]]'],
                         unexpected_keywords=['layer_top_shape=[[1,3,2]]'])
assert exit_status
