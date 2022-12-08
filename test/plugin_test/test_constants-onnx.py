# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_pow_model(onnx_path, input_size, output_size, exponent, version=14):
    ''' Create onnx model for pow op.
    '''
    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, input_size)
    X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, input_size)
    pow_tensor = helper.make_tensor_value_info('Pow', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    const_node = helper.make_node('Constant', [], ['Exponent'],
                                  value=helper.make_tensor(name='const_value',
                                                           data_type=onnx.TensorProto.FLOAT,
                                                           dims=list(exponent.shape),
                                                           vals=exponent,
                                                           ))
    pow_node = helper.make_node(
        OP_NAME,
        inputs=['X1', 'Exponent'],
        outputs=['Pow']
    )
    add_node = helper.make_node(
        'Add',
        inputs=['X2', 'Pow'],
        outputs=['Y']
    )
    graph_def = helper.make_graph(
        [const_node, pow_node, add_node],  # nodes
        OP_NAME + '-model',  # name
        [X1, X2],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


# This test works with parser plugin: test/plugin_test/plugins/aipubt_parser_plugin_onnx_pow.py
OP_NAME = 'Pow'

# Set environment variable AIPUPLUGIN_PATH
os.environ['AIPUPLUGIN_PATH'] = os.path.join(os.path.dirname(__file__), 'plugins')

input_shape = [1, 3, 5, 4, 2]
output_shape = input_shape

# Generate input data
feed_dict = dict()
feed_dict['X1'] = np.random.ranf(input_shape).astype(np.float32) * 100
feed_dict['X2'] = np.random.ranf(input_shape).astype(np.float32)

for exponent in ([-3, 2], [-5.1], ):
    model_name = '-'.join([OP_NAME, str(len(exponent))])
    model_path = model_name + '.onnx'
    # Create model
    create_pow_model(
        model_path, input_shape, output_shape, np.array(exponent), 14)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type=None, save_output=True, verify=False,
        expected_keywords=['for_test', 'exponent_offset', 'exponent_1_offset'])
    assert exit_status
