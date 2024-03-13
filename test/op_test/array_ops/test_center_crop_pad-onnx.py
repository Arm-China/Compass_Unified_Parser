# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_center_crop_pad_model(onnx_path, input_size, output_size, shape, const_input=False, version=18):
    ''' Create onnx model for CenterCropPad op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    shape_tensor = helper.make_tensor_value_info('Shape', TensorProto.INT64, [4])
    shape_node = helper.make_node('Constant', [], ['Shape'],
                                  value=helper.make_tensor(name='const_value',
                                                           data_type=onnx.TensorProto.INT64,
                                                           dims=[4],
                                                           vals=np.array(shape).astype(np.int64),
                                                           ))
    nodes = [shape_node]
    if const_input:
        helper.make_tensor_value_info('const_input', TensorProto.FLOAT, input_size)
        const_input_node = helper.make_node(
            'Constant', [], ['const_input'],
            value=helper.make_tensor(name='const_input',
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=input_size,
                                     vals=np.random.randint(-3, 4, input_size).astype(np.float32),
                                     )
        )
        crop_pad_inputs = ['const_input', 'Shape']
        nodes.append(const_input_node)
    else:
        crop_pad_inputs = ['X', 'Shape']
    Y0 = helper.make_tensor_value_info('Y0', TensorProto.FLOAT, output_size)
    center_crop_pad_node = helper.make_node(
        OP_NAME,
        inputs=crop_pad_inputs,
        outputs=['Y0']
    )
    add_node = helper.make_node(
        'Add',
        inputs=['Y0', 'X'],
        outputs=['Y']
    )
    graph_def = helper.make_graph(
        nodes + [center_crop_pad_node, add_node],  # nodes
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'CenterCropPad'
input_shape = [3, 2, 1, 5]
output_shape = input_shape

# Generate input data
feed_dict = dict()
feed_dict['X'] = np.random.randint(0, 4, input_shape).astype(np.float32)
np.save('input', feed_dict)

for idx, shape in enumerate([[3, 1, 4, 1], [1, 2, 5, 5]]):
    for const_input in (False, True, ):
        model_name = '-'.join([OP_NAME, str(idx), str(const_input)])
        model_path = model_name + '.onnx'
        # Create model
        create_center_crop_pad_model(
            model_path, input_shape, output_shape, shape, const_input)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, verify=True)
        assert exit_status
