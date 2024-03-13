# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper
import tensorflow.compat.v1 as tf


def create_conv_transpose_model(onnx_path, input_size, output_size, weight_shape, version=1):
    ''' Create onnx model for conv_transpose op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    # W = np.random.ranf(weight_shape).astype(np.float32)
    W = np.ones(weight_shape).astype(np.float32)
    W_tensor = helper.make_tensor_value_info('W', TensorProto.FLOAT, W.shape)
    const_W = helper.make_node(
        'Constant',
        [],
        ['W'],
        value=helper.make_tensor(
            name='weight_value',
            data_type=TensorProto.FLOAT,
            dims=W.shape,
            vals=W,
        )
    )
    conv_transpose = helper.make_node(
        OP_NAME,
        inputs=['X', 'W'],
        outputs=['Y'],
        kernel_shape=[7, 8],
        output_padding=[1, 2],
        output_shape=[175, 180],
        strides=[3, 4],
        auto_pad='SAME_LOWER',
    )
    graph_def = helper.make_graph(
        [const_W, conv_transpose],  # nodes, sequences matters
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


def create_conv_transpose_model_2(onnx_path, input_size, output_size, weight_shape, version=1):
    ''' Create onnx model for conv_transpose op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    # W = np.random.ranf(weight_shape).astype(np.float32)
    W = np.ones(weight_shape).astype(np.float32)
    W_tensor = helper.make_tensor_value_info('W', TensorProto.FLOAT, W.shape)
    const_W = helper.make_node(
        'Constant',
        [],
        ['W'],
        value=helper.make_tensor(
            name='weight_value',
            data_type=TensorProto.FLOAT,
            dims=W.shape,
            vals=W,
        )
    )
    conv_transpose = helper.make_node(
        OP_NAME,
        inputs=['X', 'W'],
        outputs=['Y'],
        kernel_shape=[7, 8],
        output_padding=[1, 2],
        strides=[3, 4],
        auto_pad='SAME_UPPER',
    )
    graph_def = helper.make_graph(
        [const_W, conv_transpose],  # nodes, sequences matters
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


def create_conv_transpose_model_3(onnx_path, input_size, output_size, weight_shape, version=1):
    ''' Create onnx model for conv_transpose op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    # W = np.random.ranf(weight_shape).astype(np.float32)
    W = np.ones(weight_shape).astype(np.float32)
    W_tensor = helper.make_tensor_value_info('W', TensorProto.FLOAT, W.shape)
    const_W = helper.make_node(
        'Constant',
        [],
        ['W'],
        value=helper.make_tensor(
            name='weight_value',
            data_type=TensorProto.FLOAT,
            dims=W.shape,
            vals=W,
        )
    )
    conv_transpose = helper.make_node(
        OP_NAME,
        inputs=['X', 'W'],
        outputs=['Y'],
        kernel_shape=[7, 8],
        output_padding=[1, 2],
        strides=[3, 4],
        auto_pad='SAME_LOWER',
    )
    graph_def = helper.make_graph(
        [const_W, conv_transpose],  # nodes, sequences matters
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


def create_conv_transpose_model_4(onnx_path, input_size, output_size, weight_shape, version=1):
    ''' Create onnx model for conv_transpose op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    # W = np.random.ranf(weight_shape).astype(np.float32)
    W = np.ones(weight_shape).astype(np.float32)
    W_tensor = helper.make_tensor_value_info('W', TensorProto.FLOAT, W.shape)
    const_W = helper.make_node(
        'Constant',
        [],
        ['W'],
        value=helper.make_tensor(
            name='weight_value',
            data_type=TensorProto.FLOAT,
            dims=W.shape,
            vals=W,
        )
    )
    conv_transpose = helper.make_node(
        OP_NAME,
        inputs=['X', 'W'],
        outputs=['Y'],
        kernel_shape=[7, 8],
        output_padding=[1, 2],
        output_shape=[175, 180],
        strides=[3, 4],
        auto_pad='SAME_UPPER',
    )
    graph_def = helper.make_graph(
        [const_W, conv_transpose],  # nodes, sequences matters
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


def create_conv_transpose_model_5(onnx_path, input_size, output_size, weight_shape, version=1):
    ''' Create onnx model for conv_transpose op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    # W = np.random.ranf(weight_shape).astype(np.float32)
    W = np.ones(weight_shape).astype(np.float32)
    W_tensor = helper.make_tensor_value_info('W', TensorProto.FLOAT, W.shape)
    const_W = helper.make_node(
        'Constant',
        [],
        ['W'],
        value=helper.make_tensor(
            name='weight_value',
            data_type=TensorProto.FLOAT,
            dims=W.shape,
            vals=W,
        )
    )
    conv_transpose = helper.make_node(
        OP_NAME,
        inputs=['X', 'W'],
        outputs=['Y'],
        kernel_shape=[7, 8],
        #output_padding=[1, 2],
        output_shape=[60, 63],
        #strides=[3, 4],
        auto_pad='SAME_UPPER',
    )
    graph_def = helper.make_graph(
        [const_W, conv_transpose],  # nodes, sequences matters
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


# test_1
print('test_1')
OP_NAME = 'ConvTranspose'
input_shape = [2, 6, 57, 58]
output_shape = [2, 10, 171, 232]
weight_shape = [6, 10, 7, 8]
# Generate input data
feed_dict = dict()
feed_dict['X'] = np.ones(input_shape).astype(np.float32) * 100
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)
for version in (11, ):  # 1,
    model_name = '-'.join([OP_NAME, str(version)])
    model_path = model_name + '.onnx'
    create_conv_transpose_model_2(
        model_path, input_shape, output_shape, weight_shape, version)
    exit_status = run_parser(
        model_path, feed_dict, save_output=True, verify=False)
    assert exit_status


# test_2
print('test_2')
OP_NAME = 'ConvTranspose'
input_shape = [2, 6, 57, 58]
output_shape = [2, 10, 175, 180]
weight_shape = [6, 10, 7, 8]
# Generate input data
feed_dict = dict()
feed_dict['X'] = np.ones(input_shape).astype(np.float32) * 100
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)
for version in (11, ):  # 1,
    model_name = '-'.join([OP_NAME, str(version)])
    model_path = model_name + '.onnx'
    create_conv_transpose_model(
        model_path, input_shape, output_shape, weight_shape, version)
    exit_status = run_parser(
        model_path, feed_dict, save_output=True, verify=False)
    assert exit_status


# test_3
print('test_3')
OP_NAME = 'ConvTranspose'
input_shape = [2, 6, 57, 58]
output_shape = [2, 10, 175, 180]
weight_shape = [6, 10, 7, 8]
# Generate input data
feed_dict = dict()
feed_dict['X'] = np.ones(input_shape).astype(np.float32) * 100
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)
for version in (11, ):  # 1,
    model_name = '-'.join([OP_NAME, str(version)])
    model_path = model_name + '.onnx'
    create_conv_transpose_model_4(
        model_path, input_shape, output_shape, weight_shape, version)
    exit_status = run_parser(
        model_path, feed_dict, save_output=True, verify=False)
    assert exit_status


# test_4
print('test_4')
OP_NAME = 'ConvTranspose'
input_shape = [2, 6, 57, 58]
output_shape = [2, 10, 171, 232]
weight_shape = [6, 10, 7, 8]
# Generate input data
feed_dict = dict()
feed_dict['X'] = np.ones(input_shape).astype(np.float32) * 100
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)
for version in (11, ):  # 1,
    model_name = '-'.join([OP_NAME, str(version)])
    model_path = model_name + '.onnx'
    create_conv_transpose_model_3(
        model_path, input_shape, output_shape, weight_shape, version)
    exit_status = run_parser(
        model_path, feed_dict, save_output=True, verify=False)
    assert exit_status


# test_5
print('test_5')
OP_NAME = 'ConvTranspose'
input_shape = [2, 6, 57, 58]
output_shape = [2, 10, 60, 63]
weight_shape = [6, 10, 7, 8]
# Generate input data
feed_dict = dict()
feed_dict['X'] = np.ones(input_shape).astype(np.float32) * 100
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)
for version in (11, ):
    model_name = '-'.join([OP_NAME, str(version)])
    model_path = model_name + '.onnx'
    create_conv_transpose_model_5(
        model_path, input_shape, output_shape, weight_shape, version)
    exit_status = run_parser(
        model_path, feed_dict, save_output=True, verify=False)
    assert exit_status
