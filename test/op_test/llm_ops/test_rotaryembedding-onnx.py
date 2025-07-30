# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from onnx.reference.ops.op_rotary_embedding import rotary_embedding

from utils.run import run_parser
from onnx import TensorProto, helper


def create_rope_model(onnx_path, input_shapes, output_shapes, version=23):
    ''' Create onnx model for rope op.
    '''
    inp_list = []
    for i, shape in enumerate(input_shapes):
        inp_list.append(helper.make_tensor_value_info(f'inp_{i}', TensorProto.FLOAT, shape))

    num_heads = 5
    # cos/sin_cache
    head_size = input_shapes[0][-1] if len(input_shapes[0]) == 4 else input_shapes[0][-1] // num_heads
    seq_len = input_shapes[0][2] if len(input_shapes[0]) == 4 else input_shapes[0][1]
    cache_shape = [input_shapes[0][0], seq_len, head_size // 2]
    cos_cache = np.random.ranf(cache_shape).astype(np.float32).flatten()
    sin_cache = np.random.ranf(cache_shape).astype(np.float32).flatten()
    inp_list.append(helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT, cache_shape))
    inp_list.append(helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT, cache_shape))

    cos_cache_tensor = helper.make_tensor(name='cos_cache', data_type=TensorProto.FLOAT,
                                          dims=cache_shape, vals=cos_cache, raw=False)
    sin_cache_tensor = helper.make_tensor(name='sin_cache', data_type=TensorProto.FLOAT,
                                          dims=cache_shape, vals=sin_cache, raw=False)
    initializer = [cos_cache_tensor, sin_cache_tensor]

    out_list = []
    for i, shape in enumerate(output_shapes):
        out_list.append(helper.make_tensor_value_info(f'out_{i}', TensorProto.FLOAT, shape))

    if len(input_shapes[0]) == 3:
        extra_args = {'num_heads': num_heads}
    else:
        extra_args = {}

    op = helper.make_node(
        OP_NAME,
        inputs=['inp_0', 'cos_cache', 'sin_cache'],
        outputs=[f'out_{i}' for i in range(len(output_shapes))],
        rotary_embedding_dim=2,
        **extra_args
    )
    graph_def = helper.make_graph(
        [op],  # nodes
        OP_NAME + '-model',  # name
        inp_list,  # inputs
        out_list,  # outputs
        initializer=initializer
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME + '-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'RotaryEmbedding'

cases = [
    [2, 5, 10, 4],  # 4D [batch_size, num_heads, sequence_length, head_size]
    [2, 10, 30],  # 3D [batch_size, sequence_length, hidden_size]
]

for case in cases:
    input_shapes = [case]
    output_shapes = [case]
    feed_dict = {}
    for i, shape in enumerate(input_shapes):
        feed_dict[f'inp_{i}'] = np.random.ranf(shape).astype(np.float32)
    model_path = '-'.join([OP_NAME, str(len(input_shapes[0]))]) + '.onnx'
    create_rope_model(model_path, input_shapes, output_shapes)
    exit_status = run_parser(model_path, feed_dict, verify=False)
    assert exit_status
