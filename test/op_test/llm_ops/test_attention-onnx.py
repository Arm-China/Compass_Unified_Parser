# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_attention_model(onnx_path, input_shapes, output_shapes, mask_dtype,
                           kv_num_heads, q_num_heads, version=23):
    ''' Create onnx model for attention op.
    '''
    inp_list = []
    for i, shape in enumerate(input_shapes):
        if i == 3 and mask_dtype == bool:
            inp_list.append(helper.make_tensor_value_info(f'inp_{i}', TensorProto.BOOL, shape))
        else:
            inp_list.append(helper.make_tensor_value_info(f'inp_{i}', TensorProto.FLOAT, shape))

    out_list = []
    for i, shape in enumerate(output_shapes):
        out_list.append(helper.make_tensor_value_info(f'out_{i}', TensorProto.FLOAT, shape))

    if len(input_shapes[0]) == 3:
        extra_args = {'kv_num_heads': kv_num_heads, 'q_num_heads': q_num_heads}
    else:
        extra_args = {}

    op = helper.make_node(
        OP_NAME,
        inputs=[f'inp_{i}' for i in range(len(input_shapes))],
        outputs=[f'out_{i}' for i in range(len(output_shapes))],
        is_causal=0,
        qk_matmul_output_mode=3,
        softcap=3.5,
        **extra_args
    )
    graph_def = helper.make_graph(
        [op],  # nodes
        OP_NAME + '-model',  # name
        inp_list,  # inputs
        out_list,  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME + '-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'Attention'

cases = [
    # q_num_heads, q_seq_len, head_size, kv_num_heads, kv_seq_len, v_head_size, past_seq_len
    (3, 5, 4, 3, 5, 6, 10, bool, 'MHA', 3),  # 3D input, self-attention, MHA, bool mask, 3d
]

for case in cases:
    q_num_heads, q_seq_len, head_size, kv_num_heads, kv_seq_len, v_head_size, past_seq_len, mask_dtype, att_type, input_rank = case
    total_seq_len = past_seq_len + kv_seq_len
    output_shapes = [[2, q_num_heads, q_seq_len, v_head_size], [2, kv_num_heads, total_seq_len, head_size],
                     [2, kv_num_heads, past_seq_len + kv_seq_len, v_head_size], [2, q_num_heads, q_seq_len, total_seq_len], ]
    if input_rank == 3:
        q_shape = [2, q_seq_len, q_num_heads * head_size]
        k_shape = [2, kv_seq_len, kv_num_heads * head_size]
        v_shape = [2, kv_seq_len, kv_num_heads * v_head_size]
        att_mask = [2, q_num_heads, q_seq_len, total_seq_len]
        past_k_shape = [2, kv_num_heads, past_seq_len, head_size]
        past_v_shape = [2, kv_num_heads, past_seq_len, v_head_size]

        output_shapes[0] = [2, q_seq_len, q_num_heads * v_head_size]
    else:
        q_shape = [2, q_num_heads, q_seq_len, head_size]
        k_shape = [2, kv_num_heads, kv_seq_len, head_size]
        v_shape = [2, kv_num_heads, kv_seq_len, v_head_size]
        att_mask = [2, q_num_heads, q_seq_len, total_seq_len]
        past_k_shape = [2, kv_num_heads, past_seq_len, head_size]
        past_v_shape = [2, kv_num_heads, past_seq_len, v_head_size]
    input_shapes = [q_shape, k_shape, v_shape, att_mask, past_k_shape, past_v_shape]
    feed_dict = {}
    for i, shape in enumerate(input_shapes):
        if i == 3:
            if mask_dtype == bool:
                feed_dict[f'inp_{i}'] = np.random.randint(low=0, high=2, size=shape).astype(bool)
            else:
                feed_dict[f'inp_{i}'] = np.random.ranf(shape).astype(np.float32)
        else:
            feed_dict[f'inp_{i}'] = np.random.ranf(shape).astype(np.float32)
    model_path = '-'.join([OP_NAME, str(len(input_shapes[0])), str(mask_dtype), att_type]) + '.onnx'
    create_attention_model(model_path, input_shapes, output_shapes, mask_dtype, kv_num_heads, q_num_heads)
    exit_status = run_parser(model_path, feed_dict, verify=False)
    assert exit_status
