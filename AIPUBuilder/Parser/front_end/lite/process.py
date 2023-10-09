# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


from .load import convert_tflite_to_graph
from .passes.front_passes import split_op_has_activation, split_fc, split_greater_or_less_equal, \
    split_not_equal, split_rsqrt, remove_detection_postprocess, convert_to_onnx, convert_onehot, convert_reverse_sequence, convert_square, \
    convert_unpack, convert_negative_pool_pad, convert_scatternd, convert_scatternd2, convert_special_uni_seq_lstm, convert_strided_slice, convert_square_diff, \
    convert_broadcast_to, remove_redundant_broadcast_to, remove_sub_equal_select, \
    merge_special_cast_quantize, convert_special_quantize, convert_special_dequantize, split_quatized_mean, \
    merge_quantized_instance_norm, merge_quantized_lstm_cell, convert_dequantize, merge_min_quant_max_to_clip, merge_quantized_ln, \
    merge_dqd, convert_sparse_to_dense
# merge_quantized_lstm_cell2, merge_quantized_lstm2
from ..onnx.passes.front_passes import fuse_weights_const, convert_deconv
from ..onnx.passes.common_passes import apply_subgraph_plugin, record_output_tensors, remove_useless_op, fuse_const
from ...graph.graph_algo import infer, clear_redundant_nodes
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


def process_tflite(model_path, params):
    '''Do some preprocessing on the graph under the tflite framework.'''
    graph = convert_tflite_to_graph(model_path, params)
    record_output_tensors(graph)
    if graph is not None and len(graph) > 0:
        apply_subgraph_plugin(graph)
        infer(graph, partial=True)
        fuse_weights_const(graph)

        split_op_has_activation(graph)

        if graph._attr.get('quantize', False):
            merge_dqd(graph, ['LiteMIRROR_PAD', 'LiteRSQRT',
                              'LiteSQUARED_DIFFERENCE'])
            merge_special_cast_quantize(graph)
            convert_special_quantize(graph)
            convert_special_dequantize(graph)
            # merge_quantized_lstm(graph)
            # merge_quantized_lstm2(graph)
            merge_quantized_lstm_cell(graph)
            # merge_quantized_lstm_cell2(graph)
            # merge_quantized_instance_norm(graph)
            merge_quantized_ln(graph)
            split_quatized_mean(graph)
        else:
            merge_min_quant_max_to_clip(graph)

        split_fc(graph)

        from ..tf.passes.front_passes import split_s2b, split_b2s
        split_s2b(graph, 'LiteSPACE_TO_BATCH_ND')
        split_b2s(graph, 'LiteBATCH_TO_SPACE_ND')
        split_greater_or_less_equal(graph)
        split_not_equal(graph, 'LiteNOT_EQUAL')
        split_rsqrt(graph)

        from ..tf.passes.front_passes import split_special_floormod, convert_reverse, merge_overlap_and_add, merge_sufficient_statistics, merge_sufficient_statistics2
        split_special_floormod(graph, 'LiteFLOOR_MOD')

        remove_detection_postprocess(graph, params)
        convert_onehot(graph)
        merge_sufficient_statistics(graph)
        merge_sufficient_statistics2(graph)

        convert_square(graph, 'LiteSQUARE')
        convert_square_diff(graph, 'LiteSQUARED_DIFFERENCE')
        convert_scatternd(graph, 'LiteSCATTER_ND')
        convert_scatternd2(graph, 'LiteSCATTER_ND')
        convert_reverse(graph, op_type='LiteREVERSE_V2')
        convert_sparse_to_dense(graph, 'LiteSPARSE_TO_DENSE')
        merge_overlap_and_add(graph)

        convert_reverse_sequence(graph, 'LiteREVERSE_SEQUENCE')
        convert_unpack(graph)
        convert_special_uni_seq_lstm(graph)

        clear_redundant_nodes(graph)
        infer(graph)
        fuse_const(graph)

        if not graph._attr.get('quantize', False):
            # need after fuse_const
            remove_useless_op(graph, ['LiteDEQUANTIZE'])

        from ..onnx.passes.middle_passes import convert_to_const
        convert_to_const(graph, ['LiteSHAPE', 'LiteZEROS_LIKE'])

        convert_strided_slice(graph, 'LiteSTRIDED_SLICE')

        from ..tf.passes.front_passes import convert_floordiv
        convert_floordiv(graph, op_type='LiteFLOOR_DIV')

        convert_deconv(graph)
        convert_negative_pool_pad(graph)
        remove_redundant_broadcast_to(graph)
        convert_broadcast_to(graph)
        remove_sub_equal_select(graph)

        from ..tf.passes.front_passes import convert_nms
        convert_nms(graph, params)

        convert_to_onnx(graph)

        # To support flex op in tflite model, need convert tf op to onnx as well.
        # FIXME: Other passes in tf front passes may be needed as well.
        from ..tf.passes.front_passes import convert_to_onnx as convert_tf_op_to_onnx
        convert_tf_op_to_onnx(graph)
    else:
        WARN('[Parser]: Got empty graph in process_tflite!')
    return graph
