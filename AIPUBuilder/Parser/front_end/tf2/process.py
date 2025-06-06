# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from .load import convert_tf2_to_graph
from ...graph.graph_algo import infer
from ..onnx.passes.front_passes import fuse_weights_const
from ..onnx.passes.common_passes import fuse_const, record_output_tensors, apply_subgraph_plugin, remove_useless_op
from .passes.front_passes import convert_to_onnx, convert_crelu, convert_l2_normalize, convert_lp_norm, convert_squeeze
from .passes.keras_front_passes import process_keras_op_before_infer, process_keras_op_after_infer, convert_sufficient_statistics
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


def front_process_tf2(graph, params):
    record_output_tensors(graph)
    if graph is not None and len(graph) > 0:
        apply_subgraph_plugin(graph)
        remove_useless_op(graph, ['Tfstop_gradient'])
        infer(graph, partial=True)
        fuse_const(graph)

        process_keras_op_before_infer(graph)

        convert_crelu(graph)

        from ..tf.passes.front_passes import split_b2s, convert_d2s_or_s2d, convert_onehot, \
            convert_matmul, convert_maxpoolwithargmax, convert_nms, split_special_floormod, \
            split_s2b, convert_floordiv, convert_topk, convert_unique
        split_b2s(graph, op_type='Tfbatch_to_space_nd')

        from ..lite.passes.front_passes import split_not_equal, split_rsqrt, convert_square, \
            convert_square_diff
        split_not_equal(graph, op_type='Tfnot_equal')
        split_rsqrt(graph, op_type='Tfrsqrt')
        convert_square(graph, op_type='Tfsquare')
        convert_square_diff(graph, op_type='Tfsquared_difference')
        convert_unique(graph)

        infer(graph)

        convert_d2s_or_s2d(graph)
        convert_floordiv(graph, op_type='Tffloor_div')
        convert_onehot(graph, op_type='Tfone_hot')
        convert_squeeze(graph, op_type='Tfsqueeze')
        convert_topk(graph, op_type='Tftop_k')
        convert_matmul(graph)
        convert_maxpoolwithargmax(graph, op_type='Tfmax_pool_with_argmax')
        convert_nms(graph, params)
        convert_sufficient_statistics(graph)
        split_special_floormod(graph, op_type='Tffloormod')
        split_s2b(graph, op_type='Tfspace_to_batch_nd')

        process_keras_op_after_infer(graph, params)

        convert_l2_normalize(graph)
        convert_lp_norm(graph)

        convert_to_onnx(graph)

        # To support lambda op in tf2 model, need convert tf op to onnx as well.
        # FIXME: Other passes in tf front passes may be needed as well.
        from ..tf.passes.front_passes import convert_to_onnx as convert_tf_op_to_onnx
        convert_tf_op_to_onnx(graph, params)
    else:
        WARN('[Parser]: Got empty graph for TF2 model %s in front_process_tf2!' %
             params['model_name'])


def process_tf2(graph, model_path, params):
    '''Do some preprocessing on the graph under the tensorflow framework.'''
    graph = convert_tf2_to_graph(graph, model_path, params)
    front_process_tf2(graph, params)

    return graph
