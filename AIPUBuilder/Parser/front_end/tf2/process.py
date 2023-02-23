# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from .load import convert_tf2_to_graph
from ...graph.graph_algo import infer
from ..onnx.passes.front_passes import fuse_weights_const
from ..onnx.passes.common_passes import fuse_const, record_output_tensors, apply_subgraph_plugin
from .passes.front_passes import convert_to_onnx, convert_crelu, convert_l2_normalize, convert_lp_norm, convert_squeeze
from .passes.keras_front_passes import process_keras_op_before_infer, process_keras_op_after_infer
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


def process_tf2(model_path, params):
    '''Do some preprocessing on the graph under the tensorflow framework.'''
    graph = convert_tf2_to_graph(model_path, params)
    record_output_tensors(graph)

    if graph is not None and len(graph) > 0:
        apply_subgraph_plugin(graph)
        infer(graph, partial=True)
        fuse_const(graph)

        process_keras_op_before_infer(graph)

        convert_crelu(graph)

        from ..tf.passes.front_passes import split_b2s, convert_depth_to_space, convert_onehot
        split_b2s(graph, op_type='Tfbatch_to_space_nd')

        from ..lite.passes.front_passes import split_not_equal
        split_not_equal(graph, op_type='Tfnot_equal')

        infer(graph)

        convert_depth_to_space(graph, op_type='Tfdepth_to_space')
        convert_onehot(graph, op_type='Tfone_hot')
        convert_squeeze(graph, op_type='Tfsqueeze')

        process_keras_op_after_infer(graph)

        convert_l2_normalize(graph)
        convert_lp_norm(graph)

        convert_to_onnx(graph)

    else:
        WARN('[Parser]: Got empty graph for TF2 model %s in process_tf2!' %
             params['model_name'])
    return graph
