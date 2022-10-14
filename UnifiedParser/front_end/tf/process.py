# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from .load import convert_tf_to_graph
from ...graph.graph_algo import infer, clear_redundant_nodes
from ..onnx.passes.front_passes import fuse_weights_const
from ..onnx.passes.common_passes import remove_useless_op, fuse_const, record_output_tensors, \
    apply_subgraph_plugin
from .passes.front_passes import merge_gru, merge_keras_gru, merge_keras_lstm, merge_zero_fraction, \
    remove_switch, remove_merge, \
    convert_to_onnx, split_b2s, split_s2b, split_special_floormod, \
    convert_resize_bilinear_nearest, remove_identity_n, convert_conv_backpropinput, \
    convert_special_fakequantminmaxvars, convert_maxpoolwithargmax, convert_nms, convert_fusebatchnormv3, \
    convert_matmul, remove_isfinite_select, merge_fasterrcnn, merge_keras_maskrcnn, \
    convert_gru_lstm
from ...common.errors import *


def process_tf(model_path, params):
    '''Do some preprocessing on the graph under the tensorflow framework.'''
    graph = convert_tf_to_graph(model_path, params)
    record_output_tensors(graph)

    if graph is not None and len(graph) > 0:
        from ..lite.passes.front_passes import convert_scatternd, split_rsqrt, convert_strided_slice, \
            convert_square, convert_square_diff, split_not_equal, convert_reverse_sequence, convert_unpack

        apply_subgraph_plugin(graph)
        remove_useless_op(
            graph, ['TfAssert', 'TfEnter', 'TfIdentity', 'TfStopGradient'])
        remove_identity_n(graph)
        infer(graph, partial=True)
        fuse_const(graph)
        fuse_weights_const(graph)

        merge_fasterrcnn(graph)
        merge_keras_maskrcnn(graph, params)
        merge_gru(graph)
        merge_keras_gru(graph)
        merge_keras_lstm(graph)
        merge_zero_fraction(graph)
        convert_gru_lstm(graph)
        convert_reverse_sequence(graph, op_type='TfReverseSequence')
        convert_scatternd(graph, op_type='TfScatterNd')
        split_b2s(graph)
        split_s2b(graph)
        split_special_floormod(graph)

        split_not_equal(graph, op_type='TfNotEqual')
        split_rsqrt(graph, op_type='TfRsqrt')
        convert_strided_slice(graph, 'TfStridedSlice')
        convert_square(graph, op_type='TfSquare')
        convert_square_diff(graph, op_type='TfSquaredDifference')

        infer(graph)

        remove_switch(graph)
        remove_merge(graph)
        remove_isfinite_select(graph)
        convert_resize_bilinear_nearest(graph)
        convert_fusebatchnormv3(graph)
        convert_matmul(graph)
        convert_unpack(graph, op_type='TfUnpack')
        convert_conv_backpropinput(graph)
        convert_maxpoolwithargmax(graph)
        convert_special_fakequantminmaxvars(graph)
        convert_nms(graph, params)

        convert_to_onnx(graph)

    else:
        WARN('[Parser]: Got empty graph for TF model %s in process_tf!' %
             params['model_name'])
    return graph
