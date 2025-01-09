# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from .load import convert_caffe_to_graph
from ...graph.graph_algo import infer, clear_redundant_nodes
from ..onnx.passes.front_passes import fuse_weights_const
from ..onnx.passes.common_passes import remove_useless_op, remove_redundant_reshape, apply_subgraph_plugin, \
    record_output_tensors, convert_to_const
from .passes.front_passes import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


def front_process_caffe(graph, params):
    record_output_tensors(graph)
    if graph is not None and len(graph) > 0:
        apply_subgraph_plugin(graph)

        remove_useless_op(graph, ['CaffeDROPOUT', 'CaffeSPLIT'])
        fuse_weights_const(graph)
        clear_redundant_nodes(graph)
        infer(graph)
        convert_to_const(graph, ['CaffeDUMMYDATA'])

        remove_redundant_reshape(graph, 'CaffeRESHAPE')
        remove_useless_reshape(graph)
        merge_reduce_reshape(graph)
        merge_bn_scale(graph)

        split_argmax(graph)
        split_axpy(graph)
        split_exp(graph)
        split_inner_product(graph)
        split_log(graph)
        split_mvn_special(graph)
        split_normalize(graph)
        split_power(graph)
        split_reduce_asum(graph)
        convert_channel_shuffle(graph)
        split_spp(graph)

        convert_bias(graph)
        convert_lstm(graph)
        convert_pool(graph)
        convert_proposal_roipooling(graph, params)
        convert_scale_to_bn(graph)
        convert_scale(graph)
        convert_slice(graph)
        convert_upsample(graph)

        adjust_filter(graph)

        remove_detection_postprocess(graph)
        refinedet_postprocess(graph, params)
        convert_to_onnx(graph, params)
        infer(graph)
    else:
        WARN('[Parser]: Got empty graph in front_process_caffe!')


def process_caffe(graph, model_path, params):
    '''Do some preprocessing on the graph under the caffe framework.'''
    graph = convert_caffe_to_graph(graph, model_path, params)
    front_process_caffe(graph, params)

    return graph
