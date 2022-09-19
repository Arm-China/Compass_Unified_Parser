"""
//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
"""

from .load import convert_caffe_to_graph
from ...graph.graph_algo import infer, clear_redundant_nodes
from ..onnx.passes.front_passes import fuse_weights_const
from ..onnx.passes.common_passes import remove_useless_op, remove_redundant_reshape, apply_subgraph_plugin, record_output_tensors
from .passes.front_passes import *
from ...common.errors import *


def process_caffe(model_path, params):
    '''Do some preprocessing on the graph under the caffe framework.'''
    graph = convert_caffe_to_graph(model_path, params)
    record_output_tensors(graph)
    if graph is not None and len(graph) > 0:
        apply_subgraph_plugin(graph)

        remove_useless_op(graph, ['CaffeDROPOUT', 'CaffeSPLIT'])
        fuse_weights_const(graph)
        clear_redundant_nodes(graph)
        infer(graph)

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
        convert_to_onnx(graph)
        infer(graph)
    else:
        WARN('[Parser]: Got empty graph in process_caffe!')
    return graph
