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

import numpy as np
from ....ops.op import OpHasWeights, OpHasBiases
from ....graph.node_wrap import NodeWrap
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from ....common.errors import *
from .common_passes import clear_redundant_nodes


def fuse_weights_const(graph):
    def get_src_data(src_name, edge_attr):
        src_obj = NodeWrap(graph, src_name)['object']
        if src_obj.type in ('Constant', 'TfConst'):
            data = src_obj.value
        elif (edge_attr.get('tensor', None) is not None and edge_attr['tensor'].is_const):
            data = edge_attr['tensor'].value
        else:
            data = None
        return data

    matched = False
    for node_name in graph.nodes:
        node_obj = NodeWrap(graph, node_name)['object']
        in_edges = graph.sorted_in_edges(node_name, keys=True, data=True)
        if isinstance(node_obj, OpHasWeights) and isinstance(node_obj, OpHasBiases):
            if node_obj.type in ('GRU', 'LSTM'):
                continue
            for i, edge_info in enumerate(in_edges):
                src_name, _, k, edge_attr = edge_info
                data = get_src_data(src_name, edge_attr)
                try:
                    if i == 1 and isinstance(data, np.ndarray):
                        node_obj.weights = data
                        if edge_attr.get('tensor', None) is not None \
                                and len(edge_attr['tensor'].min_max) == 2:
                            node_obj.weights_min_max = list(
                                edge_attr['tensor'].min_max)
                        matched = True
                        graph.remove_edge(src_name, node_name, key=k)
                    elif i == 2 and isinstance(data, np.ndarray):
                        node_obj.biases = data
                        matched = True
                        graph.remove_edge(src_name, node_name, key=k)
                except Exception as e:
                    WARN('[Parser]: Node(%s) meets error (%s) in fuse_weights_const!' % (
                        node_name, str(e)))
        elif isinstance(node_obj, OpHasWeights):
            for i, edge_info in enumerate(in_edges):
                src_name, _, k, edge_attr = edge_info
                data = get_src_data(src_name, edge_attr)
                if i == 1 and isinstance(data, np.ndarray):
                    node_obj.weights = data
                    if edge_attr.get('tensor', None) is not None \
                            and len(edge_attr['tensor'].min_max) == 2:
                        node_obj.weights_min_max = list(
                            edge_attr['tensor'].min_max)
                    matched = True
                    graph.remove_edge(src_name, node_name, key=k)
    if matched:
        clear_redundant_nodes(graph)


def convert_special_prelu(graph):
    matches = single_node_matcher(graph, 'PRelu')
    for m in matches:
        prelu = m['target']
        prelu_obj = NodeWrap(graph, prelu)['object']
        if prelu_obj is None:
            WARN(
                '[Parser]: Meets invalid PRelu Op (%s) in convert_special_prelu!' % prelu)
            continue
        inputs = prelu_obj.get_input_tensors()
        in_edges = graph.sorted_in_edges(prelu, data=True)
        if len(inputs) != 2 or inputs[1] is None or len(in_edges) != 2:
            WARN(
                '[Parser]: Meets invalid PRelu Op (%s) in convert_special_prelu!' % prelu)
            continue
        if in_edges[1][2]['tensor'] is not None \
                and in_edges[1][2]['tensor'].is_const \
                and inputs[1].size == 1:
            slope = np.reshape(inputs[1], [])
            graph.remove_edges_from(in_edges[1:])
            leaky_attr = prelu_obj.copied_attr()
            leaky_attr.update({'opeset_version': 6, 'alpha': float(slope)})
            NodeWrap(graph, prelu).replace_obj('LeakyRelu', leaky_attr)
