# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ....ops.op import Op, OpHasWeights, OpHasBiases, KerasOp, BaseDeconvOp
from ....graph.node_wrap import NodeWrap
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL
from .common_passes import clear_redundant_nodes


def fuse_weights_const(graph):
    def _get_src_data(src_name, edge_attr):
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
        if node_obj is None:
            ERROR('[Parser]: Meets invalid Op(%s) in fuse_weights_const!' % node_name)
            continue
        if isinstance(node_obj, KerasOp):
            continue
        in_edges = graph.sorted_in_edges(node_name, keys=True, data=True)
        if isinstance(node_obj, OpHasWeights) and isinstance(node_obj, OpHasBiases):
            if node_obj.type in ('GRU', 'LSTM', 'QLinearConv'):
                continue
            if node_obj.type == 'LiteTRANSPOSE_CONV' \
                    or node_obj.type == 'LiteCONV_3D_TRANSPOSE':
                biases_in_port = 3
            else:
                biases_in_port = 2
            for i, edge_info in enumerate(in_edges):
                src_name, _, k, edge_attr = edge_info
                data = _get_src_data(src_name, edge_attr)
                try:
                    if i == 1 and isinstance(data, np.ndarray):
                        node_obj.weights = data
                        if edge_attr.get('tensor', None) is not None:
                            if len(edge_attr['tensor'].min_max) == 2:
                                node_obj.weights_min_max = list(
                                    edge_attr['tensor'].min_max)
                            if len(edge_attr['tensor'].scale_zp) == 2:
                                node_obj.weights_scale_zp = list(
                                    edge_attr['tensor'].scale_zp)
                        matched = True
                        graph.remove_edge(src_name, node_name, key=k)
                    elif i == biases_in_port and isinstance(data, np.ndarray):
                        node_obj.biases = data
                        if edge_attr.get('tensor', None) is not None:
                            if len(edge_attr['tensor'].scale_zp) == 2:
                                node_obj.biases_scale_zp = list(
                                    edge_attr['tensor'].scale_zp)
                        matched = True
                        graph.remove_edge(src_name, node_name, key=k)
                except Exception as e:
                    ERROR('[Parser]: Node(%s) meets error (%s) in fuse_weights_const!' % (
                        node_name, str(e)))
        elif isinstance(node_obj, OpHasWeights):
            for i, edge_info in enumerate(in_edges):
                src_name, _, k, edge_attr = edge_info
                data = _get_src_data(src_name, edge_attr)
                if i == 1 and isinstance(data, np.ndarray):
                    node_obj.weights = data
                    if edge_attr.get('tensor', None) is not None:
                        if len(edge_attr['tensor'].min_max) == 2:
                            node_obj.weights_min_max = list(
                                edge_attr['tensor'].min_max)
                        if len(edge_attr['tensor'].scale_zp) == 2:
                            node_obj.weights_scale_zp = list(
                                edge_attr['tensor'].scale_zp)
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
            ERROR(
                '[Parser]: Meets invalid PRelu Op (%s) in convert_special_prelu!' % prelu)
            continue
        inputs = prelu_obj.get_input_tensors()
        in_edges = graph.sorted_in_edges(prelu, data=True)
        if len(inputs) != 2 or inputs[1] is None or len(in_edges) != 2:
            ERROR(
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


def convert_deconv(graph):
    deconv_ops = BaseDeconvOp.get_concrete_subclass_names()
    framework_ops = Op.framework_op_types(graph._attr['framework'])
    current_deconvs = list(set(deconv_ops).intersection(framework_ops))
    matches = single_node_matcher(graph, current_deconvs)
    for m in matches:
        deconv = m['target']
        deconv_obj = NodeWrap(graph, deconv)['object']
        if deconv_obj is None:
            ERROR('[Parser]: Meets invalid Deconv Op(%s) in convert_deconv!' % deconv)
            continue
        main_in_port = type(deconv_obj).main_in_port()
        input_shapes = deconv_obj.get_input_shapes()
        in_edges = graph.sorted_in_edges(deconv, data=True)
        if len(input_shapes) >= 0 \
                and len(input_shapes) > main_in_port \
                and input_shapes[main_in_port] is not None \
                and all(s is not None for s in input_shapes[main_in_port]) \
                and len(input_shapes) == len(in_edges):
            src, _, in_attr = in_edges[main_in_port]
            graph.remove_edges_from(in_edges)
            in_attr['dst_in_port'] = 0
            graph.add_edge(src, deconv, **in_attr)
            in_shape = input_shapes[main_in_port]
            spatial_in_shape = in_shape[1:-1] if deconv_obj.data_format == 'NHWC' else in_shape[2:]
            deconv_obj.update_pads(spatial_in_shape)
            new_weights = np.transpose(deconv_obj.weights, axes=type(deconv_obj).perm_lite_to_onnx())
            attrs = deconv_obj.copied_attr()
            attrs.update({'opset_version': 11, 'weights': new_weights})
            NodeWrap(graph, deconv).replace_obj('ConvTranspose', attrs)
