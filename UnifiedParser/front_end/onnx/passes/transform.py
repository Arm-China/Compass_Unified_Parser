# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow.compat.v1 as tf
import numpy as np
import copy
from ....graph.graph import Graph
from ....ops.op import Op, LayoutConcernedOp, LayoutUnawareOp, OnnxOp, CommonOp
from ....ops.common_ops import PluginOp
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL
from ....common.utils import extend_lists
from ....graph.node_wrap import NodeWrap
from ....graph.graph_algo import *
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from .common_passes import remove_useless_op, insert_transpose, insert_constant, insert_gather, \
    remove_redundant_transpose


def decompose_abnormal_reshape(graph, changed_nodes):
    matches = single_node_matcher(graph, 'Reshape')
    for m in matches:
        node_name = m['target']
        node_obj = NodeWrap(graph, node_name)['object']
        if node_obj is not None:
            in_shapes = node_obj.get_input_shapes()
            out_shapes = node_obj.get_output_shapes()
            if len(in_shapes) >= 1 \
                    and in_shapes[0] is not None \
                    and len(in_shapes[0]) != 4 \
                    and len(out_shapes) == 1 \
                    and out_shapes[0] is not None \
                    and len(out_shapes[0]) == 4:
                dst_out_shape = out_shapes[0]
                intermediate_shape = []
                pre_4d_node = None
                cur_node = node_name
                while graph.pred[cur_node]:
                    cur_node = graph.pred[cur_node][0]
                    cur_obj = NodeWrap(graph, cur_node)['object']
                    if cur_obj is not None:
                        cur_out_shapes = cur_obj.get_output_shapes()
                        if len(cur_out_shapes) == 1 \
                                and cur_out_shapes[0] is not None \
                                and len(cur_out_shapes[0]) == 4 \
                                and cur_out_shapes[0] != dst_out_shape \
                                and sorted(cur_out_shapes[0]) == sorted(dst_out_shape):
                            pre_4d_node = cur_node
                            intermediate_shape = copy.deepcopy(
                                cur_out_shapes[0])
                            changed_nodes.append(node_name)
                            break
                if pre_4d_node and intermediate_shape:
                    reshape_in_edges = graph.sorted_in_edges(
                        node_name, data=True)
                    reshape_out_edges = graph.sorted_out_edges(
                        node_name, data=True)
                    if len(reshape_in_edges) >= 1 and len(reshape_out_edges) == 1:
                        graph.remove_edges_from(reshape_in_edges[1:])
                        const = get_valid_node_name(
                            graph, node_name + '_shape')
                        graph.add_node(const)
                        const_value = np.array(intermediate_shape, np.int64)
                        const_attr = {'name': const, 'value': const_value,
                                      'data_format': 'NHWC', 'opset_version': 9}
                        NodeWrap(graph, const).replace_obj(
                            'Constant', const_attr)
                        const_edge_attr = {'src_out_port': 0, 'dst_in_port': 1, 'tensor': Tensor(
                            value=const_value, is_const=True)}
                        graph.add_edge(const, node_name, **const_edge_attr)
                        for _, dst, out_attr in reshape_out_edges:
                            tr = insert_transpose(
                                graph, node_name, dst, out_attr, [0, 2, 3, 1])
                            changed_nodes.append(tr)
                        changed_nodes.append(const)


def insert_transpose_for_layoutconcern(graph):
    supportted_frameworks = Op.framework_to_op(graph._attr['framework'])
    current_op_types = set()
    for fr in supportted_frameworks:
        current_op_types = current_op_types.union(
            set(fr.get_concrete_subclass_names()))
    layout_types = list(set(LayoutConcernedOp.get_concrete_subclass_names(
    )).intersection(current_op_types))
    layout_types = sorted(layout_types)
    matches = extend_lists([single_node_matcher(graph, type_name)
                            for type_name in layout_types])
    for m in matches:
        node = m['target']
        node_obj = NodeWrap(graph, node)['object']
        if node_obj is not None and len(node_obj.get_input_shapes()) >= 1 and len(node_obj.get_output_shapes()) >= 1:
            input_shape = node_obj.get_input_shapes()[0]
            output_shape = node_obj.get_output_shapes()[0]
            if getattr(node_obj, 'data_format', None) is None \
                    or len(node_obj.data_format) < 3 \
                    or node_obj.data_format[:2] != 'NC':
                continue
            if input_shape \
                    and output_shape \
                    and len(input_shape) >= 2 \
                    and len(output_shape) >= 2:
                input_dim = len(input_shape)
                pre_trans_perm = [0] + list(range(2, input_dim)) + [1]
                post_trans_perm = [0, input_dim-1] + \
                    list(range(1, input_dim-1))
                in_edges = graph.sorted_in_edges(node, data=True)
                out_edges = graph.sorted_out_edges(
                    node, keys=True, data=True)
                src, _, in_attr = in_edges[0]
                insert_transpose(graph, src, node, in_attr, pre_trans_perm)

                post_trans_list = []
                if node_obj.type != 'GenerateProposals':
                    out_ports = node_obj.get_out_ports()
                    for p in out_ports:
                        if (node_obj.type == 'BatchNormalization' or node_obj.type == 'TfFusedBatchNormV3') and p > 0:
                            continue
                        candidate_name = node + \
                            '_post_transpose' if len(
                                out_ports) == 1 else node + '_post_transpose_port_' + str(p)
                        post_trans = get_valid_node_name(
                            graph, candidate_name)
                        node_port_tensor = None
                        for _, dst, k, out_attr in out_edges:
                            if out_attr['src_out_port'] == p:
                                graph.remove_edge(node, dst, key=k)
                                new_out_attr = copy.deepcopy(out_attr)
                                new_out_attr['src_out_port'] = 0
                                graph.add_edge(
                                    post_trans, dst, **new_out_attr)
                                if out_attr['tensor'].value is not None:
                                    node_port_tensor = np.transpose(out_attr['tensor'].value, [
                                                                    post_trans_perm.index(i) for i in range(len(post_trans_perm))])
                        graph.add_edge(
                            node, post_trans, **{'src_out_port': p, 'dst_in_port': 0, 'tensor': Tensor(value=node_port_tensor)})
                        post_trans_attr = node_obj.copied_attr()
                        post_trans_attr.update({'name': post_trans,
                                                'opset_version': 1,
                                                'data_format': 'NHWC',
                                                'perm': post_trans_perm})
                        NodeWrap(graph, post_trans).replace_obj(
                            'Transpose', post_trans_attr)
                        post_trans_list.append(post_trans)

                    if node_obj.type == 'MaxPool' and len(out_ports) == 2:
                        if len(input_shape) != 4 or len(output_shape) != 4:
                            WARN(
                                '[Parser]: Meets invalid Maxpool Op (%s) with indices in insert_transpose_for_layoutconcern!' % node)
                            continue
                        indices_transpose = post_trans_list[-1]

                        n, c, h, w = input_shape
                        _, _, out_h, out_w = output_shape
                        hwc_compensation = np.reshape(np.arange(0, n).astype(
                            np.float32) * h * w * c, (n, 1, 1, 1))
                        hwc_compensation = np.tile(
                            hwc_compensation, (1, out_h, out_w, c))
                        hw_compensation = np.arange(
                            0, c).astype(np.float32) * h * w
                        c_compensation = np.arange(0, c).astype(np.float32)

                        sub_oprand2 = hwc_compensation + c_compensation
                        div_oprand2 = np.tile(np.reshape(
                            np.array([c], np.float32), (1, 1, 1, 1)), (n, out_h, out_w, c))
                        add_oprand2 = hwc_compensation + hw_compensation

                        cast_to_float = get_valid_node_name(
                            graph, node + '_indices_to_float')
                        sub = get_valid_node_name(
                            graph, node + '_indices_sub')
                        div = get_valid_node_name(
                            graph, node + '_indices_div')
                        add = get_valid_node_name(
                            graph, node + '_indices_add')
                        cast_to_int = get_valid_node_name(
                            graph, node + '_indices_to_int')

                        indices_transpose_in_edges = graph.sorted_in_edges(
                            indices_transpose, data=True)
                        _, _, indices_out_attr = indices_transpose_in_edges[0]
                        graph.remove_edge(node, indices_transpose)
                        graph.add_edge(node, cast_to_float,
                                       **indices_out_attr)
                        graph.add_edge(cast_to_float, sub)
                        graph.add_edge(sub, div)
                        graph.add_edge(div, add)
                        graph.add_edge(add, cast_to_int)
                        graph.add_edge(cast_to_int, indices_transpose)
                        insert_constant(
                            graph, sub + '_oprand2', sub_oprand2, sub, in_port=1, data_format='NHWC')
                        insert_constant(
                            graph, div + '_oprand2', div_oprand2, div, in_port=1, data_format='NHWC')
                        insert_constant(
                            graph, add + '_oprand2', add_oprand2, add, in_port=1, data_format='NHWC')

                        NodeWrap(graph, cast_to_float).replace_obj(
                            'Cast', {'name': cast_to_float, 'opset_version': 13, 'to': 1})
                        NodeWrap(graph, sub).replace_obj(
                            'Sub', {'name': sub, 'opset_version': 7})
                        NodeWrap(graph, div).replace_obj(
                            'Div', {'name': div, 'opset_version': 7})
                        NodeWrap(graph, add).replace_obj(
                            'Add', {'name': add, 'opset_version': 7})
                        NodeWrap(graph, cast_to_int).replace_obj(
                            'Cast', {'name': cast_to_int, 'opset_version': 13, 'to': 6})

                else:
                    post_trans_list.append(node)

                node_obj.data_format = 'NHWC'
                if node_obj.type == 'BatchNormalization':
                    in_edges = graph.sorted_in_edges(node, data=True)
                    if node_obj is not None and len(in_edges) == 5:
                        input_shape = node_obj.get_input_shapes()[1]
                        input_dim = len(input_shape)
                        pre_trans_perm = list(range(1, input_dim)) + [0]
                        for src, _, in_attr in in_edges:
                            if in_attr['dst_in_port'] == 0:
                                continue
                            insert_transpose(
                                graph, src, node, in_attr, pre_trans_perm)
                elif node_obj.type == 'MaxUnpool':
                    if len(in_edges) == 3 and len(input_shape) >= 3:
                        indices, _, in_attr_1 = in_edges[1]
                        insert_transpose(
                            graph, indices, node, in_attr_1, pre_trans_perm)
                        out_shape, _, in_attr_2 = in_edges[2]
                        insert_gather(graph,
                                      out_shape,
                                      node,
                                      np.array(
                                          [0]+list(range(2, len(input_shape)))+[1], np.int32),
                                      edge_attr=in_attr_2)
                    else:
                        WARN(
                            '[Parser]: Meets invalid MaxUnpool Node (%s) in insert_transpose_for_layoutconcern!' % node)
                elif node_obj.type == 'Resize':
                    input_tensors = node_obj.get_input_tensors()
                    dim = len(input_tensors[0].shape)
                    indices = np.array(
                        [0] + list(range(2, dim)) + [1] if dim > 1 else [0], np.int32)
                    if node_obj.scales is not None and node_obj.scales.size == indices.size:
                        node_obj.scales = np.take(
                            node_obj.scales, indices, axis=0)
                    if node_obj.sizes is not None and node_obj.sizes.size == indices.size:
                        node_obj.sizes = np.take(
                            node_obj.sizes, indices, axis=0)
                    roi_inp, _, roi_in_attr = in_edges[1]
                    scales_inp, _, scales_in_attr = in_edges[2]
                    sizes_inp, _, sizes_in_attr = in_edges[3]
                    if node_obj.coordinate_transformation_mode == 'tf_crop_and_resize' \
                            and input_tensors[1] is not None \
                            and input_tensors[1].size > 0:
                        roi_indices = np.concatenate(
                            [indices, indices + dim])
                        insert_gather(
                            graph, roi_inp, node, roi_indices, axis=0, edge_attr=roi_in_attr)
                    if input_tensors[2] is not None and input_tensors[2].size > 0:
                        insert_gather(
                            graph, scales_inp, node, indices, axis=0, edge_attr=scales_in_attr)
                    if input_tensors[3] is not None and input_tensors[3].size > 0:
                        insert_gather(graph, sizes_inp, node,
                                      indices, axis=0, edge_attr=sizes_in_attr)
                elif node_obj.type in ('GenerateProposals', 'UpsampleByIndex'):
                    if len(in_edges) >= 2:
                        src2, _, in_attr2 = in_edges[1]
                        insert_transpose(
                            graph, src2, node, in_attr2, [0, 2, 3, 1])
                    else:
                        WARN('[Parser]: Meets invalid %s Op (%s) in insert_transpose_for_layoutconcern!' % (
                            node_obj.type, node))
                if node in graph._attr['output_names'] and post_trans_list and node not in post_trans_list:
                    index = graph._attr['output_names'].index(node)
                    graph._attr['output_names'][index] = post_trans_list[0]
                    for i, name in enumerate(post_trans_list[1:]):
                        index += 1
                        graph._attr['output_names'].insert(
                            index, post_trans_list[1 + i])
            else:
                WARN(
                    '[Parser]: Meets invalid Node (%s) in insert_transpose_for_layoutconcern!' % node)
        else:
            WARN(
                '[Parser]: Meets invalid Node (%s) in insert_transpose_for_layoutconcern!' % node)


def nhwc_for_other(graph):
    ops = ['CTCGreedyDecoder', 'Pad']
    matches = [single_node_matcher(graph, op_type) for op_type in ops]
    matches = extend_lists(matches)
    for m in matches:
        node = m['target']
        node_obj = NodeWrap(graph, node)['object']
        if node_obj is not None:
            if node_obj.data_format == 'NCHW':
                if node_obj.type == 'CTCGreedyDecoder':
                    in_edges = graph.sorted_in_edges(node, data=True)
                    in_shapes = node_obj.get_input_shapes()
                    if len(in_edges) >= 2 and len(in_shapes[0]) == 3:
                        src, _, in_attr = in_edges[0]
                        insert_transpose(graph, src, node, in_attr, [1, 0, 2])
                node_obj.data_format = 'NHWC'
        else:
            WARN('[Parser]: Meets invalid Op (%s) in nhwc_for_other!' % node)


def transform_to_nhwc(graph, params):
    '''Convert the layout to the form of nhwc.'''
    input_data_format = params.get('input_data_format', 'NHWC')
    insert_transpose_for_layoutconcern(graph)
    if input_data_format == 'NCHW':
        nhwc_for_other(graph)
    remove_redundant_transpose(graph)
    remove_useless_op(graph, ['Transpose'])
    if input_data_format == 'NCHW':
        infer(graph)
