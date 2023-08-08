# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import re
import copy
import torch
from ....ops.op import ActivationOnlyOp, BaseActivationOp, OpHasWeights, TfOp, TfliteOp, OpHasPaddingStrides, OpNeedBroadcast, OpHasAxis
from ....graph.node_wrap import NodeWrap
from ....graph.graph_algo import get_valid_node_name, clear_redundant_nodes
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from ...onnx.passes.common_passes import insert_constant, remove_node_safely, insert_transpose, insert_reshape, \
    insert_gather, insert_reshape_after, insert_tile
from ....common.defs import Tensor, FLOAT_EQUAL
from ....common.utils import extend_lists
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL


def convert_broadcast_to(graph):
    matched = False
    matches = single_node_matcher(graph, 'LiteBROADCAST_TO')
    for m in matches:
        bt = m['target']
        in_edges = graph.sorted_in_edges(bt, data=True)
        bt_obj = NodeWrap(graph, bt)['object']
        if bt_obj is None or \
                len(bt_obj.get_input_shapes()) < 2 or \
                len(bt_obj.get_input_tensors()) < 2 or \
                len(in_edges) < 2:
            ERROR(
                '[Parser]: Meets invalid LiteBROADCAST_TO(%s) in convert_broadcast_to!' % bt)
            continue
        if not in_edges[1][2]['tensor'].is_const:
            WARN('[Parser]: Meets unsupported non-constant shape of LiteBROADCAST_TO(%s) in convert_broadcast_to!' % bt)
            continue
        matched = True
        bt_in_shape = bt_obj.get_input_shapes()[0]
        bt_out_shape = bt_obj.get_input_tensors()[1]
        inp, _, bt_in_attr = in_edges[0]
        if len(bt_in_shape) < len(bt_out_shape):
            extra_dim = len(bt_out_shape) - len(bt_in_shape)
            bt_in_shape = [1] * extra_dim + bt_in_shape
            insert_reshape(graph, inp, bt, bt_in_attr, bt_in_shape)
        repeats = np.divide(list(bt_out_shape), bt_in_shape).astype(np.int64)
        tile_attr = bt_obj.copied_attr()
        tile_attr.update({'opset_version': 13, 'repeats': repeats})
        NodeWrap(graph, bt).replace_obj('Tile', tile_attr)
        graph.remove_edges_from(in_edges[1:])
        insert_constant(graph, bt + '_repeats', repeats, bt, in_port=1)
    if matched:
        clear_redundant_nodes(graph)


def convert_onehot(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('depth', {'op': 'Constant'}),
                                   ('on_value', {'op': 'Constant'}),
                                   ('off_value', {'op': 'Constant'}),
                                   ('one_hot', {'op': 'LiteONE_HOT'})
                               ],
                               edges=[
                                   ('depth', 'one_hot', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('on_value', 'one_hot', {
                                    'src_out_port': 0, 'dst_in_port': 2}),
                                   ('off_value', 'one_hot', {
                                    'src_out_port': 0, 'dst_in_port': 3})
                               ])
    for m in matches:
        obj_dict = {name: NodeWrap(graph, m[name])['object']
                    for name in ['depth', 'on_value', 'off_value', 'one_hot']}
        if any(obj is None for obj in obj_dict.values()):
            ERROR('[Parser]: Meets invalid Node in convert_onehot!')
            continue
        matched = True
        obj_dict['on_value'].value = np.append(obj_dict['off_value'].value, obj_dict['on_value'].value)
        graph.remove_edge(m['off_value'], m['one_hot'])
        onehot_attr = obj_dict['one_hot'].copied_attr()
        onehot_attr.update({'opset_version': 11})
        NodeWrap(graph, m['one_hot']).replace_obj('OneHot', onehot_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_negative_pool_pad(graph):
    pool_op = ['LiteAVERAGE_POOL_2D', 'LiteMAX_POOL_2D']
    pool_op_matches = [single_node_matcher(
        graph, op_type) for op_type in pool_op]
    pool_op_matches = extend_lists(pool_op_matches)
    for m in pool_op_matches:
        pool = m['target']
        pool_obj = NodeWrap(graph, pool)['object']
        if pool_obj is not None:
            new_node_attr = pool_obj.copied_attr()
            pad = pool_obj.pads
            if (np.array(pad) < 0).any():
                if pad[0] < 0:
                    pad[2] = pad[0] + pad[2]
                    pad[0] = 0
                if pad[1] < 0:
                    pad[3] = pad[1] + pad[3]
                    pad[1] = 0
                new_node_attr.update({'pads': pad, 'auto_pad': 'NOTSET'})
                new_node_attr.update(
                    {'opset_version': pool_obj.correspond_onnx_op['version']})
                if pool_obj.type == 'LiteMAX_POOL_2D':
                    NodeWrap(graph, pool).replace_obj('MaxPool', new_node_attr)
                else:
                    NodeWrap(graph, pool).replace_obj(
                        'AveragePool', new_node_attr)


def convert_sparse_to_dense(graph, op_type='LiteSPARSE_TO_DENSE'):
    matched = False
    if op_type not in ('TfSparseToDense', 'LiteSPARSE_TO_DENSE'):
        ERROR(
            '[Parser]: Meets invalid Op type (%s) in convert_sparse_to_dense!' % op_type)
        return
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        sd = m['target']
        sd_obj = NodeWrap(graph, sd)['object']
        in_edges = graph.sorted_in_edges(sd, data=True)

        if sd_obj is None or \
                len(in_edges) < 4:
            ERROR(
                '[Parser]: Meets invalid LiteSPARSE_TO_DENSE(%s) in convert_sparse_to_dense!' % sd)
            continue

        input_tensors = sd_obj.get_input_tensors()

        if len(sd_obj.get_input_tensors()) < 4 or \
                not in_edges[1][2]['tensor'].is_const or \
                any([input_tensor is None for input_tensor in input_tensors]):
            continue

        if op_type == 'LiteSPARSE_TO_DENSE':
            if in_edges[0][2]['tensor'].is_const is False:
                WARN(
                    '[Parser]: Meets non-const indices input of SparseToDense Op (%s) in convert_sparse_to_dense!' % sd)
        else:
            if in_edges[0][2]['tensor'].is_const is False and sd_obj.validate_indices is True:
                WARN(
                    '[Parser]: Meets non-const indices input of SparseToDense Op (%s) in convert_sparse_to_dense!' % sd)

        matched = True

        sd_src0, _, _ = in_edges[0]
        sd_src2, _, _ = in_edges[2]
        sd_src3, _, _ = in_edges[3]

        sparse_indices = input_tensors[0]
        output_shape = input_tensors[1]
        sparse_values = input_tensors[2]
        mul_value = np.ones(output_shape, dtype=sparse_values.dtype)

        inp0_attr = copy.deepcopy(in_edges[0][2])
        inp0_attr.update({'dst_in_port': 1})
        inp2_attr = copy.deepcopy(in_edges[2][2])
        inp3_mul_attr = copy.deepcopy(in_edges[3][2])
        inp3_mul_attr.update({'dst_in_port': 0})

        mul = get_valid_node_name(graph, sd + '_mul')

        graph.remove_edges_from(in_edges)

        graph.add_edge(sd_src0, sd, **inp0_attr)
        graph.add_edge(sd_src2, sd, **inp2_attr)
        graph.add_edge(sd_src3, mul, **inp3_mul_attr)
        graph.add_edge(mul, sd)
        insert_constant(graph, mul + '_value', mul_value, mul, in_port=1)

        if np.ndim(sparse_indices) == 1 and np.ndim(sparse_values) == 1:
            reshape_dim = [sparse_indices.shape[0], 1]
            insert_reshape(graph, sd_src0, sd, inp0_attr, reshape_dim)
        elif np.ndim(sparse_indices) == 0 and np.ndim(sparse_values) == 0:
            insert_reshape(graph, sd_src0, sd, inp0_attr, [1])

        NodeWrap(graph, mul).replace_obj(
            'Mul', {'name': mul, 'opset_version': 7})
        NodeWrap(graph, sd).replace_obj(
            'ScatterND', {'name': sd, 'opset_version': 16, 'reduction': 'none'})

    if matched:
        clear_redundant_nodes(graph)


def convert_square(graph, op_type='TfSquare'):
    if op_type not in ('TfSquare', 'Tfsquare', 'LiteSQUARE'):
        ERROR('[Parser]: Meets invalid Op type (%s) in convert_square_diff!' % op_type)
        return
    need_clear = False
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        square = m['target']
        square_obj = NodeWrap(graph, square)['object']
        if square_obj is None:
            ERROR('[Parser]: Meets invalid Op type (%s) in convert_square!' % square)
            continue
        if square_obj.type == 'Tfsquare':
            need_clear = True
            in_edges = graph.sorted_in_edges(square)
            graph.remove_edges_from(in_edges[1:])
        pow_attr = square_obj.copied_attr()
        pow_attr.update({'opset_version': 13})
        NodeWrap(graph, square).replace_obj('Pow', pow_attr)
        insert_constant(graph, square + '_power', np.array(2, np.int32),
                        square, in_port=1, data_format='NHWC')
    if need_clear:
        clear_redundant_nodes(graph)


def merge_quantized_ln(graph):
    matched = False
    ln_matches = matched_patterns(graph,
                                  nodes=[
                                      ('reshape1', {'op': 'LiteRESHAPE'}),
                                      ('mean1', {'op': 'LiteMEAN'}),
                                      ('square_diff', {
                                          'op': 'LiteSQUARED_DIFFERENCE'}),
                                      ('mean2', {'op': 'LiteMEAN'}),
                                      ('add1', {'op': 'LiteADD'}),
                                      ('eps', {'op': 'Constant'}),
                                      ('sqrt', {'op': 'LiteRSQRT'}),
                                      ('mul1', {'op': 'LiteMUL'}),
                                      ('mul2', {'op': 'LiteMUL'}),
                                      ('sub', {'op': 'LiteSUB'}),
                                      ('add2', {'op': 'LiteADD'}),
                                      ('reshape2', {'op': 'LiteRESHAPE'}),
                                      ('mul3', {'op': 'LiteMUL'}),
                                      ('gamma', {'op': 'Constant'}),
                                      ('add3', {'op': 'LiteADD'}),
                                      ('beta', {'op': 'Constant'}),

                                  ],
                                  edges=[
                                      ('reshape1', 'mean1'),
                                      ('reshape1', 'square_diff'),
                                      ('reshape1', 'mul1'),
                                      ('mean1', 'square_diff'),
                                      ('mean1', 'mul2'),
                                      ('square_diff', 'mean2'),
                                      ('mean2', 'add1'),
                                      ('eps', 'add1', {'dst_in_port': 1}),
                                      ('add1', 'sqrt'),
                                      ('sqrt', 'mul1'),
                                      ('sqrt', 'mul2'),
                                      ('mul1', 'add2'),
                                      ('mul2', 'sub'),
                                      ('sub', 'add2'),
                                      ('add2', 'reshape2'),
                                      ('reshape2', 'mul3'),
                                      ('gamma', 'mul3', {'dst_in_port': 1}),
                                      ('mul3', 'add3'),
                                      ('beta', 'add3', {'dst_in_port': 1}),
                                  ])

    for m in ln_matches:
        key_names = ['reshape1', 'mean1', 'square_diff', 'mean2', 'add1', 'sqrt',
                     'mul1', 'mul2', 'sub', 'add2', 'reshape2', 'mul3', 'add3', 'eps', 'gamma', 'beta']
        reshape1, mean1, square_diff, mean2, add1, sqrt, mul1, mul2, sub, add2, reshape2, mul3, add3, epsilon, gamma, beta = [
            m[name] for name in key_names]
        objs_dict = {m[name]: NodeWrap(graph, m[name])[
            'object'] for name in key_names}

        if any([obj is None for obj in objs_dict.values()]):
            ERROR('[Parser]: Meets invalid nodes in merge_tflite_ln!')
            continue

        if objs_dict[epsilon].value.size > 1 \
                and not FLOAT_EQUAL(objs_dict[epsilon].value.flatten()[1:], objs_dict[epsilon].value.item(0)):
            continue

        input_shapes = objs_dict[reshape1].get_input_shapes()
        reshape1_in_edges = graph.sorted_in_edges(reshape1, data=True)
        add1_in_edges = graph.sorted_in_edges(add1, data=True)
        add3_in_edges = graph.sorted_in_edges(add3)

        if len(input_shapes) < 1 \
                or any((shape is None for shape in input_shapes))\
                or any((shape_item is None for shape in input_shapes for shape_item in shape))\
                or len(add1_in_edges) != 2\
                or len(reshape1_in_edges) != 2:
            ERROR('[Parser]: Meets invalid nodes in merge_tflite_ln!')
            continue

        reshape1_src, _, _ = reshape1_in_edges[0]
        if len(input_shapes[0]) <= 1 \
                or objs_dict[mean1].keepdims != objs_dict[mean2].keepdims \
                or (not sorted(objs_dict[mean1].axes) == sorted(objs_dict[mean2].axes))\
                or objs_dict[mean1].axes != [2]:
            continue

        in_shape = input_shapes[0]
        axes = list(range(2, len(in_shape)))

        matched = True

        biases = np.array(objs_dict[beta].value).astype(np.int32)
        biases = np.reshape(biases, [-1])
        weights = np.array(objs_dict[gamma].value).astype(np.int8)
        weights = np.reshape(weights, [-1])
        eps_q = objs_dict[epsilon].value.item(0)
        eps_s, eps_z = add1_in_edges[1][2]['tensor'].scale_zp

        inp_out_attr = copy.deepcopy(reshape1_in_edges[0][2])
        inp_out_attr.update({'dst_in_port': 0})

        graph.remove_edges_from(add3_in_edges)
        graph.add_edge(reshape1_src, add3, **inp_out_attr)
        ln_attr = objs_dict[add3].copied_attr()
        ln_attr.update({'epsilon': eps_q, 'eps_scale': eps_s, 'eps_zp': eps_z, 'weights': weights,
                       'biases': biases, 'opset_version': 6, 'non_channel_axes': axes, 'data_format': 'NCHW'})
        NodeWrap(graph, add3).replace_obj(
            'InstanceNormalization', ln_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_square_diff(graph, op_type='TfSquaredDifference'):
    matched = False
    if op_type not in ('TfSquaredDifference', 'Tfsquared_difference', 'LiteSQUARED_DIFFERENCE'):
        ERROR('[Parser]: Meets invalid Op type (%s) in convert_square_diff!' % op_type)
        return
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        squd = m['target']
        squd_obj = NodeWrap(graph, squd)['object']
        squd_in_edges = graph.sorted_in_edges(squd, data=True)
        squd_out_edges = graph.sorted_out_edges(squd, data=True)
        if squd_obj is not None \
                and len(squd_in_edges) >= 2:
            matched = True
            graph.remove_edges_from(squd_in_edges[2:])
            s_pow = get_valid_node_name(graph, squd + '_pow')
            graph.add_edge(squd, s_pow)
            for _, dst, out_attr in squd_out_edges:
                graph.remove_edge(squd, dst)
                graph.add_edge(s_pow, dst, **out_attr)
            insert_constant(graph, s_pow + '_power', np.array(2, np.int32),
                            s_pow, in_port=1, data_format='NHWC')

            sub_attr = squd_obj.copied_attr()
            pow_attr = squd_obj.copied_attr()
            sub_attr.update({'opset_version': 11})
            pow_attr.update({'name': s_pow,
                             'opset_version': 7, })
            NodeWrap(graph, squd).replace_obj(
                'Sub', sub_attr)
            NodeWrap(graph, s_pow).replace_obj(
                'Pow', pow_attr)

            if squd in graph._attr['output_names']:
                index = graph._attr['output_names'].index(squd)
                graph._attr['output_names'].remove(squd)
                graph._attr['output_names'].insert(index, s_pow)
        else:
            ERROR(
                '[Parser]: Meets invalid Node(%s) in convert_square_diff!'
                % (squd))
    if matched:
        clear_redundant_nodes(graph)


def convert_scatternd(graph, op_type='TfScatterNd'):
    # TODO: Check whether this pass is still needed and can be replaced with pass convert_scatternd2.
    if op_type not in ('TfScatterNd', 'LiteSCATTER_ND'):
        ERROR('[Parser]: Meets invalid Op type (%s) in convert_scatternd!' % op_type)
        return
    matches = matched_patterns(graph,
                               nodes=[
                                   ('indices', {
                                    'op': ['Constant', 'TfConst']}),
                                   ('update', {}),
                                   ('shape', {'op': ['Constant', 'TfConst']}),
                                   ('scatter_nd', {'op': op_type})
                               ],
                               edges=[
                                   ('indices', 'scatter_nd', {
                                    'src_out_port': 0, 'dst_in_port': 0}),
                                   ('update', 'scatter_nd', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                   ('shape', 'scatter_nd', {
                                    'src_out_port': 0, 'dst_in_port': 2})
                               ])
    matched = False
    for m in matches:
        names = ['shape', 'scatter_nd', 'indices', 'update']
        node_objs = {n: NodeWrap(graph, m[n])['object'] for n in names}
        data_obj = node_objs['shape']
        indices_obj = node_objs['indices']
        indices_name = m['indices']
        scatter_nd_obj = node_objs['scatter_nd']
        scatter_nd = m['scatter_nd']
        in_edges = graph.sorted_in_edges(scatter_nd, data=True)
        out_edge = graph.sorted_out_edges(scatter_nd, data=True)
        if data_obj is not None \
                and indices_obj is not None \
                and scatter_nd_obj is not None \
                and len(in_edges) == 3:
            matched = True
            indices = indices_obj.value
            updates = in_edges[1][2]['tensor'].value
            data = data_obj.value

            outer_dims = len(indices.shape) - 1
            indices_nd = indices.shape[-1]
            slice_size = int(np.prod(updates.shape[outer_dims:]))
            out_shape = data_obj.value

            output_flat_size = int(np.prod(out_shape[:indices_nd]))
            remain_flat_size = output_flat_size
            dims_to_count = [0] * indices_nd
            for i in range(indices_nd):
                dims_to_count[i] = remain_flat_size // out_shape[i]
                remain_flat_size = dims_to_count[i]

            indices_reshape_dim = [-1, indices_nd]
            updates_reshape_dim = [-1, slice_size]
            reshaped_indices = np.reshape(indices, indices_reshape_dim)
            unique_indices, _ = np.unique(
                reshaped_indices, return_index=True, axis=0)

            if len(unique_indices) < len(reshaped_indices):
                indices_valid_out_shape = out_shape[:indices.shape[-1]]
                appending_indices = np.expand_dims(
                    np.array(indices_valid_out_shape) - 1, 0)
                reshaped_indices = np.concatenate(
                    [reshaped_indices, appending_indices], axis=0)
                ref_indices = np.sum(reshaped_indices *
                                     np.array(dims_to_count), axis=1)
                sorted_indices = np.argsort(ref_indices)
                gathered_ref_indices = np.take(
                    ref_indices, sorted_indices, axis=0)
                gather_id = np.array(sorted_indices).astype(np.int32)

                updates_reshape = get_valid_node_name(
                    graph, scatter_nd + '_updates_reshape')
                concat = get_valid_node_name(graph, scatter_nd + '_concat')
                post_segmentsum = get_valid_node_name(
                    graph, concat + '_segmentsum')
                update_gather = get_valid_node_name(
                    graph, post_segmentsum + '_gather')
                post_reshape = get_valid_node_name(
                    graph, update_gather + '_post_reshape')

                for src, _, in_attr in in_edges:
                    if in_attr['dst_in_port'] == 0 or in_attr['dst_in_port'] == 2:
                        graph.remove_edge(src, scatter_nd)
                    elif in_attr['dst_in_port'] == 1:
                        graph.remove_edge(src, scatter_nd)
                        graph.add_edge(src, updates_reshape, **
                                       {'src_out_port': in_attr['src_out_port'], 'dst_in_port': 0})
                graph.add_edge(updates_reshape, concat, **
                               {'src_out_port': 0, 'dst_in_port': 0})
                graph.add_edge(concat, update_gather, **
                               {'src_out_port': 0, 'dst_in_port': 0})
                graph.add_edge(update_gather, post_segmentsum, **
                               {'src_out_port': 0, 'dst_in_port': 0})
                graph.add_edge(post_segmentsum, post_reshape, **
                               {'src_out_port': 0, 'dst_in_port': 0})
                for _, dst, out_attr in out_edge:
                    graph.remove_edge(scatter_nd, dst)
                    graph.add_edge(post_reshape, dst, **out_attr)

                insert_constant(graph, updates_reshape + '_indices', np.array(updates_reshape_dim),
                                updates_reshape, in_port=1, data_format='NHWC')
                insert_constant(graph, concat + '_data', np.zeros(
                    (1, slice_size), updates.dtype), concat, in_port=1, data_format='NHWC')
                insert_constant(graph, update_gather + '_indices', gather_id,
                                update_gather, in_port=1, data_format='NHWC')
                insert_constant(graph, post_segmentsum + '_segment_ids', np.array(
                    gathered_ref_indices).astype(np.int32), post_segmentsum, in_port=1, data_format='NHWC')
                insert_constant(graph, post_reshape + '_indices', out_shape,
                                post_reshape, in_port=1, data_format='NHWC')
                insert_constant(graph, indices_name + '_new', gathered_ref_indices,
                                out_edge[0][1], in_port=0, data_format='NHWC')

                NodeWrap(graph, updates_reshape).replace_obj(
                    'Reshape', {'name': updates_reshape, 'shape': np.array(updates_reshape_dim)})
                NodeWrap(graph, concat).replace_obj(
                    'Concat', {'name': concat, 'opset_version': 11, 'axis': 0})
                NodeWrap(graph, post_segmentsum).replace_obj(
                    'SegmentReduce', {'name': post_segmentsum, 'method': 'SUM'})
                NodeWrap(graph, update_gather).replace_obj(
                    'Gather', {'name': update_gather})
                NodeWrap(graph, post_reshape).replace_obj(
                    'Reshape', {'name': post_reshape, 'shape': out_shape})

                if scatter_nd in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(scatter_nd)
                    graph._attr['output_names'].remove(scatter_nd)
                    graph._attr['output_names'].insert(index, post_reshape)

            else:
                for src, _, in_attr in in_edges:
                    graph.remove_edge(src, scatter_nd)
                    if in_attr['dst_in_port'] == 1:
                        new_in_attr = copy.deepcopy(in_attr)
                        new_in_attr['dst_in_port'] = 2
                        graph.add_edge(src, scatter_nd, **new_in_attr)
                    elif in_attr['dst_in_port'] == 0:
                        new_in_attr = copy.deepcopy(in_attr)
                        new_in_attr['dst_in_port'] = 1
                        graph.add_edge(src, scatter_nd, **new_in_attr)
                data = np.zeros(
                    data_obj.value, dtype=in_edges[1][2]['tensor'].value.dtype)
                insert_constant(graph, scatter_nd + '_data', data,
                                scatter_nd, in_port=0, data_format='NHWC')
                scatter_nd_attr = scatter_nd_obj.copied_attr()
                scatter_nd_attr.update(
                    {'reduction': 'add', 'opset_version': scatter_nd_obj.correspond_onnx_op['version']})
                NodeWrap(graph, scatter_nd).replace_obj(
                    scatter_nd_obj.correspond_onnx_op['type'], scatter_nd_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_scatternd2(graph, op_type='TfScatterNd'):
    ''' Convert ScatterNd whose indices input is not constant to onnx ScatterND with reduction=add.
    '''
    if op_type not in ('TfScatterNd', 'LiteSCATTER_ND'):
        ERROR('[Parser]: Meets invalid Op type (%s) in convert_scatternd!' % op_type)
        return
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        scatternd = m['target']
        scatternd_obj = NodeWrap(graph, scatternd)['object']
        scatternd_in_edges = graph.sorted_in_edges(scatternd, data=True)
        if scatternd_obj is None or len(scatternd_in_edges) < 3:
            ERROR('[Parser]: Meets invalid %s Node(%s) in convert_scatternd2!' % (op_type, scatternd))
            continue
        indices, _, indices_in_attr = scatternd_in_edges[0]
        updates, _, updates_in_attr = scatternd_in_edges[1]
        shape, _, shape_in_attr = scatternd_in_edges[2]
        if shape_in_attr['tensor'] is None or not shape_in_attr['tensor'].is_const \
                or shape_in_attr['tensor'].value is None:
            WARN('[Parser]: Meets unsupported non-const shape input of Node(%s) in convert_scatternd2!' % scatternd)
            continue
        if updates_in_attr['tensor'] is None or updates_in_attr['tensor'].value is None:
            continue
        data = np.zeros(shape_in_attr['tensor'].value, dtype=updates_in_attr['tensor'].value.dtype)
        graph.remove_edges_from(scatternd_in_edges)
        insert_constant(graph, scatternd + '_data', data, scatternd, in_port=0, data_format='NHWC')
        indices_in_attr.update({'dst_in_port': 1})
        graph.add_edge(indices, scatternd, **indices_in_attr)
        updates_in_attr.update({'dst_in_port': 2})
        graph.add_edge(updates, scatternd, **updates_in_attr)
        scatternd_attr = scatternd_obj.copied_attr()
        scatternd_attr.update({'reduction': 'add', 'opset_version': 16})
        NodeWrap(graph, scatternd).replace_obj('ScatterND', scatternd_attr)


def convert_reverse_sequence(graph, op_type='TfReverseSequence'):
    if op_type not in ('TfReverseSequence', 'LiteREVERSE_SEQUENCE'):
        ERROR('[Parser]: Meets invalid Op type (%s) in convert_reverse_sequence!' % op_type)
        return
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        reverse_sequence = m['target']
        reverse_sequence_obj = NodeWrap(graph, reverse_sequence)['object']
        new_node_attr = reverse_sequence_obj.copied_attr()
        batch_axis = reverse_sequence_obj.batch_dim
        time_axis = reverse_sequence_obj.seq_dim
        if batch_axis in [0, 1] and time_axis in [0, 1]:
            time_axis = reverse_sequence_obj.seq_dim
            batch_axis = reverse_sequence_obj.batch_dim
            new_node_attr.update(
                {'batch_axis': batch_axis, 'time_axis': time_axis})
            new_node_attr.update(
                {'opset_version': reverse_sequence_obj.correspond_onnx_op['version']})
            NodeWrap(graph, reverse_sequence).replace_obj(
                reverse_sequence_obj.correspond_onnx_op['type'], new_node_attr)
        else:
            in_edges = graph.sorted_in_edges(reverse_sequence, data=True)
            out_edges = graph.sorted_out_edges(reverse_sequence, data=True)
            reverse_sequence_src, _, reverse_sequence_in_attr = in_edges[0]
            shape_num = len(reverse_sequence_in_attr['tensor'].value.shape)
            perm_shape = np.arange(shape_num)
            if batch_axis < 0:
                batch_axis = shape_num + batch_axis
            if time_axis < 0:
                time_axis = shape_num + time_axis
            if batch_axis > 1:
                if time_axis == 0:
                    transed_axis = 1
                else:
                    transed_axis = 0
                temp = perm_shape[batch_axis]
                perm_shape[batch_axis] = perm_shape[transed_axis]
                perm_shape[transed_axis] = temp
                batch_axis = transed_axis
            if time_axis > 1:
                if batch_axis == 0:
                    transed_axis = 1
                else:
                    transed_axis = 0
                temp = perm_shape[time_axis]
                perm_shape[time_axis] = perm_shape[transed_axis]
                perm_shape[transed_axis] = temp
                time_axis = transed_axis

            insert_transpose(graph, reverse_sequence_src,
                             reverse_sequence, reverse_sequence_in_attr, perm_shape)
            _, reverse_sequence_out, reverse_sequence_out_attr = out_edges[0]
            ret = insert_transpose(
                graph, reverse_sequence, reverse_sequence_out, reverse_sequence_out_attr, perm_shape)
            transpose_name = []
            transpose_name.append(ret)

            if reverse_sequence in graph._attr['output_names'] and transpose_name:
                index = graph._attr['output_names'].index(reverse_sequence)
                graph._attr['output_names'][index] = transpose_name[0]
            new_node_attr.update(
                {'batch_axis': batch_axis, 'time_axis': time_axis})
            new_node_attr.update(
                {'opset_version': reverse_sequence_obj.correspond_onnx_op['version']})
            NodeWrap(graph, reverse_sequence).replace_obj(
                reverse_sequence_obj.correspond_onnx_op['type'], new_node_attr)


def convert_unpack(graph, op_type='LiteUNPACK'):
    if op_type not in ('TfUnpack', 'LiteUNPACK'):
        ERROR('[Parser]: Meets invalid Op type (%s) in convert_unpack!' % op_type)
        return
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        unpack = m['target']
        unpack_obj = NodeWrap(graph, unpack)['object']
        if unpack_obj is not None \
                and len(unpack_obj.get_output_shapes()) >= 1:
            out_edges = graph.sorted_out_edges(unpack, data=True)
            split_attr = unpack_obj.copied_attr()
            split_attr.update(
                {'opset_version': 2, 'split': [1] * unpack_obj.num})
            NodeWrap(graph, unpack).replace_obj('Split', split_attr)

            output_shapes = unpack_obj.get_output_shapes()
            reshape_dim = copy.deepcopy(list(output_shapes[0]))
            last_names = []
            out_ports = unpack_obj.get_out_ports()
            for p in out_ports:
                reshape = get_valid_node_name(
                    graph, unpack + '_post_reshape_' + str(p))
                unpack_out_tensor = None
                for _, dst, out_attr in out_edges:
                    if out_attr['src_out_port'] == p:
                        unpack_out_tensor = out_attr['tensor']
                        new_out_attr = copy.deepcopy(out_attr)
                        new_out_attr['src_out_port'] = 0
                        graph.remove_edge(unpack, dst)
                        graph.add_edge(reshape, dst, **new_out_attr)
                split_out_attr = {'src_out_port': p, 'dst_in_port': 0}
                if unpack_out_tensor is not None \
                        and unpack_out_tensor.dtype is not None:
                    split_out_tensor = Tensor(dtype=unpack_out_tensor.dtype,
                                              scale_zp=unpack_out_tensor.scale_zp)
                    split_out_attr.update({'tensor': split_out_tensor})
                graph.add_edge(unpack, reshape, **split_out_attr)
                NodeWrap(graph, reshape).replace_obj(
                    'Reshape', {'name': reshape, 'opset_version': 5})
                insert_constant(graph, reshape + '_shape', np.array(reshape_dim,
                                np.int64), reshape, in_port=1, data_format='NHWC')
                last_names.append(reshape)

            if unpack in graph._attr['output_names'] and last_names:
                index = graph._attr['output_names'].index(unpack)
                graph._attr['output_names'].remove(unpack)
                for name in last_names:
                    graph._attr['output_names'].insert(index, name)
                    index += 1
        else:
            ERROR(
                '[Parser]: Meets invalid LiteUNPACK Node(%s) in convert_unpack!' % unpack)


def convert_special_uni_seq_lstm(graph):
    matches = single_node_matcher(graph, 'LiteUNIDIRECTIONAL_SEQUENCE_LSTM')
    for m in matches:
        lstm = m['target']
        lstm_obj = NodeWrap(graph, lstm)['object']
        in_edges = graph.sorted_in_edges(lstm, keys=True, data=True)
        if lstm_obj is not None and len(in_edges) == 24:
            if not FLOAT_EQUAL(lstm_obj.proj_clip, 0.0):
                WARN(
                    '[Parser]: Meets unsupported non-zero proj_clip of UNIDIRECTIONAL_SEQUENCE_LSTM (%s) in convert_special_uni_seq_lstm!' % lstm)
                continue
            inputs = lstm_obj.get_input_tensors()
            if inputs[0] is None:
                ERROR(
                    '[Parser]: Meets invalid input for TFLite UNIDIRECTIONAL_SEQUENCE_LSTM (%s)!' % lstm)
                continue
            if any([inp is None for inp in inputs[1:9]]):
                ERROR('[Parser]: Cannot convert TFLite UNIDIRECTIONAL_SEQUENCE_LSTM (%s) with empty parameter/recurrent weights to Onnx!' % lstm)
                continue
            if any([inp is not None for inp in inputs[16:18]]):
                ERROR(
                    '[Parser]: Cannot convert TFLite UNIDIRECTIONAL_SEQUENCE_LSTM (%s) with projection mode to Onnx!' % lstm)
                continue
            if any([inp is not None for inp in inputs[20:24]]):
                ERROR(
                    '[Parser]: Cannot convert TFLite UNIDIRECTIONAL_SEQUENCE_LSTM (%s) with layer_norm mode to Onnx!' % lstm)
                continue

            inp = inputs[0]
            input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights = inputs[
                1:5]
            recurrent_to_input_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights = inputs[
                5:9]
            cell_to_input_weights, cell_to_forget_weights, cell_to_output_weights = inputs[
                9:12]
            input_gate_bias, forget_gate_bias, cell_bias, output_gate_bias = inputs[12:16]
            input_activation_state, input_cell_state = inputs[18:20]

            layout = True
            if np.ndim(inp) == 3:
                if lstm_obj.time_major:
                    time_steps, batch_size, input_size = inp.shape[0:3]
                    layout = False
                else:
                    batch_size, time_steps, input_size = inp.shape[0:3]
            else:
                batch_size, time_steps = inp.shape[0], 1
                reshape_dim = [inp.shape[0], 1,
                               int(np.prod(inp.shape)) // inp.shape[0]]
                input_size = reshape_dim[-1]
                src, _, k, in_attr = in_edges[0]
                insert_reshape(graph, src, lstm, in_attr, reshape_dim,
                               key=k, type='Reshape', data_format='NHWC')
                in_edges = graph.sorted_in_edges(lstm, keys=True, data=True)

            cell_size = input_to_input_weights.shape[0]
            cell_size_zeros = np.zeros((cell_size, ), np.float32)

            # default peephole weight
            if cell_to_input_weights is None:
                cell_to_input_weights = cell_size_zeros
            if cell_to_forget_weights is None:
                cell_to_forget_weights = cell_size_zeros
            if cell_to_output_weights is None:
                cell_to_output_weights = cell_size_zeros

            # default bias
            if input_gate_bias is None:
                input_gate_bias = cell_size_zeros
            if forget_gate_bias is None:
                forget_gate_bias = cell_size_zeros
            if cell_bias is None:
                cell_bias = cell_size_zeros
            if output_gate_bias is None:
                output_gate_bias = cell_size_zeros

            # iofc
            W = np.concatenate([input_to_input_weights, input_to_output_weights,
                                input_to_forget_weights, input_to_cell_weights], axis=0)
            R = np.concatenate([recurrent_to_input_weights, recurrent_to_output_weights,
                                recurrent_to_forget_weights, recurrent_to_cell_weights], axis=0)
            B = np.concatenate([input_gate_bias, output_gate_bias,
                                forget_gate_bias, cell_bias] + [cell_size_zeros] * 4, axis=0)
            # iof
            P = np.concatenate(
                [cell_to_input_weights, cell_to_output_weights, cell_to_forget_weights], axis=0)

            W = np.expand_dims(W, 0)
            R = np.expand_dims(R, 0)
            B = np.expand_dims(B, 0)
            P = np.expand_dims(P, 0)

            sequence_lens = np.array([time_steps] * batch_size, np.int32)
            h_init, _, h_init_k, h_init_in_attr = in_edges[18]
            c_init, _, c_init_k, c_init_in_attr = in_edges[19]

            graph.remove_edges_from(in_edges[1:])
            insert_constant(graph, lstm + '_W', W, lstm,
                            in_port=1, data_format='NHWC')
            insert_constant(graph, lstm + '_R', R, lstm,
                            in_port=2, data_format='NHWC')
            insert_constant(graph, lstm + '_B', B, lstm,
                            in_port=3, data_format='NHWC')
            insert_constant(graph, lstm + '_sequence_lens',
                            sequence_lens, lstm, in_port=4, data_format='NHWC')
            insert_constant(graph, lstm + '_P', P, lstm,
                            in_port=7, data_format='NHWC')

            if input_activation_state is not None:
                new_h_init_in_attr = copy.deepcopy(h_init_in_attr)
                new_h_init_in_attr['dst_in_port'] = 5
                graph.add_edge(h_init, lstm, **new_h_init_in_attr)
                h_init_reshape_dim = [1, batch_size, cell_size] if lstm_obj.time_major else [
                    batch_size, 1, cell_size]
                insert_reshape(graph, h_init, lstm, new_h_init_in_attr,
                               h_init_reshape_dim, key=h_init_k)
            else:
                h_init_value = np.zeros([1, batch_size, cell_size] if lstm_obj.time_major else [
                                        batch_size, 1, cell_size], np.float32)
                insert_constant(graph, lstm + '_h_init', h_init_value,
                                lstm, in_port=5, data_format='NHWC')

            if input_cell_state is not None:
                new_c_init_in_attr = copy.deepcopy(c_init_in_attr)
                new_c_init_in_attr['dst_in_port'] = 6
                graph.add_edge(c_init, lstm, **new_c_init_in_attr)
                c_init_reshape_dim = [1, batch_size, cell_size] if lstm_obj.time_major else [
                    batch_size, 1, cell_size]
                insert_reshape(graph, c_init, lstm, new_c_init_in_attr,
                               c_init_reshape_dim, key=c_init_k)
            else:
                c_init_value = np.zeros([1, batch_size, cell_size] if lstm_obj.time_major else [
                                        batch_size, 1, cell_size], np.float32)
                insert_constant(graph, lstm + '_c_init', c_init_value,
                                lstm, in_port=6, data_format='NHWC')

            out_edges = graph.sorted_out_edges(lstm, keys=True, data=True)
            lstm_post_reshape = get_valid_node_name(
                graph, lstm + '_post_reshape')
            for _, dst, k, out_attr in out_edges:
                graph.remove_edge(lstm, dst, k)
                graph.add_edge(lstm_post_reshape, dst, **out_attr)
            graph.add_edge(lstm, lstm_post_reshape)

            if lstm in graph._attr['output_names']:
                index = graph._attr['output_names'].index(lstm)
                graph._attr['output_names'][index] = lstm_post_reshape

            if lstm_obj.time_major:
                post_reshape_dim = np.array(
                    [time_steps, batch_size, cell_size], np.int64)
            else:
                post_reshape_dim = np.array(
                    [batch_size, time_steps, cell_size], np.int64)
            NodeWrap(graph, lstm_post_reshape).replace_obj(
                'Reshape', {'name': lstm_post_reshape, 'opset_version': 5})
            insert_constant(graph, lstm_post_reshape + '_dim', post_reshape_dim,
                            lstm_post_reshape, in_port=1, data_format='NHWC')

            lstm_attr = lstm_obj.copied_attr()
            lstm_attr.update({'opset_version': 14,
                              'input_size': input_size,
                              'time_steps': time_steps,
                              'hidden_size': cell_size,
                              'direction': 'forward',
                              'activations': ['SIGMOID', lstm_obj.activations, lstm_obj.activations],
                              'method': 'Y',
                              'layout': layout
                              })
            NodeWrap(graph, lstm).replace_obj('LSTM', lstm_attr)

            clear_redundant_nodes(graph)
        else:
            ERROR(
                '[Parser]: Meets invalid LiteUNIDIRECTIONAL_SEQUENCE_LSTM Node(%s)!' % lstm)


def convert_strided_slice(graph, op_type='TfStridedSlice'):
    if op_type not in ('TfStridedSlice', 'LiteSTRIDED_SLICE'):
        ERROR('[Parser]: Meets invalid Op type (%s) in convert_strided_slice!' % op_type)
        return
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        strided_slice = m['target']
        slice_obj = NodeWrap(graph, strided_slice)['object']
        in_edges = graph.sorted_in_edges(strided_slice, keys=True, data=True)
        out_edges = graph.sorted_out_edges(strided_slice, data=True)
        if slice_obj is not None and len(in_edges) == 4 and len(out_edges) > 0:
            in_consts = slice_obj.sorted_in_consts()
            if len(in_consts) < 3 or (len(in_consts) == 3 and in_consts[0][0] == strided_slice):
                WARN('[Parser]: Invalid StridedSlice (%s) to convert due to dynamic range of begin/end/strides in convert_strided_slice!' % strided_slice)
                continue

            begin, end, strides = [c[2] for c in in_consts[:3]]
            input_shape = slice_obj.get_input_shapes()[0]
            if input_shape is None or any([s is None for s in input_shape]):
                ERROR(
                    '[Parser]: Invalid StridedSlice (%s) input shape in convert_strided_slice!' % strided_slice)
                continue
            axes_shape = slice_obj.get_input_shapes()[0]
            begin_inp, _, _, begin_in_attr = in_edges[1]
            end_inp, _, _, end_in_attr = in_edges[2]
            strides_inp, _, _, strides_in_attr = in_edges[3]

            last_name = strided_slice
            if slice_obj.begin_mask != 0 \
                    or slice_obj.end_mask != 0 \
                    or slice_obj.ellipsis_mask != 0\
                    or slice_obj.shrink_axis_mask != 0 \
                    or slice_obj.new_axis_mask != 0:
                shape = input_shape
                shrink_axis_mask = slice_obj.shrink_axis_mask
                new_axis_mask = slice_obj.new_axis_mask
                ellipsis_mask = slice_obj.ellipsis_mask
                begin_mask = slice_obj.begin_mask
                end_mask = slice_obj.end_mask
                out_shape = []

                from ....ops.tf_ops.array_ops import TfStridedSliceOp
                begin, end, strides, out_shape, reshape_dim1, split_axis, splits_dim, reshape_dim2 = TfStridedSliceOp.set_attr_remove_mask_tf(shape,
                                                                                                                                              begin,
                                                                                                                                              end,
                                                                                                                                              strides,
                                                                                                                                              shrink_axis_mask,
                                                                                                                                              new_axis_mask,
                                                                                                                                              ellipsis_mask,
                                                                                                                                              begin_mask,
                                                                                                                                              end_mask)

                if reshape_dim1 != None:
                    src, _, k, p_in_attr = in_edges[0]
                    insert_reshape(
                        graph, src, strided_slice, p_in_attr, reshape_dim1, key=k)

                if reshape_dim2 != None:
                    post_reshape = get_valid_node_name(
                        graph, strided_slice + '_post_reshape')
                    for _, dst, out_attr in out_edges:
                        graph.remove_edge(strided_slice, dst)
                        graph.add_edge(
                            post_reshape, dst, **out_attr)
                    _, _, out_attr = out_edges[0]

                    slice_out_attr = {'src_out_port': 0, 'dst_in_port': 0}
                    if out_edges[0][2]['tensor'].dtype is not None:
                        slice_out_tensor = Tensor(dtype=out_edges[0][2]['tensor'].dtype,
                                                  scale_zp=out_edges[0][2]['tensor'].scale_zp)
                        slice_out_attr.update({'tensor': slice_out_tensor})
                    graph.add_edge(strided_slice, post_reshape,
                                   **slice_out_attr)

                    NodeWrap(graph, post_reshape).replace_obj(
                        'Reshape', {'name': post_reshape, 'opset_version': 1, 'shape': reshape_dim2})
                    last_name = post_reshape

                    if splits_dim != None:
                        post_split = get_valid_node_name(
                            graph, strided_slice + '_post_split')
                        out = get_valid_node_name(graph, post_split + '_out')
                        out_edges = graph.sorted_out_edges(strided_slice, data=True)
                        slice_out_tensor = out_edges[0][2]['tensor']
                        slice_out_attr = {'src_out_port': 0, 'dst_in_port': 0}
                        split_out_attr = {'src_out_port': 0, 'dst_in_port': 0}
                        split_out_1_attr = {'src_out_port': 1, 'dst_in_port': 0}
                        if slice_out_tensor.dtype is not None:
                            tensor = Tensor(dtype=out_attr['tensor'].dtype,
                                            scale_zp=out_attr['tensor'].scale_zp)
                            slice_out_attr.update(
                                {'tensor': copy.deepcopy(tensor)})
                            split_out_attr.update(
                                {'tensor': copy.deepcopy(tensor)})
                            split_out_1_attr.update(
                                {'tensor': copy.deepcopy(tensor)})

                        graph.remove_edge(strided_slice, post_reshape)
                        graph.add_edge(strided_slice, post_split, **slice_out_attr)
                        graph.add_edge(post_split, post_reshape, **split_out_attr)
                        graph.add_edge(post_split, out, **split_out_1_attr)
                        NodeWrap(graph, out).replace_obj('Out', {'name': out})
                        NodeWrap(graph, post_split).replace_obj(
                            'Split', {'name': post_split, 'opset_version': 11, 'axis': split_axis, 'split': splits_dim})

                if len(out_shape) > len(input_shape):
                    axes_shape = out_shape

            axes = np.array(range(len(axes_shape)), np.int32)
            if len(input_shape) != len(begin) or len(input_shape) != len(end):
                length = len(input_shape) - len(begin)
                begin = np.array(list(begin) + [0] * length)
                end = np.array(
                    list(end) + input_shape[len(input_shape) - length:])
                strides = np.array(list(strides) + [1] * length)

            graph.remove_edges_from(in_edges[1:])

            new_begin_in_attr = copy.deepcopy(begin_in_attr)
            new_begin_in_attr['tensor'].value = begin
            NodeWrap(graph, begin_inp)['object'].value = begin
            graph.add_edge(begin_inp, strided_slice, **new_begin_in_attr)

            new_end_in_attr = copy.deepcopy(end_in_attr)
            new_end_in_attr['tensor'].value = end
            NodeWrap(graph, end_inp)['object'].value = end
            graph.add_edge(end_inp, strided_slice, **new_end_in_attr)

            insert_constant(graph, strided_slice + '_axes',
                            axes, strided_slice, in_port=3)

            new_strides_in_attr = copy.deepcopy(strides_in_attr)
            new_strides_in_attr['tensor'].value = strides
            new_strides_in_attr['dst_in_port'] = 4
            NodeWrap(graph, strides_inp)['object'].value = strides
            graph.add_edge(strides_inp, strided_slice, **new_strides_in_attr)

            slice_attr = slice_obj.copied_attr()
            slice_attr.update({'opset_version': 10})
            NodeWrap(graph, strided_slice).replace_obj('Slice', slice_attr)

            if strided_slice in graph._attr['output_names'] and last_name != strided_slice:
                index = graph._attr['output_names'].index(strided_slice)
                graph._attr['output_names'][index] = last_name
        else:
            ERROR('[Parser]: Meets invalid TFLite STRIDED_SLICE (%s) in convert_strided_slice!' %
                  strided_slice)


def merge_quantized_lstm(graph):
    def _get_forget_bias(node_name, node_obj):
        '''
        Return float_forget_bias, quant_forget_bias, forget_bias_scale_zp of the node.
        '''
        node_out_edges = graph.sorted_out_edges(node_name, data=True)
        if node_obj is None or len(node_out_edges) < 1:
            ERROR('[Parser]: Meet invalid node (%s) in _get_forget_bias of merge_quantized_lstm!' % node_name)
            return (None, None, None)
        quant_forget_bias = node_obj.value
        forget_bias_scale_zp = node_out_edges[0][2]['tensor'].scale_zp
        if len(forget_bias_scale_zp) == 2:
            float_forget_bias = (quant_forget_bias - forget_bias_scale_zp[1]) * forget_bias_scale_zp[0]
        else:
            float_forget_bias = None
        return (float_forget_bias, quant_forget_bias, forget_bias_scale_zp)

    matches = matched_patterns(graph,
                               nodes=[('concat', {'op': 'LiteCONCATENATION'}),
                                      ('fc', {'op': 'LiteFULLY_CONNECTED'}),
                                      ('split', {'op': 'LiteSPLIT'}),
                                      ('sigmoid_sp0', {'op': 'LiteLOGISTIC'}),
                                      ('tanh_sp1', {'op': 'LiteTANH'}),
                                      ('add_sp2', {'op': 'LiteADD'}),
                                      ('sigmoid_sp3', {'op': 'LiteLOGISTIC'}),
                                      ('adder', {'op': 'Constant'}),
                                      ('mul_sp01', {'op': 'LiteMUL'}),
                                      ('sigmoid_sp2', {'op': 'LiteLOGISTIC'}),
                                      ('mul_sp2', {'op': 'LiteMUL'}),
                                      ('add_cout', {'op': 'LiteADD'}),
                                      ('tanh_c', {'op': 'LiteTANH'}),
                                      ('mul_hout', {'op': 'LiteMUL'}),
                                      ],
                               edges=[('concat', 'fc'),
                                      ('fc', 'split'),
                                      ('split', 'sigmoid_sp0',
                                       {'src_out_port': 0}),
                                      ('split', 'tanh_sp1',
                                       {'src_out_port': 1}),
                                      ('split', 'add_sp2', {
                                       'src_out_port': 2}),
                                      ('split', 'sigmoid_sp3',
                                       {'src_out_port': 3}),
                                      ('sigmoid_sp0', 'mul_sp01'),
                                      ('tanh_sp1', 'mul_sp01'),
                                      ('adder', 'add_sp2'),
                                      ('add_sp2', 'sigmoid_sp2'),
                                      ('sigmoid_sp2', 'mul_sp2'),
                                      ('mul_sp01', 'add_cout'),
                                      ('mul_sp2', 'add_cout'),
                                      ('add_cout', 'tanh_c'),
                                      ('tanh_c', 'mul_hout'),
                                      ('sigmoid_sp3', 'mul_hout'),
                                      ]
                               )
    # Match lstm cell and save the matches info to a dict called lstm_matches_dict.
    # The key of the dict is the starting node of lstm(unpack),
    # value is a tuple of two elements (unpack out port, matches).
    lstm_matches_dict = {}
    for m in matches:
        names = ['concat', 'fc', 'adder', 'mul_sp2']
        objs_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(obj is None for obj in objs_dict.values()):
            ERROR('[Parser]: Meets invalid node in merge_quantized_lstm!')
            continue
        if objs_dict['adder'].value is None \
                or objs_dict['adder'].value != 127 \
                or objs_dict['fc'].weights is None \
                or objs_dict['fc'].biases is None:
            continue
        concat_in_edges = graph.sorted_in_edges(m['concat'], data=True)
        mul_in_edges = graph.sorted_in_edges(m['mul_sp2'])
        if len(concat_in_edges) != 2 \
                or len(mul_in_edges) != 2:
            continue
        mul_inputs = [src for src,
                      _ in mul_in_edges if src != m['sigmoid_sp2']]
        if len(mul_inputs) != 1:
            continue
        last_cout = mul_inputs[0]
        unpack, last_hout = None, None
        for idx, (concat_input_node, _, in_attr) in enumerate(concat_in_edges):
            node_obj = NodeWrap(graph, concat_input_node)['object']
            if node_obj is None:
                continue
            if node_obj.type in ['LiteQUANTIZE', 'Cast']:
                in_edges = graph.sorted_in_edges(concat_input_node, data=True)
                if len(in_edges) < 1:
                    continue
                in_edge = in_edges[0]
            else:
                in_edge = concat_in_edges[idx]
            if idx == 0:
                unpack, _, in_attr = in_edge
                unpack_out_port = in_attr['src_out_port']
            else:
                last_hout, _, _ = in_edge
        if unpack is None or last_hout is None:
            continue

        match_dict = copy.deepcopy(m)
        match_dict.update({'last_cout': last_cout, 'last_hout': last_hout})
        if unpack not in lstm_matches_dict:
            lstm_matches_dict.update({unpack: [(unpack_out_port, match_dict)]})
        else:
            matches_list = lstm_matches_dict[unpack]
            matches_list.append((unpack_out_port, match_dict))
            lstm_matches_dict.update({unpack: matches_list})

    matched = False
    for unpack, cell_matches in lstm_matches_dict.items():
        if not cell_matches:
            continue
        unpack_obj = NodeWrap(graph, unpack)['object']
        unpack_in_edges = graph.sorted_in_edges(unpack, data=True)
        if unpack_obj.type != 'LiteUNPACK' \
                or unpack_obj.axis != 0 \
                or unpack_obj.num != len(cell_matches) \
                or len(unpack_in_edges) < 1:
            continue
        unpack_in_shapes = unpack_obj.get_input_shapes()
        if len(unpack_in_shapes) < 1 \
                or unpack_in_shapes[0] is None \
                or len(unpack_in_shapes[0]) != 3:
            continue
        seq_length, batch_size, input_size = unpack_in_shapes[0]

        cell_matches.sort()
        _, first_cell_match = cell_matches[0]

        # Get the value of forget_bias
        first_adder_obj = NodeWrap(graph, first_cell_match['adder'])['object']
        float_forget_bias, quant_forget_bias, forget_bias_scale_zp = _get_forget_bias(
            first_cell_match['adder'], first_adder_obj)

        # Check whether weights and biases are valid
        first_fc_obj = NodeWrap(graph, first_cell_match['fc'])['object']
        weights = first_fc_obj.weights
        biases = first_fc_obj.biases
        if len(weights.shape) != 2 \
                or len(biases.shape) != 1 \
                or weights.shape[0] != biases.shape[0] \
                or biases.shape[0] % 4 != 0:
            continue
        hidden_size = int(biases.shape[0] / 4)
        if weights.shape[1] != (input_size + hidden_size):
            continue
        dequant_weights = (
            weights - first_fc_obj.weights_scale_zp[1]) * first_fc_obj.weights_scale_zp[0]
        dequant_biases = (
            biases - first_fc_obj.biases_scale_zp[1]) * first_fc_obj.biases_scale_zp[0]
        weights_scale_zp = first_fc_obj.weights_scale_zp
        biases_scale_zp = first_fc_obj.biases_scale_zp

        # Check whether initial hidden and cell node are valid
        concat_in_edges = graph.sorted_in_edges(
            first_cell_match['concat'], data=True)
        if len(concat_in_edges) < 2:
            continue
        initial_h, _, in_attr = concat_in_edges[1]
        initial_h_in_attr = copy.deepcopy(in_attr)
        initial_h_in_attr.update({'dst_in_port': 5})
        initial_c = first_cell_match['last_cout']
        initial_c_out_edges = graph.sorted_out_edges(initial_c, data=True)
        initial_c_in_attr = None
        for _, dst, out_attr in initial_c_out_edges:
            if dst == first_cell_match['mul_sp2']:
                initial_c_in_attr = copy.deepcopy(out_attr)
                initial_c_in_attr.update({'dst_in_port': 6})
                break
        if initial_c_in_attr is None:
            continue

        # Check whether all the cell_matches belong to one lstm opeartion
        final_hout_list = [first_cell_match['mul_hout']]
        is_same_lstm = True
        for idx, (parent_out_port, cell_match) in enumerate(cell_matches):
            if idx != parent_out_port:
                is_same_lstm = False
                break
            if idx == 0:
                continue
            fc = cell_match['fc']
            fc_obj = NodeWrap(graph, fc)['object']
            if fc_obj is None \
                    or not np.allclose(dequant_weights, ((fc_obj.weights - fc_obj.weights_scale_zp[1]) * fc_obj.weights_scale_zp[0]), rtol=1e-04, atol=1e-04) \
                    or not np.allclose(dequant_biases, ((fc_obj.biases - fc_obj.biases_scale_zp[1]) * fc_obj.biases_scale_zp[0]), rtol=1e-04, atol=1e-04):
                is_same_lstm = False
                break
            if cell_match['last_hout'] != cell_matches[idx - 1][1]['mul_hout'] \
                    or cell_match['last_cout'] != cell_matches[idx - 1][1]['add_cout']:
                is_same_lstm = False
                break
            # Get the value of forget_bias
            current_forget_bias = _get_forget_bias(
                cell_match['adder'], NodeWrap(graph, cell_match['adder'])['object'])[0]
            if current_forget_bias is None or not FLOAT_EQUAL(current_forget_bias, float_forget_bias):
                is_same_lstm = False
                break
            final_hout_list.append(cell_match['mul_hout'])
        if not is_same_lstm:
            continue

        # Get the final outputs of the hidden and the cell
        final_hout = cell_matches[-1][1]['mul_hout']
        final_hout_out_edges = graph.sorted_out_edges(final_hout, data=True)
        final_cout = cell_matches[-1][1]['add_cout']
        final_cout_out_edges = graph.sorted_out_edges(final_cout, data=True)

        # Check whether there is pack node to concat all the intermediate outputs of the hidden
        pack = None
        for _, dst, _ in final_hout_out_edges:
            dst_obj = NodeWrap(graph, dst)['object']
            if dst_obj is None:
                ERROR('[Parser]: Meets invalid node %s in merge_quantized_lstm!' % dst)
                continue
            if dst_obj.type == 'LitePACK':
                pack_in_edges = graph.sorted_in_edges(dst)
                if len(pack_in_edges) != unpack_obj.num \
                        or any(pack_src not in final_hout_list for pack_src, _ in pack_in_edges):
                    continue
                pack = dst
                break

        matched = True
        lstm = get_valid_node_name(graph, initial_h + '_lstm')
        # Reconnect outputs
        if pack is not None:
            pack_out_edges = graph.sorted_out_edges(pack, data=True)
            for _, dst, out_attr in pack_out_edges:
                graph.remove_edge(pack, dst)
                new_attr = copy.deepcopy(out_attr)
                new_attr.update({'src_out_port': 0})
                graph.add_edge(lstm, dst, **new_attr)
        for _, dst, out_attr in final_hout_out_edges:
            if pack is not None and dst == pack:
                continue
            graph.remove_edge(final_hout, dst)
            new_attr = copy.deepcopy(out_attr)
            new_attr.update({'src_out_port': 1})
            graph.add_edge(lstm, dst, **new_attr)
        for _, dst, out_attr in final_cout_out_edges:
            graph.remove_edge(final_cout, dst)
            new_attr = copy.deepcopy(out_attr)
            new_attr.update({'src_out_port': 2})
            graph.add_edge(lstm, dst, **new_attr)

        for idx, node in enumerate([pack, final_hout, final_cout]):
            if node is None:
                continue
            if node == pack:
                old_dim = [seq_length, 1, batch_size, hidden_size]
                new_dim = [seq_length, batch_size, hidden_size]
            else:
                old_dim = [1, batch_size, hidden_size]
                new_dim = [batch_size, hidden_size]
            post_reshape = insert_reshape_after(
                graph, lstm, new_dim, old_dim, out_port=idx)
            if node in graph._attr['output_names']:
                index = graph._attr['output_names'].index(node)
                graph._attr['output_names'][index] = post_reshape

        # Prepare inputs for onnx lstm
        kernel_weights, recurrent_weights = np.split(
            weights, [input_size], axis=1)
        input_w, cell_w, forget_w, output_w = np.split(
            kernel_weights, 4, axis=0)
        input_r, cell_r, forget_r, output_r = np.split(
            recurrent_weights, 4, axis=0)
        input_wb, cell_wb, forget_wb, output_wb = np.split(biases, 4, axis=0)
        W_value = np.stack([np.concatenate(
            [input_w, output_w, forget_w, cell_w], axis=0)])
        R_value = np.stack([np.concatenate(
            [input_r, output_r, forget_r, cell_r], axis=0)])
        biases_w = np.concatenate([input_wb, output_wb, forget_wb, cell_wb])
        biases_r = np.zeros_like(biases_w)
        B_value = np.stack([np.concatenate([biases_w, biases_r])])
        seq_length = np.array([seq_length] * batch_size, np.int64)
        initial_hc_shape = [1, batch_size, hidden_size]

        # Convert to onnx lstm
        inp, _, in_attr = unpack_in_edges[0]
        graph.add_edge(inp, lstm, **in_attr)
        graph.add_edge(initial_h, lstm, **initial_h_in_attr)
        graph.add_edge(initial_c, lstm, **initial_c_in_attr)
        insert_constant(graph, lstm + '_W', W_value, lstm,
                        in_port=1, data_format='NHWC')
        insert_constant(graph, lstm + '_R', R_value, lstm,
                        in_port=2, data_format='NHWC')
        insert_constant(graph, lstm + '_B', B_value, lstm,
                        in_port=3, data_format='NHWC')
        insert_constant(graph, lstm + '_seq_length', seq_length, lstm,
                        in_port=4, data_format='NHWC')
        insert_reshape(graph, initial_h, lstm,
                       initial_h_in_attr, initial_hc_shape)
        insert_reshape(graph, initial_c, lstm,
                       initial_c_in_attr, initial_hc_shape)
        lstm_attr = {'name': lstm,
                     'opset_version': 14,
                     'quantize': 1,
                     'layout': False,
                     'hidden_size': hidden_size,
                     'forget_bias': quant_forget_bias,
                     'forget_bias_scale_zp': forget_bias_scale_zp,
                     'weights_scale_zp': weights_scale_zp,
                     'biases_scale_zp': biases_scale_zp}
        NodeWrap(graph, lstm).replace_obj('LSTM', lstm_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_quantized_lstm2(graph):
    '''
    Merge quantized lstm(crnn). The cell matches are different with merge_quantized_lstm.
    And the first cell merges the initial hidden state(H0) to biases, which may introduce
    errors if trying to calculate H0. Therefore, only merge the latter timesteps to lstm,
    and the first timestep is not merged.
    '''
    first_cell_matches = matched_patterns(graph,
                                          nodes=[('fc', {'op': 'LiteFULLY_CONNECTED'}),
                                                 ('split', {'op': 'LiteSPLIT'}),
                                                 ('sigmoid_sp0', {'op': 'LiteLOGISTIC'}),
                                                 ('sigmoid_sp1', {'op': 'LiteLOGISTIC'}),
                                                 ('tanh_sp2', {'op': 'LiteTANH'}),
                                                 ('sigmoid_sp3', {'op': 'LiteLOGISTIC'}),
                                                 ('mul_sp01', {'op': 'LiteMUL'}),
                                                 ('mul_sp2', {'op': 'LiteMUL'}),
                                                 ('add_cout', {'op': 'LiteADD'}),
                                                 ('tanh_c', {'op': 'LiteTANH'}),
                                                 ('mul_hout', {'op': 'LiteMUL'}),
                                                 ],
                                          edges=[('fc', 'split'),
                                                 ('split', 'sigmoid_sp0', {'src_out_port': 0}),
                                                 ('split', 'sigmoid_sp1', {'src_out_port': 1}),
                                                 ('split', 'tanh_sp2', {'src_out_port': 2}),
                                                 ('split', 'sigmoid_sp3', {'src_out_port': 3}),
                                                 ('sigmoid_sp0', 'mul_sp01'),
                                                 ('tanh_sp2', 'mul_sp01'),
                                                 ('sigmoid_sp1', 'mul_sp2'),
                                                 ('mul_sp01', 'add_cout'),
                                                 ('mul_sp2', 'add_cout'),
                                                 ('add_cout', 'tanh_c'),
                                                 ('tanh_c', 'mul_hout'),
                                                 ('sigmoid_sp3', 'mul_hout'),
                                                 ])
    other_cell_matches = matched_patterns(graph,
                                          nodes=[('fc1', {'op': 'LiteFULLY_CONNECTED'}),
                                                 ('fc2', {'op': 'LiteFULLY_CONNECTED'}),
                                                 ('add_fc', {'op': 'LiteADD'}),
                                                 ('bias', {'op': 'Constant'}),
                                                 ('add_bias', {'op': 'LiteADD'}),
                                                 ('split', {'op': 'LiteSPLIT'}),
                                                 ('sigmoid_sp0', {'op': 'LiteLOGISTIC'}),
                                                 ('sigmoid_sp1', {'op': 'LiteLOGISTIC'}),
                                                 ('tanh_sp2', {'op': 'LiteTANH'}),
                                                 ('sigmoid_sp3', {'op': 'LiteLOGISTIC'}),
                                                 ('mul_sp01', {'op': 'LiteMUL'}),
                                                 ('mul_sp2', {'op': 'LiteMUL'}),
                                                 ('add_cout', {'op': 'LiteADD'}),
                                                 ('tanh_c', {'op': 'LiteTANH'}),
                                                 ('mul_hout', {'op': 'LiteMUL'}),
                                                 ],
                                          edges=[('fc1', 'add_fc'),
                                                 ('fc2', 'add_fc'),
                                                 ('add_fc', 'add_bias'),
                                                 ('bias', 'add_bias'),
                                                 ('add_bias', 'split'),
                                                 ('split', 'sigmoid_sp0', {'src_out_port': 0}),
                                                 ('split', 'sigmoid_sp1', {'src_out_port': 1}),
                                                 ('split', 'tanh_sp2', {'src_out_port': 2}),
                                                 ('split', 'sigmoid_sp3', {'src_out_port': 3}),
                                                 ('sigmoid_sp0', 'mul_sp01'),
                                                 ('tanh_sp2', 'mul_sp01'),
                                                 ('sigmoid_sp1', 'mul_sp2'),
                                                 ('mul_sp01', 'add_cout'),
                                                 ('mul_sp2', 'add_cout'),
                                                 ('add_cout', 'tanh_c'),
                                                 ('tanh_c', 'mul_hout'),
                                                 ('sigmoid_sp3', 'mul_hout'),
                                                 ])
    lstm_matches_dict = {}
    for m in first_cell_matches + other_cell_matches:
        if 'fc' in m:
            fc = m['fc']
            fc_obj = NodeWrap(graph, fc)['object']
            if fc_obj is None or fc_obj.weights is None:
                WARN('[Parser]: Meets invalid node (%s) in merge_quantized_lstm2!' % fc)
                continue
            fc_in_edges = graph.sorted_in_edges(fc, data=True)
            if len(fc_in_edges) < 1:
                continue
        else:
            fc1 = m['fc1']
            fc2 = m['fc2']
            fc_in_edges = graph.sorted_in_edges(fc1, data=True)[:1] + graph.sorted_in_edges(fc2, data=True)[:1]
            if len(fc_in_edges) < 2:
                continue
        mul_in_edges = graph.sorted_in_edges(m['mul_sp2'])
        mul_inputs = [src for src,
                      _ in mul_in_edges if src != m['sigmoid_sp1']]
        if len(mul_inputs) != 1:
            continue
        last_cout = mul_inputs[0]
        unpack_out_port, unpack, last_hout = None, None, None
        match_dict = copy.deepcopy(m)
        for idx, (fc_input_node, fc_node, in_attr) in enumerate(fc_in_edges):
            node_obj = NodeWrap(graph, fc_input_node)['object']
            if node_obj is None:
                continue
            if node_obj.type == 'LiteUNPACK':
                unpack = fc_input_node
                unpack_out_port = in_attr['src_out_port']
                match_dict.update({'fc_after_unpack': fc_node})
            else:
                last_hout = fc_input_node
                match_dict.update({'fc_after_hstate': fc_node})
        if unpack_out_port is None or unpack is None:
            continue
        mul_hout_out_edges = graph.sorted_out_edges(m['mul_hout'])
        pack = None
        for _, dst in mul_hout_out_edges:
            dst_obj = NodeWrap(graph, dst)['object']
            if dst_obj is None:
                WARN('[Parser]: Meets invalid node %s in merge_quantized_lstm2!' % dst)
                continue
            if dst_obj.type == 'LitePACK':
                pack = dst
                break
        if pack is None:
            WARN('[Parser]: No pack node found in merge_quantized_lstm2!')
            continue

        match_dict.update({'last_cout': last_cout, 'last_hout': last_hout})
        lstm_matches_dict_key = (unpack, pack)
        if lstm_matches_dict_key not in lstm_matches_dict:
            lstm_matches_dict.update({lstm_matches_dict_key: [(unpack_out_port, match_dict)]})
        else:
            matches_list = lstm_matches_dict[lstm_matches_dict_key]
            matches_list.append((unpack_out_port, match_dict))
            lstm_matches_dict.update({lstm_matches_dict_key: matches_list})

    matched = False
    for (unpack, pack), cell_matches in lstm_matches_dict.items():
        if not cell_matches:
            continue

        # Check whether unpack node is valid
        unpack_obj = NodeWrap(graph, unpack)['object']
        unpack_in_edges = graph.sorted_in_edges(unpack, data=True)
        if unpack_obj.type != 'LiteUNPACK' \
                or unpack_obj.axis != 0 \
                or unpack_obj.num != len(cell_matches) \
                or len(unpack_in_edges) < 1:
            continue
        unpack_in_shapes = unpack_obj.get_input_shapes()
        if len(unpack_in_shapes) < 1 \
                or unpack_in_shapes[0] is None \
                or len(unpack_in_shapes[0]) != 3:
            continue
        seq_length, batch_size, input_size = unpack_in_shapes[0]

        # Sort cell matches by out ports of pack, and check whether it's reversed lstm
        cell_matches.sort()
        is_reverse_lstm = False
        final_yout = pack
        if cell_matches[-1][1]['last_hout'] is None:
            cell_matches.reverse()
            is_reverse_lstm = True
            pack_out_edges = graph.sorted_out_edges(pack, data=True)
            if len(pack_out_edges) != 1:
                # TODO: Support the case that len(pack_out_edges) != 1
                continue
            reverse_node = pack_out_edges[0][1]
            reverse_node_obj = NodeWrap(graph, reverse_node)['object']
            if reverse_node_obj is None or reverse_node_obj.type != 'LiteREVERSE_V2':
                continue
            final_yout = reverse_node

        _, first_cell_match = cell_matches[0]
        first_cell_fc = first_cell_match['fc']
        first_fc_obj = NodeWrap(graph, first_cell_fc)['object']
        first_cell_fc_in_attr = graph.sorted_in_edges(first_cell_fc, data=True)[0][2]
        # Check whether weights are valid. The weights of first cell should be same as others.
        # No need to check biases here because it is different with other cells and won't be used.
        weights = first_fc_obj.weights
        if len(weights.shape) != 2 or weights.shape[1] != input_size:
            continue
        hidden_size = int(first_fc_obj.biases.shape[0] / 4)
        weights_scale_zp = first_fc_obj.weights_scale_zp
        if len(weights_scale_zp) == 2:
            dequant_weights = (
                weights - first_fc_obj.weights_scale_zp[1]) * first_fc_obj.weights_scale_zp[0]
        else:
            dequant_weights = weights

        # Check whether all the cell_matches belong to one lstm opeartion
        final_hout_list = [first_cell_match['mul_hout']]
        dequant_recurrent_weights, recurrent_weights, recurrent_weights_scale_zp = None, None, None
        dequant_biases, biases, biases_scale_zp = None, None, None
        activations_scale_list, activations_zp_list = [], []
        is_same_lstm = True
        for idx, (parent_out_port, cell_match) in enumerate(cell_matches):
            exp_out_port = (len(cell_matches) - 1 - idx) if is_reverse_lstm else idx
            if parent_out_port != exp_out_port:
                is_same_lstm = False
                break
            if idx == 0:
                continue
            fc1, fc2 = cell_match['fc_after_unpack'], cell_match['fc_after_hstate']
            fc1_obj, fc2_obj = NodeWrap(graph, fc1)['object'], NodeWrap(graph, fc2)['object']
            if fc1_obj is None or fc2_obj is None:
                is_same_lstm = False
                break
            if len(fc1_obj.weights_scale_zp) == 2:
                fc_weights = ((fc1_obj.weights - fc1_obj.weights_scale_zp[1]) * fc1_obj.weights_scale_zp[0])
            else:
                fc_weights = fc1_obj.weights
            if fc_obj is None \
                    or not np.allclose(dequant_weights, fc_weights):
                is_same_lstm = False
                break
            if len(fc2_obj.weights_scale_zp) == 2:
                current_recurrent_weights = (
                    fc2_obj.weights - fc2_obj.weights_scale_zp[1]) * fc2_obj.weights_scale_zp[0]
            else:
                current_recurrent_weights = fc2_obj.weights
            if dequant_recurrent_weights is None:
                recurrent_weights = fc2_obj.weights
                dequant_recurrent_weights = current_recurrent_weights
                recurrent_weights_scale_zp = fc2_obj.weights_scale_zp
            elif not np.allclose(dequant_recurrent_weights, current_recurrent_weights):
                is_same_lstm = False
                break
            bias = cell_match['bias']
            bias_obj = NodeWrap(graph, bias)['object']
            bias_out_edges = graph.sorted_out_edges(bias, data=True)
            if bias_obj is None or bias_obj.value is None or len(bias_out_edges) < 1 \
                    or bias_out_edges[0][2]['tensor'] is None:
                is_same_lstm = False
                break
            current_bias_value = bias_obj.value
            current_bias_scale, current_bias_zp = bias_out_edges[0][2]['tensor'].scale_zp
            if current_bias_scale and current_bias_zp:
                current_bias_value = (current_bias_value - current_bias_zp) * current_bias_scale
            if dequant_biases is None:
                biases = bias_obj.value
                dequant_biases = current_bias_value
                biases_scale_zp = bias_out_edges[0][2]['tensor'].scale_zp
            elif not np.allclose(dequant_biases, current_bias_value):
                is_same_lstm = False
                break
            if cell_match['last_hout'] != cell_matches[idx - 1][1]['mul_hout'] \
                    or cell_match['last_cout'] != cell_matches[idx - 1][1]['add_cout']:
                is_same_lstm = False
                break
            final_hout_list.append(cell_match['mul_hout'])
            # get scale/zp from the output tensor of sum0, icfo, mul_i_and_c, mul_f_and_c_prev, c_lut,
            # cout and hout
            act_nodes = [cell_match[name] for name in
                         ['add_bias', 'sigmoid_sp0', 'tanh_sp2', 'sigmoid_sp1', 'sigmoid_sp3',
                          'mul_sp01', 'mul_sp2', 'tanh_c', 'add_cout', 'mul_hout']]
            activations_scale, activations_zp = get_out_tensor_scale_zp(graph, act_nodes)
            if None in activations_scale or None in activations_zp:
                break
            activations_scale_list.append(activations_scale)
            activations_zp_list.append(activations_zp)
        if not is_same_lstm:
            continue

        # Check whether pack node to concat all the intermediate outputs of the hidden is valid
        pack_in_edges = graph.sorted_in_edges(pack)
        if len(pack_in_edges) != unpack_obj.num \
                or any(pack_src != final_hout_list[idx] for idx, (pack_src, _) in enumerate(pack_in_edges)):
            continue

        # The outputs of the first cell will be the inputs of merged lstm
        initial_h = first_cell_match['mul_hout']
        initial_c = first_cell_match['add_cout']
        initial_h_out_edges = graph.sorted_out_edges(initial_h, data=True)
        initial_c_out_edges = graph.sorted_out_edges(initial_c, data=True)
        if len(initial_h_out_edges) < 1 or len(initial_c_out_edges) < 1:
            continue

        # Get the final outputs of the hidden and the cell
        final_hout = cell_matches[-1][1]['mul_hout']
        final_hout_out_edges = graph.sorted_out_edges(final_hout, data=True)
        if len(final_hout_out_edges) < 1:
            continue
        final_cout = cell_matches[-1][1]['add_cout']
        final_cout_out_edges = graph.sorted_out_edges(final_cout, data=True)

        matched = True
        # Reconnect inputs and outputs
        # Add split node to split input into 2 parts
        split = get_valid_node_name(graph, unpack + '_split')
        inp, _, in_attr = unpack_in_edges[0]
        graph.add_edge(inp, split, **in_attr)
        lstm_timestep_num = unpack_obj.num - 1
        NodeWrap(graph, split).replace_obj('Split',
                                           {'name': split,
                                            'opset_version': 11,
                                            'quantize': 1,
                                            'axis': 0,
                                            'split': [lstm_timestep_num, 1] if is_reverse_lstm else [1, lstm_timestep_num]})
        # Connect split with fc from the first cell and merged lstm
        lstm = get_valid_node_name(graph, initial_h + '_lstm')
        graph.remove_edge(unpack, first_cell_fc)
        split_to_fc_out_attr = copy.deepcopy(first_cell_fc_in_attr)
        split_to_fc_out_attr.update({'src_out_port': 1 if is_reverse_lstm else 0})
        if first_cell_fc_in_attr['tensor'] is not None and first_cell_fc_in_attr['tensor'].value is not None:
            split_to_fc_out_attr['tensor'].value = np.expand_dims(first_cell_fc_in_attr['tensor'].value, 0)
        graph.add_edge(split, first_cell_fc, **split_to_fc_out_attr)
        split_to_lstm_out_attr = {'src_out_port': 0 if is_reverse_lstm else 1}
        if first_cell_fc_in_attr['tensor'] is not None:
            split_to_lstm_out_attr.update({'tensor': Tensor(dtype=first_cell_fc_in_attr['tensor'].dtype,
                                                            scale_zp=first_cell_fc_in_attr['tensor'].scale_zp)})
        graph.add_edge(split, lstm, **split_to_lstm_out_attr)
        insert_reshape(graph, split, first_cell_fc, split_to_fc_out_attr, [batch_size, input_size])
        # Add concat node to concat the output of the first cell and the merged lstm
        concat = get_valid_node_name(graph, lstm + '_concat')
        # Connect lstm and initial_h(which is also the output of the first cell) to concat
        lstm_y_out_attr = {'src_out_port': 0, 'dst_in_port': 0 if is_reverse_lstm else 1}
        hout_out_attr = final_hout_out_edges[0][2]
        if hout_out_attr['tensor'] is not None:
            lstm_y_out_attr.update({'tensor': Tensor(dtype=hout_out_attr['tensor'].dtype,
                                                     scale_zp=hout_out_attr['tensor'].scale_zp)})
        graph.add_edge(lstm, concat, **lstm_y_out_attr)
        first_cell_h_out_attr = copy.deepcopy(initial_h_out_edges[0][2])
        first_cell_h_out_attr.update({'dst_in_port': 1 if is_reverse_lstm else 0})
        graph.add_edge(initial_h, concat, **first_cell_h_out_attr)
        insert_reshape(graph, initial_h, concat, first_cell_h_out_attr, [1, batch_size, hidden_size])
        NodeWrap(graph, concat).replace_obj('Concat', {'name': concat, 'opset_version': 13, 'quantize': 1, 'axis': 0})
        # Connect the dst nodes of pack/reverse with concat/lstm
        final_yout_out_edges = graph.sorted_out_edges(final_yout, data=True)
        for _, dst, out_attr in final_yout_out_edges:
            graph.remove_edge(final_yout, dst)
            new_attr = copy.deepcopy(out_attr)
            new_attr.update({'src_out_port': 0})
            graph.add_edge(concat, dst, **new_attr)
        for _, dst, out_attr in final_hout_out_edges:
            if dst == pack:
                continue
            graph.remove_edge(final_hout, dst)
            new_attr = copy.deepcopy(out_attr)
            new_attr.update({'src_out_port': 1})
            graph.add_edge(lstm, dst, **new_attr)
        for _, dst, out_attr in final_cout_out_edges:
            graph.remove_edge(final_cout, dst)
            new_attr = copy.deepcopy(out_attr)
            new_attr.update({'src_out_port': 2})
            graph.add_edge(lstm, dst, **new_attr)

        # Reset output names
        for idx, node in enumerate([final_yout, final_hout, final_cout]):
            if node == final_yout:
                old_dim = [seq_length - 1, 1, batch_size, hidden_size]
                new_dim = [seq_length - 1, batch_size, hidden_size]
            else:
                old_dim = [1, batch_size, hidden_size]
                new_dim = [batch_size, hidden_size]
            post_reshape = insert_reshape_after(
                graph, lstm, new_dim, old_dim, out_port=idx)
            if node in graph._attr['output_names']:
                index = graph._attr['output_names'].index(node)
                out_node = post_reshape if node != final_yout else concat
                graph._attr['output_names'][index] = out_node

        # Prepare inputs for onnx lstm
        input_w, forget_w, cell_w, output_w = np.split(
            weights, 4, axis=0)
        input_r, forget_r, cell_r, output_r = np.split(
            recurrent_weights, 4, axis=0)
        input_wb, forget_wb, cell_wb, output_wb = np.split(biases, 4, axis=0)
        W_value = np.stack([np.concatenate(
            [input_w, output_w, forget_w, cell_w], axis=0)])
        R_value = np.stack([np.concatenate(
            [input_r, output_r, forget_r, cell_r], axis=0)])
        biases_w = np.concatenate([input_wb, output_wb, forget_wb, cell_wb])
        biases_r = np.zeros_like(biases_w)
        B_value = np.stack([np.concatenate([biases_w, biases_r])])
        seq_length = np.array([seq_length - 1] * batch_size, np.int64)
        initial_hc_shape = [1, batch_size, hidden_size]

        # Convert to onnx lstm
        insert_constant(graph, lstm + '_W', W_value, lstm,
                        in_port=1, data_format='NHWC')
        insert_constant(graph, lstm + '_R', R_value, lstm,
                        in_port=2, data_format='NHWC')
        insert_constant(graph, lstm + '_B', B_value, lstm,
                        in_port=3, data_format='NHWC')
        insert_constant(graph, lstm + '_seq_length', seq_length, lstm,
                        in_port=4, data_format='NHWC')
        initial_h_out_attr = copy.deepcopy(initial_h_out_edges[0][2])
        initial_h_out_attr.update({'dst_in_port': 5})
        initial_c_out_attr = copy.deepcopy(initial_c_out_edges[0][2])
        initial_c_out_attr.update({'dst_in_port': 6})
        graph.add_edge(initial_c, lstm, **initial_h_out_attr)
        graph.add_edge(initial_h, lstm, **initial_h_out_attr)
        insert_reshape(graph, initial_c, lstm,
                       initial_c_out_attr, initial_hc_shape)
        insert_reshape(graph, initial_h, lstm,
                       initial_h_out_attr, initial_hc_shape)
        lstm_attr = {'name': lstm,
                     'opset_version': 14,
                     'quantize': 1,
                     'layout': False,
                     'hidden_size': hidden_size,
                     'direction': 'reverse' if is_reverse_lstm else 'forward',
                     'weights_scale_zp': [np.array([weights_scale_zp[0], recurrent_weights_scale_zp[0]]),
                                          np.array([weights_scale_zp[1], recurrent_weights_scale_zp[1]])],
                     'biases_scale_zp': [np.array(biases_scale_zp[0]), np.array(biases_scale_zp[1])],
                     'activations_scale': np.array(activations_scale_list, np.float32),
                     'activations_zp': np.array(activations_zp_list, np.int32)}
        NodeWrap(graph, lstm).replace_obj('LSTM', lstm_attr)

    if matched:
        clear_redundant_nodes(graph)


def get_out_tensor_scale_zp(graph, node_names, out_port=0):
    nodes_cnt = 0 if not node_names else len(node_names)
    assert nodes_cnt > 1, 'Expect at least one node name in get_out_tensor_scale_zp!'
    act_scale_list, act_zp_list = [None] * nodes_cnt, [None] * nodes_cnt
    for idx, name in enumerate(node_names):
        node_out_edges = graph.sorted_out_edges(name, data=True)
        if len(node_out_edges) <= out_port:
            continue
        out_tensor = node_out_edges[out_port][2]['tensor']
        if out_tensor is None or len(out_tensor.scale_zp) != 2:
            continue
        scale = np.array(out_tensor.scale_zp[0])
        zp = np.array(out_tensor.scale_zp[1])
        if scale.size != 1 or zp.size != 1:
            continue
        act_scale_list[idx] = scale.item()
        act_zp_list[idx] = zp.item()
    return act_scale_list, act_zp_list


def merge_quantized_lstm_cell(graph):
    '''
    Merge quantized lstm cell(for crnn).
    '''
    matched = False
    cell_matches = matched_patterns(graph,
                                    nodes=[('fc1', {'op': 'LiteFULLY_CONNECTED'}),
                                           ('fc2', {'op': 'LiteFULLY_CONNECTED'}),
                                           ('add_fc', {'op': 'LiteADD'}),
                                           ('bias', {'op': 'Constant'}),
                                           ('add_bias', {'op': 'LiteADD'}),
                                           ('split', {'op': 'LiteSPLIT'}),
                                           ('sigmoid_sp0', {'op': 'LiteLOGISTIC'}),
                                           ('sigmoid_sp1', {'op': 'LiteLOGISTIC'}),
                                           ('tanh_sp2', {'op': 'LiteTANH'}),
                                           ('sigmoid_sp3', {'op': 'LiteLOGISTIC'}),
                                           ('mul_sp01', {'op': 'LiteMUL'}),
                                           ('mul_sp2', {'op': 'LiteMUL'}),
                                           ('add_cout', {'op': 'LiteADD'}),
                                           ('tanh_c', {'op': 'LiteTANH'}),
                                           ('mul_hout', {'op': 'LiteMUL'}),
                                           ],
                                    edges=[('fc1', 'add_fc'),
                                           ('fc2', 'add_fc'),
                                           ('add_fc', 'add_bias'),
                                           ('bias', 'add_bias'),
                                           ('add_bias', 'split'),
                                           ('split', 'sigmoid_sp0', {'src_out_port': 0}),
                                           ('split', 'sigmoid_sp1', {'src_out_port': 1}),
                                           ('split', 'tanh_sp2', {'src_out_port': 2}),
                                           ('split', 'sigmoid_sp3', {'src_out_port': 3}),
                                           ('sigmoid_sp0', 'mul_sp01'),
                                           ('tanh_sp2', 'mul_sp01'),
                                           ('sigmoid_sp1', 'mul_sp2'),
                                           ('mul_sp01', 'add_cout'),
                                           ('mul_sp2', 'add_cout'),
                                           ('add_cout', 'tanh_c'),
                                           ('tanh_c', 'mul_hout'),
                                           ('sigmoid_sp3', 'mul_hout'),
                                           ])
    for m in cell_matches:
        names = ['fc1', 'fc2', 'bias', 'mul_hout', 'add_cout']
        objs_dict = {name: NodeWrap(graph, m[name])['object'] for name in names}
        if any(obj is None for obj in objs_dict.values()):
            ERROR('[Parser]: Meet invalid node in merge_quantized_lstm_cell!')
            continue
        fc1_in_edges = graph.sorted_in_edges(m['fc1'], data=True)
        fc2_in_edges = graph.sorted_in_edges(m['fc2'], data=True)
        if len(fc1_in_edges) < 1 or len(fc2_in_edges) < 1:
            continue
        initial_c = None
        mul_in_edges = graph.sorted_in_edges(m['mul_sp2'], data=True)
        for src, _, mul_in_attr in mul_in_edges:
            if src != m['sigmoid_sp1']:
                initial_c = src
                initial_c_in_attr = copy.deepcopy(mul_in_attr)
        if initial_c is None:
            continue

        fc1_is_input = False
        fc_after_unpack, fc_after_hstate = None, None
        for idx, (fc_input_node, fc_node, in_attr) in enumerate(fc1_in_edges[:1] + fc2_in_edges[:1]):
            node_obj = NodeWrap(graph, fc_input_node)['object']
            if node_obj is None:
                ERROR('[Parser]: Meet invalid node (%s) in merge_quantized_lstm_cell!' % fc_input_node)
                continue
            if node_obj.type == 'LiteUNPACK':
                fc1_is_input = (idx == 0)
                fc_after_unpack = fc_node
            else:
                fc_after_hstate = fc_node
        if fc_after_unpack is None or fc_after_hstate is None:
            continue
        fc_after_unpack_in_edge = fc1_in_edges[0] if fc1_is_input else fc2_in_edges[0]
        fc_after_unpack_obj = objs_dict['fc1'] if fc1_is_input else objs_dict['fc2']
        fc_after_unpack_in_shapes = fc_after_unpack_obj.get_input_shapes()
        if len(fc_after_unpack_in_shapes) < 1 \
                or fc_after_unpack_in_shapes[0] is None \
                or None in fc_after_unpack_in_shapes[0] \
                or len(fc_after_unpack_in_shapes[0]) != 2:
            continue
        batch_size, input_size = fc_after_unpack_in_shapes[0]
        weights = fc_after_unpack_obj.weights
        weights_scale_zp = fc_after_unpack_obj.weights_scale_zp

        fc_after_hstate_in_edge = fc2_in_edges[0] if fc1_is_input else fc1_in_edges[0]
        fc_after_hstate_obj = objs_dict['fc2'] if fc1_is_input else objs_dict['fc1']
        fc_after_hstate_in_shapes = fc_after_hstate_obj.get_input_shapes()
        if len(fc_after_hstate_in_shapes) < 1 \
                or fc_after_hstate_in_shapes[0] is None \
                or None in fc_after_hstate_in_shapes[0] \
                or len(fc_after_hstate_in_shapes[0]) != 2:
            continue
        batch_size_from_hstate, hidden_size = fc_after_hstate_in_shapes[0]
        if batch_size_from_hstate != batch_size:
            continue
        recurrent_weights = fc_after_hstate_obj.weights
        recurrent_weights_scale_zp = fc_after_hstate_obj.weights_scale_zp

        biases = objs_dict['bias'].value
        biases_out_edges = graph.sorted_out_edges(m['bias'], data=True)
        if biases is None or len(biases_out_edges) < 1 \
                or biases_out_edges[0][2]['tensor'] is None:
            continue
        biases_scale_zp = list(biases_out_edges[0][2]['tensor'].scale_zp)

        # get scale/zp from the output tensor of sum0, icfo, mul_i_and_c, mul_f_and_c_prev, c_lut,
        # cout and hout
        act_nodes = [m[name] for name in ['add_bias', 'sigmoid_sp0', 'tanh_sp2',
                                          'sigmoid_sp1', 'sigmoid_sp3', 'mul_sp01', 'mul_sp2', 'tanh_c',
                                          'add_cout', 'mul_hout']]
        activations_scale, activations_zp = get_out_tensor_scale_zp(graph, act_nodes)
        if None in activations_scale or None in activations_zp:
            continue

        matched = True
        # Prepare inputs for onnx lstm
        input_w, forget_w, cell_w, output_w = np.split(
            weights, 4, axis=0)
        input_r, forget_r, cell_r, output_r = np.split(
            recurrent_weights, 4, axis=0)
        input_wb, forget_wb, cell_wb, output_wb = np.split(biases, 4, axis=0)
        W_value = np.stack([np.concatenate(
            [input_w, output_w, forget_w, cell_w], axis=0)])
        R_value = np.stack([np.concatenate(
            [input_r, output_r, forget_r, cell_r], axis=0)])
        biases_w = np.concatenate([input_wb, output_wb, forget_wb, cell_wb])
        biases_r = np.zeros_like(biases_w)
        B_value = np.stack([np.concatenate([biases_w, biases_r])])
        seq_length = np.array([1] * batch_size, np.int64)
        initial_hc_shape = [1, batch_size, hidden_size]

        # Create lstm node and connect input with lstm
        lstm = get_valid_node_name(graph, fc_after_unpack + '_lstm')
        src, _, in_attr = fc_after_unpack_in_edge
        graph.remove_edge(src, fc_after_unpack)
        graph.add_edge(src, lstm, **in_attr)
        src_new_dim = [1, batch_size, input_size]
        insert_reshape(graph, src, lstm, in_attr, src_new_dim)

        # Connect W, R, B and sequence_lens with lstm
        insert_constant(graph, lstm + '_W', W_value, lstm,
                        in_port=1, data_format='NHWC')
        insert_constant(graph, lstm + '_R', R_value, lstm,
                        in_port=2, data_format='NHWC')
        insert_constant(graph, lstm + '_B', B_value, lstm,
                        in_port=3, data_format='NHWC')
        insert_constant(graph, lstm + '_seq_length', seq_length, lstm,
                        in_port=4, data_format='NHWC')

        # Connect initial_h with lstm
        initial_h, _, initial_h_in_attr = fc_after_hstate_in_edge
        initial_h_in_attr.update({'dst_in_port': 5})
        graph.add_edge(initial_h, lstm, **initial_h_in_attr)
        insert_reshape(graph, initial_h, lstm,
                       initial_h_in_attr, initial_hc_shape)

        # Connect initial_c with lstm
        initial_c_in_attr.update({'dst_in_port': 6})
        graph.add_edge(initial_c, lstm, **initial_c_in_attr)
        insert_reshape(graph, initial_c, lstm,
                       initial_c_in_attr, initial_hc_shape)

        # Connect the dst nodes of the hidden and the cell with lstm
        hout = m['mul_hout']
        hout_out_edges = graph.sorted_out_edges(hout, data=True)
        for _, dst, out_attr in hout_out_edges:
            graph.remove_edge(hout, dst)
            new_attr = copy.deepcopy(out_attr)
            new_attr.update({'src_out_port': 1})
            graph.add_edge(lstm, dst, **new_attr)
        cout = m['add_cout']
        cout_out_edges = graph.sorted_out_edges(cout, data=True)
        for _, dst, out_attr in cout_out_edges:
            graph.remove_edge(cout, dst)
            new_attr = copy.deepcopy(out_attr)
            new_attr.update({'src_out_port': 2})
            graph.add_edge(lstm, dst, **new_attr)

        # Reset output names
        for idx, node in enumerate([hout, cout]):
            old_dim = [1, batch_size, hidden_size]
            new_dim = [batch_size, hidden_size]
            post_reshape = insert_reshape_after(
                graph, lstm, new_dim, old_dim, out_port=(idx + 1))
            if node in graph._attr['output_names']:
                index = graph._attr['output_names'].index(node)
                graph._attr['output_names'][index] = post_reshape

        # Convert to onnx lstm
        lstm_attr = {'name': lstm,
                     'opset_version': 14,
                     'quantize': 1,
                     'layout': False,
                     'hidden_size': hidden_size,
                     'direction': 'forward',
                     'weights_scale_zp': [np.array([weights_scale_zp[0].item(), recurrent_weights_scale_zp[0].item()], np.float32),
                                          np.array([weights_scale_zp[1].item(), recurrent_weights_scale_zp[1].item()], np.int32)],
                     'biases_scale_zp': biases_scale_zp,
                     'activations_scale': np.array(activations_scale, np.float32),
                     'activations_zp': np.array(activations_zp, np.int32)}
        NodeWrap(graph, lstm).replace_obj('LSTM', lstm_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_quantized_lstm_cell2(graph):
    '''
    Merge quantized lstm cell.
    '''
    matched = False
    cell_matches = matched_patterns(graph,
                                    nodes=[('concat', {'op': 'LiteCONCATENATION'}),
                                           ('fc', {'op': 'LiteFULLY_CONNECTED'}),
                                           ('split', {'op': 'LiteSPLIT'}),
                                           ('sigmoid_sp0', {'op': 'LiteLOGISTIC'}),
                                           ('tanh_sp1', {'op': 'LiteTANH'}),
                                           ('add_sp2', {'op': 'LiteADD'}),
                                           ('sigmoid_sp3', {'op': 'LiteLOGISTIC'}),
                                           ('adder', {'op': 'Constant'}),
                                           ('mul_sp01', {'op': 'LiteMUL'}),
                                           ('sigmoid_sp2', {'op': 'LiteLOGISTIC'}),
                                           ('mul_sp2', {'op': 'LiteMUL'}),
                                           ('add_cout', {'op': 'LiteADD'}),
                                           ('tanh_c', {'op': 'LiteTANH'}),
                                           ('mul_hout', {'op': 'LiteMUL'}),
                                           ],
                                    edges=[('concat', 'fc'),
                                           ('fc', 'split'),
                                           ('split', 'sigmoid_sp0',
                                            {'src_out_port': 0}),
                                           ('split', 'tanh_sp1',
                                            {'src_out_port': 1}),
                                           ('split', 'add_sp2', {
                                               'src_out_port': 2}),
                                           ('split', 'sigmoid_sp3',
                                            {'src_out_port': 3}),
                                           ('sigmoid_sp0', 'mul_sp01'),
                                           ('tanh_sp1', 'mul_sp01'),
                                           ('adder', 'add_sp2'),
                                           ('add_sp2', 'sigmoid_sp2'),
                                           ('sigmoid_sp2', 'mul_sp2'),
                                           ('mul_sp01', 'add_cout'),
                                           ('mul_sp2', 'add_cout'),
                                           ('add_cout', 'tanh_c'),
                                           ('tanh_c', 'mul_hout'),
                                           ('sigmoid_sp3', 'mul_hout'),
                                           ]
                                    )
    for m in cell_matches:
        names = ['concat', 'fc', 'adder', 'mul_hout', 'add_cout']
        objs_dict = {name: NodeWrap(graph, m[name])['object'] for name in names}
        if any(obj is None for obj in objs_dict.values()):
            ERROR('[Parser]: Meet invalid node in merge_quantized_lstm_cell!')
            continue

        concat = m['concat']
        concat_in_shapes = objs_dict['concat'].get_input_shapes()
        concat_in_edges = graph.sorted_in_edges(concat, data=True)
        if len(concat_in_edges) < 2 \
                or len(concat_in_shapes) < 2 \
                or concat_in_shapes[0] is None \
                or None in concat_in_shapes[0] \
                or concat_in_shapes[1] is None \
                or None in concat_in_shapes[1]:
            continue
        batch_size, input_size = concat_in_shapes[0]
        batch_size_from_hstate, hidden_size = concat_in_shapes[1]
        if batch_size_from_hstate != batch_size:
            continue

        initial_c = None
        mul_in_edges = graph.sorted_in_edges(m['mul_sp2'], data=True)
        for src, _, mul_in_attr in mul_in_edges:
            if src != m['sigmoid_sp2']:
                initial_c = src
                initial_c_in_attr = copy.deepcopy(mul_in_attr)
        if initial_c is None:
            continue

        # Get forget_bias
        adder_out_edges = graph.sorted_out_edges(m['adder'], data=True)
        if len(adder_out_edges) < 1 \
                or adder_out_edges[0][2]['tensor'] is None:
            continue
        forget_bias_scale_zp = adder_out_edges[0][2]['tensor'].scale_zp
        forget_bias = objs_dict['adder'].value

        # Get weights and biases
        weights = objs_dict['fc'].weights
        weights_scale_zp = objs_dict['fc'].weights_scale_zp
        biases = objs_dict['fc'].biases
        biases_scale_zp = objs_dict['fc'].biases_scale_zp

        # get scale/zp from the output tensor of sum0, icfo, mul_i_and_c, mul_f_and_c_prev, c_lut,
        # cout, hout and f_in if forget_bias exists
        act_nodes = [m[name] for name in ['fc', 'sigmoid_sp0', 'tanh_sp1',
                                          'sigmoid_sp2', 'sigmoid_sp3', 'mul_sp01', 'mul_sp2', 'tanh_c',
                                          'add_cout', 'mul_hout', 'add_sp2']]
        activations_scale, activations_zp = get_out_tensor_scale_zp(graph, act_nodes)
        if None in activations_scale or None in activations_zp:
            continue

        matched = True
        # Prepare inputs for onnx lstm
        kernel_weights, recurrent_weights = np.split(
            weights, [input_size], axis=1)
        input_w, cell_w, forget_w, output_w = np.split(
            kernel_weights, 4, axis=0)
        input_r, cell_r, forget_r, output_r = np.split(
            recurrent_weights, 4, axis=0)
        input_wb, cell_wb, forget_wb, output_wb = np.split(biases, 4, axis=0)
        W_value = np.stack([np.concatenate(
            [input_w, output_w, forget_w, cell_w], axis=0)])
        R_value = np.stack([np.concatenate(
            [input_r, output_r, forget_r, cell_r], axis=0)])
        biases_w = np.concatenate([input_wb, output_wb, forget_wb, cell_wb])
        biases_r = np.zeros_like(biases_w)
        B_value = np.stack([np.concatenate([biases_w, biases_r])])
        seq_length = np.array([1] * batch_size, np.int64)
        initial_hc_shape = [1, batch_size, hidden_size]

        # Create lstm node and connect input with lstm
        lstm = get_valid_node_name(graph, concat + '_lstm')
        src, _, in_attr = concat_in_edges[0]
        graph.remove_edge(src, concat)
        graph.add_edge(src, lstm, **in_attr)
        src_new_dim = [1, batch_size, input_size]
        insert_reshape(graph, src, lstm, in_attr, src_new_dim)

        # Connect W, R, B and sequence_lens with lstm
        insert_constant(graph, lstm + '_W', W_value, lstm,
                        in_port=1, data_format='NHWC')
        insert_constant(graph, lstm + '_R', R_value, lstm,
                        in_port=2, data_format='NHWC')
        insert_constant(graph, lstm + '_B', B_value, lstm,
                        in_port=3, data_format='NHWC')
        insert_constant(graph, lstm + '_seq_length', seq_length, lstm,
                        in_port=4, data_format='NHWC')

        # Connect initial_h with lstm
        initial_h, _, initial_h_in_attr = concat_in_edges[1]
        initial_h_in_attr.update({'dst_in_port': 5})
        graph.add_edge(initial_h, lstm, **initial_h_in_attr)
        insert_reshape(graph, initial_h, lstm,
                       initial_h_in_attr, initial_hc_shape)

        # Connect initial_c with lstm
        initial_c_in_attr.update({'dst_in_port': 6})
        graph.add_edge(initial_c, lstm, **initial_c_in_attr)
        insert_reshape(graph, initial_c, lstm,
                       initial_c_in_attr, initial_hc_shape)

        # Connect the dst nodes of the hidden and the cell with lstm
        hout = m['mul_hout']
        hout_out_edges = graph.sorted_out_edges(hout, data=True)
        for _, dst, out_attr in hout_out_edges:
            graph.remove_edge(hout, dst)
            new_attr = copy.deepcopy(out_attr)
            new_attr.update({'src_out_port': 1})
            graph.add_edge(lstm, dst, **new_attr)
        cout = m['add_cout']
        cout_out_edges = graph.sorted_out_edges(cout, data=True)
        for _, dst, out_attr in cout_out_edges:
            graph.remove_edge(cout, dst)
            new_attr = copy.deepcopy(out_attr)
            new_attr.update({'src_out_port': 2})
            graph.add_edge(lstm, dst, **new_attr)

        # Reset output names
        for idx, node in enumerate([hout, cout]):
            old_dim = [1, batch_size, hidden_size]
            new_dim = [batch_size, hidden_size]
            post_reshape = insert_reshape_after(
                graph, lstm, new_dim, old_dim, out_port=(idx + 1))
            if node in graph._attr['output_names']:
                index = graph._attr['output_names'].index(node)
                graph._attr['output_names'][index] = post_reshape

        # Convert to onnx lstm
        lstm_attr = {'name': lstm,
                     'opset_version': 14,
                     'quantize': 1,
                     'layout': False,
                     'hidden_size': hidden_size,
                     'forget_bias': forget_bias,
                     'forget_bias_scale_zp': forget_bias_scale_zp,
                     'direction': 'forward',
                     'weights_scale_zp': weights_scale_zp,
                     'biases_scale_zp': biases_scale_zp,
                     'activations_scale': np.array(activations_scale, np.float32),
                     'activations_zp': np.array(activations_zp, np.int32)}
        NodeWrap(graph, lstm).replace_obj('LSTM', lstm_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_quantized_instance_norm(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('mean_1', {'op': 'LiteMEAN'}),
                                   ('squared_diff', {
                                    'op': 'LiteSQUARED_DIFFERENCE'}),
                                   ('mean_2', {'op': 'LiteMEAN'}),
                                   ('eps', {'op': 'Constant'}),
                                   ('add_1', {'op': 'LiteADD'}),
                                   ('rsqrt', {'op': 'LiteRSQRT'}),
                                   ('gamma', {'op': 'Constant'}),
                                   ('mul_1', {'op': 'LiteMUL'}),
                                   ('mul_2', {'op': 'LiteMUL'}),
                                   ('mul_3', {'op': 'LiteMUL'}),
                                   ('beta', {'op': 'Constant'}),
                                   ('sub', {'op': 'LiteSUB'}),
                                   ('add_2', {'op': 'LiteADD'})
                               ],
                               edges=[
                                   ('mean_1', 'squared_diff'),
                                   ('squared_diff', 'mean_2'),
                                   ('mean_2', 'add_1'),
                                   ('eps', 'add_1', {'dst_in_port': 1}),
                                   ('add_1', 'rsqrt'),
                                   ('rsqrt', 'mul_1'),
                                   ('gamma', 'mul_1', {'dst_in_port': 1}),
                                   ('mul_1', 'mul_2'),
                                   ('mul_1', 'mul_3'),
                                   ('mean_1', 'mul_3'),
                                   ('beta', 'sub'),
                                   ('mul_3', 'sub', {'dst_in_port': 1}),
                                   ('mul_2', 'add_2'),
                                   ('sub', 'add_2')
                               ])
    for m in matches:
        names = ['mean_1', 'squared_diff', 'mean_2', 'eps', 'add_1', 'rsqrt',
                 'gamma', 'mul_1', 'mul_2', 'mul_3', 'beta', 'sub', 'add_2']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None or not obj.quantize for obj in obj_dict.values()]):
            ERROR('[Parser]: Meets invalid Op in merge_quantized_instance_norm!')
            continue
        mean1_in_edges = graph.sorted_in_edges(m['mean_1'], data=True)
        squared_diff_in_edges = graph.sorted_in_edges(
            m['squared_diff'], data=True)
        add1_in_edges = graph.sorted_in_edges(m['add_1'], data=True)
        mul1_in_edges = graph.sorted_in_edges(m['mul_1'], data=True)
        mul2_in_edges = graph.sorted_in_edges(m['mul_2'], data=True)
        sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
        if len(mean1_in_edges) < 1 \
                or len(squared_diff_in_edges) != 2 \
                or len(add1_in_edges) != 2 \
                or len(mul1_in_edges) != 2 \
                or len(mul2_in_edges) != 2 \
                or len(sub_in_edges) != 2:
            continue
        input_shapes = obj_dict['mean_1'].get_input_shapes()
        if len(input_shapes) < 1 \
                or len(input_shapes[0]) < 3 \
                or any(s is None for s in input_shapes[0]):
            continue
        need_continue = False
        src, _, in_attr1 = mean1_in_edges[0]
        src_out_port = in_attr1['src_out_port']
        for s, _, in_attr in squared_diff_in_edges:
            if s != m['mean_1']:
                if s != src or in_attr['src_out_port'] != src_out_port:
                    need_continue = True
                    break
        if need_continue:
            continue
        for s, _, in_attr in mul2_in_edges:
            if s != m['mul_1']:
                if s != src or in_attr['src_out_port'] != src_out_port:
                    need_continue = True
                    break
        if need_continue:
            continue
        mean1_axes = OpHasAxis.make_axes_non_negative(
            obj_dict['mean_1'].axes, len(input_shapes[0]))
        mean2_axes = OpHasAxis.make_axes_non_negative(
            obj_dict['mean_2'].axes, len(input_shapes[0]))
        if mean1_axes != mean2_axes \
                or mean1_axes != list(range(1, len(input_shapes[0]) - 1)):
            continue
        eps_quantized = obj_dict['eps'].value
        if eps_quantized is None or eps_quantized.size != 1:
            continue
        eps_scale_zp = add1_in_edges[1][2]['tensor'].scale_zp
        if eps_scale_zp is None or len(eps_scale_zp) != 2:
            continue
        eps = float((eps_quantized - eps_scale_zp[1]) * eps_scale_zp[0])
        channel_dim = input_shapes[0][-1]
        gamma = obj_dict['gamma'].value
        beta = obj_dict['beta'].value
        if gamma is None or beta is None:
            continue
        gamma = np.atleast_1d(np.squeeze(gamma))
        beta = np.atleast_1d(np.squeeze(beta))
        if np.ndim(gamma) != 1 \
                or np.ndim(beta) != 1 \
                or gamma.size != beta.size \
                or gamma.size != channel_dim:
            continue

        matched = True
        add_in_edges = graph.sorted_in_edges(m['add_2'])
        graph.remove_edge(src, m['mean_1'])
        graph.remove_edge(src, m['squared_diff'])
        graph.remove_edge(src, m['mul_2'])
        graph.remove_edges_from(add_in_edges)
        graph.add_edge(src, m['add_2'], **in_attr1)

        add_attr = obj_dict['add_2'].copied_attr()
        add_attr.update({'non_channel_axes': mean1_axes,
                         'epsilon': eps,
                         'weights': gamma,
                         'weights_scale_zp': list(mul1_in_edges[1][2]['tensor'].scale_zp),
                         'biases': beta,
                         'biases_scale_zp': list(sub_in_edges[0][2]['tensor'].scale_zp),
                         })
        NodeWrap(graph, m['add_2']).replace_obj(
            'InstanceNormalization', add_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_quantized_mul_add(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('mul', {'op': 'LiteMUL'}),
                                      ('multiplier', {'op': 'Constant'}),
                                      ('add', {'op': 'LiteADD'}),
                                      ('adder', {'op': 'Constant'})
                                      ],
                               edges=[('multiplier', 'mul', {'dst_in_port': 1}),
                                      ('mul', 'add'),
                                      ('adder', 'add')
                                      ]
                               )
    for m in matches:
        names = ['mul', 'multiplier', 'add', 'adder']
        objs_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in objs_dict.values()]):
            ERROR('[Parser]: Meets invalid node in merge_quantized_mul_add!')
            continue
        if objs_dict['multiplier'].value is None \
                or objs_dict['adder'].value is None:
            ERROR('[Parser]: Meets invalid node in merge_quantized_mul_add!')
            continue
        in_edges = graph.sorted_in_edges(m['mul'], data=True)
        if len(in_edges) != 2:
            ERROR('[Parser]: Meets invalid inputs in merge_quantized_mul_add!')
            continue
        out_edges = graph.sorted_out_edges(m['add'], data=True)
        if len(out_edges) < 1:
            continue
        input_shapes = objs_dict['mul'].get_input_shapes()
        if len(input_shapes) < 1 \
                or any(s is None for s in input_shapes[0]):
            ERROR('[Parser]: Meets invalid inputs in merge_quantized_mul_add!')
            continue
        if not objs_dict['mul'].quantize \
                or not objs_dict['add'].quantize:
            continue
        if np.ndim(objs_dict['multiplier'].value) not in (0, 1) \
                or np.ndim(objs_dict['adder'].value) not in (0, 1):
            continue
        matched = True

        multipliler_out_edges = graph.sorted_out_edges(
            m['multiplier'], data=True)
        adder_out_edges = graph.sorted_out_edges(m['adder'], data=True)
        _, _, multiplier_out_attr = multipliler_out_edges[0]
        _, _, adder_out_attr = adder_out_edges[0]
        src, _, in_attr = in_edges[0]
        in_shape = input_shapes[0]
        weights = np.atleast_1d(objs_dict['multiplier'].value)
        biases = np.atleast_1d(objs_dict['adder'].value)
        if weights.size != in_shape[-1]:
            weights = np.tile(weights, [in_shape[-1] // weights.size])
        if biases.size != in_shape[-1]:
            biases = np.tile(biases, [in_shape[-1] // biases.size])
        weights_scale_zp = multiplier_out_attr['tensor'].scale_zp
        biases_scale_zp = adder_out_attr['tensor'].scale_zp
        add_in_edges = graph.sorted_in_edges(m['add'])
        graph.remove_edges_from(in_edges + add_in_edges)
        graph.add_edge(src, m['add'], **in_attr)

        bn_attr = objs_dict['add'].copied_attr()
        bn_attr.update({'weights': weights,
                        'weights_scale_zp': weights_scale_zp,
                        'biases': biases,
                        'biases_scale_zp': biases_scale_zp,
                        'axis': -1,
                        'data_format': 'NHWC'})
        NodeWrap(graph, m['add']).replace_obj('ArmBatchNorm', bn_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_special_cast_quantize(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('cast', {'op': 'LiteCAST'}),
                                      ('quantize', {'op': 'LiteQUANTIZE'})
                                      ],
                               edges=[('cast', 'quantize')
                                      ]
                               )
    for m in matches:
        cast, quantize = m['cast'], m['quantize']
        cast_obj = NodeWrap(graph, cast)['object']
        quantize_obj = NodeWrap(graph, quantize)['object']
        if cast_obj is None or quantize_obj is None:
            ERROR('[Parser]: Meets invalid op in merge_special_cast_quantize!')
            continue
        if not cast_obj.quantize or not quantize_obj.quantize:
            continue
        if cast_obj.to != 'float32':
            continue
        in_edges = graph.sorted_in_edges(cast, data=True)
        out_edges = graph.sorted_out_edges(quantize, data=True)
        if len(in_edges) != 1 or len(out_edges) < 1:
            continue
        src, _, in_attr = in_edges[0]
        _, dst, out_attr = out_edges[0]
        if in_attr.get('tensor', None) is not None \
                and in_attr['tensor'].dtype is not None \
                and out_attr.get('tensor', None) is not None \
                and out_attr['tensor'].dtype is not None:
            to_dtype = out_attr['tensor'].dtype
            if to_dtype == 'float32':
                continue
            matched = True
            graph.remove_edge(cast, quantize)
            graph.add_edge(src, quantize, **in_attr)
            node_attr = quantize_obj.copied_attr()
            node_attr.update({'opset_version': 19, 'to': to_dtype, 'saturate': True})
            NodeWrap(graph, quantize).replace_obj('Cast', node_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_special_quantize(graph):
    matches = single_node_matcher(graph, 'LiteQUANTIZE')
    for m in matches:
        quantize = m['target']
        quantize_obj = NodeWrap(graph, quantize)['object']
        if quantize_obj is None:
            ERROR(
                '[Parser]: Meets invalid LiteQUANTIZE(%s) in convert_special_quantize!' % quantize)
            continue
        if not quantize_obj.quantize:
            continue
        in_edges = graph.sorted_in_edges(quantize, data=True)
        out_edges = graph.sorted_out_edges(quantize, data=True)
        if len(in_edges) != 1 \
                or len(out_edges) < 1:
            continue
        src, _, in_attr = in_edges[0]
        _, dst, out_attr = out_edges[0]
        if in_attr.get('tensor', None) is not None \
                and out_attr.get('tensor', None) is not None \
                and in_attr['tensor'].dtype is not None \
                and out_attr['tensor'].dtype is not None:
            from_dtype = in_attr['tensor'].dtype
            to_dtype = out_attr['tensor'].dtype
            if to_dtype == 'float32':
                continue
            node_attr = quantize_obj.copied_attr()
            if from_dtype == 'float32':
                node_attr.update({'scale': out_attr['tensor'].scale_zp[0],
                                  'zero_point': out_attr['tensor'].scale_zp[1],
                                  'to_dtype': to_dtype})
                NodeWrap(graph, quantize).replace_obj('ArmQuantize', node_attr)
            else:
                node_attr.update({'opset_version': 19, 'to': to_dtype, 'saturate': True})
                NodeWrap(graph, quantize).replace_obj('Cast', node_attr)
        else:
            ERROR(
                '[Parser]: Meets invalid LiteQUANTIZE(%s) in convert_special_quantize!' % quantize)


def convert_special_dequantize(graph):
    matches = single_node_matcher(graph, 'LiteDEQUANTIZE')
    for m in matches:
        dequantize = m['target']
        dequantize_obj = NodeWrap(graph, dequantize)['object']
        if dequantize_obj is None:
            ERROR(
                '[Parser]: Meets invalid LiteDEQUANTIZE(%s) in convert_special_dequantize!' % dequantize)
            continue
        if not dequantize_obj.quantize:
            continue
        in_edges = graph.sorted_in_edges(dequantize, data=True)
        out_edges = graph.sorted_out_edges(dequantize, data=True)
        if len(in_edges) != 1 \
                or len(out_edges) < 1:
            continue
        src, _, in_attr = in_edges[0]
        _, dst, out_attr = out_edges[0]
        if in_attr.get('tensor', None) is not None \
                and out_attr.get('tensor', None) is not None \
                and in_attr['tensor'].dtype is not None \
                and len(in_attr['tensor'].scale_zp) == 2 \
                and out_attr['tensor'].dtype is not None:
            from_dtype = in_attr['tensor'].dtype
            to_dtype = out_attr['tensor'].dtype
            if from_dtype == 'float32' or to_dtype != 'float32':
                continue
            node_attr = dequantize_obj.copied_attr()
            node_attr.update({'scale': in_attr['tensor'].scale_zp[0],
                              'zero_point': in_attr['tensor'].scale_zp[1],
                              'from_dtype': from_dtype})
            NodeWrap(graph, dequantize).replace_obj('ArmDeQuantize', node_attr)


def convert_dequantize(graph):
    matches = single_node_matcher(graph, 'LiteDEQUANTIZE')
    for m in matches:
        dequantize = m['target']
        dequantize_obj = NodeWrap(graph, dequantize)['object']
        in_edges = graph.sorted_in_edges(dequantize, data=True)
        out_edges = graph.sorted_out_edges(dequantize, data=True)
        if dequantize_obj is None or len(in_edges) != 1 or len(out_edges) < 1:
            ERROR(
                '[Parser]: Meets invalid LiteDEQUANTIZE(%s) in convert_dequantize!' % dequantize)
            continue
        src, _, in_attr = in_edges[0]
        _, dst, out_attr = out_edges[0]
        if in_attr['tensor'].value is None \
                or out_attr['tensor'].value is None:
            ERROR(
                '[Parser]: Meets invalid LiteDEQUANTIZE(%s) in convert_dequantize!' % dequantize)
            continue
        if in_attr['tensor'].dtype is None \
                or out_attr['tensor'].dtype is None:
            ERROR('[Parser]: Meets invalid dtype of LiteDEQUANTIZE(%s) in convert_dequantize! Should set compat_quantized_model=true in cfg!' % dequantize)
            continue
        src_dtype = np.dtype(in_attr['tensor'].dtype)
        dst_dtype = np.dtype(out_attr['tensor'].dtype)
        if not np.issubdtype(src_dtype, np.integer) or not np.issubdtype(dst_dtype, np.floating):
            continue
        scale = dequantize_obj.scale
        zero_point = dequantize_obj.zero_point
        if scale.dtype == np.float64:
            scale = scale.astype(np.float32)
        if zero_point.dtype == np.int64:
            zero_point = zero_point.astype(np.int32)
        insert_constant(graph,
                        dequantize + '_scale',
                        scale,
                        dequantize,
                        in_port=1,
                        data_format='NHWC',
                        const_ver=9,
                        scale_zp=None,
                        quantize=dequantize_obj.quantize)
        insert_constant(graph,
                        dequantize + '_zero_point',
                        zero_point,
                        dequantize,
                        in_port=2,
                        data_format='NHWC',
                        const_ver=9,
                        scale_zp=None,
                        quantize=dequantize_obj.quantize)
        node_attr = dequantize_obj.copied_attr()
        node_attr.update({'opset_version': 10})
        NodeWrap(graph, dequantize).replace_obj('DequantizeLinear', node_attr)


def remove_useless_dequantize(graph):
    if graph._attr['quantize']:
        return
    matched = False
    matches = two_nodes_matcher(graph, 'Constant', 'LiteDEQUANTIZE')
    for m in matches:
        const, dequantize = m['begin'], m['end']
        const_obj = NodeWrap(graph, const)['object']
        dequantize_obj = NodeWrap(graph, dequantize)['object']
        in_edges = graph.sorted_in_edges(dequantize, data=True)
        if const_obj is not None and dequantize_obj is not None and len(in_edges) == 1:
            if const_obj.value is not None and const_obj.value.dtype == 'float32':
                matched = True
                graph.remove_edge(const, dequantize)
                for _, dst, out_attr in graph.sorted_out_edges(dequantize, data=True):
                    graph.remove_edge(dequantize, dst)
                    graph.add_edge(const, dst, **out_attr)
        else:
            ERROR('[Parser]: Meets invalid LiteDEQUANTIZE(%s) in remove_useless_dequantize!' % dequantize)
            continue
    if matched:
        clear_redundant_nodes(graph)


def merge_dqd(graph, op_list):
    quantize = graph._attr.get('quantize', False)
    if not quantize:
        return
    if not op_list:
        WARN('[Parser]: Empty Op type in merge_dqd!')
        return
    matched = False
    op_list = list(op_list)
    matches = single_node_matcher(graph, op_list)
    for m in matches:
        n = m['target']
        if not graph.has_node(n):
            continue
        node_obj = NodeWrap(graph, n)['object']
        if node_obj is not None:
            node_type = node_obj.type
            in_edges = graph.sorted_in_edges(n, keys=True, data=True)
            out_edges = graph.sorted_out_edges(n, data=True)
            if len(in_edges) < 1 or len(out_edges) != 1:
                continue
            input_tensors = node_obj.get_input_tensors()
            output_tensors = node_obj.get_output_tensors()
            if len(input_tensors) < 1 or len(output_tensors) != 1:
                continue
            if input_tensors[0] is None or not np.issubdtype(input_tensors[0].dtype, np.floating):
                continue
            if output_tensors[0] is None or not np.issubdtype(output_tensors[0].dtype, np.floating):
                continue
            in_objs = [NodeWrap(graph, edge[0])['object'] for edge in in_edges]
            out_objs = [NodeWrap(graph, edge[1])['object'] for edge in out_edges]
            if any(obj is None for obj in (in_objs + out_objs)):
                ERROR('[Parser]: Meets invalid Op(%s) in merge_dqd!' % n)
                continue
            if out_objs[0].type != 'LiteQUANTIZE':
                continue
            if node_type == 'LiteMIRROR_PAD':
                if in_objs[0].type != 'LiteDEQUANTIZE':
                    continue
            elif node_type == 'LiteSQUARED_DIFFERENCE':
                if len(in_edges) != 2 or any(obj.type != 'LiteDEQUANTIZE' for obj in in_objs):
                    continue
            elif node_type == 'LiteRSQRT':
                if in_objs[0].type != 'LiteDEQUANTIZE':
                    continue
            else:
                WARN('[Parser]: Meets unsupportesrcd Op type(%s) in merge_dqd!' % node_type)
                continue
            matched = True
            for i, (dequant_name, _, k, in_attr) in enumerate(in_edges):
                if node_type in ['LiteMIRROR_PAD', 'LiteRSQRT'] and i > 0:
                    continue
                elif node_type == 'LiteSQUARED_DIFFERENCE' and i > 1:
                    continue
                dequant_in_edges = graph.sorted_in_edges(dequant_name, keys=True, data=True)
                if len(dequant_in_edges) < 1:
                    ERROR('[Parser]: Meets invalid LiteDEQUANTIZE Op(%s) in merge_dqd!' % dequant_name)
                    continue
                dequant_src, _, dequant_k, dequant_in_attr = dequant_in_edges[0]
                graph.remove_edge(dequant_name, n, key=k)
                new_dequant_in_attr = copy.deepcopy(dequant_in_attr)
                new_dequant_in_attr['src_out_port'] = in_attr['src_out_port']
                graph.add_edge(dequant_src, n, **new_dequant_in_attr)

            _, quant_name, out_attr = out_edges[0]
            graph.remove_edge(n, quant_name)
            for _, dst, quant_out_attr in graph.sorted_out_edges(quant_name, data=True):
                graph.remove_edge(quant_name, dst)
                graph.add_edge(n, dst, **quant_out_attr)

            node_obj.quantize = True
        else:
            ERROR('[Parser]: Meets invalid Op(%s) in merge_dqd!' % n)

    if matched:
        clear_redundant_nodes(graph)


def merge_min_quant_max_to_clip(graph):
    quantize = graph._attr.get('quantize', False)
    if quantize:
        return
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('min', {'op': 'LiteMINIMUM'}),
                                      ('quant', {'op': 'LiteQUANTIZE'}),
                                      ('max', {'op': 'LiteMAXIMUM'})
                                      ],
                               edges=[('min', 'quant'),
                                      ('quant', 'max')
                                      ]
                               )
    for m in matches:
        names = ['min', 'quant', 'max']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in obj_dict.values()]):
            ERROR('[Parser]: Meets invalid node in merge_min_quant_max_to_clip!')
            continue

        min_out_edges = graph.sorted_out_edges(m['min'], data=True)
        quant_out_edges = graph.sorted_out_edges(m['quant'], data=True)
        max_out_edges = graph.sorted_out_edges(m['max'], data=True)
        if len(min_out_edges) != 1 or len(quant_out_edges) != 1 or len(max_out_edges) < 1:
            continue
        min_in_edges = graph.sorted_in_edges(m['min'], data=True)
        max_in_edges = graph.sorted_in_edges(m['max'], data=True)
        if len(min_in_edges) != 2 or len(max_in_edges) != 2:
            ERROR('[Parser]: Meets invalid node in merge_min_quant_max_to_clip!')
            continue
        if min_in_edges[0][2]['tensor'].dtype is None \
                or min_in_edges[0][2]['tensor'].dtype != min_in_edges[1][2]['tensor'].dtype \
                or len(min_in_edges[0][2]['tensor'].scale_zp) != 2 \
                or len(min_in_edges[1][2]['tensor'].scale_zp) != 2 \
                or not FLOAT_EQUAL(min_in_edges[0][2]['tensor'].scale_zp[0], min_in_edges[1][2]['tensor'].scale_zp[0]) \
                or not FLOAT_EQUAL(min_in_edges[0][2]['tensor'].scale_zp[1], min_in_edges[1][2]['tensor'].scale_zp[1]):
            continue
        if max_in_edges[0][2]['tensor'].dtype is None \
                or max_in_edges[0][2]['tensor'].dtype != max_in_edges[1][2]['tensor'].dtype \
                or len(max_in_edges[0][2]['tensor'].scale_zp) != 2 \
                or len(max_in_edges[1][2]['tensor'].scale_zp) != 2 \
                or not FLOAT_EQUAL(max_in_edges[0][2]['tensor'].scale_zp[0], max_in_edges[1][2]['tensor'].scale_zp[0]) \
                or not FLOAT_EQUAL(max_in_edges[0][2]['tensor'].scale_zp[1], max_in_edges[1][2]['tensor'].scale_zp[1]):
            continue
        if max_out_edges[0][2]['tensor'].dtype is None \
                or len(max_out_edges[0][2]['tensor'].scale_zp) != 2 \
                or max_out_edges[0][2]['tensor'].scale_zp[0] is None \
                or max_out_edges[0][2]['tensor'].scale_zp[0].size != 1 \
                or max_out_edges[0][2]['tensor'].scale_zp[1] is None \
                or max_out_edges[0][2]['tensor'].scale_zp[1].size != 1:
            continue
        if min_in_edges[0][2]['tensor'].dtype != max_in_edges[0][2]['tensor'].dtype:
            continue

        dtype = np.dtype(min_in_edges[0][2]['tensor'].dtype)
        if not np.issubdtype(dtype, np.integer):
            continue

        if not max_in_edges[1][2]['tensor'].is_const \
                or max_in_edges[1][2]['tensor'].value is None \
                or max_in_edges[1][2]['tensor'].value.size != 1:
            continue

        src, _, in_attr = min_in_edges[0]
        graph.remove_edge(src, m['min'])
        graph.remove_edges_from(max_in_edges)
        graph.add_edge(src, m['max'], **in_attr)

        max_out_scale, max_out_zp = max_out_edges[0][2]['tensor'].scale_zp
        clip_min = float(max_in_edges[1][2]['tensor'].value)
        clip_max = (np.iinfo(dtype).max - int(max_out_zp)) * float(max_out_scale)
        clip_attr = obj_dict['max'].copied_attr()
        clip_attr.update({'opset_version': 6,
                         'quantize': False,
                          'min': clip_min,
                          'max': clip_max}
                         )
        NodeWrap(graph, m['max']).replace_obj('Clip', clip_attr)

    if matched:
        clear_redundant_nodes(graph)


def remove_sub_equal_select(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('sub', {'op': 'LiteSUB'}),
                                      ('equal', {'op': 'LiteEQUAL'}),
                                      ('equal_to', {'op': 'Constant'}),
                                      ('zeros_like', {}),
                                      ('select', {'op': 'LiteSELECT'}),
                                      ],
                               edges=[('sub', 'equal'),
                                      ('equal_to', 'equal'),
                                      ('equal', 'select', {'dst_in_port': 0}),
                                      ('zeros_like', 'select',
                                       {'dst_in_port': 2})
                                      ]
                               )
    for m in matches:
        names = ['sub', 'equal_to', 'select']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(obj is None for obj in obj_dict.values()):
            ERROR('[Parser]: Meets invalid Op in remove_sub_equal_select!')
            continue
        sub_in_edges = graph.sorted_in_edges(
            m['sub'], data=True)
        select_in_edges = graph.sorted_in_edges(
            m['select'], data=True)
        if len(sub_in_edges) != 2 \
                or sub_in_edges[0][0] != sub_in_edges[1][0] \
                or sub_in_edges[0][2]['src_out_port'] != sub_in_edges[1][2]['src_out_port'] \
                or len(select_in_edges) != 3 \
                or not FLOAT_EQUAL(obj_dict['equal_to'].value, 0.):
            continue
        sub_src, _, sub_in_attr = sub_in_edges[0]
        select_src, _, select_true_in_attr = select_in_edges[1]
        if sub_src != select_src \
                or sub_in_attr['src_out_port'] != select_true_in_attr['src_out_port'] \
                or select_true_in_attr['dst_in_port'] != 1:
            continue
        matched = True
        for _, dst, out_attr in graph.sorted_out_edges(m['select'], data=True):
            graph.remove_edge(m['select'], dst)
            new_out_attr = copy.deepcopy(out_attr)
            new_out_attr['src_out_port'] = sub_in_attr['src_out_port']
            graph.add_edge(sub_src, dst, **new_out_attr)
        if m['select'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['select'])
            if sub_src not in graph._attr['output_names']:
                graph._attr['output_names'][index] = sub_src
            else:
                graph._attr['output_names'].pop(index)
    if matched:
        clear_redundant_nodes(graph)


def split_op_has_activation(graph, is_tf_op=False):
    op_subclass_names = TfOp.get_concrete_subclass_names(
    ) if is_tf_op else TfliteOp.get_concrete_subclass_names()
    op_has_activations = list(set(BaseActivationOp.get_concrete_subclass_names(
    )).intersection(op_subclass_names))
    activation_types = ActivationOnlyOp.get_concrete_subclass_names()
    op_has_activations = list(
        set(op_has_activations).difference(activation_types))

    matches = [single_node_matcher(graph, op_type)
               for op_type in op_has_activations]
    matches = extend_lists(matches)
    for m in matches:
        node_name = m['target']
        node = NodeWrap(graph, node_name)
        node_obj = node['object']
        if node_obj.activations == 'NONE':
            continue
        onnx_op_dict = BaseActivationOp.activation_to_onnx_op(
            node_obj.activations)
        onnx_op_type = onnx_op_dict.get('type', None)
        opset_version = onnx_op_dict.get('opset_version', None)
        if onnx_op_type is None or opset_version is None:
            ERROR('[Parser]: Activation type %s not implemented in split_op_has_activation!' %
                  node_obj.activations)
        activation_name = get_valid_node_name(
            graph, node_name + '_' + node_obj.activations)
        assert not graph.has_node(
            activation_name), 'The activation op is already in the graph, no need to add in split_op_has_activation.'
        graph.add_node(activation_name)
        activation_node = NodeWrap(graph, activation_name)
        activation_attr = {'name': activation_name,
                           'activations': node_obj.activations}

        if is_tf_op is True and node_obj.activations == 'LEAKYRELU':
            alpha = node_obj.negative_slope if node_obj.negative_slope is not None else 0.2
            activation_attr.update({'alpha': alpha})

        activation_attr.update(onnx_op_dict)
        activation_node.replace_obj(onnx_op_type, activation_attr)
        node_out_edges = graph.sorted_out_edges(node_name, data=True)
        for _, out, out_attr in node_out_edges:
            graph.remove_edge(node_name, out)
            graph.add_edge(activation_name, out, **out_attr)
        graph.add_edge(node_name, activation_name)
        node_obj.activations = 'NONE'
        if node_name in graph._attr['output_names']:
            index = graph._attr['output_names'].index(node_name)
            graph._attr['output_names'][index] = activation_name


def split_fc(graph):
    matches = single_node_matcher(graph, 'LiteFULLY_CONNECTED')
    for m in matches:
        fc = m['target']
        fc_obj = NodeWrap(graph, fc)['object']
        fc_in_edges = graph.sorted_in_edges(fc, keys=True, data=True)
        if fc_obj is not None \
                and len(fc_in_edges) >= 1 \
                and len(fc_obj.get_input_shapes()) >= 1 \
                and len(fc_obj.get_input_shapes()[0]) >= 2:
            last_name = fc
            input_shapes = fc_obj.get_input_shapes()
            src, _, k, in_attr = fc_in_edges[0]
            fc_attr = fc_obj.copied_attr()
            if fc_obj.weights is not None:
                if len(input_shapes[0]) > 2:
                    dim = [-1, fc_obj.weights.shape[-1]]
                    insert_reshape(graph, src, fc, in_attr, dim, key=k)
                biases = fc_obj.biases \
                    if fc_obj.biases is not None \
                    else np.zeros((fc_obj.weights.shape[0],),
                                  dtype=np.int32 if np.issubdtype(
                                      fc_obj.weights.dtype, np.integer) else np.float32
                                  )
                fc_attr.update({'biases': biases})
                NodeWrap(graph, fc).replace_obj('FullyConnected', fc_attr)
                graph.remove_edges_from(fc_in_edges[1:])
                if fc_obj.keepdims and len(input_shapes[0]) > 2:
                    out_shape = list(
                        input_shapes[0][:-1]) + [fc_obj.weights.shape[0]]
                    last_name = insert_reshape_after(graph, fc, out_shape)
            elif len(input_shapes) >= 2 and len(input_shapes[1]) == 2:
                if len(input_shapes[0]) > 2:
                    dim = [-1, input_shapes[1][-1]]
                    insert_reshape(graph, src, fc, in_attr, dim, key=k)
                    fc_in_edges = graph.sorted_in_edges(
                        fc, keys=True, data=True)
                second_input, _, k, in_attr2 = fc_in_edges[1]
                fc_attr.update({'opset_version': 9})
                NodeWrap(graph, fc).replace_obj('MatMul', fc_attr)
                insert_transpose(graph, second_input, fc,
                                 in_attr2, [1, 0], key=k)
                if fc_obj.biases is None \
                        and len(fc_in_edges) == 3 \
                        and fc_obj.get_input_tensors()[2] is not None:
                    fc_obj.biases = fc_obj.get_input_tensors()[2]
                if fc_obj.biases is not None:
                    add = get_valid_node_name(graph, fc + '_bias_add')
                    out_edges = graph.sorted_out_edges(fc, data=True)
                    for _, dst, out_attr in out_edges:
                        graph.remove_edge(fc, dst)
                        graph.add_edge(add, dst, **out_attr)
                    graph.add_edge(fc, add)
                    insert_constant(graph, add + '_bias', fc_obj.biases,
                                    add, in_port=1, data_format='NHWC')
                    add_attr = {'name': add, 'opset_version': 7}
                    NodeWrap(graph, add).replace_obj('Add', add_attr)
                    last_name = add
                if fc_obj.keepdims and len(input_shapes[0]) > 2:
                    out_shape = list(input_shapes[0][:-1]) + [input_shapes[1][0]]
                    last_name = insert_reshape_after(graph, last_name, out_shape)
                graph.remove_edges_from(fc_in_edges[2:])
            else:
                last_name = None
                ERROR(
                    '[Parser]: Meets invalid pattern of LiteFULLY_CONNECTED Node(%s) in split_fc!' % fc)
                continue
            if fc in graph._attr['output_names'] \
                    and last_name \
                    and fc != last_name:
                index = graph._attr['output_names'].index(fc)
                graph._attr['output_names'][index] = last_name
        else:
            ERROR('[Parser]: Meets invalid LiteFULLY_CONNECTED Node(%s) in split_fc!' % fc)


def split_l2_norm(graph):
    reshape_version, const_version = 5, 9
    matches = single_node_matcher(graph, 'LiteL2_NORMALIZATION')
    for m in matches:
        l2 = m['target']
        l2_obj = NodeWrap(graph, l2)['object']
        input_shapes = l2_obj.get_input_shapes()
        if len(input_shapes) >= 1 and len(input_shapes[0]) == 4:
            in_edges = graph.sorted_in_edges(l2, data=True)
            out_edges = graph.sorted_out_edges(l2, data=True)
            if len(in_edges) > 0 and len(out_edges) > 0:
                origin_shape = copy.deepcopy(input_shapes[0])
                pre_reshape = get_valid_node_name(graph, l2 + '_pre_reshape')
                pre_const = get_valid_node_name(graph, pre_reshape + '_shape')
                post_reshape = get_valid_node_name(graph, l2 + '_post_reshape')
                post_const = get_valid_node_name(
                    graph, post_reshape + '_shape')

                graph.add_node(pre_reshape)
                reshape_dim = np.array(
                    [1, 1, 1, int(np.prod(input_shapes[0]))], np.int64)
                reshape_attr = {'name': pre_reshape,
                                'opset_version': reshape_version}
                NodeWrap(graph, pre_reshape).replace_obj(
                    'Reshape', reshape_attr)
                graph.add_node(pre_const)
                pre_const_attr = {'name': pre_const,
                                  'value': reshape_dim,
                                  'data_format': 'NHWC',
                                  'opset_version': const_version}
                NodeWrap(graph, pre_const).replace_obj(
                    'Constant', pre_const_attr)
                edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                             'tensor': Tensor(value=reshape_dim, is_const=True)}
                graph.add_edge(pre_const, pre_reshape, **edge_attr)

                graph.add_node(post_reshape)
                reshape_dim = np.array(origin_shape, np.int64)
                reshape_attr = {'name': post_reshape,
                                'opset_version': reshape_version}
                NodeWrap(graph, post_reshape).replace_obj(
                    'Reshape', reshape_attr)
                graph.add_node(post_const)
                post_const_attr = {'name': post_const,
                                   'value': reshape_dim,
                                   'data_format': 'NHWC',
                                   'opset_version': const_version}
                NodeWrap(graph, post_const).replace_obj(
                    'Constant', post_const_attr)
                edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                             'tensor': Tensor(value=reshape_dim, is_const=True)}
                graph.add_edge(post_const, post_reshape, **edge_attr)

                norm_attr = l2_obj.copied_attr()
                norm_attr.update({'axis': 3, 'p': 2})
                NodeWrap(graph, l2).replace_obj('LpNormalization', norm_attr)

                for src, _, in_attr in in_edges:
                    graph.remove_edge(src, l2)
                    graph.add_edge(src, pre_reshape, **in_attr)
                graph.add_edge(pre_reshape, l2)

                for _, dst, out_attr in out_edges:
                    graph.remove_edge(l2, dst)
                    graph.add_edge(post_reshape, dst, **out_attr)
                graph.add_edge(l2, post_reshape)

                if l2 in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(l2)
                    graph._attr['output_names'][index] = post_reshape


def split_greater_or_less_equal(graph):
    matches = [single_node_matcher(graph, op)
               for op in ['LiteGREATER_EQUAL', 'LiteLESS_EQUAL']]
    matches = extend_lists(matches)
    for m in matches:
        gl_equal = m['target']
        gl_equal_obj = NodeWrap(graph, gl_equal)['object']
        in_edges = graph.sorted_in_edges(gl_equal, data=True)
        if gl_equal_obj is not None and len(in_edges) == 2:
            operand1, _, in_attr1 = in_edges[0]
            operand2, _, in_attr2 = in_edges[1]
            greater_less = get_valid_node_name(
                graph, gl_equal + '_greater' if gl_equal_obj.type == 'LiteGREATER_EQUAL' else gl_equal + '_less')
            equal = get_valid_node_name(graph, gl_equal + '_equal')
            graph.remove_edges_from(in_edges)
            graph.add_edge(operand1, greater_less, **in_attr1)
            graph.add_edge(operand2, greater_less, **in_attr2)
            graph.add_edge(operand1, equal, **in_attr1)
            graph.add_edge(operand2, equal, **in_attr2)
            graph.add_edge(greater_less, gl_equal)
            graph.add_edge(equal, gl_equal, **
                           {'src_out_port': 0, 'dst_in_port': 1})

            less_attr = gl_equal_obj.copied_attr()
            less_attr.update({'name': greater_less, 'opset_version': 9})
            NodeWrap(graph, greater_less).replace_obj(
                'Greater' if gl_equal_obj.type == 'LiteGREATER_EQUAL' else 'Less', less_attr)
            equal_attr = gl_equal_obj.copied_attr()
            equal_attr.update({'name': equal, 'opset_version': 11})
            NodeWrap(graph, equal).replace_obj('Equal', equal_attr)
            NodeWrap(graph, gl_equal).replace_obj(
                'Or', {'name': gl_equal, 'opset_version': 7})


def split_not_equal(graph, op_type='TfNotEqual'):
    if op_type not in ('TfNotEqual', 'Tfnot_equal', 'LiteNOT_EQUAL'):
        ERROR('[Parser]: Meets invalid Op type (%s) in split_not_equal!' % op_type)
        return
    need_clear = False
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        not_equal = m['target']
        not_equal_obj = NodeWrap(graph, not_equal)['object']
        if not_equal_obj is None:
            ERROR(
                '[Parser]: Meets invalid NotEqual Op (%s) in split_not_equal!' % not_equal)
            continue
        in_edges = graph.sorted_in_edges(not_equal, data=True)
        if op_type == 'Tfnot_equal':
            graph.remove_edges_from(in_edges[2:])
            in_edges = in_edges[:2]
            need_clear = True
        equal = get_valid_node_name(graph, not_equal + '_equal')
        for src, _, in_attr in in_edges:
            graph.remove_edge(src, not_equal)
            graph.add_edge(src, equal, **in_attr)
        graph.add_edge(equal, not_equal)

        equal_attr = not_equal_obj.copied_attr()
        equal_attr.update({'name': equal, 'opset_version': 11})
        NodeWrap(graph, equal).replace_obj('Equal', equal_attr)
        not_attr = not_equal_obj.copied_attr()
        not_attr.update({'opset_version': 1})
        NodeWrap(graph, not_equal).replace_obj('Not', not_attr)
    if need_clear:
        clear_redundant_nodes(graph)


def split_quatized_mean(graph):
    matches = single_node_matcher(graph, 'LiteMEAN')
    for m in matches:
        mean = m['target']
        mean_obj = NodeWrap(graph, mean)['object']
        if mean_obj is None or not mean_obj.quantize:
            continue
        in_edges = graph.sorted_in_edges(mean, data=True)
        out_edges = graph.sorted_out_edges(mean, data=True)
        if len(in_edges) < 2 or len(out_edges) < 1:
            continue
        if in_edges[0][2]['tensor'] is None \
                or out_edges[0][2]['tensor'] is None:
            continue
        input_dtype = in_edges[0][2]['tensor'].dtype
        if input_dtype is None:
            continue
        input_scale_zp = in_edges[0][2]['tensor'].scale_zp
        output_scale_zp = out_edges[0][2]['tensor'].scale_zp
        if input_scale_zp is None \
                or len(input_scale_zp) != 2 \
                or output_scale_zp is None \
                or len(output_scale_zp) != 2:
            continue
        output_shapes = mean_obj.get_output_shapes()
        if len(output_shapes) < 1 or output_shapes[0] is None:
            continue
        axes = mean_obj.axes
        output_shape = output_shapes[0]
        add_value = np.zeros(output_shape).astype(input_dtype)
        post_add = get_valid_node_name(graph, mean + '_post_add')
        for _, dst, out_attr in out_edges:
            graph.remove_edge(mean, dst)
            graph.add_edge(post_add, dst, **out_attr)
        graph.add_edge(mean,
                       post_add,
                       **{'tensor': Tensor(dtype=input_dtype, scale_zp=input_scale_zp)})
        insert_constant(graph,
                        post_add + '_adder',
                        add_value,
                        post_add,
                        in_port=1,
                        data_format='NHWC',
                        scale_zp=(np.ones_like(input_scale_zp[0]), np.zeros_like(input_scale_zp[1])),
                        quantize=True)
        if mean in graph._attr['output_names']:
            index = graph._attr['output_names'].index(mean)
            graph._attr['output_names'][index] = post_add
        mean_attr = mean_obj.copied_attr()
        mean_attr.update({'opset_version': 11, 'axes': axes})
        NodeWrap(graph, mean).replace_obj('ReduceMean', mean_attr)
        NodeWrap(graph, post_add).replace_obj(
            'Add', {'opset_version': 7, 'name': post_add})


def split_rsqrt(graph, op_type='LiteRSQRT'):
    if op_type not in ('TfRsqrt', 'Tfrsqrt', 'LiteRSQRT'):
        ERROR('[Parser]: Meets invalid Op type (%s) in split_rsqrt!' % op_type)
        return
    matched = False
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        rsqrt = m['target']
        rsqrt_obj = NodeWrap(graph, rsqrt)['object']
        if rsqrt_obj is not None:
            matched = True
            in_edges = graph.sorted_in_edges(rsqrt, data=True)
            graph.remove_edges_from(in_edges)
            sqrt = get_valid_node_name(graph, rsqrt + '_sqrt')
            for src, _, in_attr in in_edges[:1]:
                graph.add_edge(src, sqrt, **in_attr)
            graph.add_edge(sqrt, rsqrt)

            sqrt_attr = rsqrt_obj.copied_attr()
            sqrt_attr.update({'name': sqrt, 'opset_version': 6})
            NodeWrap(graph, sqrt).replace_obj('Sqrt', sqrt_attr)
            recip_attr = rsqrt_obj.copied_attr()
            recip_attr.update({'opset_version': 6})
            NodeWrap(graph, rsqrt).replace_obj('Reciprocal', recip_attr)
    if matched and op_type == 'Tfrsqrt':
        clear_redundant_nodes(graph)


def remove_detection_postprocess(graph, params):
    matched = False
    matches = single_node_matcher(graph, 'LiteCUSTOM')
    for m in matches:
        custom = m['target']
        custom_obj = NodeWrap(graph, custom)['object']
        if custom_obj is None:
            ERROR('[Parser]: Meets invalid LiteCUSTOM Op (%s) in remove_detection_postprocess!' % custom)
            continue
        if custom_obj.method != 'TFLITE_DETECTION_POSTPROCESS' \
                or custom not in graph._attr['output_names']:
            continue
        custom_in_edges = graph.sorted_in_edges(custom, data=True)
        if not params.get('detection_postprocess', ''):
            matched = True
            WARN('[Parser]: Unsupported LiteCUSTOM Op (%s) will be removed from graph and graph outputs will be reset!' % custom)
            graph.remove_edges_from(custom_in_edges)
            graph._attr['output_names'].clear()
            for src, _, in_attr in custom_in_edges:
                if in_attr['tensor'] is not None and in_attr['tensor'].is_const:
                    continue
                out = get_valid_node_name(graph, src + '_out')
                new_in_attr = copy.deepcopy(in_attr)
                new_in_attr.update({'dst_in_port': 0})
                graph.add_edge(src, out, **new_in_attr)
                NodeWrap(graph, out).replace_obj('Out', {'name': out})
                graph._attr['output_names'].append(src)
        else:
            assert len(
                custom_in_edges) >= 2, 'The length of in_edges of custom op is invalid in remove_detection_postprocess.'
            matched = True
            for i, (src, _, in_attr) in enumerate(custom_in_edges):
                if i < 2:
                    out = get_valid_node_name(graph, src + '_out')
                    new_in_attr = copy.deepcopy(in_attr)
                    new_in_attr.update({'dst_in_port': i})
                    graph.add_edge(src, out, **new_in_attr)
                    NodeWrap(graph, out).replace_obj('Out', {'name': out})
                    graph._attr['output_names'].append(src)
    if matched:
        clear_redundant_nodes(graph)


def remove_redundant_broadcast_to(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('bt', {'op': 'LiteBROADCAST_TO'}),
                                   ('mdst', {'op': ['LiteMUL', 'LiteADD']})
                               ],
                               edges=[
                                   ('bt', 'mdst'),
                               ])
    for m in matches:
        bt, mdst = m['bt'], m['mdst']
        bt_obj = NodeWrap(graph, bt)['object']
        mdst_obj = NodeWrap(graph, mdst)['object']
        if bt_obj is None or \
                len(bt_obj.get_input_tensors()) < 2 or \
                len(graph.sorted_in_edges(bt)) < 2:
            ERROR(
                '[Parser]: Meets invalid LiteBROADCAST_TO(%s) in remove_redundant_broadcast_to!' % bt)
            continue
        if not graph.sorted_in_edges(bt, data=True)[1][2]['tensor'].is_const:
            WARN('[Parser]: Meets unsupported non-constant shape of LiteBROADCAST_TO(%s) in remove_redundant_broadcast_to!' % bt)
            continue
        if mdst_obj is None or \
                len(mdst_obj.get_input_shapes()) != 2:
            ERROR(
                '[Parser]: Meets invalid node(%s) in remove_redundant_broadcast_to!' % mdst)
            continue
        bt_out_shape = bt_obj.get_input_tensors()[1]
        inp, _, bt_in_attr = graph.sorted_in_edges(bt, data=True)[0]
        mdst_in_edges = graph.sorted_in_edges(mdst, data=True)
        if mdst_in_edges[0][0] == mdst_in_edges[1][0]:
            continue
        bt_out_edges = graph.sorted_out_edges(bt, data=True)
        mdst_in_port = [out_attr['dst_in_port']
                        for _, dst, out_attr in bt_out_edges if dst == mdst][0]
        mdst_in_shape = mdst_obj.get_input_shapes()[1 - mdst_in_port]
        if len(mdst_in_shape) != bt_out_shape.size or \
                any([mdst_in_shape[index] != int(rep) for index, rep in enumerate(bt_out_shape) if int(rep) != 1]):
            continue
        matched = True
        graph.remove_edge(bt, mdst)
        new_in_attr = copy.deepcopy(bt_in_attr)
        new_in_attr.update({'dst_in_port': mdst_in_port})
        graph.add_edge(inp, mdst, **new_in_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_to_onnx(graph):
    '''Convert the model to the onnx version.'''
    lite_ops = TfliteOp.get_concrete_subclass_names()
    matches = extend_lists([single_node_matcher(graph, op_type)
                            for op_type in lite_ops])
    for m in matches:
        node_name = m['target']
        node_obj = NodeWrap(graph, node_name)['object']
        if node_obj is not None:
            new_node_attr = node_obj.copied_attr()
            pure_type = re.sub(r'^Lite', '', node_obj.type)
            if getattr(node_obj, 'correspond_onnx_op', None) is not None:
                if isinstance(node_obj, OpHasWeights):
                    if node_obj.weights is None:
                        ERROR('[Parser]: Node(%s) dosenot contain weights in convert_to_onnx!' %
                              node_name)
                        continue
                    new_weights = np.transpose(
                        node_obj.weights, axes=type(node_obj).perm_lite_to_onnx())
                    new_node_attr.update({'weights': new_weights})
                    if pure_type == 'DEPTHWISE_CONV_2D':
                        if hasattr(node_obj, 'multiplier') \
                                and getattr(node_obj, 'multiplier') > 1 \
                                and new_weights.shape[1] == 1 \
                                and node_obj.group == 1:
                            new_node_attr.update(
                                {'group': new_weights.shape[0] // getattr(node_obj, 'multiplier')})
                if pure_type == 'CAST':
                    new_node_attr.update({'saturate': False})
                elif pure_type == 'ELU':
                    new_node_attr.update({'alpha': 1.})
                elif pure_type == 'EXPAND_DIMS':
                    in_edges = graph.sorted_in_edges(node_name)
                    if len(in_edges) < 1 \
                            or len(node_obj.get_input_tensors()) < 1 \
                            or node_obj.get_input_tensors()[0] is None:
                        ERROR(
                            '[Parser]: Invalid TFlite ExpandDims Node(%s) to convert to Onnx!' % node_name)
                        continue
                    axis = node_obj.axis
                    out_tensor = np.expand_dims(
                        node_obj.get_input_tensors()[0], axis)
                    graph.remove_edges_from(in_edges[1:])
                    insert_constant(graph,
                                    node_name + '_shape',
                                    np.array(out_tensor.shape, np.int32),
                                    node_name,
                                    in_port=1,
                                    data_format='NHWC')
                elif pure_type == 'FLOOR_MOD':
                    new_node_attr.update({'fmod': 0})
                elif pure_type == 'L2_NORMALIZATION':
                    new_node_attr.update({'p': 2})
                elif pure_type == 'LOCAL_RESPONSE_NORMALIZATION':
                    size = 2 * node_obj.radius + 1
                    alpha = node_obj.alpha * size
                    new_node_attr.update({'size': size, 'alpha': alpha})
                elif pure_type == 'MEAN':
                    in_edges = graph.sorted_in_edges(node_name)
                    graph.remove_edges_from(in_edges[1:])
                elif pure_type == 'PACK':
                    new_node_attr.update({'new_axis': True})
                elif pure_type in ('PAD', 'PADV2', 'MIRROR_PAD'):
                    pads = node_obj.sorted_in_consts()[0][2]
                    const_name = node_obj.sorted_in_consts()[0][0]
                    NodeWrap(graph, const_name)[
                        'object'].value = np.transpose(pads)
                    in_edges = graph.sorted_in_edges(node_name, data=True)
                    if len(in_edges) == 2 and in_edges[1][2].get('tensor', None) is not None:
                        in_edges[1][2]['tensor'].value = np.transpose(pads)
                    if pure_type == 'MIRROR_PAD':
                        if node_obj.mode == 'REFLECT':
                            new_node_attr.update({'mode': 'reflect'})
                        else:
                            new_node_attr.update({'mode': 'symmetric'})
                elif pure_type == 'RELU6':
                    new_node_attr.update({'min': 0., 'max': 6.})
                elif pure_type == 'RELU_N1_TO_1':
                    new_node_attr.update({'min': -1., 'max': 1.})
                elif pure_type == 'RESHAPE':
                    in_edges = graph.sorted_in_edges(
                        node_name, keys=True, data=True)
                    input_tensors = node_obj.get_input_tensors()
                    if len(in_edges) == 2 and len(input_tensors) == 2 and np.ndim(input_tensors[1]) != 1:
                        shape_inp, _, k, in_attr = in_edges[1]
                        dim = [input_tensors[1].size]
                        insert_reshape(graph, shape_inp,
                                       node_name, in_attr, dim, key=k)
                elif pure_type in ('RESIZE_BILINEAR', 'RESIZE_NEAREST_NEIGHBOR'):
                    dst_size = node_obj.sorted_in_consts()[0][2]
                    input_shape = node_obj.get_input_shapes()[0]
                    full_size = [input_shape[0]] + \
                        dst_size.tolist() + [input_shape[-1]]
                    in_edges = graph.sorted_in_edges(node_name)
                    graph.remove_edges_from(in_edges[1:])
                    assert node_obj.correspond_onnx_op['version'] >= 11, \
                        '[Parser]: Only support Resize above 11 when converting from TFLite to Onnx!'
                    insert_constant(graph, node_name + '_roi', np.array([],
                                                                        np.int64), node_name, in_port=1, data_format='NHWC')
                    insert_constant(
                        graph, node_name + '_scales', np.array([], np.float32), node_name, in_port=2, data_format='NHWC')
                    insert_constant(
                        graph, node_name + '_size', np.array(full_size, np.int32), node_name, in_port=3, data_format='NHWC')
                    mode = 'linear' if pure_type == 'RESIZE_BILINEAR' else 'nearest'
                    if node_obj.align_corners:
                        coordinate_transformation_mode = 'align_corners'
                        nearest_mode = 'round_prefer_ceil'
                    else:
                        if node_obj.half_pixel:
                            if mode == 'nearest':
                                coordinate_transformation_mode = 'tf_half_pixel_for_nn'
                            else:
                                coordinate_transformation_mode = 'half_pixel'
                        else:
                            coordinate_transformation_mode = 'asymmetric'
                        nearest_mode = 'floor'
                    new_node_attr.update({'mode': mode,
                                          'coordinate_transformation_mode': coordinate_transformation_mode,
                                          'nearest_mode': nearest_mode
                                          })
                elif pure_type == 'SEGMENT_SUM':
                    new_node_attr.update({'method': 'SUM'})
                elif pure_type == 'SELECT':
                    in_tensors = node_obj.get_input_tensors()
                    try:
                        dims_and_reps = OpNeedBroadcast.cal_reshape_and_tile(
                            [t.shape for t in in_tensors], match_from_left=True)
                    except:
                        dims_and_reps = []
                    in_edges = graph.sorted_in_edges(
                        node_name, keys=True, data=True)
                    if len(dims_and_reps) != len(in_edges):
                        ERROR(
                            '[Parser]: Fail to calculate LITESELECT op (%s) broadcast in convert_to_onnx!' % node_name)
                        continue
                    for i, dr in enumerate(dims_and_reps):
                        if dr['reshape'] is not None:
                            src, _, k, in_attr = in_edges[i]
                            insert_reshape(graph, src, node_name,
                                           in_attr, dr['reshape'], key=k)
                            in_edges = graph.sorted_in_edges(
                                node_name, keys=True, data=True)
                        if dr['tile'] is not None:
                            src, _, k, in_attr = in_edges[i]
                            insert_tile(graph, src, node_name,
                                        in_attr, dr['tile'], key=k)
                            in_edges = graph.sorted_in_edges(
                                node_name, keys=True, data=True)
                elif pure_type == 'SLICE':
                    starts = node_obj.sorted_in_consts()[0][2]
                    ends_name = node_obj.sorted_in_consts()[1][0]
                    NodeWrap(graph, ends_name)[
                        'object'].value = starts + node_obj.size
                elif pure_type in ('SPLIT', 'SPLIT_V'):
                    split = []
                    for _, _, e in graph.sorted_out_edges(node_name, data=True):
                        out_port = e['src_out_port']
                        if out_port >= len(split):
                            split += [0] * (out_port - len(split) + 1)
                        split[out_port] = e['tensor'].shape[node_obj.axis]
                    if not(node_obj.num_splits and (len(split) == node_obj.num_splits)):
                        ERROR('[Parser]: Invalid split nums of node %s in convert_to_onnx!' %
                              (node_name))
                    new_node_attr.update({'split': split})
                    in_edges = graph.sorted_in_edges(node_name)
                    if pure_type == 'SPLIT':
                        graph.remove_edges_from(in_edges[:1])
                    else:
                        graph.remove_edges_from(in_edges[1:])
                elif pure_type == 'TRANSPOSE':
                    const_name, const_value = node_obj.sorted_in_consts(
                    )[0][0], node_obj.sorted_in_consts()[0][2]
                    new_node_attr.update({'perm': const_value.tolist()})
                    graph.remove_node(const_name)

                new_node_attr.update(
                    {'opset_version': node_obj.correspond_onnx_op['version']})
                NodeWrap(graph, node_name).replace_obj(
                    node_obj.correspond_onnx_op['type'], new_node_attr)
            else:
                ERROR('[Parser]: TFLite op %s cannot be converted to Onnx' % pure_type)
        else:
            ERROR(
                '[Parser]: Meets invalid TFLite op for Node(%s) in convert_to_onnx!' % node_name)
