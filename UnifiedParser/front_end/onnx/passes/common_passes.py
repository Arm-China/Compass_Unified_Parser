# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import copy
import itertools
from ....plugin_loader import PARSER_OP_DICT

from ....common.defs import Tensor, Framework, FLOAT_EQUAL
from ....common.errors import *
from ....common.utils import extend_lists, get_converted_dtype
from ....ops.op import Op, OpHasWeights, OpHasBiases, OpHasOneOutPort, ConstLikeOp
from ....ops.onnx_ops.array_ops import CastOp
from ....ops.release_ops import ArmCastOp, ArmTransposeOp
from ....graph.node_wrap import NodeWrap
from ....graph.graph_algo import has_path, get_valid_node_name, all_simple_paths, clear_redundant_nodes
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher


def fuse_const(graph):
    matches = single_node_matcher(graph, '')
    for m in matches:
        node_name = m['target']
        if node_name in graph.nodes:
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is not None \
                    and not isinstance(node_obj, ConstLikeOp) \
                    and isinstance(node_obj, OpHasOneOutPort) \
                    and node_obj.is_all_inputs_const():
                out_edge = graph.sorted_out_edges(node_name, data=True)
                if len(out_edge) >= 1 and out_edge[0][2]['tensor'] is not None and out_edge[0][2]['tensor'].value is not None:
                    const_value = out_edge[0][2]['tensor'].value
                    if str(const_value.dtype) == 'int64':
                        const_value = const_value.astype(np.int32)
                    elif str(const_value.dtype) == 'float64':
                        const_value = const_value.astype(np.float32)
                    const_attr = {'name': node_name,
                                  'value': const_value,
                                  'data_format': node_obj.data_format,
                                  'opset_version': 9}
                    NodeWrap(graph, node_name).replace_obj(
                        'Constant', const_attr)
                    in_edges = graph.sorted_in_edges(node_name)
                    graph.remove_edges_from(in_edges)
    clear_redundant_nodes(graph)


def remove_node_safely(graph, n):
    assert graph.has_node(
        n), 'The node %s does not exist, cannot remove_node_safely.' % (n)
    in_edges = graph.sorted_in_edges(n, data=True)
    out_edges = graph.sorted_out_edges(n, data=True)
    if len(in_edges) >= 1:
        src_name, _, in_attr = in_edges[0]
        new_attr = copy.deepcopy(in_attr)
        for _, dst_name, out_attr in out_edges:
            if NodeWrap(graph, src_name)['object'].type == 'Constant':
                const = NodeWrap(graph, src_name)['object'].value
                if new_attr['tensor'] is None:
                    new_attr['tensor'] = Tensor(value=const)
                else:
                    new_attr['tensor'].value = const
            if new_attr['tensor'] is None:
                new_attr['tensor'] = Tensor(min_max=out_attr['tensor'].min_max)
            else:
                new_attr['tensor'].min_max = out_attr['tensor'].min_max if out_attr['tensor'].min_max else in_attr['tensor'].min_max
            new_attr.update({'dst_in_port': out_attr['dst_in_port']})
            graph.add_edge(src_name, dst_name, **new_attr)
        graph.remove_node(n)
    elif len(in_edges) == 0 and len(out_edges) == 0:
        graph.remove_node(n)


def remove_useless_op(graph, op_type_list):
    '''Remove some redundant unwanted OP.'''
    for op_type in op_type_list:
        removing_nodes = []
        matched = single_node_matcher(graph, op_type)
        for m in matched:
            node_name = m['target']
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is None:
                continue
            in_tensors = node_obj.get_input_tensors()
            if op_type == 'Cast':
                if node_obj.to not in ArmCastOp.attributes()['to_dtype']['options']:
                    if node_obj.to == 'bool':
                        continue
                    new_dtype = get_converted_dtype(
                        node_obj.to, return_type_string=True)
                    if new_dtype:
                        WARN('[Parser]: Change unsupported dtype (%s) in Cast op (%s) to %s!' %
                             (node_obj.to, node_name, new_dtype))
                        node_obj.to = new_dtype
                if len(in_tensors) > 0 \
                        and in_tensors[0] is not None \
                        and str(in_tensors[0].dtype) == node_obj.to:
                    removing_nodes.append(node_name)
                else:
                    continue
            elif op_type == 'ArmCast':
                if len(in_tensors) > 0 \
                        and in_tensors[0] is not None \
                        and str(in_tensors[0].dtype) == node_obj.to_dtype:
                    removing_nodes.append(node_name)
                else:
                    continue
            elif op_type == 'Concat':
                in_edges = graph.sorted_in_edges(node_name)
                if len(in_edges) <= 1:
                    removing_nodes.append(node_name)
            elif op_type in ('Dropout', 'Dummy', 'Identity'):
                removing_nodes.append(node_name)
            elif op_type == 'Expand':
                in_edges = graph.sorted_in_edges(node_name, data=True)
                if len(in_edges) == 2 \
                        and in_edges[1][2].get('tensor', None) is not None \
                        and in_edges[1][2]['tensor'].is_const \
                        and FLOAT_EQUAL(in_edges[1][2]['tensor'].value, 1):
                    shape_node, _, _ = in_edges[1]
                    graph.remove_edge(shape_node, node_name)
                    removing_nodes.append(node_name)
            elif op_type in ('AveragePool', 'MaxPool'):
                in_shapes = node_obj.get_input_shapes()
                out_shapes = node_obj.get_output_shapes()
                if in_shapes[0] is None or out_shapes[0] is None:
                    WARN(
                        '[Parser]: Meets invalid input/output shape for node (%s) in remove_useless_op.' % node_name)
                    continue
                if (node_obj.data_format == 'NCHW' and in_shapes[0][2:] == [1, 1] and out_shapes[0][2:] == [1, 1]) \
                        or (node_obj.data_format == 'NHWC' and in_shapes[0][1:3] == [1, 1] and out_shapes[0][1:3] == [1, 1]):
                    removing_nodes.append(node_name)
            elif op_type == 'Pad':
                if np.all(np.array(node_obj.pads, np.int64) == 0):
                    removing_nodes.append(node_name)
            elif op_type in ('Transpose', 'ArmTranspose'):
                trans_in_edges = graph.sorted_in_edges(node_name, data=True)
                if len(trans_in_edges) != 1:
                    WARN(
                        '[Parser]: Meets invalid Transpose(%s) to remove in remove_useless_op!' % node_name)
                    continue
                trans_in_tensor = trans_in_edges[0][2]['tensor'].value
                perm = node_obj.perm
                src_name = trans_in_edges[0][0]
                src_node_obj = NodeWrap(graph, src_name)['object']
                if src_node_obj is None:
                    continue
                if src_node_obj.type == 'Constant' \
                        and src_node_obj.value is not None \
                        and len(graph.sorted_out_edges(src_name)) == 1:
                    src_node_obj.value = np.transpose(
                        src_node_obj.value, axes=perm)
                    removing_nodes.append(node_name)
                elif trans_in_tensor is not None and list(range(len(trans_in_tensor.shape))) == perm:
                    removing_nodes.append(node_name)
                elif perm and list(perm) == list(range(max(perm) + 1)):
                    removing_nodes.append(node_name)
            elif op_type in ('Reshape', 'ArmReshape'):
                reshape_in_edges = graph.sorted_in_edges(node_name, data=True)
                src_name = reshape_in_edges[0][0]
                src_node_obj = NodeWrap(graph, src_name)['object']
                if src_node_obj is None:
                    continue
                in_shapes = node_obj.get_input_shapes()
                out_shapes = node_obj.get_output_shapes()
                if src_node_obj.type == 'Constant' \
                        and src_node_obj.value is not None \
                        and len(graph.sorted_out_edges(src_name)) == 1:
                    new_shape = node_obj.shape if op_type == 'Reshape' else node_obj.dim
                    src_node_obj.value = np.reshape(
                        src_node_obj.value, newshape=new_shape)
                    removing_nodes.append(node_name)
                elif len(in_shapes) >= 1 \
                        and in_shapes[0] is not None \
                        and len(out_shapes) >= 1 \
                        and in_shapes[0] == out_shapes[0]:
                    removing_nodes.append(node_name)
                elif len(in_shapes) >= 1 \
                        and in_shapes[0] is not None \
                        and len(in_shapes[0]) == 2 \
                        and len(out_shapes) == 1 \
                        and out_shapes[0] is not None \
                        and len(out_shapes[0]) == 4 \
                        and in_shapes[0] == out_shapes[0][2:] \
                        and out_shapes[0][0:2] == [1, 1] \
                        and NodeWrap(graph, graph.succ[node_name][0])['object'].type == 'Out':
                    remove_node_safely(graph, node_name)
            elif op_type == 'Resize':
                in_shapes = node_obj.get_input_shapes()
                if node_obj.cur_version == 10:
                    if len(in_tensors) == 2 and node_obj.cur_version == 10 and in_tensors[1].size > 0 and np.all(in_tensors[1] == 1):
                        removing_nodes.append(node_name)
                else:
                    if len(in_tensors) >= 3 and in_tensors[2].size > 0 and np.all(in_tensors[2] == 1):
                        removing_nodes.append(node_name)
                    elif len(in_tensors) == 4 and in_tensors[3].size > 0 and in_shapes[0] == in_tensors[3].tolist():
                        removing_nodes.append(node_name)
            elif op_type == 'Slice':
                out_tensors = node_obj.get_output_tensors()
                if len(in_tensors) >= 1 \
                        and len(out_tensors) == 1 \
                        and in_tensors[0] is not None \
                        and out_tensors[0] is not None \
                        and list(in_tensors[0].shape) == list(out_tensors[0].shape) \
                        and all([d == 0 for d in node_obj.starts]) \
                        and list(in_tensors[0].shape) == node_obj.ends:
                    removing_nodes.append(node_name)
            elif op_type == 'Split':
                if len(in_tensors) >= 1 and len(node_obj.split) == 1 and \
                        in_tensors[0] is not None and \
                        in_tensors[0].shape[node_obj.axis] == node_obj.split[0]:
                    removing_nodes.append(node_name)
            elif op_type == 'Upsample':
                if node_obj.scales and np.all(np.array(node_obj.scales, np.float32) == 1):
                    upsample_in_edges = graph.sorted_in_edges(node_name)
                    if len(upsample_in_edges) >= 1:
                        graph.remove_edges_from(upsample_in_edges[1:])
                    removing_nodes.append(node_name)
            else:
                removing_nodes.append(node_name)

        for rn in removing_nodes:
            remove_node_safely(graph, rn)


def remove_redundant_bn(graph, max_branches=6):
    for b in range(max_branches, 0, -1):
        nodes = [('pre_bn', {'op': 'ArmBatchNorm'})] \
            + [('post_bn_%s' % str(i + 1),
                {'op': 'ArmBatchNorm'}) for i in range(b)]
        edges = [('pre_bn', 'post_bn_%s' % str(i + 1),
                  {'dst_in_port': 0}) for i in range(b)]
        matches = matched_patterns(graph, nodes, edges)
        for m in matches:
            pre_bn = m['pre_bn']
            post_bns = [m['post_bn_%s' % str(i + 1)] for i in range(b)]
            obj_dict = {k: NodeWrap(graph, k)['object']
                        for k in [pre_bn] + post_bns}
            if all([obj is not None for obj in obj_dict.values()]):
                pre_bn_out_edges = graph.sorted_out_edges(pre_bn)
                if len(pre_bn_out_edges) != b:
                    continue
                if any([obj_dict[name].axis != obj_dict[pre_bn].axis for name in post_bns]):
                    continue
                pre_bn_in_edges = graph.sorted_in_edges(pre_bn, data=True)
                if len(pre_bn_in_edges) < 1:
                    WARN(
                        '[Parser]: Meets invalid BatchNorm Op (%s) in remove_redundant_bn!' % pre_bn)
                    continue
                src, _, in_attr = pre_bn_in_edges[0]
                graph.remove_edge(src, pre_bn)
                for name in post_bns:
                    graph.remove_edge(pre_bn, name)
                    graph.add_edge(src, name, **in_attr)
                    new_weights = obj_dict[pre_bn].weights * \
                        obj_dict[name].weights
                    new_biases = obj_dict[pre_bn].biases * \
                        obj_dict[name].weights + obj_dict[name].biases
                    obj_dict[name].weights = new_weights
                    obj_dict[name].biases = new_biases
                remove_node_safely(graph, pre_bn)
            else:
                WARN('[Parser]: Meets invalid BatchNorm Op in remove_redundant_bn!')


def remove_redundant_reshape(graph, type='Reshape'):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('reshape_1', {'op': type}), ('reshape_2', {'op': type})],
                               edges=[('reshape_1', 'reshape_2')]
                               )
    for m in matches:
        reshape_1, reshape_2 = m['reshape_1'], m['reshape_2']
        reshape_1_obj = NodeWrap(graph, reshape_1)['object']
        reshape_2_obj = NodeWrap(graph, reshape_2)['object']
        reshape_1_in_shapes = reshape_1_obj.get_input_shapes()
        reshape_1_out_shapes = reshape_1_obj.get_output_shapes()
        reshape_2_out_shapes = reshape_2_obj.get_output_shapes()
        if len(reshape_1_in_shapes) >= 1 \
                and len(reshape_1_out_shapes) == 1 \
                and len(reshape_2_out_shapes) >= 1:
            remove_node_safely(graph, reshape_1)


def remove_redundant_transpose(graph):
    transpose_types = ['Transpose', 'ArmTranspose']
    matches = [matched_patterns(graph,
                                nodes=[('transpose1', {'op': tt}),
                                       ('transpose2', {'op': tt})],
                                edges=[('transpose1', 'transpose2')]
                                ) for tt in transpose_types]
    matches = extend_lists(matches)
    for m in matches:
        trans1, trans2 = m['transpose1'], m['transpose2']
        trans1_obj, trans2_obj = NodeWrap(
            graph, trans1)['object'], NodeWrap(graph, trans2)['object']
        if trans1_obj is not None \
                and trans2_obj is not None \
                and len(trans1_obj.perm) == len(trans2_obj.perm):
            trans1_out_edges = graph.sorted_out_edges(trans1)
            if len(trans1_out_edges) == 1:
                trans2_obj.perm = ArmTransposeOp.cal_merged_perm(
                    trans1_obj.perm, trans2_obj.perm)
                remove_node_safely(graph, trans1)
        else:
            WARN('[Parser]: Meets invalid Transpose (%s or %s) in remove_redundant_transpose!'
                 % (trans1, trans2))


def remove_redundant_transpose_pro(graph, trans_type='Transpose', max_branches=6):
    assert max_branches > 0 and trans_type in (
        'Transpose', 'ArmTranspose'), 'trans_type and max_branches are not valid in remove_redundant_transpose_pro.'
    for b in range(max_branches, 0, -1):
        matched = False
        nodes = [('trans', {'op': trans_type})] + \
            [('trans_out_%s' % str(i+1), {}) for i in range(b)]
        edges = [('trans', 'trans_out_%s' % str(i+1)) for i in range(b)]
        matches = matched_patterns(graph, nodes, edges)
        for m in matches:
            trans = m['trans']
            trans_out_names = [m['trans_out_%s' % str(i+1)] for i in range(b)]
            if any([not graph.has_node(name) for name in [trans] + trans_out_names]):
                WARN(
                    '[Parser]: Meets invalid name that does not exist in graph in remove_redundant_transpose_pro!')
                continue
            trans_obj = NodeWrap(graph, trans)['object']
            trans_out_objs = [NodeWrap(graph, name)['object']
                              for name in trans_out_names]
            if trans_obj is None or any([out_obj is None for out_obj in trans_out_objs]):
                WARN(
                    '[Parser]: Meets invalid Node object in remove_redundant_transpose_pro!')
                continue
            out_trans_objs = {
                out_obj.name: out_obj for out_obj in trans_out_objs if out_obj.type == trans_type}
            if len(out_trans_objs) == 0:
                continue
            matched = True
            trans_in_edges = graph.sorted_in_edges(trans, data=True)
            trans_out_edges = graph.sorted_out_edges(trans, data=True)
            src, _, in_attr = trans_in_edges[0]
            for _, dst, out_attr in trans_out_edges:
                if dst in out_trans_objs:
                    graph.remove_edge(trans, dst)
                    new_out_attr = copy.deepcopy(out_attr)
                    new_out_attr.update(
                        {'src_out_port': in_attr['src_out_port']})
                    graph.add_edge(src, dst, **new_out_attr)
                    out_trans_objs[dst].perm = ArmTransposeOp.cal_merged_perm(
                        trans_obj.perm, out_trans_objs[dst].perm)
            trans_out_edges = graph.sorted_out_edges(trans)
            if len(trans_out_edges) == 0:
                graph.remove_edge(src, trans)
        if matched:
            clear_redundant_nodes(graph)


def remove_redundant_cast(graph):
    cast_types = ['Cast', 'ArmCast']
    cast_combinations = itertools.product(cast_types, cast_types)
    for cast1_type, cast2_type in cast_combinations:
        matches = two_nodes_matcher(graph, cast1_type, cast2_type)
        for m in matches:
            cast1, cast2 = m['begin'], m['end']
            cast1_obj = NodeWrap(graph, cast1)['object']
            cast2_obj = NodeWrap(graph, cast2)['object']
            if cast1_obj is not None and cast2_obj is not None:
                cast1_dst_type = cast1_obj.to if cast1_type == 'Cast' else cast1_obj.to_dtype
                cast2_dst_type = cast2_obj.to if cast2_type == 'Cast' else cast2_obj.to_dtype
                if 'int' in cast1_dst_type and 'float' in cast2_dst_type:
                    # int+float is not same as float. For example, if input is 1.5, int+float
                    # will get 1.0, while float will get 1.5.
                    continue
                cast1_out_edges = graph.sorted_out_edges(cast1)
                if len(cast1_out_edges) == 1:
                    remove_node_safely(graph, cast1)
            else:
                WARN('[Parser]: Meets invalid Cast Node (%s or %s) in remove_redundant_cast!' % (
                    cast1, cast2))


def insert_cast(graph, src, dst, dst_type, in_attr=None, key=None, type='Cast'):
    ret = None
    if graph is not None \
            and len(graph) >= 2 \
            and dst_type in ArmCastOp.attributes()['to_dtype']['options'] \
            and graph.has_node(src) \
            and graph.has_node(dst) \
            and type in ('Cast', 'ArmCast'):
        if not in_attr:
            in_attr = {'src_out_port': 0, 'dst_in_port': 0}
        if has_path(graph, src, dst):
            graph.remove_edge(src, dst, key=key)
        cast = get_valid_node_name(
            graph, dst + '_pre_cast_' + str(in_attr.get('dst_in_port', 0)))
        cast_in_attr = copy.deepcopy(in_attr)
        cast_in_attr['dst_in_port'] = 0
        if in_attr.get('tensor', None) is not None and in_attr['tensor'].value is not None:
            casted_value = in_attr['tensor'].value.astype(np.dtype(dst_type))
        else:
            casted_value = None
        cast_out_attr = {'src_out_port': 0, 'dst_in_port': in_attr.get(
            'dst_in_port', 0), 'tensor': Tensor(value=casted_value)}
        graph.add_edge(src, cast, **cast_in_attr)
        graph.add_edge(cast, dst, **cast_out_attr)
        if type == 'Cast':
            cast_attr = {'name': cast, 'opset_version': 1, 'to': dst_type}
        else:
            cast_attr = {'name': cast, 'to_dtype': dst_type}
        NodeWrap(graph, cast).replace_obj(type, cast_attr)
        ret = cast
    else:
        WARN('[Parser]: Invalid params for insert_cast!')
    return ret


def insert_cast_after(graph, src, from_dtype, to_dtype, out_port=0, type='Cast'):
    ret = None
    if graph.has_node(src) \
            and to_dtype in ArmCastOp.attributes()['to_dtype']['options'] \
            and type in ('Cast', 'ArmCast') \
            and NodeWrap(graph, src)['object'] is not None \
            and out_port in NodeWrap(graph, src)['object'].get_out_ports():
        if out_port == 0:
            cast = src + '_post_cast'
        else:
            cast = src + '_post_cast_' + str(out_port)
        cast = get_valid_node_name(graph, cast)
        cast_in_tensor_value = None
        for _, dst, k, out_attr in graph.sorted_out_edges(src, keys=True, data=True):
            if out_attr['src_out_port'] == out_port:
                new_out_attr = copy.deepcopy(out_attr)
                new_out_attr['src_out_port'] = 0
                if new_out_attr['tensor'] is not None \
                        and new_out_attr['tensor'].value is not None:
                    new_out_attr['tensor'].value = new_out_attr['tensor'].value.astype(
                        np.dtype(to_dtype))
                    cast_in_tensor_value = new_out_attr['tensor'].value.astype(
                        np.dtype(from_dtype))
                graph.remove_edge(src, dst, k)
                graph.add_edge(cast, dst, **new_out_attr)
        graph.add_edge(src, cast, **{'src_out_port': out_port,
                                     'dst_in_port': 0, 'tensor': Tensor(value=cast_in_tensor_value)})
        if type == 'Cast':
            cast_attr = {'name': cast, 'opset_version': 1, 'to': to_dtype}
        else:
            cast_attr = {'name': cast, 'to_dtype': to_dtype}
        NodeWrap(graph, cast).replace_obj(type, cast_attr)
        ret = cast
    else:
        WARN('[Parser]: Invalid params for insert_cast_after!')
    return ret


def insert_constant(graph, name, value, dst, in_port=0, data_format='NCHW', const_ver=9):
    if graph.has_node(dst) and value is not None and isinstance(value, np.ndarray):
        const_name = get_valid_node_name(graph, name)
        graph.add_node(const_name)
        const_attr = {'name': const_name,
                      'value': value,
                      'data_format': data_format,
                      'opset_version': const_ver}
        NodeWrap(graph, const_name).replace_obj('Constant', const_attr)
        edge_attr = {'src_out_port': 0, 'dst_in_port': in_port,
                     'tensor': Tensor(value=value, is_const=True)}
        graph.add_edge(const_name, dst, **edge_attr)
    else:
        WARN('[Parser]: Invalid params for insert_constant (%s)!' % name)


def insert_gather(graph, src, dst, indices, axis=0, edge_attr=dict(), key=None, type='Gather'):
    ret = None
    assert type in (
        'Gather', 'ArmGather'), 'The type of node is invalid in insert_gather.'
    if graph.has_node(src) and graph.has_node(dst) and indices is not None:
        if has_path(graph, src, dst):
            graph.remove_edge(src, dst, key=key)
        if not edge_attr:
            edge_attr = {'src_out_port': 0, 'dst_in_port': 0}
        gather = get_valid_node_name(graph, dst + '_pre_gather')
        gather_in_attr = copy.deepcopy(edge_attr)
        gather_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, gather, **gather_in_attr)
        insert_constant(graph, gather + '_indices', indices,
                        gather, in_port=1, data_format='NHWC')
        gather_out_attr = {'src_out_port': 0,
                           'dst_in_port': edge_attr['dst_in_port']}
        if edge_attr and edge_attr.get('tensor', None) is not None and getattr(edge_attr['tensor'], 'value', None) is not None:
            out_tensor = np.take(edge_attr['tensor'].value, indices, axis=axis)
            gather_out_attr.update({'tensor': Tensor(value=out_tensor)})
        graph.add_edge(gather, dst, **gather_out_attr)
        gather_attr = {'name': gather, 'axis': axis}
        if type == 'Gather':
            gather_attr.update({'opset_version': 11})
        NodeWrap(graph, gather).replace_obj(type,  gather_attr)
        ret = gather
    else:
        WARN('[Parser]: Invalid params for insert_gather!')
    return ret


def insert_reshape(graph, src, dst, in_attr, dim, key=None, type='Reshape', data_format='NHWC'):
    ret = None
    if graph.has_node(src) and graph.has_node(dst) and dim:
        graph.remove_edge(src, dst, key=key)
        reshape = get_valid_node_name(graph, src + '_post_reshape')
        graph.add_node(reshape)
        reshape_attr = {'name': reshape, 'opset_version': 5}
        if type == 'Reshape':
            reshape_dim = np.array(dim, np.int32)
            const = get_valid_node_name(graph, reshape + '_shape')
            graph.add_node(const)
            const_attr = {'name': const, 'value': reshape_dim,
                          'data_format': data_format, 'opset_version': 9}
            NodeWrap(graph, const).replace_obj('Constant', const_attr)
            const_edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                               'tensor': Tensor(value=reshape_dim, is_const=True)}
            graph.add_edge(const, reshape, **const_edge_attr)
        else:
            reshape_attr.update({'dim': dim})
        NodeWrap(graph, reshape).replace_obj(type, reshape_attr)

        reshape_in_attr = copy.deepcopy(in_attr)
        reshape_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, reshape, **reshape_in_attr)

        reshape_out_attr = copy.deepcopy(in_attr)
        out_tensor = Tensor()
        if in_attr.get('tensor', None) is not None:
            out_tensor = copy.deepcopy(in_attr['tensor'])
            if in_attr['tensor'].value is not None:
                out_tensor.value = np.reshape(
                    in_attr['tensor'].value, newshape=dim)
                out_tensor.shape = out_tensor.value.shape
        reshape_out_attr.update({'src_out_port': 0, 'tensor': out_tensor})
        graph.add_edge(reshape, dst, **reshape_out_attr)
        ret = reshape
    else:
        WARN('[Parser]: Invalid params for insert_reshape!')
    return ret


def insert_reshape_after(graph, src, new_dim, old_dim=list(), out_port=0, type='Reshape'):
    ret = None
    if graph.has_node(src) and type in ('Reshape', 'ArmReshape'):
        if out_port == 0:
            reshape_name = src + '_post_reshape'
        else:
            reshape_name = src + '_post_reshape_' + str(out_port)
        reshape = get_valid_node_name(graph, reshape_name)
        graph.add_node(reshape)
        reshape_attr = {'name': reshape, 'opset_version': 5}
        if type == 'Reshape':
            reshape_dim = np.array(new_dim, np.int64)
            const = get_valid_node_name(graph, reshape + '_shape')
            graph.add_node(const)
            const_attr = {'name': const, 'value': reshape_dim,
                          'data_format': 'NHWC', 'opset_version': 9}
            NodeWrap(graph, const).replace_obj('Constant', const_attr)
            const_edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                               'tensor': Tensor(value=reshape_dim, is_const=True)}
            graph.add_edge(const, reshape, **const_edge_attr)
        else:
            reshape_attr.update({'dim': new_dim})
        NodeWrap(graph, reshape).replace_obj(type, reshape_attr)

        src_out_attr = {'src_out_port': out_port, 'dst_in_port': 0}
        out_edges = graph.sorted_out_edges(src, keys=True, data=True)
        for _, dst, key, out_attr in out_edges:
            if out_attr.get('src_out_port', 0) == out_port:
                graph.remove_edge(src, dst, key)
                new_out_attr = copy.deepcopy(out_attr)
                new_out_attr['src_out_port'] = 0
                graph.add_edge(reshape, dst, **new_out_attr)
                if new_out_attr.get('tensor', None) is not None \
                        and new_out_attr['tensor'].value is not None:
                    if old_dim:
                        new_src_out_tensor = np.reshape(
                            new_out_attr['tensor'].value, newshape=old_dim)
                        src_out_attr.update(
                            {'tensor': Tensor(value=new_src_out_tensor)})
                    elif new_dim and new_dim != list(new_out_attr['tensor'].value.shape):
                        new_src_out_tensor = np.reshape(
                            new_out_attr['tensor'].value, newshape=new_dim)
                        src_out_attr.update(
                            {'tensor': Tensor(value=new_src_out_tensor)})
        graph.add_edge(src, reshape, **src_out_attr)
        ret = reshape
    else:
        WARN('[Parser]: Invalid params for insert_reshape_after!')
    return ret


def place_reshape(graph, reshape, dim, data_format='NHWC'):
    if not isinstance(dim, np.ndarray):
        dim = np.array(dim, np.int32)
    NodeWrap(graph, reshape).replace_obj('Reshape',
                                         {'name': reshape,
                                          'opset_version': 5}
                                         )
    insert_constant(graph,
                    reshape + '_shape',
                    dim,
                    reshape,
                    in_port=1,
                    data_format=data_format)


def insert_slice(graph, src, dst, in_attr, begin, size, key=None, type='Slice', data_format='NHWC'):
    ret = None
    if graph.has_node(src) and graph.has_node(dst) and begin and size and type in ('Slice', 'ArmSlice'):
        graph.remove_edge(src, dst, key=key)
        slice = get_valid_node_name(graph, src + '_post_slice')
        graph.add_node(slice)
        slice_attr = {'name': slice}

        starts = np.array(begin, np.int32)
        size = np.array(size, np.int32)
        ends = starts + size

        if type == 'Slice':
            slice_attr.update({'opset_version': 10})
            insert_constant(graph, slice + '_starts', starts,
                            slice, in_port=1, data_format=data_format)
            insert_constant(graph, slice + '_ends', ends, slice,
                            in_port=2, data_format=data_format)
        else:
            slice_attr.update({'starts': starts.tolist(),
                               'ends': ends.tolist(),
                               'steps': [1] * starts.size
                               })
        NodeWrap(graph, slice).replace_obj(type, slice_attr)

        slice_in_attr = copy.deepcopy(in_attr)
        slice_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, slice, **slice_in_attr)

        slice_out_attr = copy.deepcopy(in_attr)
        slice_out_attr.update({'src_out_port': 0})
        graph.add_edge(slice, dst, **slice_out_attr)
        ret = slice
    else:
        WARN('[Parser]: Invalid params for insert_slice!')
    return ret


def insert_slice_after(graph, src, begin, size, out_port=0, type='Slice', data_format='NHWC'):
    ret = None
    if not (graph.has_node(src) and begin and size and type in ('Slice', 'ArmSlice')):
        WARN('[Parser]: Invalid params for insert_slice_after!')
        return ret
    if out_port == 0:
        name = src + '_post_slice'
    else:
        name = src + '_post_slice_' + str(out_port)
    slice_name = get_valid_node_name(graph, name)
    graph.add_node(slice_name)
    slice_attr = {'name': slice_name}

    starts = np.array(begin, np.int32)
    size = np.array(size, np.int32)
    ends = starts + size

    if type == 'Slice':
        slice_attr.update({'opset_version': 10})
        insert_constant(graph, slice_name + '_starts', starts,
                        slice_name, in_port=1, data_format=data_format)
        insert_constant(graph, slice_name + '_ends', ends,
                        slice_name, in_port=2, data_format=data_format)
    else:
        slice_attr.update({'starts': starts.tolist(),
                           'ends': ends.tolist(),
                           'steps': [1] * starts.size
                           })
    NodeWrap(graph, slice_name).replace_obj(type, slice_attr)

    src_out_attr = {'src_out_port': out_port, 'dst_in_port': 0}
    out_edges = graph.sorted_out_edges(src, data=True)
    for _, dst, out_attr in out_edges:
        if out_attr.get('src_out_port', 0) != out_port:
            continue
        graph.remove_edge(src, dst)
        new_out_attr = copy.deepcopy(out_attr)
        new_out_attr['src_out_port'] = 0
        graph.add_edge(slice_name, dst, **new_out_attr)
    graph.add_edge(src, slice_name, **src_out_attr)
    ret = slice_name
    return ret


def insert_tile(graph, src, dst, in_attr, reps, key=None, type='Tile', data_format='NHWC'):
    ret = None
    if graph.has_node(src) \
            and graph.has_node(dst) \
            and has_path(graph, src, dst) \
            and reps is not None \
            and ((isinstance(reps, (list, tuple)) and len(reps) > 0) or (isinstance(reps, np.ndarray) and reps.size > 0)) \
            and type in ('Tile', 'ArmTile'):
        if isinstance(reps, (list, tuple)):
            reps = np.array(reps, np.int32)
        graph.remove_edge(src, dst, key=key)
        tile = get_valid_node_name(graph, dst + '_pre_tile')
        graph.add_node(tile)

        tile_in_attr = copy.deepcopy(in_attr)
        tile_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, tile, **tile_in_attr)
        dst_in_attr = copy.deepcopy(in_attr)
        if dst_in_attr['tensor'].value is not None:
            tensor = Tensor(min_max=in_attr['tensor'].min_max, value=np.tile(
                dst_in_attr['tensor'].value, reps.tolist()))
        else:
            tensor = Tensor(min_max=in_attr['tensor'].min_max)
        dst_in_attr.update({'src_out_port': 0, 'tensor': tensor})
        graph.add_edge(tile, dst, **dst_in_attr)

        tile_attr = NodeWrap(graph, dst)['object'].copied_attr()
        if type == 'Tile':
            const = get_valid_node_name(graph, tile + '_reps')
            graph.add_node(const)
            const_edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                               'tensor': Tensor(value=reps, is_const=True)}
            graph.add_edge(const, tile, **const_edge_attr)
            NodeWrap(graph, const).replace_obj('Constant', {'name': const,
                                                            'value': reps,
                                                            'data_format': data_format,
                                                            'opset_version': 9
                                                            }
                                               )
            tile_attr.update({'name': tile, 'opset_version': 6})
            NodeWrap(graph, tile).replace_obj('Tile', tile_attr)
        else:
            tile_attr.update({'name': tile, 'reps': reps.tolist()})
            NodeWrap(graph, tile).replace_obj('ArmTile', tile_attr)

        ret = tile
    else:
        WARN('[Parser]: Invalid params for insert_tile!')
    return ret


def insert_transpose(graph, src, dst, in_attr, perm, key=None):
    ret = None
    if graph.has_node(src) \
            and graph.has_node(dst) \
            and perm is not None \
            and isinstance(perm, (list, np.ndarray)):
        if isinstance(perm, np.ndarray):
            perm = perm.tolist()
        if has_path(graph, src, dst):
            graph.remove_edge(src, dst, key=key)
        transpose = get_valid_node_name(graph, src + '_post_transpose')
        graph.add_node(transpose)
        transpose_attr = {'name': transpose, 'opset_version': 1}
        transpose_attr.update({'perm': perm})
        NodeWrap(graph, transpose).replace_obj('Transpose', transpose_attr)

        transpose_in_attr = copy.deepcopy(in_attr)
        transpose_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, transpose, **transpose_in_attr)

        transpose_out_attr = copy.deepcopy(in_attr)
        out_tensor = Tensor()
        if in_attr['tensor'] is not None:
            out_tensor = copy.deepcopy(in_attr['tensor'])
            if in_attr['tensor'].value is not None:
                out_tensor.value = np.transpose(
                    in_attr['tensor'].value, axes=perm)
                out_tensor.shape = out_tensor.value.shape
        transpose_out_attr.update({'src_out_port': 0, 'tensor': out_tensor})
        graph.add_edge(transpose, dst, **transpose_out_attr)
        ret = transpose
    else:
        WARN('[Parser]: Invalid params for insert_transpose!')
    return ret


def insert_transpose_after(graph, src, perm, port=0):
    ret = None
    if graph.has_node(src) \
            and perm is not None \
            and isinstance(perm, (list, np.ndarray)):
        if isinstance(perm, np.ndarray):
            perm = perm.tolist()
        transpose = get_valid_node_name(graph, src + '_post_transpose')
        found_port = False
        out_tensor = None
        for _, dst, out_attr in graph.sorted_out_edges(src, data=True):
            if out_attr['src_out_port'] == port:
                found_port = True
                new_out_attr = copy.deepcopy(out_attr)
                new_out_attr['src_out_port'] = 0
                graph.remove_edge(src, dst)
                graph.add_edge(transpose, dst, **new_out_attr)
                out_tensor = new_out_attr['tensor'].value
        if found_port:
            out_edge_attr = {'src_out_port': port, 'dst_in_port': 0}
            if out_tensor is not None:
                inverse_perm = Op.cal_inverse_perm(perm)
                out_tensor = np.transpose(out_tensor, inverse_perm)
                out_edge_attr.update({'tensor': Tensor(value=out_tensor)})
            graph.add_edge(src, transpose, **out_edge_attr)
            NodeWrap(graph, transpose).replace_obj('Transpose',
                                                   {'name': transpose,
                                                    'opset_version': 1,
                                                    'perm': perm
                                                    })
            ret = transpose
        else:
            WARN('[Parser]: Cannot find port=%d in insert_transpose_after!' % port)
    else:
        WARN('[Parser]: Invalid params for insert_transpose_after!')
    return ret


def apply_subgraph_plugin(graph):
    try:
        apply_named_subgraph_plugin(graph)
        apply_pattern_subgraph_plugin(graph)
    except Exception as e:
        import traceback
        WARN('Applying Subgraph plugin Failed. %s' % (str(e)))
        print(traceback.format_exc())


def merge_pattern_to_plugin(graph, plugin_node_optype, innodes, outnodes, match=None):
    assert len(
        outnodes) > 0, '[Parser]: Meet invalid outnodes in merge_pattern_to_plugin!'
    plugin_node = get_valid_node_name(graph, outnodes[0] + '_plugin')

    def add_plugin_in_edge(src, in_attr, in_port):
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr.update({'dst_in_port': in_port})
        graph.remove_edge(src, innode)
        graph.add_edge(src, plugin_node, **new_in_attr)

    all_nodes = set()
    for src in innodes:
        for dst in outnodes:
            if src == dst:
                all_nodes.add(src)
                continue
            for path in all_simple_paths(graph, src, dst):
                all_nodes.update(path)

    in_port = 0
    input_index_map = []
    for innode in innodes:
        input_map = []
        innode_in_edges = graph.sorted_in_edges(innode, data=True)
        graph.remove_edges_from(innode_in_edges)
        for src, _, in_attr in innode_in_edges:
            add_plugin_in_edge(src, in_attr, in_port)
            input_map.append(in_port)
            in_port += 1
        input_index_map.append(input_map)

    out_port = 0
    for outnode in outnodes:
        input_map = []
        outnode_in_edges = graph.sorted_in_edges(outnode, data=True)
        graph.remove_edges_from(outnode_in_edges)
        for src, _, in_attr in outnode_in_edges:
            if any((node == src or has_path(graph, node, src)) for node in innodes):
                continue
            add_plugin_in_edge(src, in_attr, in_port)
            input_map.append(in_port)
            in_port += 1
        input_index_map.append(input_map)
        outnode_out_edges = graph.sorted_out_edges(outnode, data=True)
        graph.remove_edges_from(outnode_out_edges)
        for _, dst, out_attr in outnode_out_edges:
            new_out_attr = copy.deepcopy(out_attr)
            new_out_attr.update({'src_out_port': out_port})
            graph.remove_edge(outnode, dst)
            graph.add_edge(plugin_node, dst, **new_out_attr)
            out_port += 1

    # get attributes before remove it
    attr_names_map = {}
    if graph._attr['framework'].name == 'TENSORFLOW':
        from ...tf.load import tf_attr_names_map
        attr_names_map.update({v: k for k, v in tf_attr_names_map.items()})

    attrs = {}
    for name in all_nodes:
        attrs[name] = {}
        for k, v in getattr(NodeWrap(graph, name)['object'],
                            '_attr',
                            {}).items():
            if k == 'keepdims':
                v.value = bool(v.value)
            attrs[name].update({k: v.value})
            if k in attr_names_map:
                attrs[name].update({attr_names_map[k]: v.value})

    if match:
        inverse_match = {v: k for k, v in match.items()}
        attrs = {inverse_match[k]: v for k, v in attrs.items()}

    new_attrs = NodeWrap(graph, outnodes[0])['object'].copied_attr()
    new_attrs.update(attrs)
    new_attrs.update({'name': plugin_node, '_nest_inputs': input_index_map})
    NodeWrap(graph, plugin_node).replace_obj(plugin_node_optype, new_attrs)

    for node in all_nodes:
        if node in graph._attr['output_names']:
            index = graph._attr['output_names'].index(node)
            if plugin_node not in graph._attr['output_names']:
                graph._attr['output_names'][index] = plugin_node
            else:
                graph._attr['output_names'].remove(index)
        remove_node_safely(graph, node)


def apply_pattern_subgraph_plugin(graph):
    pattern_subgraph = set()
    for name, plugin in PARSER_OP_DICT.items():
        if hasattr(plugin, '_subgraph_type') and plugin._subgraph_type == 'pattern_subgraph':
            pattern_subgraph.add(plugin)
    if not pattern_subgraph:
        return

    pattern_subgraph = list(pattern_subgraph)
    pattern_subgraph.sort(key=lambda x: x.priority, reverse=True)

    def get_op_name(optype):
        optype_prefix = {
            Framework.TFLITE: lambda x: 'Lite'+x,
            Framework.CAFFE: lambda x: 'Caffe'+x.upper(),
            Framework.TENSORFLOW: lambda x: 'Tf'+x,
        }

        framework = graph._attr.get('framework', Framework.NONE)
        op_name = optype_prefix[framework](optype) if \
            framework in optype_prefix else optype
        return op_name

    def get_io_nodes(nodes, edges):
        has_successors = set()
        has_precusors = set()
        for edge in edges:
            has_successors.add(edge[0])
            has_precusors.add(edge[1])
        inputs = []
        outputs = []
        for node in nodes:
            if node[0] not in has_precusors:
                inputs.append(node[0])
            if node[0] not in has_successors:
                outputs.append(node[0])
        return inputs, outputs

    for plugin in pattern_subgraph:
        innodes, outnodes = get_io_nodes(
            plugin.pattern_nodes, plugin.pattern_edges)

        nodes = [(name, {'op': get_op_name(optype)})
                 for name, optype in plugin.pattern_nodes]
        matches = matched_patterns(graph,
                                   nodes=nodes,
                                   edges=plugin.pattern_edges)
        for m in matches:
            innodes = [m[i] for i in innodes]
            outnodes = [m[i] for i in outnodes]
            merge_pattern_to_plugin(
                graph, plugin.op_type, innodes, outnodes, m)
            DEBUG('[Parser]: pattern based subgraph plugin applied: {[%s]->[%s]} merged to %s' %
                  (','.join([str(n) for n in innodes]), ','.join([str(n) for n in outnodes]), plugin.op_type))


def apply_named_subgraph_plugin(graph):
    named_subgraph = set()
    for _, plugin in PARSER_OP_DICT.items():
        if hasattr(plugin, '_subgraph_type') and plugin._subgraph_type == 'named_subgraph':
            named_subgraph.add(plugin)
    named_subgraph = list(named_subgraph)
    named_subgraph.sort(key=lambda x: x.priority, reverse=True)
    for plugin in named_subgraph:
        if not all(graph.has_node(n) for n in plugin.start_nodes+plugin.end_nodes):
            continue

        merge_pattern_to_plugin(
            graph, '.named_subgraph.'+plugin.op_type, plugin.start_nodes, plugin.end_nodes)
        DEBUG('[Parser]: name_based subgraph plugin applied: {[%s]->[%s]} merged to %s' % (
            ','.join(plugin.start_nodes), ','.join(plugin.end_nodes), plugin.op_type))


def record_output_tensors(graph):
    # using Out Op to record the output tensors order
    out_tensors = graph._attr['output_tensor_names']

    matches = single_node_matcher(graph, 'Out')
    out_nodes = [None]*len(out_tensors)
    for m in matches:
        node_name = m['target']
        if node_name in graph.nodes:
            try:
                _, _, _, info = graph.sorted_in_edges(
                    node_name, keys=True, data=True)[0]
                t = info.get('tensor', None)
                if t is not None and t.name in out_tensors:
                    idx = out_tensors.index(t.name)
                    out_nodes[idx] = node_name
            except Exception:
                pass
    graph._attr['output_nodes'] = out_nodes
