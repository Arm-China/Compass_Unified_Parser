# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
from ....ops.op import Op, OpHasWeights, OpHasBiases, KerasOp, BaseDeconvOp
from ....graph.node_wrap import NodeWrap
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL
from .common_passes import clear_redundant_nodes, FLOAT_EQUAL, insert_constant


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
                            if len(edge_attr['tensor'].min_max) == 2:
                                node_obj.biases_min_max = list(
                                    edge_attr['tensor'].min_max)
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


def convert_special_sequence_construct(graph):
    matches = single_node_matcher(graph, 'SequenceConstruct')
    for m in matches:
        seq_construct = m['target']
        seq_construct_obj = NodeWrap(graph, seq_construct)['object']
        if seq_construct_obj is None:
            ERROR(
                '[Parser]: Meets invalid SequenceConstruct Op (%s) in convert_special_sequence_construct!' % seq_construct)
            continue
        in_edges = graph.sorted_in_edges(seq_construct)
        if len(in_edges) != 1:
            WARN('[Parser]: Only supports SequenceConstruct Op (%s) with 1 input now, but got %d in convert_special_sequence_construct!' % (
                seq_construct, len(in_edges)))
            continue

        WARN('[Parser]: SequenceConstruct Op (%s) is unsupported and will be treated as Identity!' % seq_construct)
        identity_attr = seq_construct_obj.copied_attr()
        identity_attr.update({'opset_version': 1})
        NodeWrap(graph, seq_construct).replace_obj('Identity', identity_attr)


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


def merge_qconv(graph):
    if not graph._attr.get('quantize', False):
        return
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('x_dequant', {'op': 'DequantizeLinear'}),
                                   ('w_dequant', {'op': 'DequantizeLinear'}),
                                   ('b_dequant', {'op': 'DequantizeLinear'}),
                                   ('conv', {'op': 'Conv'}),
                                   ('y_quant', {'op': 'QuantizeLinear'}),
                               ],
                               edges=[
                                   ('x_dequant', 'conv'),
                                   ('w_dequant', 'conv', {'dst_in_port': 1}),
                                   ('b_dequant', 'conv', {'dst_in_port': 2}),
                                   ('conv', 'y_quant')
                               ])
    for m in matches:
        names = ['x_dequant', 'w_dequant', 'b_dequant', 'conv', 'y_quant']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(v is None for v in obj_dict.values()):
            error_node = [n for n in obj_dict if obj_dict[n] is None][0]
            ERROR('[Parser]: Meets invalid Op(%s) in merge_qconv!' % error_node)
            continue
        x_dequant_in_edges = graph.sorted_in_edges(m['x_dequant'], data=True)
        if len(x_dequant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid Dequantize Op(%s) in merge_qconv!' % m['x_dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in x_dequant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in x_dequant_in_edges[1:]):
            continue
        w_dequant_in_edges = graph.sorted_in_edges(m['w_dequant'], data=True)
        if len(w_dequant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid Dequantize Op(%s) in merge_qconv!' % m['w_dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in w_dequant_in_edges) \
                or any(not e[2]['tensor'].is_const for e in w_dequant_in_edges):
            continue
        b_dequant_in_edges = graph.sorted_in_edges(m['b_dequant'], data=True)
        if len(b_dequant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid Dequantize Op(%s) in merge_qconv!' % m['b_dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in b_dequant_in_edges) \
                or any(not e[2]['tensor'].is_const for e in b_dequant_in_edges):
            continue
        conv_out_edges = graph.sorted_out_edges(m['conv'], data=True)
        if len(conv_out_edges) != 1:
            continue
        y_quant_in_edges = graph.sorted_in_edges(m['y_quant'], data=True)
        if len(y_quant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid Quantize Op(%s) in merge_qconv!' % m['y_quant'])
            continue
        if any(e[2]['tensor'].value is None for e in y_quant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in y_quant_in_edges[1:]):
            continue

        matched = True

        src, _, in_attr = x_dequant_in_edges[0]
        x_scale, x_zp = obj_dict['x_dequant'].x_scale, obj_dict['x_dequant'].x_zero_point
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['tensor'].dtype = str(x_zp.dtype)
        new_in_attr['tensor'].scale_zp = (x_scale, x_zp)

        weights = w_dequant_in_edges[0][2]['tensor'].value
        w_scale, w_zp = obj_dict['w_dequant'].x_scale, obj_dict['w_dequant'].x_zero_point
        biases = b_dequant_in_edges[0][2]['tensor'].value
        b_scale, b_zp = obj_dict['b_dequant'].x_scale, obj_dict['b_dequant'].x_zero_point
        y_scale, y_zp = obj_dict['y_quant'].y_scale, obj_dict['y_quant'].y_zero_point

        graph.remove_edges_from(graph.sorted_in_edges(m['conv']))
        graph.remove_edge(m['conv'], m['y_quant'])
        graph.add_edge(src, m['conv'], **new_in_attr)
        for _, dst, out_attr in graph.sorted_out_edges(m['y_quant'], data=True):
            graph.remove_edge(m['y_quant'], dst)
            out_attr['tensor'].dtype = str(y_zp.dtype)
            out_attr['tensor'].scale_zp = (y_scale, y_zp)
            graph.add_edge(m['conv'], dst, **out_attr)

        if m['y_quant'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['y_quant'])
            graph._attr['output_names'][index] = m['conv']

        conv_attr = obj_dict['conv'].copied_attr()
        conv_attr.update({'quantize': True,
                          'weights': weights,
                          'weights_scale_zp': [w_scale, w_zp],
                          'biases': biases,
                          'biases_scale_zp': [b_scale, b_zp]
                          })
        NodeWrap(graph, m['conv']).replace_obj('Conv', conv_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_qmatmul(graph):
    if not graph._attr.get('quantize', False):
        return
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('a_dequant', {'op': 'DequantizeLinear'}),
                                   ('b_dequant', {'op': 'DequantizeLinear'}),
                                   ('matmul', {'op': 'MatMul'}),
                                   ('y_quant', {'op': 'QuantizeLinear'}),
                               ],
                               edges=[
                                   ('a_dequant', 'matmul'),
                                   ('b_dequant', 'matmul', {'dst_in_port': 1}),
                                   ('matmul', 'y_quant')
                               ])
    for m in matches:
        names = ['a_dequant', 'b_dequant', 'matmul', 'y_quant']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(v is None for v in obj_dict.values()):
            error_node = [n for n in obj_dict if obj_dict[n] is None][0]
            ERROR('[Parser]: Meets invalid Op(%s) in merge_qmatmul!' % error_node)
            continue
        a_dequant_in_edges = graph.sorted_in_edges(m['a_dequant'], data=True)
        if len(a_dequant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid Dequantize Op(%s) in merge_qmatmul!' % m['x_dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in a_dequant_in_edges[1:]):
            continue
        b_dequant_in_edges = graph.sorted_in_edges(m['b_dequant'], data=True)
        if len(b_dequant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid Dequantize Op(%s) in merge_qmatmul!' % m['b_dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in b_dequant_in_edges[1:]):
            continue
        matmul_out_edges = graph.sorted_out_edges(m['matmul'], data=True)
        if len(matmul_out_edges) != 1:
            continue
        y_quant_in_edges = graph.sorted_in_edges(m['y_quant'], data=True)
        if len(y_quant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid Quantize Op(%s) in merge_qmatmul!' % m['y_quant'])
            continue
        if any(e[2]['tensor'].value is None for e in y_quant_in_edges[1:]):
            continue

        matched = True
        a_zp = obj_dict['a_dequant'].x_zero_point
        b_zp = obj_dict['b_dequant'].x_zero_point
        y_zp = obj_dict['y_quant'].y_zero_point

        matmul_in_edges = graph.sorted_in_edges(m['matmul'])
        graph.remove_edges_from(matmul_in_edges)
        for src, _, in_attr in a_dequant_in_edges:
            new_in_attr = copy.deepcopy(in_attr)
            graph.add_edge(src, m['matmul'], **new_in_attr)
        if len(a_dequant_in_edges) == 2:
            insert_constant(graph, m['matmul'] + '_a_zero_point', a_zp, m['matmul'], in_port=2, data_format='NHWC')
        for src, _, in_attr in b_dequant_in_edges:
            new_in_attr = copy.deepcopy(in_attr)
            new_in_attr['dst_in_port'] += 3
            graph.add_edge(src, m['matmul'], **new_in_attr)
        if len(b_dequant_in_edges) == 2:
            insert_constant(graph, m['matmul'] + '_b_zero_point', b_zp, m['matmul'], in_port=5, data_format='NHWC')
        for src, _, in_attr in y_quant_in_edges[1:]:
            new_in_attr = copy.deepcopy(in_attr)
            new_in_attr['dst_in_port'] += 5
            graph.add_edge(src, m['matmul'], **new_in_attr)
        if len(y_quant_in_edges) == 2:
            insert_constant(graph, m['matmul'] + '_y_zero_point', y_zp, m['matmul'], in_port=7, data_format='NHWC')

        graph.remove_edge(m['matmul'], m['y_quant'])
        y_quant_out_edges = graph.sorted_out_edges(m['y_quant'], data=True)
        for _, dst, out_attr in y_quant_out_edges:
            graph.remove_edge(m['y_quant'], dst)
            graph.add_edge(m['matmul'], dst, **out_attr)

        if m['y_quant'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['y_quant'])
            graph._attr['output_names'][index] = m['matmul']

        matmul_attr = obj_dict['matmul'].copied_attr()
        matmul_attr.update({'opset_version': 10, 'quantize': True})
        NodeWrap(graph, m['matmul']).replace_obj('QLinearMatMul', matmul_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_q_multiple(graph, op_list):
    if not graph._attr.get('quantize', False):
        return
    if not op_list:
        return
    if not isinstance(op_list, (list, tuple)):
        op_list = [op_list]
    else:
        op_list = list(op_list)

    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('float_op', {'op': op_list}),
                                   ('quant', {'op': 'QuantizeLinear'}),
                               ],
                               edges=[
                                   ('float_op', 'quant')
                               ])
    for m in matches:
        in_edges = graph.sorted_in_edges(m['float_op'], data=True)
        if len(in_edges) < 1:
            ERROR('[Parser]: Meets invalid Concat Op(%s) in merge_q_multiple!' % m['float_op'])
            continue
        out_edges = graph.sorted_out_edges(m['float_op'], data=True)
        if len(out_edges) != 1:
            continue

        op_in_names = [e[0] for e in in_edges]
        names = op_in_names + [m['float_op'], m['quant']]
        obj_dict = {n: NodeWrap(graph, n)['object'] for n in names}
        if any(v is None for v in obj_dict.values()):
            error_node = [n for n in obj_dict if obj_dict[n] is None][0]
            ERROR('[Parser]: Meets invalid Op(%s) in merge_q_multiple!' % error_node)
            continue
        if any(obj_dict[n].type != 'DequantizeLinear' for n in op_in_names):
            continue

        found_invalid_dequant = False
        for dequant in op_in_names:
            dequant_in_edges = graph.sorted_in_edges(dequant, data=True)
            if len(dequant_in_edges) not in (2, 3):
                ERROR('[Parser]: Meets invalid Quantize Op(%s) in merge_q_multiple!' % dequant)
                found_invalid_dequant = True
                continue
            if any(e[2]['tensor'].value is None for e in dequant_in_edges[1:]) \
                    or any(not e[2]['tensor'].is_const for e in dequant_in_edges[1:]):
                found_invalid_dequant = True
                continue
        if found_invalid_dequant:
            continue

        quant_in_edges = graph.sorted_in_edges(m['quant'], data=True)
        if len(quant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid Quantize Op(%s) in merge_q_multiple!' % m['quant'])
            continue
        if any(e[2]['tensor'].value is None for e in quant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in quant_in_edges[1:]):
            continue

        matched = True

        y_scale, y_zp = obj_dict[m['quant']].y_scale, obj_dict[m['quant']].y_zero_point

        graph.remove_edges_from(in_edges)
        graph.remove_edge(m['float_op'], m['quant'])

        for i, dequant in enumerate(op_in_names):
            dequant_in_edges = graph.sorted_in_edges(dequant, data=True)
            src, _, in_attr = dequant_in_edges[0]
            new_in_attr = copy.deepcopy(in_attr)
            new_in_attr['dst_in_port'] = i
            x_scale, x_zp = obj_dict[dequant].x_scale, obj_dict[dequant].x_zero_point
            new_in_attr['tensor'].dtype = str(x_zp.dtype)
            new_in_attr['tensor'].scale_zp = (x_scale, x_zp)
            graph.add_edge(src, m['float_op'], **new_in_attr)

        for _, dst, out_attr in graph.sorted_out_edges(m['quant'], data=True):
            graph.remove_edge(m['quant'], dst)
            out_attr['tensor'].dtype = str(y_zp.dtype)
            out_attr['tensor'].scale_zp = (y_scale, y_zp)
            graph.add_edge(m['float_op'], dst, **out_attr)

        if m['quant'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['quant'])
            graph._attr['output_names'][index] = m['float_op']

        obj_dict[m['float_op']].quantize = True

    if matched:
        clear_redundant_nodes(graph)


def merge_q_unary(graph, op_list):
    if not graph._attr.get('quantize', False):
        return

    if not op_list:
        return
    if not isinstance(op_list, (list, tuple)):
        op_list = [op_list]
    else:
        op_list = list(op_list)

    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('dequant', {'op': 'DequantizeLinear'}),
                                   ('float_op', {'op': op_list}),
                                   ('quant', {'op': 'QuantizeLinear'}),
                               ],
                               edges=[
                                   ('dequant', 'float_op'),
                                   ('float_op', 'quant')
                               ])
    for m in matches:
        names = ['dequant', 'float_op', 'quant']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(v is None for v in obj_dict.values()):
            error_node = [n for n in obj_dict if obj_dict[n] is None][0]
            ERROR('[Parser]: Meets invalid Op(%s) in merge_q_unary!' % error_node)
            continue
        dequant_in_edges = graph.sorted_in_edges(m['dequant'], data=True)
        if len(dequant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid Dequantize Op(%s) in merge_q_unary!' % m['dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in dequant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in dequant_in_edges[1:]):
            continue

        op_in_edges = graph.sorted_in_edges(m['float_op'], data=True)
        if len(op_in_edges) < 1:
            ERROR('[Parser]: Meets invalid Op(%s) in merge_q_unary!' % m['float_op'])
            continue
        op_out_edges = graph.sorted_out_edges(m['float_op'], data=True)
        if len(op_out_edges) != 1:
            continue

        quant_in_edges = graph.sorted_in_edges(m['quant'], data=True)
        if len(quant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid Quantize Op(%s) in merge_q_unary!' % m['quant'])
            continue
        if any(e[2]['tensor'].value is None for e in quant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in quant_in_edges[1:]):
            continue

        matched = True

        x_scale, x_zp = obj_dict['dequant'].x_scale, obj_dict['dequant'].x_zero_point
        y_scale, y_zp = obj_dict['quant'].y_scale, obj_dict['quant'].y_zero_point

        src, _, in_attr = dequant_in_edges[0]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['tensor'].dtype = str(x_zp.dtype)
        new_in_attr['tensor'].scale_zp = (x_scale, x_zp)

        graph.remove_edges_from(op_in_edges[:1])
        graph.remove_edge(m['float_op'], m['quant'])
        graph.add_edge(src, m['float_op'], **new_in_attr)
        for _, dst, out_attr in graph.sorted_out_edges(m['quant'], data=True):
            graph.remove_edge(m['quant'], dst)
            out_attr['tensor'].dtype = str(y_zp.dtype)
            out_attr['tensor'].scale_zp = (y_scale, y_zp)
            graph.add_edge(m['float_op'], dst, **out_attr)
        if m['quant'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['quant'])
            graph._attr['output_names'][index] = m['float_op']
        obj_dict['float_op'].quantize = True

    if matched:
        clear_redundant_nodes(graph)


def merge_sequence_construct_and_at(graph):
    matched = False
    matches = two_nodes_matcher(graph, 'SequenceConstruct', 'SequenceAt')
    for m in matches:
        seq_construct, seq_at = m['begin'], m['end']
        seq_construct_obj = NodeWrap(graph, seq_construct)['object']
        seq_at_obj = NodeWrap(graph, seq_at)['object']
        construct_in_edges = graph.sorted_in_edges(seq_construct, data=True)
        seq_num = len(construct_in_edges)
        if seq_construct_obj is None or seq_at_obj is None or seq_num < 1:
            ERROR(
                '[Parser]: Meets invalid SequenceConstruct/SequenceAt Op in merge_sequence_construct_and_at!')
            continue
        at_in_edges = graph.sorted_in_edges(seq_at, data=True)
        if len(at_in_edges) != 2 or at_in_edges[1][2]['tensor'] is None \
                or not at_in_edges[1][2]['tensor'].is_const:
            WARN('[Parser]: Only supports SequenceAt Op (%s) with constant position in merge_sequence_construct_and_at!' % seq_construct)
            continue
        position = at_in_edges[1][2]['tensor'].value
        if position < 0:
            position = position + seq_num
        if position < 0 or position >= seq_num:
            ERROR(
                '[Parser]: Meets invalid position(%d) of SequenceAt Op (%s) in merge_sequence_construct_and_at!' % (position, seq_at))
            continue
        matched = True
        at_out_edges = graph.sorted_out_edges(seq_at, data=True)
        graph.remove_edges_from(at_out_edges)
        src, _, in_attr = construct_in_edges[position]
        for _, dst, out_attr in at_out_edges:
            dst_in_attr = copy.deepcopy(in_attr)
            dst_in_attr.update({'dst_in_port': out_attr['dst_in_port']})
            graph.add_edge(src, dst, **dst_in_attr)
        if seq_at in graph._attr['output_names']:
            index = graph._attr['output_names'].index(seq_at)
            graph._attr['output_names'][index] = src
    if matched:
        clear_redundant_nodes(graph)
