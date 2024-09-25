# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import copy
import numpy as np
from collections import OrderedDict
from networkx.algorithms import shortest_path_length
from ....common.defs import Tensor
from ....ops.op import Op, OpHasWeights, OpHasBiases, KerasOp, BaseDeconvOp, ConstLikeOp, OpHasOneOutPort
from ....graph.node_wrap import NodeWrap
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from ....graph.graph_algo import get_valid_node_name, determined_sort
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL, WARN_EXCEPTION
from .common_passes import clear_redundant_nodes, FLOAT_EQUAL, insert_constant, insert_reshape, insert_reshape_after, \
    insert_transpose, insert_transpose_after


def fuse_weights_const(graph):
    def _get_src_data(src_name, edge_attr):
        src_obj = NodeWrap(graph, src_name)['object']
        if src_obj.type in ('Constant', 'TfConst'):
            data = src_obj.value
        elif src_obj.type == 'DequantizeLinear' and edge_attr['tensor'].is_const:
            input_tensors = src_obj.get_input_tensors()
            data = None if len(input_tensors) < 1 else input_tensors[0]
            edge_attr['tensor'].scale_zp = [src_obj.x_scale, src_obj.x_zero_point]
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
            if node_obj.type in ('GRU', 'LSTM', 'QLinearConv', 'DeformConv'):
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
                                node_obj.weights_range = list(
                                    edge_attr['tensor'].min_max)
                            if len(edge_attr['tensor'].scale_zp) == 2:
                                node_obj.weights_scale_zp = list(
                                    edge_attr['tensor'].scale_zp)
                                node_obj.quantize = True
                        matched = True
                        graph.remove_edge(src_name, node_name, key=k)
                    elif i == biases_in_port and isinstance(data, np.ndarray):
                        node_obj.biases = data
                        if edge_attr.get('tensor', None) is not None:
                            if len(edge_attr['tensor'].min_max) == 2:
                                node_obj.biases_range = list(
                                    edge_attr['tensor'].min_max)
                            if len(edge_attr['tensor'].scale_zp) == 2:
                                node_obj.biases_scale_zp = list(
                                    edge_attr['tensor'].scale_zp)
                                node_obj.quantize = True
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
                            node_obj.weights_range = list(
                                edge_attr['tensor'].min_max)
                        if len(edge_attr['tensor'].scale_zp) == 2:
                            node_obj.weights_scale_zp = list(
                                edge_attr['tensor'].scale_zp)
                            node_obj.quantize = True
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


def decompose_loop(graph, params):
    matched = False
    matches = single_node_matcher(graph, 'Loop')
    for m in matches:
        loop = m['target']
        loop_obj = NodeWrap(graph, loop)['object']
        loop_in_edges = graph.sorted_in_edges(loop, data=True)
        loop_out_edges = graph.sorted_out_edges(loop, data=True)
        if loop_obj is not None \
                and len(loop_in_edges) >= 2 and len(loop_out_edges) >= 1 and \
                loop_in_edges[0][2]['tensor'].is_const and \
                loop_in_edges[0][2]['tensor'].value is not None and \
                loop_in_edges[1][2]['tensor'].is_const and \
                loop_in_edges[1][2]['tensor'].value is not None:

            condition = loop_in_edges[1][2]['tensor'].value

            N = len(loop_obj.body._attr['input_tensors']) - 2  # loop carried dependencies
            K = len(loop_obj.body._attr['output_names']) - 1 - N  # scan_outputs

            matched = True
            sub_graph_nodes = determined_sort(loop_obj.body, loop_obj.body._attr['output_names'])
            k_carried_dict = OrderedDict()
            for i in range(K):
                scan_outs_name = get_valid_node_name(graph, f'{loop}_scan_outs_{i}')
                k_carried_dict[scan_outs_name] = []

            graph.remove_edges_from(loop_in_edges)
            if not condition:
                loop_out_ports = loop_obj.get_out_ports()
                if any(p >= 2 for p in loop_out_ports) \
                        or (1 in loop_out_ports and len(loop_in_edges) != 3):
                    WARN('[Parser]: Meets unsupported Loop Node(%s) in decompose_const_loop!' % loop)
                    continue
                const = None
                if 1 in loop_out_ports:
                    v_initial, _, v_initial_in_attr = loop_in_edges[2]
                    shape = get_valid_node_name(graph, loop + '_shape')
                    shape_in_attr = copy.deepcopy(v_initial_in_attr)
                    shape_in_attr.update({'dst_in_port': 0})
                    graph.add_edge(v_initial, shape, **shape_in_attr)
                    concat = get_valid_node_name(graph, shape + '_concat')
                    graph.add_edge(shape, concat, **{'src_out_port': 0, 'dst_in_port': 1})
                    insert_constant(graph, concat + '_zero', np.array([0], dtype=np.int64), concat, in_port=0)
                    const = get_valid_node_name(graph, shape + '_const')
                    graph.add_edge(concat, const)
                    NodeWrap(graph, shape).replace_obj('Shape', {'name': shape, 'opset_version': 1})
                    NodeWrap(graph, concat).replace_obj('Concat', {'name': concat, 'opset_version': 13, 'axis': 0})
                    NodeWrap(graph, const).replace_obj('ConstantOfShape', {'name': const, 'opset_version': 9})
                for _, dst, out_attr in loop_out_edges:
                    graph.remove_edge(loop, dst)
                    if out_attr['src_out_port'] == 0:
                        graph.add_edge(loop_in_edges[2][0], dst, **out_attr)
                    else:
                        const_out_attr = copy.deepcopy(out_attr)
                        const_out_attr.update({'src_out_port': 0})
                        graph.add_edge(const, dst, **const_out_attr)
                if loop in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(loop)
                    if loop_in_edges[-1][0] not in graph._attr['output_names']:
                        graph._attr['output_names'][index] = loop_in_edges[-1][0]
                    else:
                        graph._attr['output_names'].pop(index)
                    if const is not None:
                        WARN('[Parser]: The output of Node(%s) has zero shape, which will be removed from graph!' % loop)
                # clear subgraph
                if loop in graph._attr['subgraphs']:
                    graph._attr['subgraphs'].pop(loop)
                continue

            if loop_obj.real_loop_cnt is None:
                continue
            loop_cnt = loop_obj.real_loop_cnt
            last_loop_res = OrderedDict()
            sub_main_node_map = {}
            for i in range(loop_cnt):
                for n in sub_graph_nodes:
                    n_obj = loop_obj.body.nodes[n]['object']
                    if n_obj is None:
                        ERROR(
                            f'[Parser]: Meet invalid Node({n}) of root node({loop}) in decompose_loop.')
                    n_in_edges = loop_obj.body.sorted_in_edges(n, data=True)

                    if n_obj.type in ['Input', 'DummyInput']:
                        if n_obj.type == 'Input' and list(loop_obj.body._attr['input_tensors'].keys()).index(n) == 0:
                            # iter_num
                            iter_num_node_name = get_valid_node_name(graph, f'{loop}_iter_{i}')
                            graph.add_node(iter_num_node_name)
                            sub_main_node_map[n] = iter_num_node_name
                            iter_value = np.array(i, dtype=np.int64)
                            cur_obj_attr = n_obj.copied_attr()
                            cur_obj_attr.update({'in_subgraph': False, 'name': iter_num_node_name,
                                                 'value': iter_value})
                            NodeWrap(graph, iter_num_node_name).replace_obj('Constant', cur_obj_attr)
                        continue
                    elif n_obj.type == 'Constant':
                        sub_main_node_map[n] = n
                        if not graph.has_node(n):
                            graph.add_node(n)
                            cur_obj_attr = n_obj.copied_attr()
                            cur_obj_attr.update({'in_subgraph': False})
                            NodeWrap(graph, n).replace_obj('Constant', cur_obj_attr)
                    else:
                        main_g_node_name = get_valid_node_name(graph, n)
                        graph.add_node(main_g_node_name)
                        sub_main_node_map[n] = main_g_node_name
                        cur_obj_attr = n_obj.copied_attr()
                        cur_obj_attr.update({'in_subgraph': False, 'name': main_g_node_name})
                        if n_obj.type.startswith('Plugin'):
                            NodeWrap(graph, main_g_node_name).replace_obj(n_obj.type[6:], cur_obj_attr)
                        else:
                            NodeWrap(graph, main_g_node_name).replace_obj(n_obj.type, cur_obj_attr)
                        for in_e in n_in_edges:
                            src, dst, n_in_attr = in_e
                            src_obj = loop_obj.body.nodes[src]['object']
                            if src_obj.type == 'Input':
                                assert src in loop_obj.body._attr['input_tensors'], f'{src} is Input but not in subgraph input tensors.'
                                inp_idx = list(loop_obj.body._attr['input_tensors'].keys()).index(src)
                                if i == 0:
                                    if inp_idx == 0:
                                        in_attr = copy.deepcopy(n_in_attr)
                                        graph.add_edge(sub_main_node_map[src], main_g_node_name, **in_attr)
                                    else:
                                        in_attr = copy.deepcopy(loop_in_edges[inp_idx][-1])
                                        in_attr['dst_in_port'] = n_in_attr['dst_in_port']
                                        graph.add_edge(loop_in_edges[inp_idx][0], main_g_node_name, **in_attr)
                                else:
                                    in_attr = copy.deepcopy(n_in_attr)
                                    if inp_idx == 0:
                                        graph.add_edge(sub_main_node_map[src], main_g_node_name, **in_attr)
                                    else:
                                        graph.add_edge(last_loop_res[inp_idx - 1], main_g_node_name, **in_attr)
                            elif src_obj.type == 'DummyInput':
                                assert graph.has_node(src), f'{src} is DummyInput but not in main graph.'
                                in_attr = copy.deepcopy(n_in_attr)
                                graph.add_edge(src, main_g_node_name, **in_attr)
                            elif src_obj.type == 'Constant':
                                if not graph.has_node(src):
                                    graph.add_node(src)
                                    sub_main_node_map[src] = src
                                    cur_obj_attr = src_obj.copied_attr()
                                    cur_obj_attr.update({'in_subgraph': False})
                                    NodeWrap(graph, src).replace_obj('Constant', cur_obj_attr)
                                else:
                                    in_attr = copy.deepcopy(n_in_attr)
                                    graph.add_edge(src, main_g_node_name, **in_attr)
                            else:
                                in_attr = copy.deepcopy(n_in_attr)
                                graph.add_edge(sub_main_node_map[src], main_g_node_name, **in_attr)

                    if n in loop_obj.body._attr['output_names']:
                        # 1+N+K
                        out_idx = loop_obj.body._attr['output_names'].index(n)
                        if out_idx < 1 + N:
                            last_loop_res[out_idx] = sub_main_node_map[n]
                        else:
                            scan_outs_name = list(k_carried_dict.keys())[out_idx - 1 - N]
                            k_carried_dict[scan_outs_name].append(sub_main_node_map[n])

            graph.remove_edges_from(loop_out_edges)

            # Loop have N + K outputs
            for i in range(N):
                _, dst, out_edge = loop_out_edges[i]
                out_attr = copy.deepcopy(out_edge)
                out_attr['src_out_port'] = 0
                graph.add_edge(last_loop_res[i + 1], dst, **out_attr)
            for i in range(K):
                scan_outs_name = list(k_carried_dict.keys())[i]
                graph.add_node(scan_outs_name)
                cur_obj_attr = {'name': scan_outs_name,
                                'opset_version': 11,
                                'axis': 0,
                                'new_axis': 1,
                                'in_subgraph': False}
                NodeWrap(graph, scan_outs_name).replace_obj('ConcatFromSequence', cur_obj_attr)
                _, _, out_attr = loop_obj.body.sorted_out_edges(k_carried_dict[scan_outs_name][0], data=True)[0]
                for idx, src in enumerate(k_carried_dict[scan_outs_name]):
                    in_attr = copy.deepcopy(out_attr)
                    in_attr['dst_in_port'] = idx
                    graph.add_edge(src, scan_outs_name, **in_attr)
                _, dst, out_attr = loop_out_edges[i + N]
                new_out_attr = copy.deepcopy(out_attr)
                new_out_attr['src_out_port'] = 0
                graph.add_edge(scan_outs_name, dst, **new_out_attr)

            if loop in graph._attr['output_names']:
                index = graph._attr['output_names'].index(loop)
                loop_outputs = []
                # N+K outputs
                graph._attr['output_names'].pop(index)
                for i in range(N):
                    loop_outputs.append(last_loop_res[i + 1])
                for i in range(K):
                    loop_outputs.append(list(k_carried_dict.keys())[i])
                graph._attr['output_names'][index:index] = loop_outputs

            # clear subgraph
            if loop in graph._attr['subgraphs']:
                graph._attr['subgraphs'].pop(loop)

    if matched:
        clear_redundant_nodes(graph)


def convert_special_sequence_construct(graph):
    '''Add Out node after inputs of sequence_construct and update graph outputs if
    the sequence_construct node is graph output.
    sequence_construct will be removed by clear_redundant_nodes if there is no path
    between it and other graph output, or be processed and removed by other passes
    otherwise.
    '''
    matched = False
    matches = single_node_matcher(graph, 'SequenceConstruct')
    for m in matches:
        seq_construct = m['target']
        seq_construct_obj = NodeWrap(graph, seq_construct)['object']
        if seq_construct_obj is None:
            ERROR(
                '[Parser]: Meets invalid SequenceConstruct Op (%s) in convert_special_sequence_construct!' % seq_construct)
            continue
        if seq_construct not in graph._attr['output_names']:
            continue
        matched = True
        WARN('[Parser]: SequenceConstruct Op (%s) will be converted to deconstructed tensors in graph outputs!' % seq_construct)
        index = graph._attr['output_names'].index(seq_construct)
        graph._attr['output_names'].pop(index)
        in_edges = graph.sorted_in_edges(seq_construct, data=True)
        for idx, (name, _, in_attr) in enumerate(in_edges):
            out_name = get_valid_node_name(graph, name + '_out')
            out_in_attr = copy.deepcopy(in_attr)
            out_in_attr.update({'dst_in_port': 0})
            graph.add_edge(name, out_name, **out_in_attr)
            NodeWrap(graph, out_name).replace_obj('Out', {'name': out_name})
            graph._attr['output_names'].insert(index + idx, name)
    if matched:
        clear_redundant_nodes(graph)


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


def convert_mmcv_deform_conv(graph):
    matches = single_node_matcher(graph, 'MMCVModulatedDeformConv2d')
    for m in matches:
        deform_conv = m['target']
        deform_conv_obj = NodeWrap(graph, deform_conv)['object']
        in_edges = graph.sorted_in_edges(deform_conv, data=True)
        if deform_conv_obj is None or len(in_edges) < 4:
            ERROR('[Parser]: Meets invalid MMCVModulatedDeformConv2d Op(%s) in convert_mmcv_deform_conv!' % deconv)
            continue
        graph.remove_edges_from(in_edges[1:])
        # inputs of MMCVModulatedDeformConv2d: input, offset, mask, weight, bias(optional)
        offset, _, offset_in_attr = in_edges[1]
        mask, _, mask_in_attr = in_edges[2]
        weight, _, weight_in_attr = in_edges[3]
        # inputs of onnx DeformConv: input, weight, offset, bias, mask
        weight_in_attr.update({'dst_in_port': 1})
        graph.add_edge(weight, deform_conv, **weight_in_attr)
        offset_in_attr.update({'dst_in_port': 2})
        graph.add_edge(offset, deform_conv, **offset_in_attr)
        if len(in_edges) > 4:
            bias, _, bias_in_attr = in_edges[4]
            bias_in_attr.update({'dst_in_port': 3})
            graph.add_edge(bias, deform_conv, **bias_in_attr)
        else:
            bias_value = np.zeros(deform_conv_obj.num_output, np.float32)
            insert_constant(graph, deform_conv + '_bias', bias_value, deform_conv, in_port=3)
        mask_in_attr.update({'dst_in_port': 4})
        graph.add_edge(mask, deform_conv, **mask_in_attr)
        deform_conv_attr = deform_conv_obj.copied_attr()
        deform_conv_attr.update({'offset_group': deform_conv_obj.deform_groups, 'opset_version': 19})
        NodeWrap(graph, deform_conv).replace_obj('DeformConv', deform_conv_attr)


def uplift_quant(graph):
    '''For DequantizeLinear+Gemm/Conv/...+Relu/...+QuantizeLinear, switch QuantizeLinear and Relu(quantized).
    Note: 1) src type could be other types(only if Relu will be 'with_activation' in IR);
          2) float_op could be other activations op or float op whose input/output have same scale/zp.
    '''
    if not graph._attr.get('quantize', False):
        return
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('x_dequant', {'op': 'DequantizeLinear', 'unique': False}),
                                   ('src', {'op': ['Conv', 'ConvTranspose', 'Gemm']}),
                                   ('float_op', {'op': ['LeakyRelu', 'Relu']}),
                                   ('quant', {'op': 'QuantizeLinear'}),
                               ],
                               edges=[
                                   ('x_dequant', 'src', {'dst_in_port': 0}),
                                   ('src', 'float_op', {'dst_in_port': 0}),
                                   ('float_op', 'quant', {'dst_in_port': 0}),
                               ])
    for m in matches:
        float_op, quant, src = m['float_op'], m['quant'], m['src']
        float_op_obj = NodeWrap(graph, float_op)['object']
        quant_obj = NodeWrap(graph, quant)['object']
        float_op_in_edges = graph.sorted_in_edges(float_op, data=True)
        if float_op_obj is None or len(float_op_in_edges) != 1:
            ERROR('[Parser]: Meets invalid node(%s) in uplift_quant!' % float_op)
            continue
        if quant_obj is None:
            ERROR('[Parser]: Meets invalid QuantizeLinear node(%s) in uplift_quant!' % quant)
            continue
        float_op_out_edges = graph.sorted_out_edges(float_op, data=True)
        if len(float_op_out_edges) != 1:
            continue
        quant_out_edges = graph.sorted_out_edges(quant, data=True)
        if len(quant_out_edges) < 1:
            continue
        quant_in_edges = graph.sorted_in_edges(quant, data=True)
        if len(quant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid QuantizeLinear Op(%s) in uplift_quant!' % quant)
            continue
        if any(e[2]['tensor'].value is None for e in quant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in quant_in_edges[1:]):
            continue
        matched = True
        graph.remove_edges_from(float_op_in_edges + quant_out_edges)
        graph.remove_edge(float_op, quant)
        src, _, in_attr = float_op_in_edges[0]
        graph.add_edge(src, quant, **in_attr)
        quant_out_attr = copy.deepcopy(quant_out_edges[0][2])
        quant_out_attr.update({'dst_in_port': 0})
        y_scale, y_zp = quant_obj.y_scale, quant_obj.y_zero_point
        quant_out_attr['tensor'].dtype = str(y_zp.dtype)
        quant_out_attr['tensor'].scale_zp = (y_scale, y_zp)
        quant_out_attr['tensor'].activation_quantization_axis = quant_obj.axis
        graph.add_edge(quant, float_op, **quant_out_attr)
        for _, dst, out_attr in quant_out_edges:
            new_out_attr = copy.deepcopy(quant_out_attr)
            new_out_attr.update({'dst_in_port': out_attr['dst_in_port']})
            graph.add_edge(float_op, dst, **new_out_attr)
        float_op_obj.quantize = True
        if quant in graph._attr['output_names']:
            index = graph._attr['output_names'].index(quant)
            graph._attr['output_names'][index] = float_op
    if matched:
        clear_redundant_nodes(graph)


def uplift_quant_through_concat(graph):
    '''Convert x1/x2/...->Concat->QuantizeLinear to the following pattern so that
    QuantizeLinear can merge with other nodes before Concat(x1/x2/...):
          x1             x2           ...
          |              |            ...
    QuantizeLinear QuantizeLinear     ...
               \         |           /
               Concat(quantized)
    '''
    if not graph._attr.get('quantize', False):
        return
    matched = False
    matches = two_nodes_matcher(graph, 'Concat', 'QuantizeLinear')
    for m in matches:
        float_op, quant = m['begin'], m['end']
        float_obj = NodeWrap(graph, float_op)['object']
        quant_obj = NodeWrap(graph, quant)['object']
        if float_obj is None or quant_obj is None:
            ERROR('[Parser]: Meets invalid nodes in uplift_quant_through_concat!')
            continue
        float_out_edges = graph.sorted_out_edges(float_op, data=True)
        if len(float_out_edges) != 1:
            continue
        quant_in_edges = graph.sorted_in_edges(quant, data=True)
        if len(quant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid QuantizeLinear Op(%s) in uplift_quant_through_concat!' % quant)
            continue
        if any(e[2]['tensor'].value is None for e in quant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in quant_in_edges[1:]):
            continue
        matched = True
        quant_attr = quant_obj.copied_attr()
        y_scale_val, y_zp_val, axis = quant_obj.y_scale, quant_obj.y_zero_point, quant_obj.axis
        y_scale, _, y_scale_in_attr = quant_in_edges[1]
        y_zp, y_zp_in_attr = None, None
        if len(quant_in_edges) == 3:
            y_zp, _, y_zp_in_attr = quant_in_edges[2]
        quant_out_edges = graph.sorted_out_edges(quant, data=True)
        float_in_edges = graph.sorted_in_edges(float_op, data=True)
        graph.remove_edges_from(quant_in_edges + quant_out_edges + float_in_edges)
        for idx, (src, _, in_attr) in enumerate(float_in_edges):
            pre_quant = get_valid_node_name(graph, src + '_quant' + str(idx))
            src_out_attr = copy.deepcopy(in_attr)
            src_out_attr.update({'dst_in_port': 0})
            graph.add_edge(src, pre_quant, **src_out_attr)
            graph.add_edge(y_scale, pre_quant, **y_scale_in_attr)
            if y_zp is not None:
                graph.add_edge(y_zp, pre_quant, **y_zp_in_attr)
            in_attr.update({'src_out_port': 0})
            in_attr['tensor'].dtype = str(y_zp_val.dtype)
            in_attr['tensor'].scale_zp = (y_scale_val, y_zp_val)
            in_attr['tensor'].activation_quantization_axis = axis
            graph.add_edge(pre_quant, float_op, **in_attr)
            pre_quant_attr = copy.deepcopy(quant_attr)
            pre_quant_attr.update({'name': pre_quant})
            NodeWrap(graph, pre_quant).replace_obj('QuantizeLinear', pre_quant_attr)
        for _, dst, out_attr in quant_out_edges:
            out_attr['tensor'].dtype = str(y_zp_val.dtype)
            out_attr['tensor'].scale_zp = (y_scale_val, y_zp_val)
            out_attr['tensor'].activation_quantization_axis = axis
            graph.add_edge(float_op, dst, **out_attr)
        float_obj.quantize = True
        if quant in graph._attr['output_names']:
            index = graph._attr['output_names'].index(quant)
            graph._attr['output_names'][index] = float_op
    if matched:
        clear_redundant_nodes(graph)


def merge_qconv(graph):
    if not graph._attr.get('quantize', False):
        return
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('x_dequant', {'op': 'DequantizeLinear', 'unique': False}),
                                   ('w_dequant', {'op': 'DequantizeLinear', 'unique': False}),
                                   ('b_dequant', {'op': 'DequantizeLinear', 'unique': False}),
                                   ('conv', {'op': ['Conv', 'ConvTranspose']}),
                                   ('y_quant', {'op': 'QuantizeLinear'}),
                               ],
                               edges=[
                                   ('x_dequant', 'conv'),
                                   ('w_dequant', 'conv', {'dst_in_port': 1}),
                                   ('b_dequant', 'conv', {'dst_in_port': 2}),
                                   ('conv', 'y_quant')
                               ])
    matches_with_bias = matched_patterns(graph,
                                         nodes=[
                                             ('x_dequant', {
                                              'op': 'DequantizeLinear', 'unique': False}),
                                             ('w_dequant', {
                                              'op': 'DequantizeLinear', 'unique': False}),
                                             ('bias', {
                                              'op': 'Constant', 'unique': False}),
                                             ('conv', {'op': 'Conv'}),
                                             ('y_quant', {
                                              'op': 'QuantizeLinear'}),
                                         ],
                                         edges=[
                                             ('x_dequant', 'conv'),
                                             ('w_dequant', 'conv',
                                              {'dst_in_port': 1}),
                                             ('bias', 'conv',
                                              {'dst_in_port': 2}),
                                             ('conv', 'y_quant'),
                                         ])
    for m in matches + matches_with_bias:
        is_float_bias = ('bias' in m)
        names = ['x_dequant', 'w_dequant', 'conv',
                 'y_quant'] + (['bias'] if is_float_bias else ['b_dequant'])
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(v is None for v in obj_dict.values()):
            error_node = [n for n in obj_dict if obj_dict[n] is None][0]
            ERROR('[Parser]: Meets invalid Op(%s) in merge_qconv!' % error_node)
            continue
        x_dequant_in_edges = graph.sorted_in_edges(m['x_dequant'], data=True)
        if len(x_dequant_in_edges) not in (2, 3):
            ERROR(
                '[Parser]: Meets invalid Dequantize Op(%s) in merge_qconv!' % m['x_dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in x_dequant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in x_dequant_in_edges[1:]):
            continue
        w_dequant_in_edges = graph.sorted_in_edges(m['w_dequant'], data=True)
        if len(w_dequant_in_edges) not in (2, 3):
            ERROR(
                '[Parser]: Meets invalid Dequantize Op(%s) in merge_qconv!' % m['w_dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in w_dequant_in_edges) \
                or any(not e[2]['tensor'].is_const for e in w_dequant_in_edges):
            continue
        conv_out_edges = graph.sorted_out_edges(m['conv'], data=True)
        if len(conv_out_edges) != 1:
            continue
        y_quant_in_edges = graph.sorted_in_edges(m['y_quant'], data=True)
        if len(y_quant_in_edges) not in (2, 3):
            ERROR('[Parser]: Meets invalid Quantize Op(%s) in merge_qconv!' %
                  m['y_quant'])
            continue
        if any(e[2]['tensor'].value is None for e in y_quant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in y_quant_in_edges[1:]):
            continue

        src, _, in_attr = x_dequant_in_edges[0]
        x_scale, x_zp = obj_dict['x_dequant'].x_scale, obj_dict['x_dequant'].x_zero_point
        w_scale, w_zp = obj_dict['w_dequant'].x_scale, obj_dict['w_dequant'].x_zero_point
        y_scale, y_zp = obj_dict['y_quant'].y_scale, obj_dict['y_quant'].y_zero_point
        weights = w_dequant_in_edges[0][2]['tensor'].value
        if not is_float_bias:
            b_dequant_in_edges = graph.sorted_in_edges(m['b_dequant'], data=True)
            if len(b_dequant_in_edges) not in (2, 3):
                ERROR(
                    '[Parser]: Meets invalid Dequantize Op(%s) in merge_qconv!' % m['b_dequant'])
                continue
            if any(e[2]['tensor'].value is None for e in b_dequant_in_edges) \
                    or any(not e[2]['tensor'].is_const for e in b_dequant_in_edges):
                continue
            b_scale, b_zp = obj_dict['b_dequant'].x_scale, obj_dict['b_dequant'].x_zero_point
            if not FLOAT_EQUAL(w_scale * x_scale, b_scale) or not np.all(b_zp == 0):
                continue
            biases = b_dequant_in_edges[0][2]['tensor'].value
        else:
            b_scale, b_zp = None, None
            biases = obj_dict['bias'].value

        matched = True
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['tensor'].dtype = str(x_zp.dtype)
        new_in_attr['tensor'].scale_zp = (x_scale, x_zp)
        new_in_attr['tensor'].activation_quantization_axis = obj_dict['x_dequant'].axis
        graph.remove_edges_from(
            graph.sorted_in_edges(m['conv']) + conv_out_edges)
        graph.add_edge(src, m['conv'], **new_in_attr)
        for _, dst, out_attr in graph.sorted_out_edges(m['y_quant'], data=True):
            graph.remove_edge(m['y_quant'], dst)
            out_attr['tensor'].dtype = str(y_zp.dtype)
            out_attr['tensor'].scale_zp = (y_scale, y_zp)
            out_attr['tensor'].activation_quantization_axis = obj_dict['y_quant'].axis
            graph.add_edge(m['conv'], dst, **out_attr)

        if m['y_quant'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['y_quant'])
            graph._attr['output_names'][index] = m['conv']

        conv_attr = obj_dict['conv'].copied_attr()
        conv_attr.update({'quantize': True})
        if obj_dict['conv'].type == 'Conv':
            if is_float_bias:
                op_type = 'Conv'
                conv_attr.update({'opset_version': 11,
                                  'weights': weights, 'weights_scale_zp': [w_scale, w_zp],
                                  'biases': biases, 'biases_scale_zp': [b_scale, b_zp]})
            else:
                op_type = 'QLinearConv'
                conv_attr.update({'opset_version': 10})
                insert_constant(graph, m['conv'] + '_x_scale',
                                x_scale, m['conv'], in_port=1, data_format='NHWC')
                insert_constant(graph, m['conv'] + '_x_zero_point',
                                x_zp, m['conv'], in_port=2, data_format='NHWC')
                insert_constant(graph, m['conv'] + '_w', weights,
                                m['conv'], in_port=3, data_format='NHWC')
                insert_constant(graph, m['conv'] + '_w_scale',
                                w_scale, m['conv'], in_port=4, data_format='NHWC')
                insert_constant(graph, m['conv'] + '_w_zero_point',
                                w_zp, m['conv'], in_port=5, data_format='NHWC')
                insert_constant(graph, m['conv'] + '_y_scale',
                                y_scale, m['conv'], in_port=6, data_format='NHWC')
                insert_constant(graph, m['conv'] + '_y_zero_point',
                                y_zp, m['conv'], in_port=7, data_format='NHWC')
                insert_constant(graph, m['conv'] + '_B', biases,
                                m['conv'], in_port=8, data_format='NHWC')
        else:
            op_type = 'ConvTranspose'
            conv_attr.update({'opset_version': 11,
                              'weights': weights, 'weights_scale_zp': [w_scale, w_zp],
                              'biases': biases, 'biases_scale_zp': [b_scale, b_zp]})

        NodeWrap(graph, m['conv']).replace_obj(op_type, conv_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_qgemm(graph):
    # Merge patterns into QGemmMs. This pass should be done after infer(to get input/output
    # shapes for the inserted Reshape nodes) and before fuse_const(avoid DequantizeLinear
    # being merged into Constant).
    if not graph._attr.get('quantize', False):
        return
    matched = False
    matches_non_2d = matched_patterns(graph,
                                      nodes=[
                                          ('x_dequant', {'op': 'DequantizeLinear', 'unique': False}),
                                          ('w_dequant', {'op': 'DequantizeLinear', 'unique': False}),
                                          ('trans', {'op': 'Transpose'}),
                                          ('b_dequant', {'op': 'DequantizeLinear', 'unique': False}),
                                          ('gemm', {'op': 'MatMul'}),
                                          ('add', {'op': 'Add'}),
                                          ('y_quant', {'op': 'QuantizeLinear'}),
                                      ],
                                      edges=[
                                          ('x_dequant', 'gemm', {'dst_in_port': 0}),
                                          ('w_dequant', 'trans'),
                                          ('trans', 'gemm', {'dst_in_port': 1}),
                                          ('b_dequant', 'add'),
                                          ('gemm', 'add'),
                                          ('add', 'y_quant')
                                      ])
    matches_non_2d_with_relu = matched_patterns(graph,
                                                nodes=[
                                                    ('x_dequant', {'op': 'DequantizeLinear', 'unique': False}),
                                                    ('w_dequant', {'op': 'DequantizeLinear', 'unique': False}),
                                                    ('trans', {'op': 'Transpose'}),
                                                    ('b_dequant', {'op': 'DequantizeLinear', 'unique': False}),
                                                    ('gemm', {'op': 'MatMul'}),
                                                    ('add', {'op': 'Add'}),
                                                    ('relu', {'op': 'Relu'}),
                                                    ('y_quant', {'op': 'QuantizeLinear'}),
                                                ],
                                                edges=[
                                                    ('x_dequant', 'gemm', {'dst_in_port': 0}),
                                                    ('w_dequant', 'trans'),
                                                    ('trans', 'gemm', {'dst_in_port': 1}),
                                                    ('b_dequant', 'add'),
                                                    ('gemm', 'add'),
                                                    ('add', 'relu'),
                                                    ('relu', 'y_quant')
                                                ])
    for m in matches_non_2d + matches_non_2d_with_relu:
        names = ['x_dequant', 'w_dequant', 'b_dequant', 'gemm', 'add', 'y_quant', 'trans'] \
            + (['relu'] if 'relu' in m else [])
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(v is None for v in obj_dict.values()):
            error_node = [n for n in obj_dict if obj_dict[n] is None][0]
            ERROR('[Parser]: Meets invalid Op(%s) in merge_qgemm!' % error_node)
            continue
        x_dequant_in_edges = graph.sorted_in_edges(m['x_dequant'], data=True)
        x_dequant_in_shapes = obj_dict['x_dequant'].get_input_shapes()
        if len(x_dequant_in_edges) not in (2, 3) or len(x_dequant_in_shapes) < 1 \
                or x_dequant_in_shapes[0] is None or None in x_dequant_in_shapes[0]:
            ERROR(
                '[Parser]: Meets invalid Dequantize Op(%s) in merge_qgemm!' % m['x_dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in x_dequant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in x_dequant_in_edges[1:]):
            continue
        w_dequant_in_edges = graph.sorted_in_edges(m['w_dequant'], data=True)
        if len(w_dequant_in_edges) not in (2, 3):
            ERROR(
                '[Parser]: Meets invalid Dequantize Op(%s) in merge_qgemm!' % m['w_dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in w_dequant_in_edges) \
                or any(not e[2]['tensor'].is_const for e in w_dequant_in_edges):
            continue
        b_dequant_in_edges = graph.sorted_in_edges(m['b_dequant'], data=True)
        if len(b_dequant_in_edges) not in (2, 3):
            ERROR(
                '[Parser]: Meets invalid Dequantize Op(%s) in merge_qgemm!' % m['b_dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in b_dequant_in_edges) \
                or any(not e[2]['tensor'].is_const for e in b_dequant_in_edges):
            continue
        gemm_out_edges = graph.sorted_out_edges(m['gemm'], data=True)
        if len(gemm_out_edges) != 1:
            continue
        relu = m['relu'] if 'relu' in m else None
        if relu is not None and len(graph.sorted_out_edges(relu)) != 1:
            continue
        if len(graph.sorted_out_edges(m['add'])) != 1 or obj_dict['trans'].perm != [1, 0]:
            continue
        y_quant_in_edges = graph.sorted_in_edges(m['y_quant'], data=True)
        y_quant_out_shapes = obj_dict['y_quant'].get_output_shapes()
        if len(y_quant_in_edges) not in (2, 3) or len(y_quant_out_shapes) < 1 \
                or y_quant_out_shapes[0] is None or None in y_quant_out_shapes[0]:
            ERROR('[Parser]: Meets invalid Quantize Op(%s) in merge_qgemm!' % m['y_quant'])
            continue
        if any(e[2]['tensor'].value is None for e in y_quant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in y_quant_in_edges[1:]):
            continue

        src, _, in_attr = x_dequant_in_edges[0]
        x_scale, x_zp = obj_dict['x_dequant'].x_scale, obj_dict['x_dequant'].x_zero_point
        w_scale, w_zp = obj_dict['w_dequant'].x_scale, obj_dict['w_dequant'].x_zero_point
        b_scale, b_zp = obj_dict['b_dequant'].x_scale, obj_dict['b_dequant'].x_zero_point
        y_scale, y_zp = obj_dict['y_quant'].y_scale, obj_dict['y_quant'].y_zero_point
        weights = w_dequant_in_edges[0][2]['tensor'].value
        biases = b_dequant_in_edges[0][2]['tensor'].value

        if not FLOAT_EQUAL(w_scale * x_scale, b_scale) or not np.all(b_zp == 0):
            continue

        matched = True
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['tensor'].dtype = str(x_zp.dtype)
        new_in_attr['tensor'].scale_zp = (x_scale, x_zp)
        new_in_attr['tensor'].activation_quantization_axis = obj_dict['x_dequant'].axis
        graph.remove_edges_from(
            graph.sorted_in_edges(m['gemm']) + gemm_out_edges)
        graph.add_edge(src, m['gemm'], **new_in_attr)
        last_node = m['gemm']
        if relu is not None:
            graph.remove_edges_from(graph.sorted_in_edges(relu) + graph.sorted_out_edges(relu))
            gemm_out_attr = gemm_out_edges[0][2]
            gemm_out_attr['tensor'].dtype = str(y_zp.dtype)
            gemm_out_attr['tensor'].scale_zp = (y_scale, y_zp)
            graph.add_edge(m['gemm'], relu, **gemm_out_attr)
            last_node = relu
            obj_dict['relu'].quantize = True
        for _, dst, out_attr in graph.sorted_out_edges(m['y_quant'], data=True):
            graph.remove_edge(m['y_quant'], dst)
            out_attr['tensor'].dtype = str(y_zp.dtype)
            out_attr['tensor'].scale_zp = (y_scale, y_zp)
            out_attr['tensor'].activation_quantization_axis = obj_dict['y_quant'].axis
            graph.add_edge(last_node, dst, **out_attr)
        if len(x_dequant_in_shapes[0]) != 2 and len(y_quant_out_shapes[0]) != 2:
            pre_shape = [int(np.prod(x_dequant_in_shapes[0][:-1])), x_dequant_in_shapes[0][-1]]
            insert_reshape(graph, src, m['gemm'], new_in_attr, pre_shape, quantize=True)
            post_shape = y_quant_out_shapes[0]
            post_new_shape = [int(np.prod(y_quant_out_shapes[0][:-1])), y_quant_out_shapes[0][-1]]
            last_node = insert_reshape_after(graph, last_node, post_shape, post_new_shape, quantize=True)

        if m['y_quant'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['y_quant'])
            graph._attr['output_names'][index] = last_node

        gemm_attr = obj_dict['gemm'].copied_attr()
        gemm_attr.update({'quantize': True, 'opset_version': 1, 'transB': True})
        insert_constant(graph, m['gemm'] + '_x_scale',
                        x_scale, m['gemm'], in_port=1, data_format='NHWC')
        insert_constant(graph, m['gemm'] + '_x_zero_point',
                        x_zp, m['gemm'], in_port=2, data_format='NHWC')
        insert_constant(graph, m['gemm'] + '_w', weights,
                        m['gemm'], in_port=3, data_format='NHWC')
        insert_constant(graph, m['gemm'] + '_w_scale',
                        w_scale, m['gemm'], in_port=4, data_format='NHWC')
        insert_constant(graph, m['gemm'] + '_w_zero_point',
                        w_zp, m['gemm'], in_port=5, data_format='NHWC')
        insert_constant(graph, m['gemm'] + '_B', biases,
                        m['gemm'], in_port=6, data_format='NHWC')
        insert_constant(graph, m['gemm'] + '_y_scale',
                        y_scale, m['gemm'], in_port=7, data_format='NHWC')
        insert_constant(graph, m['gemm'] + '_y_zero_point',
                        y_zp, m['gemm'], in_port=8, data_format='NHWC')

        NodeWrap(graph, m['gemm']).replace_obj('QGemmMs', gemm_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_qmatmul(graph):
    if not graph._attr.get('quantize', False):
        return
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('a_dequant', {'op': 'DequantizeLinear', 'unique': False}),
                                   ('b_dequant', {'op': 'DequantizeLinear', 'unique': False}),
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
    matches = single_node_matcher(graph, op_list)
    for m in matches:
        float_op = m['target']
        in_edges = graph.sorted_in_edges(float_op, data=True)
        if len(in_edges) < 1:
            ERROR('[Parser]: Meets invalid float Op(%s) in merge_q_multiple!' % float_op)
            continue
        out_edges = graph.sorted_out_edges(float_op, data=True)
        if len(out_edges) < 1:
            continue

        op_in_names = [e[0] for e in in_edges]
        op_out_names = [e[1] for e in out_edges]
        names = op_in_names + [float_op] + op_out_names
        obj_dict = {n: NodeWrap(graph, n)['object'] for n in names}
        if any(v is None for v in obj_dict.values()):
            error_node = [n for n in obj_dict if obj_dict[n] is None][0]
            ERROR('[Parser]: Meets invalid Op(%s) in merge_q_multiple!' % error_node)
            continue

        # For some ops, need special treatment. For example, the second input of Split
        # could be split(length of each output), and should not check DequantizeLinear
        # op for it.
        if obj_dict[float_op].type == 'Split':
            op_in_names = op_in_names[:1]

        if any(obj_dict[n].type != 'DequantizeLinear' for n in op_in_names):
            continue
        if any(obj_dict[n].type != 'QuantizeLinear' for n in op_out_names):
            continue

        found_invalid_dequant = False
        for dequant in op_in_names + op_out_names:
            dequant_in_edges = graph.sorted_in_edges(dequant, data=True)
            if len(dequant_in_edges) not in (2, 3):
                ERROR('[Parser]: Meets invalid Quantize Op(%s) in merge_q_multiple!' % dequant)
                found_invalid_dequant = True
                continue
            if any(e[2]['tensor'].value is None for e in dequant_in_edges[1:]) \
                    or any(not e[2]['tensor'].is_const for e in dequant_in_edges[1:]):
                found_invalid_dequant = True
                break
        if found_invalid_dequant:
            continue

        matched = True

        for i, dequant in enumerate(op_in_names):
            graph.remove_edge(dequant, float_op)
            dequant_in_edges = graph.sorted_in_edges(dequant, data=True)
            src, _, in_attr = dequant_in_edges[0]
            src_obj = NodeWrap(graph, src)['object']
            if src_obj is not None and not src_obj.quantize:
                src_obj.quantize = True
            new_in_attr = copy.deepcopy(in_attr)
            new_in_attr['dst_in_port'] = i
            x_scale, x_zp = obj_dict[dequant].x_scale, obj_dict[dequant].x_zero_point
            new_in_attr['tensor'].dtype = str(x_zp.dtype)
            new_in_attr['tensor'].scale_zp = (x_scale, x_zp)
            new_in_attr['tensor'].activation_quantization_axis = obj_dict[dequant].axis
            graph.add_edge(src, float_op, **new_in_attr)

        for quant in op_out_names:
            src_out_port = graph.sorted_in_edges(quant, data=True)[0][2]['src_out_port']
            y_scale, y_zp = obj_dict[quant].y_scale, obj_dict[quant].y_zero_point
            for _, dst, out_attr in graph.sorted_out_edges(quant, data=True):
                graph.remove_edge(quant, dst)
                out_attr['tensor'].dtype = str(y_zp.dtype)
                out_attr['tensor'].scale_zp = (y_scale, y_zp)
                out_attr['tensor'].activation_quantization_axis = obj_dict[quant].axis
                dst_in_attr = copy.deepcopy(out_attr)
                dst_in_attr.update({'src_out_port': src_out_port})
                graph.add_edge(float_op, dst, **dst_in_attr)

            if quant in graph._attr['output_names']:
                index = graph._attr['output_names'].index(quant)
                if float_op in graph._attr['output_names']:
                    graph._attr['output_names'].pop(index)
                else:
                    graph._attr['output_names'][index] = float_op

        obj_dict[float_op].quantize = True

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
                                   ('dequant', {
                                    'op': 'DequantizeLinear', 'unique': False}),
                                   ('float_op', {'op': op_list}),
                                   ('quant', {'op': 'QuantizeLinear'}),
                               ],
                               edges=[
                                   ('dequant', 'float_op', {'dst_in_port': 0}),
                                   ('float_op', 'quant')
                               ])
    for m in matches:
        names = ['dequant', 'float_op', 'quant']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(v is None for v in obj_dict.values()):
            error_node = [n for n in obj_dict if obj_dict[n] is None][0]
            ERROR('[Parser]: Meets invalid Op(%s) in merge_q_unary!' %
                  error_node)
            continue
        dequant_in_edges = graph.sorted_in_edges(m['dequant'], data=True)
        if len(dequant_in_edges) not in (2, 3):
            ERROR(
                '[Parser]: Meets invalid Dequantize Op(%s) in merge_q_unary!' % m['dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in dequant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in dequant_in_edges[1:]):
            continue

        op_in_edges = graph.sorted_in_edges(m['float_op'], data=True)
        if len(op_in_edges) < 1:
            ERROR('[Parser]: Meets invalid Op(%s) in merge_q_unary!' %
                  m['float_op'])
            continue
        op_out_edges = graph.sorted_out_edges(m['float_op'], data=True)
        if len(op_out_edges) != 1:
            continue

        quant_in_edges = graph.sorted_in_edges(m['quant'], data=True)
        if len(quant_in_edges) not in (2, 3):
            ERROR(
                '[Parser]: Meets invalid Quantize Op(%s) in merge_q_unary!' % m['quant'])
            continue
        if any(e[2]['tensor'].value is None for e in quant_in_edges[1:]) \
                or any(not e[2]['tensor'].is_const for e in quant_in_edges[1:]):
            continue
        if obj_dict['float_op'].type == 'Clip':
            if len(op_in_edges) != 3\
                    or op_in_edges[1][2]['tensor'] is None \
                    or not op_in_edges[1][2]['tensor'].is_const\
                    or op_in_edges[2][2]['tensor'] is None \
                    or not op_in_edges[2][2]['tensor'].is_const:
                WARN(
                    '[Parser]: Meets invaild clip value for Op (%s) in merge_q_unary!' % m['float_op'])
                continue

        matched = True

        x_scale, x_zp = obj_dict['dequant'].x_scale, obj_dict['dequant'].x_zero_point
        y_scale, y_zp = obj_dict['quant'].y_scale, obj_dict['quant'].y_zero_point

        if obj_dict['float_op'].type == 'Clip':
            graph.remove_edges_from(op_in_edges[1:])
            clip_min = op_in_edges[1][2]['tensor'].value
            clip_max = op_in_edges[2][2]['tensor'].value

            q_min = np.iinfo(x_zp.dtype).min
            q_max = np.iinfo(x_zp.dtype).max

            q_clip_min = np.array(
                np.clip(clip_min / x_scale + x_zp, q_min, q_max)).astype(x_zp.dtype)
            q_clip_max = np.array(
                np.clip(clip_max / x_scale + x_zp, q_min, q_max)).astype(x_zp.dtype)

            insert_constant(graph, m['float_op'] + '_q_clip_min',
                            q_clip_min, m['float_op'], in_port=1)
            insert_constant(graph, m['float_op'] + '_q_clip_max',
                            q_clip_max, m['float_op'], in_port=2)
        # elif obj_dict['float_op'].type in ('Sigmoid', 'LeakyRelu', 'HardSwish', 'HardSigmoid', 'Relu') \
        #         and y_zp.dtype == 'int32':
        #     y_zp = y_zp.astype(np.int16)
        #     WARN(
        #         '[Parser]: Op (%s) output zeropoint dtype is int32, now convert it to int16!' % m['float_op'])

        src, _, in_attr = dequant_in_edges[0]
        src_obj = NodeWrap(graph, src)['object']
        if src_obj is not None and not src_obj.quantize:
            src_obj.quantize = True
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['tensor'].dtype = str(x_zp.dtype)
        new_in_attr['tensor'].scale_zp = (x_scale, x_zp)
        new_in_attr['tensor'].activation_quantization_axis = obj_dict['dequant'].axis

        graph.remove_edges_from(op_in_edges[:1])
        graph.remove_edge(m['float_op'], m['quant'])
        graph.add_edge(src, m['float_op'], **new_in_attr)
        for _, dst, out_attr in graph.sorted_out_edges(m['quant'], data=True):
            graph.remove_edge(m['quant'], dst)
            out_attr['tensor'].dtype = str(y_zp.dtype)
            out_attr['tensor'].scale_zp = (y_scale, y_zp)
            out_attr['tensor'].activation_quantization_axis = obj_dict['quant'].axis
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


def merge_sequence_construct_and_concat(graph):
    '''Merge inputs->SequenceConstruct->ConcatFromSequence to inputs->ConcatFromSequence.
    '''
    matched = False
    matches = two_nodes_matcher(graph, 'SequenceConstruct', 'ConcatFromSequence')
    for m in matches:
        seq_construct, seq_concat = m['begin'], m['end']
        seq_construct_obj = NodeWrap(graph, seq_construct)['object']
        seq_concat_obj = NodeWrap(graph, seq_concat)['object']
        construct_in_edges = graph.sorted_in_edges(seq_construct, data=True)
        seq_num = len(construct_in_edges)
        if seq_construct_obj is None or seq_concat_obj is None or seq_num < 1:
            ERROR(
                '[Parser]: Meets invalid SequenceConstruct/ConcatFromSequence Op in merge_sequence_construct_and_concat!')
            continue
        matched = True
        graph.remove_edge(seq_construct, seq_concat)
        for src, _, in_attr in construct_in_edges:
            graph.add_edge(src, seq_concat, **in_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_rcnn(graph, params):
    def _convert_to_x_first(graph, node_name):
        '''Add split and concat to convert output from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax).
        Return the name of the concat node.'''
        split_node = get_valid_node_name(graph, node_name + '_split')
        graph.add_edge(node_name, split_node, **{'src_out_port': 0})
        split_node_attr = {'name': split_node, 'opset_version': 11, 'axis': -1, 'split': [1, 1, 1, 1]}
        NodeWrap(graph, split_node).replace_obj('Split', split_node_attr)
        concat_node = get_valid_node_name(graph, node_name + '_concat')
        graph.add_edge(split_node, concat_node, **{'src_out_port': 1, 'dst_in_port': 0})
        graph.add_edge(split_node, concat_node, **{'src_out_port': 0, 'dst_in_port': 1})
        graph.add_edge(split_node, concat_node, **{'src_out_port': 3, 'dst_in_port': 2})
        graph.add_edge(split_node, concat_node, **{'src_out_port': 2, 'dst_in_port': 3})
        concat_node_attr = {'name': concat_node, 'opset_version': 11, 'axis': -1}
        NodeWrap(graph, concat_node).replace_obj('Concat', concat_node_attr)
        return concat_node

    if params.get('detection_postprocess', '').upper() not in ('FASTERRCNN', 'MASKRCNN'):
        return
    is_maskrcnn = (params['detection_postprocess'].upper() == 'MASKRCNN')

    pred_box_matches = matched_patterns(graph,
                                        nodes=[
                                            ('conv', {'op': 'Conv'}),
                                            ('weights', {'op': 'Constant'}),
                                            ('reshape1', {'op': 'Reshape'}),
                                            ('trans', {'op': 'Transpose'}),
                                            ('reshape2', {'op': 'Reshape'}),
                                            ('concat', {'op': 'Concat'}),
                                            ('reshape3', {'op': 'Reshape'})],
                                        edges=[
                                            ('weights', 'conv', {'dst_in_port': 1}),
                                            ('conv', 'reshape1', {'dst_in_port': 0}),
                                            ('reshape1', 'trans'),
                                            ('trans', 'reshape2'),
                                            ('reshape2', 'concat', {'dst_in_port': 0}),
                                            ('concat', 'reshape3', {'dst_in_port': 0})])
    if not pred_box_matches or len(pred_box_matches) != 2:
        return
    pred_bbox_deltas, objectness = None, None
    for m in pred_box_matches:
        if 'bbox_pred' in m['weights']:
            pred_bbox_deltas = m['reshape3']
        elif 'cls_logits' in m['weights']:
            objectness = m['reshape3']
    if not pred_bbox_deltas or not objectness:
        return

    anchors_matches = matched_patterns(graph,
                                       nodes=[
                                           ('reshape1', {'op': 'Reshape'}),
                                           ('cast', {'op': 'Cast'}),
                                           ('add', {'op': 'Add'}),
                                           ('reshape2', {'op': 'Reshape'}),
                                           ('concat', {'op': 'Concat'})],
                                       edges=[
                                           ('reshape1', 'cast'),
                                           ('cast', 'add'),
                                           ('add', 'reshape2', {'dst_in_port': 0}),
                                           ('reshape2', 'concat', {'dst_in_port': 0})])
    if len(anchors_matches) != 1:
        return
    anchors = anchors_matches[0]['concat']

    roi_pool_matches = matched_patterns(graph,
                                        nodes=[
                                            ('pool_out', {}),
                                            ('reshape1', {'op': ['Reshape', 'Flatten']}),
                                            ('gemm1', {'op': 'Gemm'}),
                                            ('relu1', {'op': 'Relu'}),
                                            ('gemm2', {'op': 'Gemm'}),
                                            ('relu2', {'op': 'Relu'}),
                                            ('reshape2', {'op': ['Reshape', 'Flatten']}),
                                            ('gemm3', {'op': 'Gemm'}),
                                            ('softmax', {'op': 'Softmax'}),
                                            ('gemm4', {'op': 'Gemm'})],
                                        edges=[
                                            ('pool_out', 'reshape1', {'dst_in_port': 0}),
                                            ('reshape1', 'gemm1', {'dst_in_port': 0}),
                                            ('gemm1', 'relu1'),
                                            ('relu1', 'gemm2', {'dst_in_port': 0}),
                                            ('gemm2', 'relu2'),
                                            ('relu2', 'reshape2', {'dst_in_port': 0}),
                                            ('reshape2', 'gemm3', {'dst_in_port': 0}),
                                            ('gemm3', 'softmax'),
                                            ('reshape2', 'gemm4', {'dst_in_port': 0})])
    if len(roi_pool_matches) != 1:
        return
    roi_heads_pool_out = roi_pool_matches[0]['pool_out']
    roi_scores = roi_pool_matches[0]['softmax']
    roi_boxes = roi_pool_matches[0]['gemm4']

    feature_matches = matched_patterns(graph,
                                       nodes=[
                                           ('inp', {'op': ['Conv', 'Add']}),
                                           ('backbone_out', {'op': 'Conv'}),
                                           ('conv1', {'op': 'Conv'}),
                                           ('relu', {'op': 'Relu'}),
                                           ('conv2', {'op': 'Conv'})],
                                       edges=[
                                           ('inp', 'backbone_out', {'dst_in_port': 0}),
                                           ('backbone_out', 'conv1', {'dst_in_port': 0}),
                                           ('conv1', 'relu'),
                                           ('relu', 'conv2', {'dst_in_port': 0})])
    features = []
    begin_node, begin_feature_node = None, None
    for m in feature_matches:
        inp_obj = NodeWrap(graph, m['inp'])['object']
        if inp_obj is not None and inp_obj.type == 'Conv':
            begin_node = m['inp']
            begin_feature_node = m['backbone_out']
        features.append(m['backbone_out'])
    if not begin_node or not begin_feature_node:
        return
    length_list = []
    for feat in features:
        length_list.append(0 if feat == begin_feature_node else shortest_path_length(graph, begin_node, feat))
    sorted_idx = np.argsort(length_list)[::-1]
    features = [features[idx] for idx in sorted_idx]

    resize_box_matches = matched_patterns(graph,
                                          nodes=[
                                              ('split', {'op': 'Split'}),
                                              ('squeeze_x', {'op': 'Squeeze'}),
                                              ('mul_x', {'op': 'Mul'}),
                                              ('div_x', {'op': 'Div'}),
                                              ('div_x_A', {'op': 'Constant'}),
                                              ('unsqueeze_x', {'op': 'Unsqueeze'}),
                                              ('squeeze_y', {'op': 'Squeeze'}),
                                              ('mul_y', {'op': 'Mul'}),
                                              ('div_y', {'op': 'Div'}),
                                              ('div_y_A', {'op': 'Constant'}),
                                              ('unsqueeze_y', {'op': 'Unsqueeze'}),
                                              ('concat', {'op': 'Concat'})],
                                          edges=[
                                              ('split', 'squeeze_x', {'src_out_port': 0}),
                                              ('squeeze_x', 'mul_x'),
                                              ('div_x_A', 'div_x', {'dst_in_port': 0}),
                                              ('div_x', 'mul_x'),
                                              ('mul_x', 'unsqueeze_x'),
                                              ('unsqueeze_x', 'concat', {'dst_in_port': 0}),
                                              ('split', 'squeeze_y', {'src_out_port': 1}),
                                              ('squeeze_y', 'mul_y'),
                                              ('div_y_A', 'div_y', {'dst_in_port': 0}),
                                              ('div_y', 'mul_y'),
                                              ('mul_y', 'unsqueeze_y'),
                                              ('unsqueeze_y', 'concat', {'dst_in_port': 1})])
    if len(resize_box_matches) != 1:
        return
    height_obj, width_obj = [NodeWrap(graph, resize_box_matches[0][name])['object'] for name in ['div_y_A', 'div_x_A']]
    if height_obj is None or width_obj is None:
        ERROR('[Parser]: Meets invalid Constant Node in merge_rcnn!')
        return
    if height_obj.value.size != 1 or width_obj.value.size != 1:
        return
    original_image_height = int(height_obj.value.item())
    original_image_width = int(width_obj.value.item())
    ret_boxes_split = resize_box_matches[0]['split']
    resized_boxes = resize_box_matches[0]['concat']

    self_min_size_f = 800
    self_max_size_f = 1333
    min_size = float(min(original_image_height, original_image_width))
    max_size = float(max(original_image_height, original_image_width))
    scale = min(self_min_size_f / min_size, self_max_size_f / max_size)
    image_height = int(scale * original_image_height)
    image_width = int(scale * original_image_width)
    bbox_xform_clip = np.log(1000.0 / 16)

    # rpn parameters
    rpn_num_class = 1
    rpn_score_threshold = float(params.get('rpn_score_threshold', 0.5))
    rpn_nms_score_threshold = float(params.get('rpn_score_threshold', 0.7))
    rpn_iou_threshold = float(params.get('rpn_iou_threshold', 0.7))
    rpn_max_box_num = np.iinfo(np.int32).max
    rpn_nms_max_box_num = int(params.get('rpn_max_box_num', 1000))
    # roi heads parameters
    roi_num_class = int(params.get('class_num', 91))
    roi_score_threshold = float(params.get('box_score_threshold', 0.7))
    roi_nms_score_threshold = float(params.get('box_score_threshold', 0.8))
    roi_iou_threshold = float(params.get('box_iou_threshold', 0.7))
    roi_max_box_num = rpn_nms_max_box_num
    roi_nms_max_box_num = int(params.get('box_max_box_num', 100))

    # rpn Sigmoid
    scores = get_valid_node_name(graph, 'rpn_sigmoid')
    graph.add_edge(objectness, scores)
    NodeWrap(graph, scores).replace_obj('Sigmoid', {'name': scores, 'opset_version': 13})
    concat_scores = scores

    # rpn DetectionOutput
    rpn_detect_out = get_valid_node_name(graph, 'rpn_detect_out')
    # DetectionOutput input: score, boxes, anchors
    scores_out_attr = {'dst_in_port': 0}
    graph.add_edge(concat_scores, rpn_detect_out, **scores_out_attr)
    boxes_out_attr = {'dst_in_port': 1}
    graph.add_edge(pred_bbox_deltas, rpn_detect_out, **boxes_out_attr)
    anchors_out_attr = {'dst_in_port': 2}
    graph.add_edge(anchors, rpn_detect_out, **anchors_out_attr)
    insert_reshape(graph, concat_scores, rpn_detect_out, scores_out_attr, [1, -1, rpn_num_class])
    insert_reshape(graph, pred_bbox_deltas, rpn_detect_out, boxes_out_attr, [1, -1, rpn_num_class, 4])
    insert_reshape(graph, anchors, rpn_detect_out, anchors_out_attr, [1, -1, 4])
    # DetectionOutput output: scores, boxes, box_num_perClass, label_perclass, total_class_num
    # boxes format: (xmin, ymin, xmax, ymax); thus set anchor_mode as caffe_detection
    rpn_detect_out_attr = {'name': rpn_detect_out, 'score_threshold': rpn_score_threshold,
                           'class_num': rpn_num_class, 'max_box_num': rpn_max_box_num,
                           'image_height': image_height, 'image_width': image_width,
                           'variance': [1.0, 1.0, 1.0, 1.0], 'anchor_mode': 'caffe_detection',
                           'bbox_xform_clip': bbox_xform_clip}
    NodeWrap(graph, rpn_detect_out).replace_obj('ArmDetectionOutput', rpn_detect_out_attr)

    # rpn FilterBoxes
    rpn_filter_boxes = get_valid_node_name(graph, 'rpn_filter_boxes')
    # FilterBoxes input/output: boxes(ymin, xmin, ymax, xmax), scores, box_num_perClass, label_perclass, total_class_num
    graph.add_edge(rpn_detect_out, rpn_filter_boxes, **{'src_out_port': 1, 'dst_in_port': 0})
    graph.add_edge(rpn_detect_out, rpn_filter_boxes, **{'src_out_port': 0, 'dst_in_port': 1})
    for port in range(2, 5):
        graph.add_edge(rpn_detect_out, rpn_filter_boxes, **{'src_out_port': port, 'dst_in_port': port})
    rpn_filter_boxes_attr = {'name': rpn_filter_boxes, 'min_size': [0.001, 0.001],
                             'maxnum': rpn_max_box_num}
    NodeWrap(graph, rpn_filter_boxes).replace_obj('ArmFilterBoxes', rpn_filter_boxes_attr)
    # Add Out node for output label_perclass(not used) of FilterBoxes
    label_perclass_out = get_valid_node_name(graph, 'rpn_filter_boxes_out')
    graph.add_edge(rpn_filter_boxes, label_perclass_out, **{'src_out_port': 3})
    NodeWrap(graph, label_perclass_out).replace_obj('Out', {'name': label_perclass_out})

    # rpn NMS
    rpn_nms = get_valid_node_name(graph, 'rpn_nms')
    # nms input: boxes(ymin, xmin, ymax, xmax), box_num_perclass, total_class_num, scores
    # FilterBoxes to NMS
    graph.add_edge(rpn_filter_boxes, rpn_nms, **{'src_out_port': 0, 'dst_in_port': 0})
    graph.add_edge(rpn_filter_boxes, rpn_nms, **{'src_out_port': 2, 'dst_in_port': 1})
    graph.add_edge(rpn_filter_boxes, rpn_nms, **{'src_out_port': 4, 'dst_in_port': 2})
    graph.add_edge(rpn_filter_boxes, rpn_nms, **{'src_out_port': 1, 'dst_in_port': 3})
    # nms output: boxes(ymin, xmin, ymax, xmax), box_num_perclass, scores, keep
    nms_attr = {'name': rpn_nms, 'center_point_box': 0,
                'image_height': image_height, 'image_width': image_width, 'max_box_num': rpn_nms_max_box_num,
                'iou_threshold': rpn_iou_threshold, 'score_threshold': rpn_nms_score_threshold}
    NodeWrap(graph, rpn_nms).replace_obj('ArmNMS', nms_attr)
    # Add Out node for outputs that are not used(only boxes is used)
    for idx in range(1, 4):
        out_name = get_valid_node_name(graph, 'rpn_nms_out_' + str(idx))
        graph.add_edge(rpn_nms, out_name, **{'src_out_port': idx})
        NodeWrap(graph, out_name).replace_obj('Out', {'name': out_name})

    # Convert output from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
    rpn_concat = _convert_to_x_first(graph, rpn_nms)

    # multi_scale_roi_align
    roi_heads_roi_align = get_valid_node_name(graph, 'rpn_roi_align')
    # PyramidROIAlign input: boxes(ymin, xmin, ymax, xmax), feature list
    graph.add_edge(rpn_nms, roi_heads_roi_align, **{'src_out_port': 0, 'dst_in_port': 0})
    for idx, feat in enumerate(features):
        feat_out_attr = {'dst_in_port': idx + 1}
        graph.add_edge(feat, roi_heads_roi_align, **feat_out_attr)
        # transpose from NCHW to NHWC
        insert_transpose(graph, feat, roi_heads_roi_align, feat_out_attr, perm=[0, 2, 3, 1])
    # PyramidROIAlign output: RoI pooled output
    # Set spatial_scale in infer_shape stage because need to know the shapes of features
    roi_heads_roi_align_attr = {'name': roi_heads_roi_align, 'resize_width': 7, 'resize_height': 7,
                                'image_width': image_width, 'image_height': image_height,
                                'sample_ratio': [2, 2], 'spatial_scale': [], 'proposal_normalized': False}
    NodeWrap(graph, roi_heads_roi_align).replace_obj('ArmPyramidROIAlign', roi_heads_roi_align_attr)
    # Connect PyramidROIAlign with box_head and box_predictor
    roi_heads_pool_out_edges = graph.sorted_out_edges(roi_heads_pool_out, data=True)
    graph.remove_edges_from(roi_heads_pool_out_edges)
    for _, dst, out_attr in roi_heads_pool_out_edges:
        out_attr.update({'src_out_port': 0})
        graph.add_edge(roi_heads_roi_align, dst, **out_attr)
    # Transpose back from NHWC to NCHW
    insert_transpose_after(graph, roi_heads_roi_align, [0, 3, 1, 2])

    # roi_heads DetectionOutput
    roi_detect_out = get_valid_node_name(graph, 'roi_detect_out')
    # DetectionOutput input: score, boxes(xmin, ymin, xmax, ymax), anchors(xmin, ymin, xmax, ymax)
    roi_scores_out_attr = {'dst_in_port': 0}
    graph.add_edge(roi_scores, roi_detect_out, **roi_scores_out_attr)
    roi_boxes_out_attr = {'dst_in_port': 1}
    graph.add_edge(roi_boxes, roi_detect_out, **roi_boxes_out_attr)
    proposals_out_attr = {'src_out_port': 0, 'dst_in_port': 2}
    graph.add_edge(rpn_concat, roi_detect_out, **proposals_out_attr)
    insert_reshape(graph, roi_scores, roi_detect_out, roi_scores_out_attr, [1, -1, roi_num_class])
    insert_reshape(graph, roi_boxes, roi_detect_out, roi_boxes_out_attr, [1, -1, roi_num_class, 4])
    insert_reshape(graph, rpn_concat, roi_detect_out, proposals_out_attr, [1, -1, 4])
    # DetectionOutput output: scores, boxes(ymin, xmin, ymax, xmax), box_num_perClass, label_perclass, total_class_num
    roi_detect_out_attr = {'name': roi_detect_out, 'score_threshold': roi_score_threshold,
                           'class_num': roi_num_class, 'max_box_num': roi_max_box_num,
                           'image_height': image_height, 'image_width': image_width,
                           'variance': [10.0, 10.0, 5.0, 5.0], 'anchor_mode': 'caffe_detection',
                           'bbox_xform_clip': bbox_xform_clip}
    NodeWrap(graph, roi_detect_out).replace_obj('ArmDetectionOutput', roi_detect_out_attr)

    # roi FilterBoxes
    roi_filter_boxes = get_valid_node_name(graph, 'roi_filter_boxes')
    # FilterBoxes input/output: boxes(ymin, xmin, ymax, xmax), scores, box_num_perClass, label_perclass, total_class_num
    graph.add_edge(roi_detect_out, roi_filter_boxes, **{'src_out_port': 1, 'dst_in_port': 0})
    graph.add_edge(roi_detect_out, roi_filter_boxes, **{'src_out_port': 0, 'dst_in_port': 1})
    for port in range(2, 5):
        graph.add_edge(roi_detect_out, roi_filter_boxes, **{'src_out_port': port, 'dst_in_port': port})
    roi_filter_boxes_attr = {'name': roi_filter_boxes, 'min_size': [0.01, 0.01],
                             'maxnum': roi_max_box_num}
    NodeWrap(graph, roi_filter_boxes).replace_obj('ArmFilterBoxes', roi_filter_boxes_attr)
    # Add Out node for output label_perclass(not used) of FilterBoxes
    roi_label_perclass_out = get_valid_node_name(graph, 'roi_label_perclass_out')
    graph.add_edge(roi_filter_boxes, roi_label_perclass_out, **{'src_out_port': 3})
    NodeWrap(graph, roi_label_perclass_out).replace_obj('Out', {'name': roi_label_perclass_out})

    # roi_heads NMS
    roi_nms = get_valid_node_name(graph, 'roi_nms')
    # nms input: boxes(ymin, xmin, ymax, xmax), box_num_perclass, total_class_num, scores
    # FilterBoxes to NMS
    graph.add_edge(roi_filter_boxes, roi_nms, **{'src_out_port': 0, 'dst_in_port': 0})
    graph.add_edge(roi_filter_boxes, roi_nms, **{'src_out_port': 2, 'dst_in_port': 1})
    graph.add_edge(roi_filter_boxes, roi_nms, **{'src_out_port': 4, 'dst_in_port': 2})
    graph.add_edge(roi_filter_boxes, roi_nms, **{'src_out_port': 1, 'dst_in_port': 3})
    # nms output: boxes(ymin, xmin, ymax, xmax), box_num_perclass, scores, keep
    roi_nms_attr = {'name': roi_nms, 'center_point_box': 0,
                    'image_height': image_height, 'image_width': image_width, 'max_box_num': roi_nms_max_box_num,
                    'iou_threshold': roi_iou_threshold, 'score_threshold': roi_nms_score_threshold}
    NodeWrap(graph, roi_nms).replace_obj('ArmNMS', roi_nms_attr)
    # Add Out node for NMS
    for idx in range(1, 4):
        out_name = get_valid_node_name(graph, 'roi_nms_out_' + str(idx))
        graph.add_edge(roi_nms, out_name, **{'src_out_port': idx})
        NodeWrap(graph, out_name).replace_obj('Out', {'name': out_name})

    # Convert output from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
    roi_concat = _convert_to_x_first(graph, roi_nms)

    # Resize boxes back to original size
    split_in_edges = graph.sorted_in_edges(ret_boxes_split)
    graph.remove_edges_from(split_in_edges)
    split_in_attr = {'src_out_port': 0, 'dst_in_port': 0}
    graph.add_edge(roi_concat, ret_boxes_split, **split_in_attr)
    insert_reshape(graph, roi_concat, ret_boxes_split, split_in_attr, [-1, 4])

    graph._attr['output_names'].clear()
    graph._attr['output_names'] = [roi_filter_boxes, roi_nms, resized_boxes]

    if is_maskrcnn:
        mask_pool_matches = matched_patterns(graph,
                                             nodes=[
                                                 ('scatter', {'op': 'ScatterElements'}),
                                                 ('conv', {'op': 'Conv'}),
                                                 ('relu', {'op': 'Relu'})],
                                             edges=[
                                                 ('scatter', 'conv', {'dst_in_port': 0}),
                                                 ('conv', 'relu')])
        mask_out_matches = matched_patterns(graph,
                                            nodes=[
                                                ('sigmoid', {'op': 'Sigmoid'}),
                                                ('flatten', {'op': 'Flatten'}),
                                                ('gather', {'op': 'Gather'}),
                                                ('reshape1', {'op': 'Reshape'}),
                                                ('reshape2', {'op': 'Reshape'}),
                                                ('unsqueeze', {'op': 'Unsqueeze'}),
                                                ('slice', {'op': 'Slice'})],
                                            edges=[
                                                ('sigmoid', 'flatten'),
                                                ('flatten', 'gather', {'dst_in_port': 0}),
                                                ('gather', 'reshape1', {'dst_in_port': 0}),
                                                ('reshape1', 'reshape2', {'dst_in_port': 0}),
                                                ('reshape2', 'unsqueeze'),
                                                ('unsqueeze', 'slice', {'dst_in_port': 0})])
        if len(mask_pool_matches) == 1 and len(mask_out_matches) == 1:
            mask_pool_out = mask_pool_matches[0]['scatter']
            mask_sigmoid = mask_out_matches[0]['sigmoid']
            # multi_scale_roi_align
            mask_roi_align = get_valid_node_name(graph, 'mask_roi_align')
            # PyramidROIAlign input: boxes(ymin, xmin, ymax, xmax), feature list
            graph.add_edge(roi_nms, mask_roi_align, **{'src_out_port': 0, 'dst_in_port': 0})
            for idx, feat in enumerate(features):
                feat_out_attr = {'dst_in_port': idx + 1}
                graph.add_edge(feat, mask_roi_align, **feat_out_attr)
                # transpose from NCHW to NHWC
                insert_transpose(graph, feat, mask_roi_align, feat_out_attr, perm=[0, 2, 3, 1])
            # PyramidROIAlign output: RoI pooled output
            mask_roi_align_attr = {'name': mask_roi_align, 'resize_width': 14, 'resize_height': 14,
                                   'image_width': image_width, 'image_height': image_height,
                                   'sample_ratio': [2, 2], 'spatial_scale': [], 'proposal_normalized': False}
            NodeWrap(graph, mask_roi_align).replace_obj('ArmPyramidROIAlign', mask_roi_align_attr)
            # Connect PyramidROIAlign with box_head and box_predictor
            mask_pool_out_edges = graph.sorted_out_edges(mask_pool_out, data=True)
            graph.remove_edges_from(mask_pool_out_edges)
            for _, dst, out_attr in mask_pool_out_edges:
                out_attr.update({'src_out_port': 0})
                graph.add_edge(mask_roi_align, dst, **out_attr)
            # Transpose back from NHWC to NCHW
            insert_transpose_after(graph, mask_roi_align, [0, 3, 1, 2])

            mask_out_edges = graph.sorted_out_edges(mask_sigmoid)
            graph.remove_edges_from(mask_out_edges)
            mask_expand = get_valid_node_name(graph, mask_sigmoid + '_expand')
            graph.add_edge(mask_sigmoid, mask_expand)
            broadcast_shape = np.array([1, 1, 1, 1, 1])
            insert_constant(graph, mask_expand + '_shape', broadcast_shape, mask_expand, in_port=1)
            NodeWrap(graph, mask_expand).replace_obj('Expand', {'name': mask_expand, 'opset_version': 13})

            mask_out_node = get_valid_node_name(graph, mask_expand + '_out')
            graph.add_edge(mask_expand, mask_out_node)
            NodeWrap(graph, mask_out_node).replace_obj('Out', {'name': mask_out_node})
            # add mask output to graph outputs
            graph._attr['output_names'].append(mask_expand)

    clear_redundant_nodes(graph)
