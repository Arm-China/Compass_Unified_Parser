# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import copy
import numpy as np
from networkx.algorithms import shortest_path_length
from ....common.defs import Tensor
from ....ops.op import Op, OpHasWeights, OpHasBiases, KerasOp, BaseDeconvOp, ConstLikeOp, OpHasOneOutPort
from ....graph.node_wrap import NodeWrap
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from ....graph.graph_algo import get_valid_node_name, determined_sort
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL
from .common_passes import clear_redundant_nodes, FLOAT_EQUAL, insert_constant, insert_reshape, insert_reshape_after, \
    insert_transpose, insert_transpose_after


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
        in_edges = graph.sorted_in_edges(loop, data=True)
        loop_out_edges = graph.sorted_out_edges(loop, data=True)
        if loop_obj is not None \
                and len(in_edges) >= 2 + len(loop_obj.body._attr['root_in_ports']) \
                and len(loop_out_edges) >= 1:
            if not (len(in_edges) == (2 + len(loop_obj.body._attr['root_in_ports']))
                    or len(in_edges) == (3 + len(loop_obj.body._attr['root_in_ports'])))\
                    or not in_edges[0][2]['tensor'].is_const \
                    or not in_edges[1][2]['tensor'].is_const \
                    or in_edges[0][2]['tensor'].value is None \
                    or in_edges[1][2]['tensor'].value is None:
                continue

            condition = in_edges[1][2]['tensor'].value

            if len(loop_obj.body._attr['output_names']) == 3:
                subgraph_main_out = loop_obj.body._attr['output_names'][-2]
            else:
                DEBUG('invalid loop, need to support more forms.')
                continue

            subgraph_main_outport = loop_obj.body._attr['output_ports'][subgraph_main_out]
            subgraph_main_nodes = determined_sort(
                loop_obj.body, [subgraph_main_out])

            # some constant nodes have been fused, skip checking them.
            subgraph_main_nodes = [
                x for x in subgraph_main_nodes if x in graph.nodes]

            subgraph_main_nodes_objs = {n: NodeWrap(
                graph, n)['object'] for n in subgraph_main_nodes}

            const_node_list = []
            for (node_obj_name, node_obj) in subgraph_main_nodes_objs.items():
                if node_obj is not None \
                        and not isinstance(node_obj, ConstLikeOp) \
                        and isinstance(node_obj, OpHasOneOutPort) \
                        and node_obj.is_all_inputs_const():
                    const_node_list.append(node_obj_name)

            if len(subgraph_main_nodes) > 0 \
                    and subgraph_main_out not in subgraph_main_nodes:
                WARN('[Parser]: Meets invalid Subgraph Nodes in decompose_const_loop!')
                continue

            try:
                if len(subgraph_main_nodes_objs[subgraph_main_out].get_output_tensors()) < 1:
                    continue
                main_out_tensor = subgraph_main_nodes_objs[subgraph_main_out].get_output_tensors()[
                    0]
            except:
                # TODO: subgraph_main_out node is None. Need to support more forms.
                pass

            matched = True
            count = int(in_edges[0][2]['tensor'].value)
            stack = get_valid_node_name(graph, loop + '_stack')

            for n in loop_obj.body._filter_node:
                try:
                    NodeWrap(graph, n)['object'].in_subgraph = False
                except:
                    pass

            graph.remove_edges_from(in_edges)
            # TODO: Condition False
            if not condition:
                for index, (_, dst, out_attr) in enumerate(loop_out_edges):
                    graph.remove_edge(loop, dst)
                    graph.add_edge(in_edges[-1][0], dst, **out_attr)
                continue

            last_loop_res = subgraph_main_out
            for i in range(count):
                if i == 0:
                    for n in subgraph_main_nodes:
                        n_obj = subgraph_main_nodes_objs[n]
                        n_in_edges = graph.sorted_in_edges(n, data=True)

                        for sub_src, _, in_attr in n_in_edges:
                            # reset iter_num in first subgraph
                            if sub_src == in_edges[0][0] and graph.nodes[sub_src]['op'] in ['Dummy', 'Constant']:
                                cur_count_value = np.array(
                                    i, np.dtype(np.int32))
                                in_attr['tensor'].value = cur_count_value
                                NodeWrap(graph, sub_src).replace_obj('Constant', {
                                    'name': sub_src, 'opset_version': 9, 'value': cur_count_value})

                        # TODO: some special nodes need to reset attr.
                        if n_obj.type == 'Slice':
                            cur_obj_attr = n_obj.copied_attr()
                            cur_obj_attr.update({'starts': None, 'ends': None})
                            NodeWrap(graph, n).replace_obj(
                                n_obj.type, cur_obj_attr)

                    graph.add_edge(subgraph_main_out,
                                   stack,
                                   **{'src_out_port': subgraph_main_outport,
                                      'dst_in_port': i,
                                      'tensor': Tensor(value=main_out_tensor)})

                else:
                    for n in subgraph_main_nodes:
                        name_suffix = '_loop_%s' % i
                        new_n = get_valid_node_name(graph, n + name_suffix)
                        n_obj = subgraph_main_nodes_objs[n]
                        n_in_edges = graph.sorted_in_edges(n, data=True)
                        if len(n_in_edges) == 0:
                            continue
                        for src, _, in_attr in n_in_edges:
                            if src not in subgraph_main_nodes and not src.endswith(name_suffix):
                                # nodes not in the sub graph.
                                if len(loop_obj.body._attr['output_names']) == 3 and not n in const_node_list:
                                    # add edge between last loop res with the first node of next loop.
                                    graph.add_edge(
                                        last_loop_res, new_n, **in_attr)
                                    last_loop_res = new_n
                                elif src == in_edges[0][0]:
                                    # change iter num for constant node.
                                    new_const = get_valid_node_name(
                                        graph, src + name_suffix)
                                    cur_count_value = np.array(
                                        i, np.dtype(np.int32))
                                    new_in_attr = copy.deepcopy(in_attr)
                                    new_in_attr['tensor'].value = cur_count_value
                                    new_in_attr['tensor'].name = new_const
                                    graph.add_edge(
                                        new_const, new_n, **new_in_attr)

                                    NodeWrap(graph, new_const).replace_obj('Constant', {
                                        'name': new_const, 'opset_version': 9, 'value': cur_count_value})
                                else:
                                    graph.add_edge(src, new_n, **in_attr)
                            elif src in subgraph_main_nodes:
                                # nodes in the sub graph
                                new_in_attr = copy.deepcopy(in_attr)

                                if n in subgraph_main_nodes:
                                    graph.add_edge(
                                        src + name_suffix, new_n, **new_in_attr)
                                    if graph.nodes[src + name_suffix]['op'] is None:
                                        src_obj = subgraph_main_nodes_objs[src]
                                        src_obj_attr = src_obj.copied_attr()
                                        src_obj_attr.update({'name': new_n})
                                        NodeWrap(
                                            graph, src + name_suffix).replace_obj(src_obj.type, src_obj_attr)
                                else:
                                    graph.add_edge(
                                        src + name_suffix, new_n, **new_in_attr)
                            else:
                                WARN(
                                    '[Parser]: Invalid in edges for Node(%s)!' % new_n)
                        cur_obj_attr = n_obj.copied_attr()
                        cur_obj_attr.update({'name': new_n})
                        if n_obj.type == 'Slice':
                            cur_obj_attr.update({'starts': None, 'ends': None})

                        NodeWrap(graph, new_n).replace_obj(
                            n_obj.type, cur_obj_attr)
                        if n == subgraph_main_out:
                            graph.add_edge(new_n,
                                           stack,
                                           **{'src_out_port': subgraph_main_outport,
                                              'dst_in_port': i,
                                              'tensor': Tensor(value=main_out_tensor)
                                              })
            if len(loop_out_edges) == 1:
                for _, dst, out_attr in loop_out_edges:
                    graph.remove_edge(loop, dst)
                    graph.add_edge(stack, dst, **out_attr)
            elif len(loop_out_edges) == 2:
                for index, (_, dst, out_attr) in enumerate(loop_out_edges):
                    graph.remove_edge(loop, dst)
                    if index == 1:
                        graph.add_edge(stack, dst, **out_attr)
            else:
                WARN('invalid loop out_edges, need to support.')
            NodeWrap(graph, stack).replace_obj('ConcatFromSequence', {
                'name': stack, 'opset_version': 11, 'axis': 0, 'new_axis': 1})

        else:
            ERROR(
                '[Parser]: Meets invalid Loop Op (%s) in decompose_const_loop!' % loop)

    if matched:
        if graph._attr.get('subgraph_output_names', None) is not None:
            graph._attr['output_names'] = list(set(graph._attr['output_names']).difference(
                list(graph._attr['subgraph_output_names'])))
            if loop in list(set(graph._attr['output_names'])):
                index = graph._attr['output_names'].index(loop)
                graph._attr['output_names'].pop(index)
                if condition:
                    graph._attr['output_names'].append(last_loop_res)
                    graph._attr['output_names'].append(stack)
                else:
                    graph._attr['output_names'].append(in_edges[-1][0])
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
            graph._attr['output_names'].insert(index+idx, name)
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
    matches_with_relu = matched_patterns(graph,
                                         nodes=[
                                             ('x_dequant', {
                                              'op': 'DequantizeLinear', 'unique': False}),
                                             ('w_dequant', {
                                              'op': 'DequantizeLinear', 'unique': False}),
                                             ('b_dequant', {
                                              'op': 'DequantizeLinear', 'unique': False}),
                                             ('conv', {'op': 'Conv'}),
                                             ('relu', {'op': 'Relu'}),
                                             ('y_quant', {
                                              'op': 'QuantizeLinear'}),
                                         ],
                                         edges=[
                                             ('x_dequant', 'conv'),
                                             ('w_dequant', 'conv',
                                              {'dst_in_port': 1}),
                                             ('b_dequant', 'conv',
                                              {'dst_in_port': 2}),
                                             ('conv', 'relu'),
                                             ('relu', 'y_quant'),
                                         ])
    for m in matches + matches_with_relu:
        names = ['x_dequant', 'w_dequant', 'b_dequant', 'conv',
                 'y_quant'] + (['relu'] if 'relu' in m else [])
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
        b_dequant_in_edges = graph.sorted_in_edges(m['b_dequant'], data=True)
        if len(b_dequant_in_edges) not in (2, 3):
            ERROR(
                '[Parser]: Meets invalid Dequantize Op(%s) in merge_qconv!' % m['b_dequant'])
            continue
        if any(e[2]['tensor'].value is None for e in b_dequant_in_edges) \
                or any(not e[2]['tensor'].is_const for e in b_dequant_in_edges):
            continue
        conv_out_edges = graph.sorted_out_edges(m['conv'], data=True)
        if len(conv_out_edges) != 1:
            continue
        relu = m['relu'] if 'relu' in m else None
        if relu is not None and len(graph.sorted_out_edges(relu)) != 1:
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
        b_scale, b_zp = obj_dict['b_dequant'].x_scale, obj_dict['b_dequant'].x_zero_point
        y_scale, y_zp = obj_dict['y_quant'].y_scale, obj_dict['y_quant'].y_zero_point
        weights = w_dequant_in_edges[0][2]['tensor'].value
        biases = b_dequant_in_edges[0][2]['tensor'].value

        if not FLOAT_EQUAL(w_scale*x_scale, b_scale) or not np.all(b_zp == 0):
            continue

        matched = True
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['tensor'].dtype = str(x_zp.dtype)
        new_in_attr['tensor'].scale_zp = (x_scale, x_zp)
        graph.remove_edges_from(
            graph.sorted_in_edges(m['conv']) + conv_out_edges)
        graph.add_edge(src, m['conv'], **new_in_attr)
        last_node = m['conv']
        if relu is not None:
            graph.remove_edges_from(graph.sorted_out_edges(relu))
            conv_out_attr = conv_out_edges[0][2]
            conv_out_attr['tensor'].dtype = str(y_zp.dtype)
            conv_out_attr['tensor'].scale_zp = (y_scale, y_zp)
            graph.add_edge(m['conv'], relu, **conv_out_attr)
            last_node = relu
            obj_dict['relu'].quantize = True
        for _, dst, out_attr in graph.sorted_out_edges(m['y_quant'], data=True):
            graph.remove_edge(m['y_quant'], dst)
            out_attr['tensor'].dtype = str(y_zp.dtype)
            out_attr['tensor'].scale_zp = (y_scale, y_zp)
            graph.add_edge(last_node, dst, **out_attr)

        if m['y_quant'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['y_quant'])
            graph._attr['output_names'][index] = last_node

        conv_attr = obj_dict['conv'].copied_attr()
        conv_attr.update({'quantize': True})
        if obj_dict['conv'].type == 'Conv':
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

        graph.remove_edges_from(in_edges)

        for i, dequant in enumerate(op_in_names):
            dequant_in_edges = graph.sorted_in_edges(dequant, data=True)
            src, _, in_attr = dequant_in_edges[0]
            new_in_attr = copy.deepcopy(in_attr)
            new_in_attr['dst_in_port'] = i
            x_scale, x_zp = obj_dict[dequant].x_scale, obj_dict[dequant].x_zero_point
            new_in_attr['tensor'].dtype = str(x_zp.dtype)
            new_in_attr['tensor'].scale_zp = (x_scale, x_zp)
            graph.add_edge(src, float_op, **new_in_attr)

        for quant in op_out_names:
            src_out_port = graph.sorted_in_edges(quant, data=True)[0][2]['src_out_port']
            y_scale, y_zp = obj_dict[quant].y_scale, obj_dict[quant].y_zero_point
            for _, dst, out_attr in graph.sorted_out_edges(quant, data=True):
                graph.remove_edge(quant, dst)
                out_attr['tensor'].dtype = str(y_zp.dtype)
                out_attr['tensor'].scale_zp = (y_scale, y_zp)
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
                np.clip(clip_min/x_scale+x_zp, q_min, q_max)).astype(x_zp.dtype)
            q_clip_max = np.array(
                np.clip(clip_max/x_scale+x_zp, q_min, q_max)).astype(x_zp.dtype)

            insert_constant(graph, m['float_op'] + '_q_clip_min',
                            q_clip_min, m['float_op'], in_port=1)
            insert_constant(graph, m['float_op'] + '_q_clip_max',
                            q_clip_max, m['float_op'], in_port=2)
        elif obj_dict['float_op'].type in ('Sigmoid', 'LeakyRelu', 'HardSwish', 'HardSigmoid', 'Relu') \
                and y_zp.dtype == 'int32':
            y_zp = y_zp.astype(np.int16)
            WARN(
                '[Parser]: Op (%s) output zeropoint dtype is int32, now convert it to int16!' % m['float_op'])

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
