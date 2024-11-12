# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import itertools
from functools import reduce
import copy
from ....common.defs import Tensor, FLOAT_EQUAL, Framework
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL
from ....common.utils import extend_lists, list_string_to_list, float_string_to_list, get_converted_dtype, get_closest_dtype
from ....graph.node_wrap import NodeWrap
from ....graph.graph_algo import determined_sort, get_valid_node_name, clear_redundant_nodes, has_path, infer
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from ....ops.op import Op, LayoutUnawareOp, BaseLinearOp, BaseActivationOp, BaseReluOp, OpHasWeights, OpHasBiases, \
    ArmOp, OpHasAxis, BaseQuantizeDequantizeOp, BaseRnnOp, OpHasAnchors, OpNeedUniBroadcast
from ....ops.onnx_ops.array_ops import ReshapeOp
from ....ops.release_ops import ArmCastOp, ArmConvolutionOp, ArmConvolution3DOp, ArmConvIntegerOp, ArmDecodeBoxOp, \
    ArmDepthwiseConvOp, ArmConvTransposeOp, ArmConvTranspose3DOp, ArmActivationOp
from ....ops.common_ops import PluginOp
from .rename_ops import simple_rename
from .common_passes import remove_node_safely, insert_cast, insert_cast_after, insert_tile, \
    insert_reshape, insert_reshape_after, insert_constant, \
    insert_slice, insert_transpose, remove_redundant_bn, remove_redundant_reshape, remove_redundant_transpose, \
    remove_redundant_transpose2, remove_useless_op, fuse_const, insert_gather, remove_redundant_cast, \
    insert_transpose_after, merge_same_op_at_out_port, remove_redundant_transpose_unaware
from ....plugin_loader import PARSER_OP_DICT


def adjust_5d_to_4d(graph):
    matches = [single_node_matcher(graph, type_name) for type_name in (
        'ArmActivation', 'ArmBatchNorm', 'ArmDiv', 'ArmInstanceNorm', 'ArmLRN', 'ArmMatMul', 'ArmSlice')]
    matches = extend_lists(matches)
    for m in matches:
        node_name = m['target']
        node_obj = NodeWrap(graph, node_name)['object']
        if node_obj is not None:
            input_shapes = node_obj.get_input_shapes()
            output_shapes = node_obj.get_output_shapes()
            if len(input_shapes) in (1, 2) \
                    and len(output_shapes) >= 1:
                if node_obj.type == 'ArmMatMul':
                    if len(input_shapes) != 2 \
                            or any((in_s is None or None in in_s or len(in_s) < 5) for in_s in input_shapes) \
                            or (output_shapes[0] is None or None in output_shapes[0]):
                        continue
                elif node_obj.type == 'ArmDiv':
                    if len(input_shapes) != 2 \
                            or any((in_s is None or None in in_s or len(in_s) < 6) for in_s in input_shapes) \
                            or (output_shapes[0] is None or None in output_shapes[0]):
                        continue
                else:
                    if any((in_s is None or None in in_s or len(in_s) != 5) for in_s in input_shapes):
                        continue
                if node_obj.type == 'ArmSlice' and input_shapes[0][0] != 1:
                    continue
                in_shape = input_shapes[0]
                in_edges = graph.sorted_in_edges(node_name, data=True)
                pre_dim = None
                old_dim = None
                if node_obj.type == 'ArmBatchNorm':
                    last_2_dim = np.gcd(int(np.prod(in_shape[1:-1])), 1920)
                    pre_dim = [in_shape[0],
                               int(np.prod(in_shape[1:-1])) // last_2_dim,
                               last_2_dim,
                               in_shape[-1]]
                    node_obj.axis = 3
                elif node_obj.type == 'ArmInstanceNorm':
                    node_obj.non_channel_axes = [1, 2]
                elif node_obj.type == 'ArmSlice':
                    pre_dim = list(in_shape[1:])
                    node_obj.starts = node_obj.starts[1:]
                    node_obj.ends = node_obj.ends[1:]
                    node_obj.steps = node_obj.steps[1:]

                for idx, in_edge in enumerate(in_edges):
                    if pre_dim is None or idx != 0:
                        in_shape_idx = input_shapes[idx]
                        pre_dim = [in_shape_idx[0],
                                   int(np.prod(in_shape_idx[1:-2])),
                                   in_shape_idx[-2],
                                   in_shape_idx[-1]]
                    src, _, in_attr = in_edge
                    insert_reshape(graph, src, node_name, in_attr,
                                   pre_dim, type='ArmReshape', quantize=node_obj.quantize)

                if node_obj.type in ('ArmMatMul', 'ArmDiv'):
                    old_dim = [output_shapes[0][0],
                               int(np.prod(output_shapes[0][1:-2])),
                               output_shapes[0][-2],
                               output_shapes[0][-1]]
                post_dim = copy.deepcopy(output_shapes[0])
                post_reshape = insert_reshape_after(
                    graph, node_name, post_dim, old_dim, type='ArmReshape', quantize=node_obj.quantize)

                if node_name in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(node_name)
                    graph._attr['output_names'][index] = post_reshape
        else:
            ERROR(
                '[Parser]: Meets invalid Activation Node (%s) in adjust_5d_activation_to_4d!' % node_name)


def adjust_pow(graph):
    matches = single_node_matcher(graph, 'ArmPow')
    for m in matches:
        pow = m['target']
        pow_obj = NodeWrap(graph, pow)['object']
        if pow_obj is not None:
            input_shapes = pow_obj.get_input_shapes()
            in_edges = graph.sorted_in_edges(pow, data=True)
            if len(input_shapes) == 2 \
                    and (len(input_shapes[1]) == 0 or (len(input_shapes[1]) == 1 and input_shapes[1][0] == 1)) \
                    and len(in_edges) == 2 \
                    and NodeWrap(graph, in_edges[1][0])['object'].type in ('Constant', 'ArmConstant') \
                    and NodeWrap(graph, in_edges[1][0])['object'].value is not None:
                main_in_shape = input_shapes[0]
                exp_const_obj = NodeWrap(graph, in_edges[1][0])['object']
                exp_const_obj.value = np.tile(np.reshape(
                    exp_const_obj.value, [1] * len(main_in_shape)), main_in_shape)
                in_edges[1][2]['tensor'] = Tensor(
                    value=exp_const_obj.value, shape=exp_const_obj.value.shape, is_const=True)
        else:
            ERROR('[Parser]: Meets invalid Pow Node (%s) in adjust_pow!' % pow)


def convert_uni_gru(graph):
    matched = False
    matches = single_node_matcher(graph, 'GRU')
    for m in matches:
        gru = m['target']
        gru_obj = NodeWrap(graph, gru)['object']
        in_edges = graph.sorted_in_edges(gru, data=True)
        out_edges = graph.sorted_out_edges(gru, data=True)
        if gru_obj is not None and len(in_edges) == 6:
            if gru_obj.direction != 'bidirectional':
                inputs = gru_obj.get_input_tensors()
                if inputs[4] is not None and np.any(inputs[4] != gru_obj.time_steps):
                    WARN(
                        '[Parser]: Cannot support GRU Op (%s) with different seq_length in convert_uni_gru!' % gru)
                    continue
                matched = True
                input_shapes = gru_obj.get_input_shapes()
                batch_size = input_shapes[0][1] if not gru_obj.layout else input_shapes[0][0]
                time_steps, hidden_size = gru_obj.time_steps, gru_obj.hidden_size

                _, _, inp_in_attr = in_edges[0]
                init_h, _, init_h_in_attr = in_edges[5]
                graph.remove_edges_from(in_edges[1:])
                init_h_in_attr['dst_in_port'] = 1
                graph.add_edge(init_h, gru, **init_h_in_attr)
                in_edges = graph.sorted_in_edges(gru, data=True)
                inp, _, inp_in_attr = in_edges[0]
                init_h, _, init_h_in_attr = in_edges[1]
                if not gru_obj.layout:
                    inp, _, inp_in_attr = in_edges[0]
                    insert_transpose(graph, inp, gru, inp_in_attr, [1, 0, 2])
                    insert_transpose(graph, init_h, gru,
                                     init_h_in_attr, [1, 0, 2])
                    in_edges = graph.sorted_in_edges(gru, data=True)
                    inp, _, inp_in_attr = in_edges[0]
                    init_h, _, init_h_in_attr = in_edges[1]
                insert_reshape(graph, init_h, gru, init_h_in_attr, [
                               batch_size, hidden_size],
                               quantize=gru_obj.quantize)

                out_rev = None
                if gru_obj.direction == 'reverse':
                    rev = get_valid_node_name(graph, gru + '_reverse')
                    graph.remove_edge(inp, gru)
                    graph.add_edge(inp, rev, **inp_in_attr)
                    gru_in_attr = copy.deepcopy(inp_in_attr)
                    if inp_in_attr['tensor'] is not None and inp_in_attr['tensor'].value is not None:
                        gru_in_attr['tensor'].value = np.flip(inp_in_attr['tensor'].value, 1)
                    gru_in_attr.update({'src_out_port': 0})
                    graph.add_edge(rev, gru, **gru_in_attr)
                    seq_len = np.array([time_steps] * batch_size, np.int32)
                    insert_constant(graph, rev + '_seq_len', seq_len, rev, in_port=1)
                    rev_seq_attr = {'name': rev, 'time_axis': 1, 'batch_axis': 0,
                                    'opset_version': 10}
                    NodeWrap(graph, rev).replace_obj('ReverseSequence', rev_seq_attr)
                    gru_obj.direction = 'forward'
                    if 0 in gru_obj.get_out_ports():
                        out_rev = get_valid_node_name(graph, gru + '_output_reverse')
                        out_rev_in_attr = {}
                        for _, dst, out_attr in out_edges:
                            if out_attr['src_out_port'] == 0:
                                graph.remove_edge(gru, dst)
                                graph.add_edge(out_rev, dst, **out_attr)
                                if not out_rev_in_attr:
                                    out_rev_in_attr = out_attr
                        out_rev_in_attr.update({'dst_in_port': 0})
                        graph.add_edge(gru, out_rev, **out_rev_in_attr)
                        insert_constant(graph, out_rev + '_seq_len', seq_len, out_rev, in_port=1)
                        out_rev_attr = {'name': out_rev, 'time_axis': 1, 'batch_axis': 0,
                                        'opset_version': 10}
                        NodeWrap(graph, out_rev).replace_obj('ReverseSequence', out_rev_attr)

                last_names = []
                out_ports = gru_obj.get_out_ports()
                for p in out_ports:
                    old_dim = [batch_size, time_steps, hidden_size] if p == 0 else [
                        batch_size, hidden_size]
                    reshape_dim = [batch_size, time_steps, 1, hidden_size] if p == 0 else [
                        batch_size, 1, hidden_size]
                    gru_out_node = (out_rev if p == 0 and out_rev is not None else gru)
                    reshape = insert_reshape_after(
                        graph, gru_out_node, reshape_dim, old_dim=old_dim, out_port=p,
                        quantize=gru_obj.quantize)
                    last_name = reshape
                    if not gru_obj.layout:
                        post_trans_perm = [1, 2, 0, 3] if p == 0 else [1, 0, 2]
                        post_trans = get_valid_node_name(
                            graph, reshape + '_post_transpose')
                        reshape_out_edges = graph.sorted_out_edges(
                            reshape, data=True)
                        for _, dst, out_attr in reshape_out_edges:
                            graph.remove_edge(reshape, dst)
                            graph.add_edge(post_trans, dst, **out_attr)
                        reshape_out_tensor = Tensor()
                        if len(reshape_out_edges) > 0 and reshape_out_edges[0][2].get('tensor', None) is not None:
                            reshape_out_shape = out_attr['tensor'].get_shape()
                            if reshape_out_shape is not None:
                                reshape_out_tensor.shape = [reshape_out_shape[i]
                                                            for i in Op.cal_inverse_perm(post_trans_perm)]
                        graph.add_edge(reshape, post_trans, **{
                                       'src_out_port': 0, 'dst_in_port': 0, 'tensor': reshape_out_tensor})
                        post_trans_attr = gru_obj.copied_attr()
                        post_trans_attr.update(
                            {'name': post_trans, 'perm': post_trans_perm, 'opset_version': 1})
                        NodeWrap(graph, post_trans).replace_obj(
                            'Transpose', post_trans_attr)
                        last_name = post_trans
                    last_names.append(last_name)

                if gru in graph._attr['output_names'] and last_names:
                    index = graph._attr['output_names'].index(gru)
                    graph._attr['output_names'].remove(gru)
                    for name in last_names:
                        graph._attr['output_names'].insert(index, name)
                        index += 1

                new_gru_attr = gru_obj.copied_attr()
                new_gru_attr['weights'] = np.squeeze(
                    new_gru_attr['weights'], axis=0)
                new_gru_attr['biases'] = np.squeeze(
                    new_gru_attr['biases'], axis=0)
                if new_gru_attr.get('clip', None) is not None:
                    new_gru_attr.update(
                        {'threshold': float(new_gru_attr['clip'])})
                gru_type = 'ArmGRUv1' if gru_obj.linear_before_reset else 'ArmGRUv3'
                NodeWrap(graph, gru).replace_obj(gru_type, new_gru_attr)
        else:
            ERROR('[Parser]: Meets invalid GRU Node (%s) in convert_gru!' % gru)
    if matched:
        clear_redundant_nodes(graph)


def convert_bi_gru(graph):
    matched = False
    matches = single_node_matcher(graph, 'GRU')
    for m in matches:
        gru = m['target']
        gru_obj = NodeWrap(graph, gru)['object']
        in_edges = graph.sorted_in_edges(gru, keys=True, data=True)
        out_edges = graph.sorted_out_edges(gru, keys=True, data=True)
        if gru_obj is not None and len(in_edges) == 6 and len(out_edges) >= 1:
            init_h, _, _, init_h_in_attr = in_edges[5]
            init_h_obj = NodeWrap(graph, init_h)['object']
            if init_h_obj is not None:
                if gru_obj.direction == 'bidirectional':
                    matched = True
                    time_steps = gru_obj.time_steps
                    input_size = gru_obj.input_size
                    hidden_size = gru_obj.hidden_size
                    batch_size = gru_obj.get_input_shapes(
                    )[0][0] if gru_obj.layout else gru_obj.get_input_shapes()[0][1]

                    inp, _, _, inp_in_attr = in_edges[0]
                    _, _, _, init_h_in_attr = in_edges[5]
                    graph.remove_edges_from(in_edges[1:])
                    new_init_h_in_attr = copy.deepcopy(init_h_in_attr)
                    new_init_h_in_attr['dst_in_port'] = 1
                    graph.add_edge(init_h, gru, **new_init_h_in_attr)
                    in_edges = graph.sorted_in_edges(gru, keys=True, data=True)
                    if not gru_obj.layout:
                        inp, _, k0, inp_in_attr = in_edges[0]
                        _, _, k1, init_h_in_attr = in_edges[1]
                        insert_transpose(graph, inp, gru, inp_in_attr, [
                                         1, 0, 2], key=k0)
                        insert_transpose(graph, init_h, gru,
                                         init_h_in_attr, [1, 0, 2], key=k1)
                        in_edges = graph.sorted_in_edges(
                            gru, keys=True, data=True)

                    inp, _, k0, inp_in_attr = in_edges[0]
                    init_h, _, k1, init_h_in_attr = in_edges[1]
                    graph.remove_edges_from(in_edges)

                    state_split = get_valid_node_name(graph, init_h + '_split')
                    state_reshape_fw = get_valid_node_name(
                        graph, init_h + '_reshape_fw')
                    state_reshape_bw = get_valid_node_name(
                        graph, init_h + '_reshape_bw')
                    fw_gru = get_valid_node_name(graph, gru + '_fw')
                    bw_gru = get_valid_node_name(graph, gru + '_bw')
                    reverse1 = get_valid_node_name(graph, bw_gru + '_reverse1')
                    reverse2 = get_valid_node_name(graph, bw_gru + '_reverse2')
                    fw_reshape = get_valid_node_name(
                        graph, fw_gru + '_reshape')
                    bw_reshape = get_valid_node_name(
                        graph, bw_gru + '_reshape')
                    concat = get_valid_node_name(graph, gru + '_seq_concat')

                    new_init_h_in_attr = copy.deepcopy(init_h_in_attr)
                    new_init_h_in_attr['dst_in_port'] = 0
                    init_state_tensor = new_init_h_in_attr['tensor'].value if new_init_h_in_attr.get(
                        'tensor', None) is not None else None
                    if init_state_tensor is not None:
                        split_state_fw_tensor, split_state_bw_tensor = np.split(
                            init_state_tensor, 2, axis=1)
                        state_reshape_fw_tensor = np.reshape(
                            split_state_fw_tensor, [batch_size, hidden_size])
                        state_reshape_bw_tensor = np.reshape(
                            split_state_bw_tensor, [batch_size, hidden_size])
                    else:
                        split_state_fw_tensor, split_state_bw_tensor = None, None
                        state_reshape_fw_tensor, state_reshape_bw_tensor = None, None

                    graph.add_edge(init_h, state_split, **new_init_h_in_attr)
                    graph.add_edge(state_split, state_reshape_fw, **
                                   {'tensor': Tensor(value=split_state_fw_tensor)})
                    graph.add_edge(state_split, state_reshape_bw, **{
                                   'src_out_port': 1, 'dst_in_port': 0, 'tensor': Tensor(value=split_state_bw_tensor)})
                    graph.add_edge(state_reshape_fw, fw_gru, **{
                                   'src_out_port': 0, 'dst_in_port': 1, 'tensor': Tensor(value=state_reshape_fw_tensor)})
                    graph.add_edge(state_reshape_bw, bw_gru, **{
                                   'src_out_port': 0, 'dst_in_port': 1, 'tensor': Tensor(value=state_reshape_bw_tensor)})
                    graph.add_edge(inp, fw_gru, **inp_in_attr)
                    graph.add_edge(inp, reverse1, **inp_in_attr)
                    graph.add_edge(reverse1, bw_gru)
                    graph.add_edge(bw_gru, bw_reshape)
                    graph.add_edge(fw_gru, fw_reshape)
                    graph.add_edge(bw_reshape, reverse2)
                    graph.add_edge(fw_reshape, concat)
                    graph.add_edge(reverse2, concat, **
                                   {'src_out_port': 0, 'dst_in_port': 1})

                    state_split_attr = {
                        'name': state_split, 'opset_version': 2, 'axis': 1, 'split': [1, 1]}
                    NodeWrap(graph, state_split).replace_obj(
                        'Split', state_split_attr)

                    NodeWrap(graph, state_reshape_fw).replace_obj(
                        'Reshape', {'name': state_reshape_fw, 'opset_version': 5})
                    insert_constant(graph,
                                    state_reshape_fw + '_shape',
                                    np.array(
                                        [batch_size, hidden_size], np.int64),
                                    state_reshape_fw,
                                    in_port=1)

                    NodeWrap(graph, state_reshape_bw).replace_obj(
                        'Reshape', {'name': state_reshape_bw, 'opset_version': 5})
                    insert_constant(graph,
                                    state_reshape_bw + '_shape',
                                    np.array(
                                        [batch_size, hidden_size], np.int64),
                                    state_reshape_bw,
                                    in_port=1)

                    fw_gru_attr = gru_obj.copied_attr()
                    fw_gru_attr.update({'name': fw_gru,
                                        'time_steps': time_steps,
                                        'input_size': input_size,
                                        'hidden_size': hidden_size,
                                        'weights': gru_obj.weights[0, ...],
                                        'biases': gru_obj.biases[0, ...],
                                        'direction': 'forward',
                                        'activations': gru_obj.activations[:2],
                                        'method': 'Y'
                                        })
                    if fw_gru_attr.get('clip', None) is not None:
                        fw_gru_attr.update(
                            {'threshold': float(fw_gru_attr['clip'])})
                    if gru_obj.activation_alpha:
                        fw_gru_attr.update({'activation_alpha': gru_obj.activation_alpha[0: len(
                            gru_obj.activation_alpha) // 2]})
                    if gru_obj.activation_beta:
                        fw_gru_attr.update(
                            {'activation_beta': gru_obj.activation_beta[0: len(gru_obj.activation_beta) // 2]})
                    NodeWrap(graph, fw_gru).replace_obj(
                        'ArmGRUv3' if not gru_obj.linear_before_reset else 'ArmGRUv1', fw_gru_attr)

                    bw_gru_attr = gru_obj.copied_attr()
                    bw_gru_attr.update({'name': bw_gru,
                                        'time_steps': time_steps,
                                        'input_size': input_size,
                                        'hidden_size': hidden_size,
                                        'weights': gru_obj.weights[1, ...],
                                        'biases': gru_obj.biases[1, ...],
                                        'direction': 'forward',
                                        'activations': gru_obj.activations[2:],
                                        'method': 'Y'
                                        })
                    if bw_gru_attr.get('clip', None) is not None:
                        bw_gru_attr.update(
                            {'threshold': float(bw_gru_attr['clip'])})
                    if gru_obj.activation_alpha:
                        bw_gru_attr.update(
                            {'activation_alpha': gru_obj.activation_alpha[len(gru_obj.activation_alpha) // 2:]})
                    if gru_obj.activation_beta:
                        bw_gru_attr.update(
                            {'activation_beta': gru_obj.activation_beta[len(gru_obj.activation_beta) // 2:]})
                    NodeWrap(graph, bw_gru).replace_obj(
                        'ArmGRUv3' if not gru_obj.linear_before_reset else 'ArmGRUv1', bw_gru_attr)

                    reverse1_attr = gru_obj.copied_attr()
                    reverse1_attr.update({'name': reverse1,
                                          'batch_axis': 0,
                                          'time_axis': 1,
                                          'opset_version': 10})
                    reverse1_seq_len = np.ndarray([batch_size, ], np.int32)
                    for b in range(batch_size):
                        reverse1_seq_len[b] = time_steps
                    insert_constant(graph, reverse1 + '_seq_len',
                                    reverse1_seq_len, reverse1, in_port=1)
                    NodeWrap(graph, reverse1).replace_obj(
                        'ReverseSequence', reverse1_attr)

                    reverse2_attr = gru_obj.copied_attr()
                    reverse2_attr.update({'name': reverse2,
                                          'batch_axis': 0,
                                          'time_axis': 1,
                                          'opset_version': 10})
                    reverse2_seq_len = np.ndarray([batch_size, ], np.int32)
                    for b in range(batch_size):
                        reverse2_seq_len[b] = time_steps
                    insert_constant(graph, reverse2 + '_seq_len', reverse2_seq_len,
                                    reverse2, in_port=1, data_format='NHWC')
                    NodeWrap(graph, reverse2).replace_obj(
                        'ReverseSequence', reverse2_attr)

                    NodeWrap(graph, fw_reshape).replace_obj(
                        'Reshape', {'name': fw_reshape, 'opset_version': 5})
                    insert_constant(graph,
                                    fw_reshape + '_shape',
                                    np.array([batch_size, time_steps,
                                              1, hidden_size], np.int64),
                                    fw_reshape,
                                    in_port=1)

                    NodeWrap(graph, bw_reshape).replace_obj(
                        'Reshape', {'name': bw_reshape, 'opset_version': 5})
                    insert_constant(graph,
                                    bw_reshape + '_shape',
                                    np.array([batch_size, time_steps,
                                              1, hidden_size], np.int64),
                                    bw_reshape,
                                    in_port=1)

                    NodeWrap(graph, concat).replace_obj(
                        'Concat', {'name': concat, 'opset_version': 4, 'axis': 2})

                    if not gru_obj.layout:
                        concat_transpose = get_valid_node_name(
                            graph, concat + '_transpose')
                        graph.add_edge(concat, concat_transpose)
                        NodeWrap(graph, concat_transpose).replace_obj('Transpose',
                                                                      {'name': concat_transpose,
                                                                       'opset_version': 1,
                                                                       'perm': [1, 2, 0, 3]}
                                                                      )
                        bi_gru_out = concat_transpose
                    else:
                        bi_gru_out = concat

                    last_names = []
                    out_ports = gru_obj.get_out_ports()
                    for p in out_ports:
                        if p == 0:
                            gru_out_name = bi_gru_out
                        else:
                            slice_in_shape = [batch_size, time_steps, 2, hidden_size]
                            concat_hidden = get_valid_node_name(graph, gru + '_out_state_concat')
                            slice = get_valid_node_name(
                                graph, gru + '_out_state_slice')
                            reshape = get_valid_node_name(
                                graph, gru + '_out_state_reshape')
                            graph.add_edge(fw_reshape, concat_hidden)
                            graph.add_edge(bw_reshape, concat_hidden, **{'dst_in_port': 1})
                            graph.add_edge(
                                concat_hidden, slice, **{'tensor': Tensor(shape=slice_in_shape)})
                            graph.add_edge(slice, reshape)

                            concat_hidden_attr = {'name': concat_hidden, 'opset_version': 4, 'axis': 2}
                            NodeWrap(graph, concat_hidden).replace_obj('Concat', concat_hidden_attr)

                            slice_attr = {'name': slice,
                                          'opset_version': 1,
                                          'axes': [0, 1, 2, 3],
                                          'starts': [0, time_steps - 1, 0, 0],
                                          'ends': [batch_size, time_steps, 2, hidden_size]
                                          }
                            NodeWrap(graph, slice).replace_obj(
                                'Slice', slice_attr)

                            reshape_attr = {
                                'name': reshape, 'opset_version': 5}
                            NodeWrap(graph, reshape).replace_obj(
                                'Reshape', reshape_attr)
                            insert_constant(graph, reshape + '_shape', np.array(
                                [batch_size, 2, hidden_size], np.int64), reshape, in_port=1)
                            if not gru_obj.layout:
                                transpose = get_valid_node_name(
                                    graph, reshape + '_post_transpose')
                                graph.add_edge(reshape, transpose)
                                NodeWrap(graph, transpose).replace_obj('Transpose', {
                                    'name': transpose, 'opset_version': 1, 'perm': [1, 0, 2]})
                                gru_out_name = transpose
                            else:
                                gru_out_name = reshape

                        for _, dst, k, out_attr in out_edges:
                            if out_attr['src_out_port'] == p:
                                graph.remove_edge(gru, dst, key=k)
                                new_out_attr = copy.deepcopy(out_attr)
                                new_out_attr['src_out_port'] = 0
                                graph.add_edge(
                                    gru_out_name, dst, **new_out_attr)
                        last_names.append(gru_out_name)

                    if gru in graph._attr['output_names'] and last_names:
                        index = graph._attr['output_names'].index(gru)
                        graph._attr['output_names'].remove(gru)
                        for name in last_names:
                            graph._attr['output_names'].insert(index, name)
                            index += 1
            else:
                ERROR('[Parser]: Meets invalid GRU Node (%s) in convert_bi_gru!' % gru)
        else:
            ERROR('[Parser]: Meets invalid GRU Node (%s) in convert_bi_gru!' % gru)

    if matched:
        clear_redundant_nodes(graph)


def convert_uni_lstm(graph):
    matches = single_node_matcher(graph, 'LSTM')
    matched = False
    for m in matches:
        lstm = m['target']
        lstm_obj = NodeWrap(graph, lstm)['object']
        if lstm_obj is not None:
            in_edges = graph.sorted_in_edges(lstm, data=True)
            out_edges = graph.sorted_out_edges(lstm, keys=True, data=True)
            if lstm_obj.direction != 'bidirectional' and len(in_edges) == 8 and len(out_edges) >= 1:
                if lstm_obj.input_forget:
                    WARN(
                        '[Parser]: Cannot support LSTM Op (%s) with input_forget=True in convert_uni_lstm!' % lstm)
                    continue
                inputs = lstm_obj.get_input_tensors()
                if inputs[4] is not None and np.any(inputs[4] != lstm_obj.time_steps):
                    WARN(
                        '[Parser]: Cannot support LSTM Op (%s) with different seq_length in convert_uni_lstm!' % lstm)
                    continue
                if inputs[7] is not None and np.any(inputs[7] != 0):
                    WARN(
                        '[Parser]: Cannot support LSTM Op (%s) with none-zero peepholes in convert_uni_lstm!' % lstm)
                    continue
                matched = True
                if graph._attr.get('quantize', False) \
                        and lstm_obj.quantize:
                    quantize = True
                else:
                    quantize = False
                time_steps = lstm_obj.time_steps
                input_size = lstm_obj.input_size
                hidden_size = lstm_obj.hidden_size
                batch_size = lstm_obj.get_input_shapes(
                )[0][0] if lstm_obj.layout else lstm_obj.get_input_shapes()[0][1]

                inp, _, inp_in_attr = in_edges[0]
                init_h, _, init_h_in_attr = in_edges[5]
                init_c, _, init_c_in_attr = in_edges[6]
                graph.remove_edges_from(in_edges[1:])
                init_h_in_attr['dst_in_port'] = 1
                init_c_in_attr['dst_in_port'] = 2
                graph.add_edge(init_h, lstm, **init_h_in_attr)
                graph.add_edge(init_c, lstm, **init_c_in_attr)

                in_edges = graph.sorted_in_edges(lstm, keys=True, data=True)
                if not lstm_obj.layout:
                    _, _, k0, inp_in_attr = in_edges[0]
                    _, _, k1, init_h_in_attr = in_edges[1]
                    _, _, k2, init_c_in_attr = in_edges[2]
                    insert_transpose(graph, inp, lstm,
                                     inp_in_attr, [1, 0, 2], key=k0, quantize=quantize)
                    insert_transpose(graph, init_h, lstm,
                                     init_h_in_attr, [1, 0, 2], key=k1, quantize=quantize)
                    insert_transpose(graph, init_c, lstm,
                                     init_c_in_attr, [1, 0, 2], key=k2, quantize=quantize)
                    in_edges = graph.sorted_in_edges(
                        lstm, keys=True, data=True)
                init_h, _, k1, init_h_in_attr = in_edges[1]
                init_c, _, k2, init_c_in_attr = in_edges[2]
                insert_reshape(graph, init_h, lstm, init_h_in_attr, [
                               batch_size, hidden_size], key=k1, quantize=quantize)
                insert_reshape(graph, init_c, lstm, init_c_in_attr, [
                               batch_size, hidden_size], key=k2, quantize=quantize)

                lstm_attr = lstm_obj.copied_attr()
                lstm_attr.update({'time_steps': time_steps,
                                  'input_size': input_size,
                                  'hidden_size': hidden_size,
                                  'weights': lstm_obj.weights[0, ...],
                                  'biases': lstm_obj.biases[0, ...],
                                  'direction': lstm_obj.direction
                                  })
                if lstm_obj.activation_alpha:
                    lstm_attr.update(
                        {'activation_alpha': lstm_obj.activation_alpha})
                if lstm_obj.activation_beta:
                    lstm_attr.update(
                        {'activation_beta': lstm_obj.activation_beta})
                if lstm_obj.clip:
                    lstm_attr.update({'threshold': float(lstm_obj.clip)})
                if lstm_obj.cell_clip:
                    lstm_attr.update({'cell_clip': float(lstm_obj.cell_clip)})
                NodeWrap(graph, lstm).replace_obj('ArmBasicLSTM', lstm_attr)

                last_names = []
                out_ports = lstm_obj.get_out_ports()
                for p in out_ports:
                    old_dim = [batch_size, time_steps, hidden_size] if p == 0 else [
                        batch_size, hidden_size]
                    reshape_dim = [batch_size, time_steps, 1, hidden_size] if p == 0 else [
                        batch_size, 1, hidden_size]
                    reshape = insert_reshape_after(
                        graph, lstm, reshape_dim, old_dim=old_dim, out_port=p, quantize=quantize)
                    last_name = reshape
                    if not lstm_obj.layout:
                        post_trans_perm = [1, 2, 0, 3] if p == 0 else [1, 0, 2]
                        last_name = insert_transpose_after(graph, reshape, post_trans_perm, quantize=quantize)
                    last_names.append(last_name)

                if lstm in graph._attr['output_names'] and last_names:
                    index = graph._attr['output_names'].index(lstm)
                    graph._attr['output_names'].remove(lstm)
                    for name in last_names:
                        if any([has_path(graph, name, cur_out_name) for cur_out_name in graph._attr['output_names']]):
                            continue
                        graph._attr['output_names'].insert(index, name)
                        index += 1
        else:
            ERROR('[Parser]: Meets invalid Node in convert_lstm!')

    if matched:
        clear_redundant_nodes(graph)


def rename_quantize_dequantize(graph):
    matches = single_node_matcher(
        graph, ['QuantizeLinear', 'DequantizeLinear'])
    for m in matches:
        quantize = m['target']
        quantize_obj = NodeWrap(graph, quantize)['object']
        in_edges = graph.sorted_in_edges(quantize, keys=True, data=True)
        out_edges = graph.sorted_out_edges(quantize, keys=True, data=True)

        if quantize_obj is None or len(in_edges) != 3 or len(out_edges) < 1:
            ERROR('[Parser]: Meets invalid QuantizeLinear/DequantizeLinear Op(%s) in rename_quantize_dequantize!' % quantize)
            continue

        if quantize_obj.type == 'QuantizeLinear':
            dtype = out_edges[0][3]['tensor'].get_dtype()
        else:
            dtype = in_edges[0][3]['tensor'].get_dtype()
        if dtype is None:
            ERROR('[Parser]: Meets invalid dtype of QuantizeLinear/DequantizeLinear Op(%s) in rename_quantize_dequantize!' % quantize)
            continue
        dtype_str = str(dtype)

        scale_name, _, k1, scale_in_attr = in_edges[1]
        zp_name, _, k2, zp_in_attr = in_edges[2]
        _, _, _, inp_attr = in_edges[0]

        if not scale_in_attr['tensor'].is_const \
                or not zp_in_attr['tensor'].is_const:
            WARN('[Parser]: QuantizeLinear/DequantizeLinear Op(%s) with non-constant scale/zp should have been split before in rename_quantize_dequantize!' % quantize)
            continue
        if scale_in_attr['tensor'].value is None \
                or zp_in_attr['tensor'].value is None \
                or list(scale_in_attr['tensor'].value.shape) != list(zp_in_attr['tensor'].value.shape):
            ERROR('[Parser]: QuantizeLinear/DequantizeLinear Op(%s) with invalid scale/zp in rename_quantize_dequantize!' % quantize)
            continue
        scale = scale_in_attr['tensor'].value
        zp = zp_in_attr['tensor'].value
        if quantize_obj.type == 'DequantizeLinear':
            if not inp_attr['tensor'].scale_zp:
                inp_attr['tensor'].scale_zp = (scale, zp)
                inp_attr['tensor'].activation_quantization_axis = quantize_obj.axis
        else:
            for _, _, _, out_attr in out_edges:
                if not out_attr['tensor'].scale_zp:
                    out_attr['tensor'].scale_zp = (scale, zp)
                    out_attr['tensor'].activation_quantization_axis = quantize_obj.axis
        graph.remove_edge(scale_name, quantize, key=k1)
        graph.remove_edge(zp_name, quantize, key=k2)
        quantize_attr = quantize_obj.copied_attr()
        quantize_attr.update({'scale': scale, 'zero_point': zp})
        if quantize_obj.type == 'QuantizeLinear':
            quantize_attr.update({'to_dtype': dtype_str, 'round_mode': 'ROUND_TO_EVEN'})
            NodeWrap(graph, quantize).replace_obj('ArmQuantize', quantize_attr)
        else:
            quantize_attr.update({'from_dtype': dtype_str})
            NodeWrap(graph, quantize).replace_obj(
                'ArmDeQuantize', quantize_attr)


def rename_dilation_erosion(graph):
    ero_dila = ['Erosion', 'Dilation']
    matches = [single_node_matcher(graph, ero_dila_type)
               for ero_dila_type in ero_dila]
    matches = extend_lists(matches)
    for m in matches:
        ero_dila = m['target']
        ero_dila_obj = NodeWrap(graph, ero_dila)['object']
        if ero_dila_obj is None:
            ERROR('[Parser]: Meets invalid node(%s) in rename_dilation_erosion!' % ero_dila)
            continue
        ero_dila_attr = ero_dila_obj.copied_attr()
        if len(ero_dila_obj.weights.shape) == 3:
            new_weights = np.reshape(ero_dila_obj.weights, list(ero_dila_obj.weights.shape) + [1])
            ero_dila_attr.update({'weights': new_weights})
            if ero_dila_obj.type == 'Erosion':
                NodeWrap(graph, ero_dila).replace_obj('ArmErosion', ero_dila_attr)
            elif ero_dila_obj.type == 'Dilation':
                NodeWrap(graph, ero_dila).replace_obj('ArmDilation', ero_dila_attr)
        else:
            ERROR('[Parser]: Meets invalid weights(%s) in rename_dilation_erosion!' % ero_dila)


def rename_div(graph):
    matches = single_node_matcher(graph, 'Div')
    for m in matches:
        div = m['target']
        div_obj = NodeWrap(graph, div)['object']
        if div_obj is None:
            ERROR('[Parser]: Meets invalid Div node(%s) in rename_div!' % div)
            continue
        div_in_edges = graph.sorted_in_edges(div, data=True)
        if len(div_in_edges) < 2 \
                or div_in_edges[0][2]['tensor'] is None \
                or div_in_edges[0][2]['tensor'].get_dtype() is None \
                or div_in_edges[1][2]['tensor'] is None \
                or div_in_edges[1][2]['tensor'].get_dtype() is None:
            ERROR('[Parser]: Meets invalid inputs of Div node(%s) in rename_div!' % div)
            continue
        src1_type = div_in_edges[0][2]['tensor'].get_dtype()
        src2_type = div_in_edges[1][2]['tensor'].get_dtype()
        if src1_type is None or src2_type is None:
            ERROR('[Parser]: Meets invalid inputs of Div node(%s) in rename_div!' % div)
            continue
        div_attr = div_obj.copied_attr()
        if 'int' in src1_type and 'int' in src2_type:
            # DivMod op has 2 outputs, which are quotient(x//y) and remainder(x%y).
            # Need to add Out node after the second output.
            remainder_out = get_valid_node_name(graph, div + '_remainder')
            graph.add_edge(div, remainder_out, **{'src_out_port': 1})
            NodeWrap(graph, remainder_out).replace_obj('Out', {'name': remainder_out})
            div_attr.update({'mode': 'trunc'})
            NodeWrap(graph, div).replace_obj('ArmDivMod', div_attr)
        else:
            NodeWrap(graph, div).replace_obj('ArmDiv', div_attr)


def convert_bi_lstm(graph):
    matches = single_node_matcher(graph, 'LSTM')
    matched = False
    for m in matches:
        lstm = m['target']
        lstm_obj = NodeWrap(graph, lstm)['object']
        in_edges = graph.sorted_in_edges(lstm, keys=True, data=True)
        out_edges = graph.sorted_out_edges(lstm, keys=True, data=True)
        if lstm_obj is not None and len(in_edges) == 8 and len(out_edges) >= 1:
            if lstm_obj.direction == 'bidirectional':
                if lstm_obj.input_forget:
                    WARN(
                        '[Parser]: Cannot support LSTM Op (%s) with input_forget=True in convert_bi_lstm!' % lstm)
                    continue
                inputs = lstm_obj.get_input_tensors()
                if inputs[4] is not None and np.any(inputs[4] != lstm_obj.time_steps):
                    WARN(
                        '[Parser]: Cannot support LSTM Op (%s) with different seq_length in convert_bi_lstm!' % lstm)
                    continue
                if inputs[7] is not None and np.any(inputs[7] != 0):
                    WARN(
                        '[Parser]: Cannot support LSTM Op (%s) with none-zero peepholes in convert_bi_lstm!' % lstm)
                    continue

                matched = True
                time_steps = lstm_obj.time_steps
                input_size = lstm_obj.input_size
                hidden_size = lstm_obj.hidden_size
                batch_size = lstm_obj.get_input_shapes(
                )[0][0] if lstm_obj.layout else lstm_obj.get_input_shapes()[0][1]

                inp, _, inp_k, inp_in_attr = in_edges[0]
                init_h, _, init_h_k, init_h_in_attr = in_edges[5]
                init_c, _, init_c_k, init_c_in_attr = in_edges[6]
                if not lstm_obj.layout:
                    insert_transpose(graph, inp, lstm, inp_in_attr, [
                                     1, 0, 2], key=inp_k)
                    insert_transpose(graph, init_h, lstm, init_h_in_attr, [
                                     1, 0, 2], key=init_h_k)
                    insert_transpose(graph, init_c, lstm, init_c_in_attr, [
                                     1, 0, 2], key=init_c_k)
                    in_edges = graph.sorted_in_edges(
                        lstm, keys=True, data=True)
                    inp, _, inp_k, inp_in_attr = in_edges[0]
                    init_h, _, init_h_k, init_h_in_attr = in_edges[5]
                    init_c, _, init_c_k, init_c_in_attr = in_edges[6]

                init_h_split = get_valid_node_name(
                    graph, init_h + '_split')        # axis=1, split=[1,1]
                init_h_fw_reshape = get_valid_node_name(
                    graph, init_h + '_fw_reshape')  # [batch_size, hidden_size]
                init_h_bw_reshape = get_valid_node_name(
                    graph, init_h + '_bw_reshape')  # [batch_size, hidden_size]
                init_c_split = get_valid_node_name(
                    graph, init_c + '_split')        # axis=1, split=[1,1]
                init_c_fw_reshape = get_valid_node_name(
                    graph, init_c + '_fw_reshape')  # [batch_size, hidden_size]
                init_c_bw_reshape = get_valid_node_name(
                    graph, init_c + '_bw_reshape')  # [batch_size, hidden_size]
                fw_lstm = get_valid_node_name(graph, lstm + '_fw')
                bw_lstm = get_valid_node_name(graph, lstm + '_bw')
                reverse1 = get_valid_node_name(graph, lstm + '_bw_reverse1')
                reverse2 = get_valid_node_name(graph, lstm + '_bw_reverse2')

                fw_y_reshape = get_valid_node_name(
                    graph, lstm + '_fw_y_reshape')
                bw_y_reshape = get_valid_node_name(
                    graph, lstm + '_bw_y_reshape')
                y_concat = get_valid_node_name(graph, lstm + '_y_concat')

                fw_y_h_reshape = get_valid_node_name(
                    graph, lstm + '_fw_y_h_reshape')
                bw_y_h_reshape = get_valid_node_name(
                    graph, lstm + '_bw_y_h_reshape')
                y_h_concat = get_valid_node_name(graph, lstm + '_y_h_concat')

                fw_y_c_reshape = get_valid_node_name(
                    graph, lstm + '_fw_y_c_reshape')
                bw_y_c_reshape = get_valid_node_name(
                    graph, lstm + '_bw_y_c_reshape')
                y_c_concat = get_valid_node_name(graph, lstm + '_y_c_concat')

                graph.remove_edges_from(in_edges)

                new_init_h_in_attr = copy.deepcopy(init_h_in_attr)
                new_init_h_in_attr['dst_in_port'] = 0
                graph.add_edge(init_h, init_h_split, **new_init_h_in_attr)
                graph.add_edge(init_h_split, init_h_fw_reshape,
                               **{'src_out_port': 0, 'dst_in_port': 0})
                graph.add_edge(init_h_split, init_h_bw_reshape,
                               **{'src_out_port': 1, 'dst_in_port': 0})
                graph.add_edge(init_h_fw_reshape, fw_lstm, **
                               {'src_out_port': 0, 'dst_in_port': 1})
                graph.add_edge(init_h_bw_reshape, bw_lstm, **
                               {'src_out_port': 0, 'dst_in_port': 1})

                new_init_c_in_attr = copy.deepcopy(init_c_in_attr)
                new_init_c_in_attr['dst_in_port'] = 0
                graph.add_edge(init_c, init_c_split, **new_init_c_in_attr)
                graph.add_edge(init_c_split, init_c_fw_reshape,
                               **{'src_out_port': 0, 'dst_in_port': 0})
                graph.add_edge(init_c_split, init_c_bw_reshape,
                               **{'src_out_port': 1, 'dst_in_port': 0})
                graph.add_edge(init_c_fw_reshape, fw_lstm, **
                               {'src_out_port': 0, 'dst_in_port': 2})
                graph.add_edge(init_c_bw_reshape, bw_lstm, **
                               {'src_out_port': 0, 'dst_in_port': 2})

                graph.add_edge(inp, fw_lstm, **inp_in_attr)
                graph.add_edge(inp, reverse1, **inp_in_attr)
                graph.add_edge(reverse1, bw_lstm)
                graph.add_edge(bw_lstm, reverse2)
                graph.add_edge(fw_lstm, fw_y_reshape)
                graph.add_edge(reverse2, bw_y_reshape)
                graph.add_edge(fw_y_reshape, y_concat)
                graph.add_edge(bw_y_reshape, y_concat, **
                               {'src_out_port': 0, 'dst_in_port': 1})
                graph.add_edge(fw_lstm, fw_y_h_reshape, **
                               {'src_out_port': 1, 'dst_in_port': 0})
                graph.add_edge(bw_lstm, bw_y_h_reshape, **
                               {'src_out_port': 1, 'dst_in_port': 0})
                graph.add_edge(fw_y_h_reshape, y_h_concat)
                graph.add_edge(bw_y_h_reshape, y_h_concat, **
                               {'src_out_port': 0, 'dst_in_port': 1})
                graph.add_edge(fw_lstm, fw_y_c_reshape, **
                               {'src_out_port': 2, 'dst_in_port': 0})
                graph.add_edge(bw_lstm, bw_y_c_reshape, **
                               {'src_out_port': 2, 'dst_in_port': 0})
                graph.add_edge(fw_y_c_reshape, y_c_concat)
                graph.add_edge(bw_y_c_reshape, y_c_concat, **
                               {'src_out_port': 0, 'dst_in_port': 1})

                y_out, y_h_out, y_c_out = y_concat, y_h_concat, y_c_concat
                if not lstm_obj.layout:
                    y_out_trans = get_valid_node_name(
                        graph, y_out + '_post_trans')
                    y_h_out_trans = get_valid_node_name(
                        graph, y_h_out + '_post_trans')
                    y_c_out_trans = get_valid_node_name(
                        graph, y_c_out + '_post_trans')
                    graph.add_edge(y_out, y_out_trans)
                    graph.add_edge(y_h_out, y_h_out_trans)
                    graph.add_edge(y_c_out, y_c_out_trans)
                    NodeWrap(graph, y_out_trans).replace_obj('Transpose', {
                        'name': y_out_trans, 'perm': [1, 2, 0, 3], 'opset_version': 1})
                    NodeWrap(graph, y_h_out_trans).replace_obj('Transpose', {
                        'name': y_h_out_trans, 'perm': [1, 0, 2], 'opset_version': 1})
                    NodeWrap(graph, y_c_out_trans).replace_obj('Transpose', {
                        'name': y_c_out_trans, 'perm': [1, 0, 2], 'opset_version': 1})
                    y_out, y_h_out, y_c_out = y_out_trans, y_h_out_trans, y_c_out_trans

                lstm_end_names = []
                out_ports = lstm_obj.get_out_ports()
                for p in out_ports:
                    out_node_name = y_out if p == 0 else (
                        y_h_out if p == 1 else y_c_out)
                    lstm_end_names.append(out_node_name)
                    for _, dst, k, out_attr in out_edges:
                        if out_attr['src_out_port'] == p:
                            graph.remove_edge(lstm, dst, key=k)
                            new_out_attr = copy.deepcopy(out_attr)
                            new_out_attr['src_out_port'] = 0
                            graph.add_edge(out_node_name, dst, **new_out_attr)

                if 'Y' not in list(lstm_obj.method):
                    y_out_noop = get_valid_node_name(graph, y_out + '_out')
                    graph.add_edge(y_out, y_out_noop)
                    NodeWrap(graph, y_out_noop).replace_obj(
                        'Out', {'name': y_out_noop})
                if 'H' not in list(lstm_obj.method):
                    y_h_out_noop = get_valid_node_name(graph, y_h_out + '_out')
                    graph.add_edge(y_h_out, y_h_out_noop)
                    NodeWrap(graph, y_h_out_noop).replace_obj(
                        'Out', {'name': y_h_out_noop})
                if 'C' not in list(lstm_obj.method):
                    y_c_out_noop = get_valid_node_name(graph, y_c_out + '_out')
                    graph.add_edge(y_c_out, y_c_out_noop)
                    NodeWrap(graph, y_c_out_noop).replace_obj(
                        'Out', {'name': y_c_out_noop})

                if lstm in graph._attr['output_names'] and lstm_end_names:
                    index = graph._attr['output_names'].index(lstm)
                    graph._attr['output_names'].pop(index)
                    for i, name in enumerate(lstm_end_names):
                        graph._attr['output_names'].insert(
                            index, lstm_end_names[i])
                        index += 1

                NodeWrap(graph, init_h_split).replace_obj(
                    'Split', {'name': init_h_split, 'opset_version': 2, 'axis': 1, 'split': [1, 1]})
                NodeWrap(graph, init_c_split).replace_obj(
                    'Split', {'name': init_c_split, 'opset_version': 2, 'axis': 1, 'split': [1, 1]})

                NodeWrap(graph, init_h_fw_reshape).replace_obj(
                    'Reshape', {'name': init_h_fw_reshape, 'opset_version': 5})
                insert_constant(graph, init_h_fw_reshape + '_shape', np.array(
                    [batch_size, hidden_size], np.int64), init_h_fw_reshape, in_port=1)
                NodeWrap(graph, init_h_bw_reshape).replace_obj(
                    'Reshape', {'name': init_h_bw_reshape, 'opset_version': 5})
                insert_constant(graph, init_h_bw_reshape + '_shape', np.array(
                    [batch_size, hidden_size], np.int64), init_h_bw_reshape, in_port=1)

                NodeWrap(graph, init_c_fw_reshape).replace_obj(
                    'Reshape', {'name': init_c_fw_reshape, 'opset_version': 5})
                insert_constant(graph, init_c_fw_reshape + '_shape', np.array(
                    [batch_size, hidden_size], np.int64), init_c_fw_reshape, in_port=1)
                NodeWrap(graph, init_c_bw_reshape).replace_obj(
                    'Reshape', {'name': init_c_bw_reshape, 'opset_version': 5})
                insert_constant(graph, init_c_bw_reshape + '_shape', np.array(
                    [batch_size, hidden_size], np.int64), init_c_bw_reshape, in_port=1)

                fw_lstm_attr = lstm_obj.copied_attr()
                fw_lstm_attr.update({'name': fw_lstm,
                                     'time_steps': time_steps,
                                     'input_size': input_size,
                                     'hidden_size': hidden_size,
                                     'weights': lstm_obj.weights[0, ...],
                                     'biases': lstm_obj.biases[0, ...],
                                     'direction': 'forward',
                                     'method': 'YHC'
                                     })

                bw_lstm_attr = lstm_obj.copied_attr()
                bw_lstm_attr.update({'name': bw_lstm,
                                     'time_steps': time_steps,
                                     'input_size': input_size,
                                     'hidden_size': hidden_size,
                                     'weights': lstm_obj.weights[1, ...],
                                     'biases': lstm_obj.biases[1, ...],
                                     'direction': 'forward',
                                     'method': 'YHC'
                                     })

                if lstm_obj.activations:
                    fw_lstm_attr.update(
                        {'activations': lstm_obj.activations[0: len(lstm_obj.activations) // 2]})
                    bw_lstm_attr.update(
                        {'activations': lstm_obj.activations[len(lstm_obj.activations) // 2:]})
                if lstm_obj.activation_alpha:
                    fw_lstm_attr.update({'activation_alpha': lstm_obj.activation_alpha[0: len(
                        lstm_obj.activation_alpha) // 2]})
                    bw_lstm_attr.update({'activation_alpha': lstm_obj.activation_alpha[len(
                        lstm_obj.activation_alpha) // 2:]})
                if lstm_obj.activation_beta:
                    fw_lstm_attr.update(
                        {'activation_beta': lstm_obj.activation_beta[0: len(lstm_obj.activation_beta) // 2]})
                    bw_lstm_attr.update(
                        {'activation_beta': lstm_obj.activation_beta[len(lstm_obj.activation_beta) // 2:]})
                if lstm_obj.clip:
                    fw_lstm_attr.update({'threshold': float(lstm_obj.clip)})
                    bw_lstm_attr.update({'threshold': float(lstm_obj.clip)})
                if lstm_obj.cell_clip:
                    fw_lstm_attr.update({'cell_clip': float(lstm_obj.cell_clip)})
                    bw_lstm_attr.update({'cell_clip': float(lstm_obj.cell_clip)})
                NodeWrap(graph, fw_lstm).replace_obj(
                    'ArmBasicLSTM', fw_lstm_attr)
                NodeWrap(graph, bw_lstm).replace_obj(
                    'ArmBasicLSTM', bw_lstm_attr)

                seq_len = np.ones([batch_size], np.int32) * time_steps
                reverse1_attr = {'name': reverse1, 'batch_axis': 0,
                                 'time_axis': 1, 'opset_version': 10}
                insert_constant(graph, reverse1 + '_seq_len',
                                seq_len, reverse1, in_port=1)
                NodeWrap(graph, reverse1).replace_obj(
                    'ReverseSequence', reverse1_attr)
                reverse2_attr = {'name': reverse2, 'batch_axis': 0,
                                 'time_axis': 1, 'opset_version': 10}
                insert_constant(graph, reverse2 + '_seq_len',
                                seq_len, reverse2, in_port=1)
                NodeWrap(graph, reverse2).replace_obj(
                    'ReverseSequence', reverse2_attr)

                NodeWrap(graph, fw_y_reshape).replace_obj(
                    'Reshape', {'name': fw_y_reshape, 'opset_version': 5})
                insert_constant(graph, fw_y_reshape + '_shape', np.array(
                    [batch_size, time_steps, 1, hidden_size], np.int64), fw_y_reshape, in_port=1)
                NodeWrap(graph, bw_y_reshape).replace_obj(
                    'Reshape', {'name': bw_y_reshape, 'opset_version': 5})
                insert_constant(graph, bw_y_reshape + '_shape', np.array(
                    [batch_size, time_steps, 1, hidden_size], np.int64), bw_y_reshape, in_port=1)
                NodeWrap(graph, y_concat).replace_obj(
                    'Concat', {'name': y_concat, 'opset_version': 11, 'axis': 2})

                NodeWrap(graph, fw_y_h_reshape).replace_obj(
                    'Reshape', {'name': fw_y_h_reshape, 'opset_version': 5})
                insert_constant(graph, fw_y_h_reshape + '_shape', np.array(
                    [batch_size, 1, hidden_size], np.int64), fw_y_h_reshape, in_port=1)
                NodeWrap(graph, bw_y_h_reshape).replace_obj(
                    'Reshape', {'name': bw_y_h_reshape, 'opset_version': 5})
                insert_constant(graph, bw_y_h_reshape + '_shape', np.array(
                    [batch_size, 1, hidden_size], np.int64), bw_y_h_reshape, in_port=1)
                NodeWrap(graph, y_h_concat).replace_obj(
                    'Concat', {'name': y_h_concat, 'opset_version': 11, 'axis': 1})

                NodeWrap(graph, fw_y_c_reshape).replace_obj(
                    'Reshape', {'name': fw_y_c_reshape, 'opset_version': 5})
                insert_constant(graph, fw_y_c_reshape + '_shape', np.array(
                    [batch_size, 1, hidden_size], np.int64), fw_y_c_reshape, in_port=1)
                NodeWrap(graph, bw_y_c_reshape).replace_obj(
                    'Reshape', {'name': bw_y_c_reshape, 'opset_version': 5})
                insert_constant(graph, bw_y_c_reshape + '_shape', np.array(
                    [batch_size, 1, hidden_size], np.int64), bw_y_c_reshape, in_port=1)
                NodeWrap(graph, y_c_concat).replace_obj(
                    'Concat', {'name': y_c_concat, 'opset_version': 11, 'axis': 1})

                graph.remove_node(lstm)
    if matched:
        clear_redundant_nodes(graph)


def merge_b2s(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('transpose1', {'op': 'Transpose'}),
                                   ('d2s', {'op': 'DepthToSpace'}),
                                   ('transpose2', {'op': 'Transpose'}),
                                   ('output', {}),
                               ],
                               edges=[
                                   ('transpose1', 'd2s'),
                                   ('d2s', 'transpose2'),
                                   ('transpose2', 'output'),
                               ])
    matched = False
    for m in matches:
        transpose1, d2s, transpose2, out = m['transpose1'], m['d2s'], m['transpose2'], m['output']
        transpose1_obj, d2s_obj, transpose2_obj, out_obj = [
            NodeWrap(graph, name)['object'] for name in [transpose1, d2s, transpose2, out]]
        if transpose1_obj is None \
                or d2s_obj is None \
                or transpose2_obj is None \
                or out_obj is None:
            ERROR('[Parser]: Meets invalid node in merge_b2s!')
            continue
        if d2s_obj.mode != 'DCR':
            continue
        transpose_perm = np.arange(0, len(transpose2_obj.perm))
        transpose_perm[0], transpose_perm[len(
            transpose2_obj.perm) - 1] = transpose_perm[len(transpose2_obj.perm) - 1], transpose_perm[0]
        if transpose2_obj.perm == transpose_perm.tolist() and transpose1_obj.perm == transpose_perm.tolist():
            transpose_2_out_shapes = transpose2_obj.get_output_shapes()
            if transpose_2_out_shapes and all([shape for shape in transpose_2_out_shapes]):
                matched = True
                if out_obj.type == 'Slice':
                    starts, ends = out_obj.starts, out_obj.ends
                    valid_slice = True
                else:
                    starts = [0] * len(transpose_2_out_shapes[0])
                    ends = transpose_2_out_shapes[0]
                    valid_slice = False
                block_size = d2s_obj.blocksize
                crops_starts = starts
                crops_ends = (np.array(
                    transpose_2_out_shapes[0], np.int64) - np.array(ends, np.int64)).tolist()
                true_last_node = out if valid_slice else transpose2

                d2s_in_edges = graph.sorted_in_edges(d2s)
                d2s_out_edges = graph.sorted_out_edges(d2s)
                graph.remove_edges_from(d2s_in_edges + d2s_out_edges)

                in_edges = graph.sorted_in_edges(transpose1, data=True)
                out_edges = graph.sorted_out_edges(true_last_node, data=True)
                for src, _, in_attr in in_edges:
                    graph.remove_edge(src, transpose1)
                    graph.add_edge(src, d2s, **in_attr)
                for _, dst, out_attr in out_edges:
                    graph.remove_edge(true_last_node, dst)
                    graph.add_edge(d2s, dst, **out_attr)
                if true_last_node in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(true_last_node)
                    graph._attr['output_names'][index] = d2s

                b2s_attr = d2s_obj.copied_attr()
                b2s_attr.update({'block_size_x': block_size, 'block_size_y': block_size,
                                 'crops': crops_starts[1:3] + crops_ends[1:3]})
                NodeWrap(graph, d2s).replace_obj('ArmBatchToSpace', b2s_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_b2s_nd(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('reshape1', {'op': 'Reshape'}),
                                   ('transpose', {'op': 'Transpose'}),
                                   ('reshape2', {'op': 'Reshape'}),
                                   ('slice', {'op': 'Slice'}),
                               ],
                               edges=[
                                   ('reshape1', 'transpose'),
                                   ('transpose', 'reshape2'),
                                   ('reshape2', 'slice'),
                               ])
    matched = False
    for m in matches:
        names = ['reshape1', 'transpose', 'reshape2', 'slice']
        node_objs = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if all([obj is not None for obj in node_objs.values()]) \
                and all([len(graph.sorted_out_edges(m[n])) == 1 for n in names[:-1]]) \
                and len(node_objs['reshape1'].get_input_shapes()[0]) == 4 \
                and len(node_objs['transpose'].get_input_shapes()[0]) == 6 \
                and len(node_objs['reshape2'].get_output_shapes()[0]) == 4 \
                and node_objs['transpose'].perm == [2, 3, 0, 4, 1, 5]:
            block_y, block_x = node_objs['transpose'].get_input_shapes()[0][:2]
            input_shape1, input_shape2 = node_objs['reshape1'].get_input_shapes()[0][1:3]
            reshape2_out_shape = node_objs['reshape2'].get_output_shapes()[0]
            output_shape1, output_shape2 = reshape2_out_shape[1:3]
            if block_y * input_shape1 == output_shape1 \
                    and block_x * input_shape2 == output_shape2:
                matched = True
                tr_in_edges = graph.sorted_in_edges(m['transpose'])
                tr_out_edges = graph.sorted_out_edges(m['transpose'])
                graph.remove_edges_from(tr_in_edges + tr_out_edges)
                src, _, in_attr = graph.sorted_in_edges(
                    m['reshape1'], data=True)[0]
                graph.remove_edge(src, m['reshape1'])
                graph.add_edge(src, m['transpose'], **in_attr)
                for _, dst, out_attr in graph.sorted_out_edges(m['slice'], data=True):
                    graph.remove_edge(m['slice'], dst)
                    graph.add_edge(m['transpose'], dst, **out_attr)
                if m['slice'] in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(m['slice'])
                    graph._attr['output_names'][index] = m['transpose']
                crops_starts = node_objs['slice'].starts
                crops_ends = (np.array(reshape2_out_shape, np.int64) -
                              np.array(node_objs['slice'].ends, np.int64)).tolist()
                b2s_attr = node_objs['transpose'].copied_attr()
                b2s_attr.update({'block_size_x': block_x,
                                 'block_size_y': block_y,
                                 'crops': crops_starts[1:3] + crops_ends[1:3]})
                NodeWrap(graph, m['transpose']).replace_obj(
                    'ArmBatchToSpace', b2s_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_square(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('const', {'op': 'Constant'}),
                                   ('pow', {'op': 'Pow'}),
                               ],
                               edges=[
                                   ('const', 'pow', {
                                    'src_out_port': 0, 'dst_in_port': 1})
                               ])
    for m in matches:
        names = ['const', 'pow']
        node_objs = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in node_objs.values()]):
            ERROR('[Parser]: Meets invalid node in merge_square!')
            continue
        power = node_objs['const'].value
        if FLOAT_EQUAL(power, 2.):
            matched = True
            graph.remove_edge(m['const'], m['pow'])
            NodeWrap(graph, m['pow']).replace_obj(
                'ArmSquare', node_objs['pow'].copied_attr())
    if matched:
        clear_redundant_nodes(graph)


def merge_square2(graph):
    matched = False
    matches = single_node_matcher(graph, 'Mul')
    for m in matches:
        mul = m['target']
        mul_obj = NodeWrap(graph, mul)['object']
        if mul_obj is None or len(mul_obj.get_input_shapes()) != 2:
            ERROR('[Parser]: Meets invalid node(%s) in merge_square2!' % mul)
            continue
        mul_in_edges = graph.sorted_in_edges(mul, keys=True, data=True)
        src1, _, _, in_attr1 = mul_in_edges[0]
        src2, _, k2, in_attr2 = mul_in_edges[1]
        if src1 != src2 or in_attr1['src_out_port'] != in_attr2['src_out_port']:
            continue
        matched = True
        graph.remove_edge(src2, mul, key=k2)
        NodeWrap(graph, mul).replace_obj('ArmSquare', mul_obj.copied_attr())
    if matched:
        clear_redundant_nodes(graph)


def merge_squared_diff(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('sub', {'op': 'Sub'}),
                                   ('pow', {'op': 'Pow'}),
                               ],
                               edges=[
                                   ('sub', 'pow', {'dst_in_port': 0}),
                               ])
    for m in matches:
        names = ['sub', 'pow']
        node_objs = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if all([obj is not None for obj in node_objs.values()]):
            sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
            pow_in_edges = graph.sorted_in_edges(m['pow'], data=True)
            sub_out_edges = graph.sorted_out_edges(m['sub'])
            if len(sub_out_edges) == 1 \
                    and len(sub_in_edges) == 2 \
                    and FLOAT_EQUAL(node_objs['pow'].get_input_tensors()[1], 2):
                matched = True
                graph.remove_edges_from(sub_in_edges + pow_in_edges)
                for src, _, in_attr in sub_in_edges:
                    graph.add_edge(src, m['pow'], **in_attr)
                NodeWrap(graph, m['pow']).replace_obj(
                    'ArmSquaredDifference', {'name': m['pow'], 'quantize': node_objs['pow'].quantize})
        else:
            ERROR('[Parser]: Meets invalid node in merge_squared_diff!')
    if matched:
        clear_redundant_nodes(graph)


def merge_rsqrt(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('sqrt', {'op': 'Sqrt'}),
                                   ('reciprocal', {'op': 'Reciprocal'}),
                               ],
                               edges=[
                                   ('sqrt', 'reciprocal'),
                               ])
    for m in matches:
        sqrt, reciprocal = m['sqrt'], m['reciprocal']
        sqrt_obj, reciprocal_obj = [NodeWrap(graph, name)['object'] for name in [
            sqrt, reciprocal]]
        if sqrt_obj is not None and reciprocal_obj is not None:
            sqrt_out_edges = graph.sorted_out_edges(sqrt, data=True)
            if len(sqrt_out_edges) == 1:
                reciprocal_out_edges = graph.sorted_out_edges(
                    reciprocal, data=True)
                for _, dst, out_attr in reciprocal_out_edges:
                    graph.remove_edge(reciprocal, dst)
                    graph.add_edge(sqrt, dst, **out_attr)
                rsqrt_attr = sqrt_obj.copied_attr()
                NodeWrap(graph, sqrt).replace_obj('ArmRsqrt', rsqrt_attr)
                remove_node_safely(graph, reciprocal)
                if reciprocal in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(reciprocal)
                    graph._attr['output_names'][index] = sqrt


def merge_s2b(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('input', {}),
                                   ('transpose1', {'op': 'Transpose'}),
                                   ('s2d', {'op': 'SpaceToDepth'}),
                                   ('transpose2', {'op': 'Transpose'})
                               ],
                               edges=[
                                   ('input', 'transpose1'),
                                   ('transpose1', 's2d'),
                                   ('s2d', 'transpose2'),
                               ])
    for m in matches:
        inp, transpose1, s2d, transpose2 = m['input'], m['transpose1'], m['s2d'], m['transpose2']
        inp_obj, transpose1_obj, s2d_obj, transpose2_obj = [
            NodeWrap(graph, name)['object'] for name in [inp, transpose1, s2d, transpose2]]
        if inp_obj is not None and s2d_obj is not None and transpose1_obj is not None and transpose2_obj is not None:
            input_shapes = transpose1_obj.get_input_shapes()
            if len(input_shapes) == 1 and input_shapes[0] and len(input_shapes[0]) >= 3:
                input_rank = len(input_shapes[0])
                if transpose1_obj.perm == transpose2_obj.perm \
                        and transpose1_obj.perm == [input_rank - 1] + list(range(1, input_rank - 1)) + [0]:
                    matched = True
                    inp_in_edges = graph.sorted_in_edges(inp)
                    inp_out_edges = graph.sorted_out_edges(inp)
                    if inp_obj.type == 'Pad' and len(inp_out_edges) == 1:
                        pads = inp_obj.pads[1:3] + inp_obj.pads[5:7]
                        valid_pad = True
                    else:
                        pads = [0] * 4
                        valid_pad = False
                    block_size = s2d_obj.blocksize
                    true_input = inp_in_edges[0][0] if valid_pad else inp
                    true_first_node = inp if valid_pad else transpose1

                    s2d_in_edges = graph.sorted_in_edges(s2d)
                    s2d_out_edges = graph.sorted_out_edges(s2d)
                    graph.remove_edges_from(s2d_in_edges + s2d_out_edges)

                    in_edges = graph.sorted_in_edges(
                        true_first_node, data=True)
                    out_edges = graph.sorted_out_edges(transpose2, data=True)
                    for src, _, in_attr in in_edges:
                        if src == true_input:
                            graph.remove_edge(src, true_first_node)
                            graph.add_edge(src, s2d, **in_attr)
                    for _, dst, out_attr in out_edges:
                        graph.remove_edge(transpose2, dst)
                        graph.add_edge(s2d, dst, **out_attr)
                    if transpose2 in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(transpose2)
                        graph._attr['output_names'][index] = s2d

                    s2b_attr = s2d_obj.copied_attr()
                    s2b_attr.update(
                        {'pads': pads, 'block_size_x': block_size, 'block_size_y': block_size})
                    NodeWrap(graph, s2d).replace_obj(
                        'ArmSpaceToBatch', s2b_attr)
        else:
            ERROR('[Parser]: Meets invalid Node in merge_s2b!')
    if matched:
        clear_redundant_nodes(graph)


def merge_s2b_nd(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('inp', {}),
                                   ('reshape1', {'op': 'Reshape'}),
                                   ('transpose', {'op': 'Transpose'}),
                                   ('reshape2', {'op': 'Reshape'}),

                               ],
                               edges=[
                                   ('inp', 'reshape1'),
                                   ('reshape1', 'transpose'),
                                   ('transpose', 'reshape2'),
                               ])
    matched = False
    for m in matches:
        names = ['inp', 'reshape1', 'transpose', 'reshape2']
        inp, reshape1, transpose, reshape2 = [m[n] for n in names]
        inp_obj, reshape1_obj, transpose_obj, reshape2_obj = \
            [NodeWrap(graph, m[n])['object'] for n in names]
        if all([obj is not None for obj in [inp_obj, reshape1_obj, transpose_obj, reshape2_obj]]):
            reshape1_out_edges = graph.sorted_out_edges(reshape1)
            transpose_out_edges = graph.sorted_out_edges(transpose)
            reshape1_in_shapes = reshape1_obj.get_input_shapes()
            trans_in_shapes = transpose_obj.get_input_shapes()
            reshape2_out_shapes = reshape2_obj.get_output_shapes()
            if (inp_obj.type != 'Pad'
                    or (inp_obj.type == 'Pad'
                        and len(inp_obj.pads) == 8
                        and inp_obj.pads[0] == 0
                        and inp_obj.pads[3] == 0
                        and inp_obj.pads[4] == 0
                        and inp_obj.pads[7] == 0
                        )) \
                    and len(reshape1_out_edges) == 1 \
                    and len(transpose_out_edges) == 1 \
                    and len(reshape1_in_shapes) >= 1 \
                    and len(trans_in_shapes) >= 1 \
                    and len(reshape2_out_shapes) >= 1 \
                    and len(reshape1_obj.get_input_shapes()[0]) == 4 \
                    and len(transpose_obj.get_input_shapes()[0]) == 6 \
                    and len(reshape2_obj.get_output_shapes()[0]) == 4 \
                    and transpose_obj.perm == [2, 4, 0, 1, 3, 5] \
                    and trans_in_shapes[0][2] * trans_in_shapes[0][1] == reshape1_in_shapes[0][1] \
                    and trans_in_shapes[0][4] * trans_in_shapes[0][3] == reshape1_in_shapes[0][2]:
                matched = True
                block_y, block_x = trans_in_shapes[0][2], trans_in_shapes[0][4]
                if inp_obj.type == 'Pad':
                    spatial_pads = inp_obj.pads[1:3] + inp_obj.pads[5:7]
                    in_edges = graph.sorted_in_edges(inp, keys=True, data=True)
                else:
                    spatial_pads = [0] * 4
                    in_edges = graph.sorted_in_edges(
                        reshape1, keys=True, data=True)
                src, _, k, in_attr = in_edges[0]
                reshape1_in_edges = graph.sorted_in_edges(reshape1)
                reshape2_in_edges = graph.sorted_in_edges(reshape2)
                graph.remove_edges_from(reshape1_in_edges + reshape2_in_edges)
                graph.add_edge(src, reshape2, **in_attr)

                if inp_obj.type == 'Pad' and len(graph.sorted_out_edges(inp)) == 0:
                    pad_in_edges = graph.sorted_in_edges(inp)
                    graph.remove_edges_from(pad_in_edges)

                s2b_attr = reshape2_obj.copied_attr()
                s2b_attr.update(
                    {'pads': spatial_pads, 'block_size_x': block_x, 'block_size_y': block_y})
                NodeWrap(graph, reshape2).replace_obj(
                    'ArmSpaceToBatch', s2b_attr)
        else:
            ERROR('[Parser]: Meets invalid Node in merge_s2b_nd!')
    if matched:
        clear_redundant_nodes(graph)


def merge_s2b_pool_b2s(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('s2b', {'op': 'ArmSpaceToBatch'}),
                                   ('pool', {'op': 'ArmPooling'}),
                                   ('b2s', {'op': 'ArmBatchToSpace'})
                               ],
                               edges=[
                                   ('s2b', 'pool'),
                                   ('pool', 'b2s')
                               ])
    for m in matches:
        s2b, pool, b2s = m['s2b'], m['pool'], m['b2s']
        s2b_obj = NodeWrap(graph, s2b)['object']
        pool_obj = NodeWrap(graph, pool)['object']
        b2s_obj = NodeWrap(graph, b2s)['object']
        if s2b_obj is None or pool_obj is None or b2s_obj is None:
            ERROR('[Parser]: Meets invalid Op in merge_s2b_avgpool_b2s!')
            continue
        in_edges = graph.sorted_in_edges(s2b, data=True)
        pool_out_edges = graph.sorted_out_edges(pool, data=True)
        if len(in_edges) < 1 or len(pool_out_edges) != 1:
            continue
        if s2b_obj.block_size_x != b2s_obj.block_size_x \
                or s2b_obj.block_size_y != b2s_obj.block_size_y \
                or any(c != 0 for c in b2s_obj.crops):
            continue
        matched = True
        out_edges = graph.sorted_out_edges(b2s, data=True)
        src, _, in_attr = in_edges[0]
        graph.remove_edge(s2b, pool)
        graph.remove_edge(pool, b2s)
        graph.add_edge(src, pool, **in_attr)
        for _, dst, out_attr in out_edges:
            graph.remove_edge(b2s, dst)
            graph.add_edge(pool, dst, **out_attr)
        pool_obj.dilations[0] *= s2b_obj.block_size_y
        pool_obj.dilations[1] *= s2b_obj.block_size_x
        pool_obj.pads = (np.array(pool_obj.pads) +
                         np.array(s2b_obj.pads)).tolist()
        if b2s in graph._attr['output_names']:
            index = graph._attr['output_names'].index(b2s)
            graph._attr['output_names'].pop(index)
            if pool not in graph._attr['output_names']:
                graph._attr['output_names'].insert(index, pool)
    if matched:
        clear_redundant_nodes(graph)


def merge_group_conv(graph, max_groups=16):
    for group in list(range(2, max_groups)):
        matched = False
        nodes = [('split', {'op': 'ArmSplit'})] + [('conv_%s' % str(i + 1), {'op': 'ArmConvolution'})
                                                   for i in range(group)] + [('concat', {'op': 'ArmConcat'})]
        edges = [('split', 'conv_%s' % str(i + 1), {'src_out_port': i, 'dst_in_port': 0}) for i in range(group)] \
            + [('conv_%s' % str(i + 1), 'concat',
                {'src_out_port': 0, 'dst_in_port': i}) for i in range(group)]
        matches = matched_patterns(graph, nodes, edges)
        for m in matches:
            split, concat = m['split'], m['concat']
            conv_names = [m['conv_%s' % str(i + 1)] for i in range(group)]
            split_obj, concat_obj = [
                NodeWrap(graph, name)['object'] for name in [split, concat]]
            conv_objs = [NodeWrap(graph, name)['object']
                         for name in conv_names]
            if all([obj is not None for obj in [split_obj, concat_obj] + conv_objs]) \
                    and len(split_obj.split) == group \
                    and len(graph.sorted_in_edges(concat)) == group \
                    and all([len(graph.sorted_out_edges(conv)) == 1 for conv in conv_names]) \
                    and all([conv_obj.group == 1 for conv_obj in conv_objs]) \
                    and len(set([conv_obj.activations for conv_obj in conv_objs])) == 1:
                matched = True
                group_conv = get_valid_node_name(
                    graph, conv_names[0] + '_group')
                weights = np.concatenate(
                    [conv_obj.weights for conv_obj in conv_objs], axis=0)
                biases = np.concatenate(
                    [conv_obj.biases for conv_obj in conv_objs], axis=0)
                num_output = group * conv_objs[0].num_output

                in_edges = graph.sorted_in_edges(split, data=True)
                src, _, in_attr = in_edges[0]
                graph.remove_edge(src, split)
                graph.add_edge(src, group_conv, **in_attr)

                out_edges = graph.sorted_out_edges(concat, data=True)
                for _, dst, out_attr in out_edges:
                    graph.remove_edge(concat, dst)
                    graph.add_edge(group_conv, dst, **out_attr)

                g_conv_attr = conv_objs[0].copied_attr()
                g_conv_attr.update({'name': group_conv, 'weights': weights,
                                    'biases': biases, 'num_output': num_output, 'group': group})
                NodeWrap(graph, group_conv).replace_obj(
                    'ArmConvolution', g_conv_attr)

                if concat in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(concat)
                    graph._attr['output_names'][index] = group_conv

        if matched:
            clear_redundant_nodes(graph)


def merge_transpose_matmul_gemm(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('trans', {'op': 'ArmTranspose', 'unique': False}),
                                   ('matmul_gemm', {'op': ['ArmMatMul', 'ArmGemm']}),
                               ],
                               edges=[
                                   ('trans', 'matmul_gemm'),
                               ])
    for m in matches:
        matmul_gemm = m['matmul_gemm']
        matmul_gemm_obj = NodeWrap(graph, matmul_gemm)['object']
        if matmul_gemm_obj is not None:
            matmul_gemm_in_edges = graph.sorted_in_edges(matmul_gemm)
            for matmul_gemm_in_edge in matmul_gemm_in_edges[:2]:
                in_edge_obj = NodeWrap(graph, matmul_gemm_in_edge[0])['object']
                if in_edge_obj is not None:
                    if in_edge_obj.type == 'ArmTranspose':
                        trans = in_edge_obj.name
                        trans_out_edges = graph.sorted_out_edges(trans, data=True)
                        trans_perm = in_edge_obj.perm
                        inp_rank = len(trans_perm)
                        exp_perm = list(range(inp_rank - 2)) + [inp_rank - 1, inp_rank - 2]
                        if trans_perm != exp_perm:
                            continue

                        matched = True
                        if trans_out_edges[0][-1]['dst_in_port'] == 0:
                            trans_a = matmul_gemm_obj.trans_a
                            matmul_gemm_obj.trans_a = not trans_a
                        else:
                            trans_b = matmul_gemm_obj.trans_b
                            matmul_gemm_obj.trans_b = not trans_b

                        graph.remove_edge(trans, matmul_gemm)
                        trans_in_edges = graph.sorted_in_edges(trans, data=True)
                        src, _, in_attr = trans_in_edges[0]
                        in_attr_copy = copy.deepcopy(in_attr)
                        in_attr_copy['dst_in_port'] = trans_out_edges[0][-1]['dst_in_port']
                        graph.add_edge(src, matmul_gemm, **in_attr_copy)
                else:
                    ERROR(f'[Parser]: Meets invalid Node: {matmul_gemm_in_edge[0]} in merge_transpose_matmul_gemm!')
        else:
            ERROR(f'[Parser]: Meets invalid Node: {matmul_gemm} in merge_transpose_matmul_gemm!')

    if matched:
        clear_redundant_nodes(graph)


def merge_not_equal(graph):
    matches = two_nodes_matcher(graph, 'Equal', 'Not')
    for m in matches:
        equal_name, not_name = m['begin'], m['end']
        equal_obj, not_obj = [NodeWrap(graph, name)['object'] for name in [
            equal_name, not_name]]
        if equal_obj is not None and not_obj is not None and len(graph.sorted_out_edges(equal_name)) == 1:
            not_equal_attr = equal_obj.copied_attr()
            not_equal_attr.update({'method': 'NOT_EQUAL'})
            NodeWrap(graph, equal_name).replace_obj(
                'ArmLogical', not_equal_attr)
            remove_node_safely(graph, not_name)
            if not_name in graph._attr['output_names']:
                index = graph._attr['output_names'].index(not_name)
                graph._attr['output_names'][index] = equal_name


def merge_greater_less_equal_or(graph):
    matched = False
    matches = [matched_patterns(graph,
                                nodes=[
                                    ('inp1', {}),
                                    ('inp2', {}),
                                    ('greater_less', {'op': op}),
                                    ('equal', {'op': 'Equal'}),
                                    ('or', {'op': 'Or'})
                                ],
                                edges=[
                                    ('inp1', 'greater_less',
                                     {'dst_in_port': 0}),
                                    ('inp2', 'greater_less',
                                     {'dst_in_port': 1}),
                                    ('inp1', 'equal', {'dst_in_port': 0}),
                                    ('inp2', 'equal', {'dst_in_port': 1}),
                                    ('greater_less', 'or'),
                                    ('equal', 'or'),
                                ]) for op in ('Greater', 'Less')]
    matches = extend_lists(matches)
    for m in matches:
        inp1, inp2, greater_less, equal, or_name = m['inp1'], m[
            'inp2'], m['greater_less'], m['equal'], m['or']
        greater_less_in_edges = graph.sorted_in_edges(greater_less, data=True)
        greater_less_out_edges = graph.sorted_out_edges(greater_less)
        equal_in_edges = graph.sorted_in_edges(equal)
        equal_out_edges = graph.sorted_out_edges(equal)
        greater_less_obj = NodeWrap(graph, greater_less)['object']
        or_obj = NodeWrap(graph, or_name)['object']
        if greater_less_obj is None or or_obj is None:
            ERROR('[Parser]: Meets invalid Node in merge_greater_less_equal_or!')
            continue
        if len(greater_less_in_edges) == 2 \
                and len(greater_less_out_edges) == 1 \
                and len(equal_in_edges) == 2 \
                and len(equal_out_edges) == 1:
            matched = True
            _, _, in_attr1 = greater_less_in_edges[0]
            _, _, in_attr2 = greater_less_in_edges[1]
            graph.remove_edges_from(
                greater_less_in_edges + equal_in_edges + greater_less_out_edges + equal_out_edges)
            graph.add_edge(inp1, or_name, **in_attr1)
            graph.add_edge(inp2, or_name, **in_attr2)
            op_method = 'GREATER_EQUAL' if greater_less_obj.type == 'Greater' else 'LESS_EQUAL'
            logical_attr = or_obj.copied_attr()
            logical_attr.update({'name': or_name, 'method': op_method})
            NodeWrap(graph, or_name).replace_obj('ArmLogical', logical_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_nhwc_maxpoolargmax(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('argmaxpool', {'op': 'ArmMaxPoolingWithArgMax'}),
                                      ('cast1', {'op': 'ArmCast'}),
                                      ('sub', {'op': 'ArmEltwise'}),
                                      ('div', {'op': 'ArmDiv'}),
                                      ('add', {'op': 'ArmEltwise'}),
                                      ('cast2', {'op': 'ArmCast'}),
                                      ('const1', {'op': 'Constant'}),
                                      ('const2', {'op': 'Constant'}),
                                      ('const3', {'op': 'Constant'}),
                                      ],
                               edges=[
                                   ('argmaxpool', 'cast1', {
                                    'src_out_port': 1, 'dst_in_port': 0}),
                                   ('cast1', 'sub'),
                                   ('const1', 'sub', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('sub', 'div'),
                                   ('const2', 'div', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('div', 'add'),
                                   ('const3', 'add', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('add', 'cast2'),
                               ])
    for m in matches:
        names = ['argmaxpool', 'cast1', 'sub', 'div',
                 'add', 'cast2', 'const1', 'const2', 'const3']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if all([obj is not None for obj in obj_dict.values()]):
            if obj_dict['sub'].method != 'SUB' \
                    or obj_dict['add'].method != 'ADD':
                continue
            out_edges_need_check = [graph.sorted_out_edges(m[n])
                                    for n in ['cast1', 'sub', 'div', 'add', 'const1', 'const2', 'const3']]
            if all([len(out_edges) == 1 for out_edges in out_edges_need_check]):
                argmaxpool_obj = obj_dict['argmaxpool']
                input_shapes = argmaxpool_obj.get_input_shapes()
                output_shapes = argmaxpool_obj.get_output_shapes()
                if len(input_shapes) == 1 \
                        and len(input_shapes[0]) == 4 \
                        and len(output_shapes) == 2 \
                        and len(output_shapes[1]) == 4:
                    b, h, w, c = input_shapes[0]
                    _, out_h, out_w, _ = output_shapes[1]
                    hwc_compensation = np.reshape(np.arange(0, b).astype(
                        np.float32) * h * w * c, (b, 1, 1, 1))
                    hwc_compensation = np.tile(
                        hwc_compensation, (1, out_h, out_w, c))
                    hw_compensation = np.arange(
                        0, c).astype(np.float32) * h * w
                    c_compensation = np.arange(0, c).astype(np.float32)

                    sub_oprand2 = hwc_compensation + c_compensation
                    div_oprand2 = np.tile(np.reshape(
                        np.array([c], np.float32), (1, 1, 1, 1)), (b, out_h, out_w, c))
                    add_oprand2 = hwc_compensation + hw_compensation

                    if FLOAT_EQUAL(obj_dict['const1'].value, sub_oprand2) \
                            and FLOAT_EQUAL(obj_dict['const2'].value, div_oprand2) \
                            and FLOAT_EQUAL(obj_dict['const3'].value, add_oprand2):
                        matched = True
                        cast1_in_edges = graph.sorted_in_edges(m['cast1'])
                        cast2_out_edges = graph.sorted_out_edges(
                            m['cast2'], data=True)
                        graph.remove_edges_from(cast1_in_edges)
                        for _, dst, out_attr in cast2_out_edges:
                            out_attr['src_out_port'] = 1
                            graph.remove_edge(m['cast2'], dst)
                            graph.add_edge(m['argmaxpool'], dst, **out_attr)

                        argmaxpool_obj.flatten_dim = 'NCHW'
                        if m['cast2'] in graph._attr['output_names']:
                            if m['argmaxpool'] not in graph._attr['output_names']:
                                index = graph._attr['output_names'].index(
                                    m['cast2'])
                                graph._attr['output_names'][index] = m['argmaxpool']
                            else:
                                graph._attr['output_names'].remove(m['cast2'])
        else:
            ERROR('[Parser]: Meets invalid Node in merge_nhwc_maxpoolargmax!')

    if matched:
        clear_redundant_nodes(graph)


def merge_hwc_maxpoolargmax(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('argmaxpool', {'op': 'ArmMaxPoolingWithArgMax'}),
                                      ('const', {'op': 'Constant'}),
                                      ('sub', {'op': 'ArmEltwise'}),
                                      ('cast', {'op': 'ArmCast'}),
                                      ],
                               edges=[('argmaxpool', 'sub', {
                                       'src_out_port': 1, 'dst_in_port': 0}),
                                      ('const', 'sub', {'dst_in_port': 1}),
                                      ('sub', 'cast'),
                                      ])
    for m in matches:
        names = ['argmaxpool', 'const', 'sub']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in obj_dict.values()]):
            ERROR('[Parser]: Meets invalid node in merge_hwc_maxpoolargmax!')
            continue
        if obj_dict['sub'].method != 'SUB' \
                or obj_dict['argmaxpool'].flatten_dim != 'NHWC':
            continue
        out_edges_need_check = [graph.sorted_out_edges(m[n]) for n in [
            'const', 'sub']]
        if any([len(out_edges) != 1 for out_edges in out_edges_need_check]):
            continue
        argmaxpool_obj = obj_dict['argmaxpool']
        input_shapes = argmaxpool_obj.get_input_shapes()
        output_shapes = argmaxpool_obj.get_output_shapes()
        if len(input_shapes) < 1 or len(input_shapes[0]) != 4 or \
                len(output_shapes) < 1 or len(output_shapes[0]) != 4:
            ERROR('[Parser]: Meets invalid input/output shapes for node (%s) in merge_hwc_maxpoolargmax!' % m['argmaxpool'])
            continue
        in_n, in_h, in_w, in_c = input_shapes[0]
        sub_oprand = np.reshape(
            np.arange(0, in_n), (in_n, 1, 1, 1)) * in_h * in_w * in_c
        _, out_h, out_w, out_c = output_shapes[0]
        sub_oprand = np.tile(
            sub_oprand, [1, out_h, out_w, out_c]).astype(np.float32)
        if not FLOAT_EQUAL(obj_dict['const'].value, sub_oprand):
            continue
        matched = True
        sub_in_edges = graph.sorted_in_edges(m['sub'])
        graph.remove_edges_from(sub_in_edges)
        cast_out_edges = graph.sorted_out_edges(m['cast'], data=True)
        for _, dst, out_attr in cast_out_edges:
            graph.remove_edge(m['cast'], dst)
            new_out_attr = copy.deepcopy(out_attr)
            new_out_attr.update({'src_out_port': 1})
            graph.add_edge(m['argmaxpool'], dst, **new_out_attr)
        argmaxpool_obj.flatten_dim = 'HWC'

        if m['cast'] in graph._attr['output_names']:
            if m['argmaxpool'] not in graph._attr['output_names']:
                index = graph._attr['output_names'].index(m['cast'])
                graph._attr['output_names'][index] = m['argmaxpool']
            else:
                graph._attr['output_names'].remove(m['cast'])
    if matched:
        clear_redundant_nodes(graph)


def merge_hw_maxpoolargmax(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('argmaxpool', {'op': 'ArmMaxPoolingWithArgMax'}),
                                      ('transpose', {'op': 'ArmTranspose'}),
                                      ('cast', {'op': 'ArmCast'}),
                                      ('const', {'op': 'Constant'}),
                                      ('tile', {'op': 'ArmTile'}),
                                      ('sub', {'op': 'ArmEltwise'}),
                                      ],
                               edges=[
                                   ('argmaxpool', 'transpose', {
                                    'src_out_port': 1, 'dst_in_port': 0}),
                                   ('transpose', 'cast'),
                                   ('cast', 'sub'),
                                   ('const', 'tile'),
                                   ('tile', 'sub', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                               ])
    for m in matches:
        names = ['argmaxpool', 'transpose', 'cast', 'const', 'tile', 'sub']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if all([obj is not None for obj in obj_dict.values()]):
            if obj_dict['sub'].method != 'SUB' \
                    or obj_dict['argmaxpool'].flatten_dim != 'NCHW':
                continue
            out_edges_need_check = [graph.sorted_out_edges(m[n])
                                    for n in ['const', 'tile']]
            if all([len(out_edges) == 1 for out_edges in out_edges_need_check]):
                argmaxpool_obj = obj_dict['argmaxpool']
                input_shapes = argmaxpool_obj.get_input_shapes()
                output_shapes = argmaxpool_obj.get_output_shapes()
                if len(input_shapes) == 1 \
                        and len(input_shapes[0]) == 4 \
                        and len(output_shapes) == 2 \
                        and len(output_shapes[1]) == 4:
                    n, h, w, c = input_shapes[0]
                    sub_oprand = np.reshape(np.arange(
                        0, n), (n, 1, 1, 1)) * c * h * w + np.reshape(np.arange(0, c), (c, 1, 1)) * h * w
                    sub_oprand = sub_oprand.astype(np.float32)
                    if FLOAT_EQUAL(obj_dict['const'].value, sub_oprand):
                        matched = True
                        sub_in_edges = graph.sorted_in_edges(m['sub'])
                        sub_out_edges = graph.sorted_out_edges(
                            m['sub'], data=True)
                        graph.remove_edges_from(sub_in_edges)
                        for _, dst, out_attr in sub_out_edges:
                            graph.remove_edge(m['sub'], dst)
                            graph.add_edge(m['cast'], dst, **out_attr)
                        argmaxpool_obj.flatten_dim = 'HW'
                        if m['sub'] in graph._attr['output_names']:
                            if m['cast'] not in graph._attr['output_names']:
                                index = graph._attr['output_names'].index(
                                    m['sub'])
                                graph._attr['output_names'][index] = m['cast']
                            else:
                                graph._attr['output_names'].remove(m['sub'])
        else:
            ERROR('[Parser]: Meets invalid Node in merge_hw_maxpoolargmax!')
    if matched:
        clear_redundant_nodes(graph)


def merge_hw_maxunpool(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('const', {'op': 'Constant'}),
                                      ('add', {'op': 'ArmEltwise'}),
                                      ('cast', {'op': 'ArmCast'}),
                                      ('transpose', {'op': 'ArmTranspose'}),
                                      ('maxunpool', {'op': 'ArmMaxUnpool'}),
                                      ],
                               edges=[('const', 'add'),
                                      ('add', 'cast'),
                                      ('cast', 'transpose'),
                                      ('transpose', 'maxunpool', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                      ])
    matches2 = matched_patterns(graph,
                                nodes=[('const', {'op': 'Constant'}),
                                       ('add', {'op': 'ArmEltwise'}),
                                       ('transpose', {'op': 'ArmTranspose'}),
                                       ('maxunpool', {'op': 'ArmMaxUnpool'}),
                                       ],
                                edges=[('const', 'add'),
                                       ('add', 'transpose'),
                                       ('transpose', 'maxunpool', {
                                           'src_out_port': 0, 'dst_in_port': 1}),
                                       ])
    for m in matches + matches2:
        names = ['const', 'add', 'transpose', 'maxunpool'] + (['cast'] if 'cast' in m else [])
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if all([obj is not None for obj in obj_dict.values()]):
            if obj_dict['add'].method != 'ADD' \
                    or obj_dict['maxunpool'].flatten_dim != 'NCHW':
                continue
            out_edges_need_check = [graph.sorted_out_edges(m[n])
                                    for n in ['const', 'add']]
            if any([len(out_edges) != 1 for out_edges in out_edges_need_check]):
                continue
            maxunpool_obj = obj_dict['maxunpool']
            input_shapes = maxunpool_obj.get_input_shapes()
            output_shapes = maxunpool_obj.get_output_shapes()
            if len(input_shapes) != 2 \
                    or len(input_shapes[0]) != 4 \
                    or len(input_shapes[1]) != 4 \
                    or len(output_shapes) < 1 \
                    or len(output_shapes[0]) != 4:
                continue
            n, in_h, in_w, c = input_shapes[0]
            out_h, out_w = output_shapes[0][1:-1]
            add_oprand = np.reshape(np.arange(
                0, n), (n, 1, 1, 1)) * c * out_h * out_w + np.reshape(np.arange(0, c), (c, 1, 1)) * out_h * out_w
            add_oprand = np.tile(add_oprand, [1, 1, in_h, in_w]).astype(np.float32)
            if not FLOAT_EQUAL(obj_dict['const'].value, add_oprand):
                continue
            matched = True
            add_out_edges = graph.sorted_out_edges(m['add'])
            graph.remove_edges_from(add_out_edges)
            add_in_edges = graph.sorted_in_edges(m['add'], data=True)
            for src, _, in_attr in add_in_edges:
                graph.remove_edge(src, m['add'])
                if src != m['const']:
                    indices_inp = m['cast'] if 'cast' in m else m['transpose']
                    indices_inp_in_attr = copy.deepcopy(in_attr)
                    indices_inp_in_attr.update({'dst_in_port': 0})
                    graph.add_edge(src, indices_inp, **indices_inp_in_attr)
            maxunpool_obj.flatten_dim = 'HW'
        else:
            ERROR('[Parser]: Meets invalid Node in merge_hw_maxunpool!')
    if matched:
        clear_redundant_nodes(graph)


def rename_activations(graph):
    activations = ['Celu', 'Clip', 'Elu', 'Gelu', 'HardSigmoid', 'HardSwish',
                   'LeakyRelu', 'Mish', 'PRelu', 'Relu',
                   'Selu', 'Shrink', 'Sigmoid', 'Silu', 'Softplus', 'Softsign', 'Swish',
                   'Tanh', 'ThresholdedRelu', ]
    matches = [single_node_matcher(graph, act_type)
               for act_type in activations]
    matches = extend_lists(matches)
    for m in matches:
        act = m['target']
        act_obj = NodeWrap(graph, act)['object']
        if act_obj is None:
            ERROR('[Parser]: Meets invalid node(%s) in rename_activations!' % act)
            continue
        act_attr = act_obj.copied_attr()
        if act_obj.type == 'Sigmoid':
            method = 'SIGMOID'
            if graph._attr['framework'].name == 'TFLITE':
                out_edges = graph.sorted_out_edges(act, data=True)
                if len(out_edges) >= 1 and out_edges[0] is not None and not out_edges[0][2]['tensor'].min_max:
                    out_edges[0][2]['tensor'].min_max = (0, 1)
        elif act_obj.type == 'HardSigmoid':
            method = 'HARDSIGMOID'
            act_attr.update({'alpha': act_obj.alpha, 'beta': act_obj.beta})
        elif act_obj.type == 'Celu':
            method = 'CELU'
            act_attr.update({'alpha': act_obj.alpha})
        elif act_obj.type == 'Elu':
            method = 'ELU'
            act_attr.update({'alpha': act_obj.alpha})
        elif act_obj.type == 'LeakyRelu':
            method = 'LEAKYRELU'
            act_attr.update({'alpha': act_obj.alpha})
        elif act_obj.type == 'PRelu':
            in_edges = graph.sorted_in_edges(act, data=True)
            if len(in_edges) != 2:
                ERROR(
                    '[Parser]: Meets invalid PRelu node(%s) in rename_activations!' % act)
                continue
            method = 'PRELU'
            slope = act_obj.get_input_tensors()[1]
            if len(slope.shape) in (2, 3) and slope.shape[0] == 1:
                if len(slope.shape) == 3 and list(slope.shape[0:2]) == [1, 1]:
                    axis = (0, 1)
                else:
                    axis = 0
                slope = np.squeeze(slope, axis=axis)
            if len(slope.shape) == 5:
                in_shape = slope.shape
                pre_dim = [in_shape[0],
                           int(np.prod(in_shape[1:3])),
                           in_shape[3],
                           in_shape[-1]]
                slope = np.reshape(slope, pre_dim)
            act_attr.update({'negative_slope': slope})
            if act_obj.quantize and np.issubdtype(slope.dtype, np.integer):
                scale_zp = in_edges[1][2]['tensor'].scale_zp
                act_attr.update({'negative_slope_scale': np.array(scale_zp[0]),
                                 'negative_slope_zp': np.array(scale_zp[1])
                                 })
            if len(in_edges[1][2]['tensor'].min_max) == 2:
                act_attr.update({'negative_slope_range': list(in_edges[1][2]['tensor'].min_max)})
        elif act_obj.type == 'Clip':
            if FLOAT_EQUAL(act_obj.min, 0) and FLOAT_EQUAL(act_obj.max, 6):
                method = 'RELU6'
            else:
                method = 'CLIP'
                act_attr.update(
                    {'clip_min': act_obj.min, 'clip_max': act_obj.max})
        elif act_obj.type == 'Selu':
            method = 'SELU'
            act_attr.update({'alpha': act_obj.alpha, 'gamma': act_obj.gamma})
        elif act_obj.type == 'Swish':
            method = 'SWISH'
            act_attr.update({'alpha': act_obj.alpha})
        elif act_obj.type == 'Gelu':
            method = 'GELU'
            act_attr.update({'approximate': act_obj.approximate})
        elif act_obj.type == 'ThresholdedRelu':
            method = 'THRESHOLDEDRELU'
            act_attr.update({'alpha': act_obj.alpha})
        elif act_obj.type == 'Shrink':
            method = 'SHRINK'
            act_attr.update({'bias': act_obj.bias, 'lambd': act_obj.lambd})
        else:
            method = str(act_obj.type).upper()
        act_attr.update({'method': method})
        NodeWrap(graph, act).replace_obj('ArmActivation', act_attr)

        if act_obj.type in ['Clip', 'Relu', 'Shrink'] \
                and len(act_obj.get_input_tensors()) >= 1 \
                and act_obj.get_input_tensors()[0] is not None \
                and np.issubdtype(act_obj.get_input_tensors()[0].dtype, np.integer)\
                and not graph._attr.get('quantize', False):
            in_edges = graph.sorted_in_edges(act, keys=True, data=True)
            src, _, k, in_attr = in_edges[0]
            insert_cast(graph, src, act, 'float32',
                        in_attr=in_attr, key=k, type='ArmCast')
            post_cast = insert_cast_after(
                graph, act, 'float32', 'int32', type='ArmCast')
            if act in graph._attr['output_names']:
                index = graph._attr['output_names'].index(act)
                graph._attr['output_names'][index] = post_cast


def rename_bitwise(graph):
    matches = single_node_matcher(
        graph, ['BitwiseAnd', 'BitwiseNot', 'BitwiseOr', 'BitwiseXor'])
    for m in matches:
        bit = m['target']
        bit_obj = NodeWrap(graph, bit)['object']
        if bit_obj is None:
            ERROR('[Parser]: Meets invalid node(%s) in rename_bitwise!' % bit)
            continue
        bit_attr = bit_obj.copied_attr()
        method = bit_obj.type.split('Bitwise')[1].upper()
        bit_attr.update({'method': method})
        NodeWrap(graph, bit).replace_obj('ArmBitwise', bit_attr)


def rename_cum(graph):
    matches = single_node_matcher(graph, ['CumProd', 'CumSum'])
    for m in matches:
        cum = m['target']
        cum_obj = NodeWrap(graph, cum)['object']
        if cum_obj is None:
            ERROR('[Parser]: Meets invalid node(%s) in rename_cum!' % cum)
            continue
        in_edges = graph.sorted_in_edges(cum, data=True)
        if cum_obj.type == 'CumSum':
            if len(in_edges) < 2:
                ERROR(
                    '[Parser]: Meets invalid in_edge for cumlative Op(%s) in rename_cum!' % cum)
                continue
            else:
                if in_edges[1][2]['tensor'] is None or not in_edges[1][2]['tensor'].is_const:
                    WARN(
                        '[Parser]: Meets unsupported non-constant axis for cumlative Op(%s) in rename_cum!' % cum)
                    continue

        cum_attr = cum_obj.copied_attr()
        method = cum_obj.type.split('Cum')[1].upper()
        if cum_obj.type == 'CumSum':
            cum_attr.update({'axis': cum_obj.axis})
        cum_attr.update({'method': method})
        NodeWrap(graph, cum).replace_obj('ArmCumulate', cum_attr)


def rename_argminmax(graph):
    arg_types = ['ArgMin', 'ArgMax']
    matches = extend_lists([single_node_matcher(graph, op)
                            for op in arg_types])
    for m in matches:
        arg = m['target']
        arg_obj = NodeWrap(graph, arg)['object']
        if arg_obj is not None:
            input_shapes = arg_obj.get_input_shapes()
            output_shapes = arg_obj.get_output_shapes()
            if len(input_shapes) >= 1 and len(output_shapes) >= 1 \
                    and input_shapes[0] is not None \
                    and output_shapes[0] is not None:
                arg_attr = arg_obj.copied_attr()
                axis = arg_obj.axis
                axis = (len(input_shapes[0]) + axis) if axis < 0 else axis
                arg_attr.update({'method': 'MIN' if arg_obj.type == 'ArgMin' else 'MAX',
                                 'axis': axis
                                 })
                NodeWrap(graph, arg).replace_obj('ArmArgMinMax', arg_attr)
                if not arg_obj.keepdims:
                    reshape_dim = output_shapes[0]
                    old_dim = reshape_dim[:]
                    old_dim.insert(axis, 1)
                    reshape = insert_reshape_after(
                        graph, arg, new_dim=reshape_dim, old_dim=old_dim, quantize=arg_obj.quantize)
                    if reshape is not None and arg in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(arg)
                        graph._attr['output_names'][index] = reshape


def rename_bn(graph):
    bn_types = ['BatchNormalization', ]
    matches = extend_lists([single_node_matcher(graph, op) for op in bn_types])
    for m in matches:
        bn = m['target']
        bn_obj = NodeWrap(graph, bn)['object']
        if bn_obj.training_mode:
            WARN(
                '[Parser]: Meets unsupported training mode for BatchNormalization Op in rename_bn(%s)' % bn)
            continue
        gamma, beta, mean, var = [c[2] for c in bn_obj.sorted_in_consts()]
        if len(gamma.shape) > 1 \
                or len(beta.shape) > 1 \
                or len(mean.shape) > 1 \
                or len(var.shape) > 1:
            continue
        input_dtypes = bn_obj.get_input_dtypes()
        if len(input_dtypes) < 1 or input_dtypes[0] is None:
            WARN(
                '[Parser]: Meets invalid dtype of BatchNormalization Op(%s) in rename_bn!' % bn)
            continue
        weights = gamma / np.sqrt(var + bn_obj.epsilon)
        biases = beta - gamma * mean / np.sqrt(var + bn_obj.epsilon)
        new_attr_dict = bn_obj.copied_attr()
        new_attr_dict.update(
            {'num_output': biases.size, 'weights': weights.astype(input_dtypes[0]),
             'biases': biases.astype(input_dtypes[0]), 'axis': -1})
        NodeWrap(graph, bn).replace_obj('ArmBatchNorm', new_attr_dict)
        in_edges = graph.sorted_in_edges(bn)
        graph.remove_edges_from(in_edges[1:])


def rename_cast(graph):
    matches = single_node_matcher(graph, 'Cast')
    for m in matches:
        cast = m['target']
        cast_obj = NodeWrap(graph, cast)['object']
        out_edges = graph.sorted_out_edges(cast, data=True)
        if cast_obj is not None and len(out_edges) > 0:
            to_dtype_map = {'bool': 'uint8', 'int64': 'int32', 'float64': 'float32'}
            if cast_obj.to in ArmCastOp.attributes()['to_dtype']['options'] + list(to_dtype_map.keys()):
                cast_attr = cast_obj.copied_attr()
                if cast_obj.to in to_dtype_map:
                    to_dtype = to_dtype_map[cast_obj.to]
                    WARN('[Parser]: Change unsupported to_dtype %s of Cast Op (%s) to %s!' %
                         (cast_obj.to, cast, to_dtype))
                else:
                    to_dtype = cast_obj.to
                ignore_scale_zp = True
                if cast_obj.saturate:
                    mode = 'saturation'
                    # ignore_scale_zp cannot be true if there is no scale/zp
                    out_attr = out_edges[0][2]
                    if out_attr.get('tensor', None) is not None \
                            and len(out_attr['tensor'].scale_zp) == 2:
                        ignore_scale_zp = False
                else:
                    mode = 'truncation'
                cast_attr.update({'to_dtype': to_dtype, 'clip_mode': mode, 'ignore_scale_zp': ignore_scale_zp})
                NodeWrap(graph, cast).replace_obj('ArmCast', cast_attr)
            else:
                ERROR('[Parser]: Meets Cast Op (%s) with invalid dtype (%s) that cannot be converted in rename_cast!' % (
                    cast, cast_obj.to))
        else:
            ERROR('[Parser]: Meets invalid Cast Op (%s) in rename_cast!' % cast)


def rename_col2im(graph):
    matched = False
    matches = single_node_matcher(graph, 'Col2Im')
    for m in matches:
        col2im = m['target']
        col2im_obj = NodeWrap(graph, col2im)['object']
        in_edges = graph.sorted_in_edges(col2im, data=True)
        if col2im_obj is None or len(in_edges) != 3:
            ERROR('[Parser]: Meets invalid Col2Im Op (%s) in rename_col2im!' % col2im)
            continue
        image_shape_tensor = in_edges[1][2]['tensor']
        block_shape_tensor = in_edges[2][2]['tensor']
        if image_shape_tensor is None or not image_shape_tensor.is_const \
                or block_shape_tensor is None or not block_shape_tensor.is_const:
            WARN('[Parser]: Meets unsupported non-constant image_shape/block_shape of Col2Im Op (%s) in rename_col2im!' % col2im)
            continue
        image_shape = image_shape_tensor.value.tolist()
        block_shape = block_shape_tensor.value.tolist()
        if len(image_shape) > 3 or len(block_shape) > 3:
            WARN('[Parser]: Meets unsupported non-constant image_shape/block_shape of Col2Im Op (%s) in rename_col2im!' % col2im)
            continue
        matched = True
        col2im_attr = col2im_obj.copied_attr()
        NodeWrap(graph, col2im).replace_obj('ArmCol2Im', col2im_attr)
    if matched:
        clear_redundant_nodes(graph)


def rename_compress(graph):
    need_clear = False
    matches = single_node_matcher(graph, 'Compress')
    for m in matches:
        compress = m['target']
        compress_obj = NodeWrap(graph, compress)['object']
        in_edges = graph.sorted_in_edges(compress, data=True)
        if compress_obj is not None and len(in_edges) == 2:
            if not in_edges[1][2]['tensor'].is_const:
                WARN(
                    '[Parser]: Meets unsupported non-constant condition for Compress Op(%s) in rename_compress!' % compress)
                continue
            inputs = compress_obj.get_input_tensors()
            if len(inputs) != 2 \
                    or inputs[0] is None \
                    or inputs[1] is None \
                    or np.ndim(inputs[0]) < 1 \
                    or np.ndim(inputs[1]) != 1:
                ERROR('[Parser]: Meets invalid inputs for Compress Op(%s)' % compress)
                continue
            inp, condition = inputs
            if compress_obj.axis is None and condition.size < inp.size:
                extend_size = int(inp.size - condition.size)
            elif compress_obj.axis is not None and condition.size < inp.shape[compress_obj.axis]:
                extend_size = int(
                    inp.shape[compress_obj.axis] - condition.size)
            else:
                extend_size = 0
            condition = np.concatenate([condition, np.array(
                [False] * extend_size)]).astype(condition.dtype)
            if compress_obj.axis is None:
                compress_obj.axis = 0
                src, _, in_attr = in_edges[0]
                insert_reshape(graph, src, compress, in_attr,
                               [-1], type='ArmReshape', quantize=compress_obj.quantize)
                in_edges = graph.sorted_in_edges(compress, data=True)
            if NodeWrap(graph, in_edges[1][0])['object'].type not in ('Constant', 'ArmConstant'):
                need_clear = True
                graph.remove_edges_from(in_edges[1:])
                insert_constant(graph, compress + '_condition', condition,
                                compress, in_port=1, data_format='NHWC')
            elif extend_size != 0:
                if NodeWrap(graph, in_edges[1][0])['object'].type == 'Constant':
                    NodeWrap(graph, in_edges[1][0])['object'].value = condition
                    in_edges[1][2]['tensor'].value = condition
                    in_edges[1][2]['tensor'].shape = condition.shape
                elif NodeWrap(graph, in_edges[1][0])['object'].type == 'ArmConstant':
                    NodeWrap(graph, in_edges[1][0])[
                        'object'].weights = condition
                    in_edges[1][2]['tensor'].value = condition
                    in_edges[1][2]['tensor'].shape = condition.shape
                else:
                    ERROR(
                        '[Parser]: Meets invalid condition of Compress Op(%s) in rename_compress!' % compress)
            NodeWrap(graph, compress).replace_obj(
                'ArmCompress', compress_obj.copied_attr())
        else:
            ERROR('[Parser]: Meets invalid Compress Op(%s) in rename_compress!' % compress)
    if need_clear:
        clear_redundant_nodes(graph)


def rename_conv(graph):
    conv_types = ['Conv', 'ConvTranspose', 'ConvInteger']
    matches = matched_patterns(
        graph, nodes=[('conv', {'op': conv_types})], edges=[])
    for m in matches:
        conv = m['conv']
        conv_node = NodeWrap(graph, conv)
        conv_obj = conv_node['object']
        if conv_obj is None \
                or len(conv_obj.get_input_shapes()) < 1 \
                or conv_obj.weights is None \
                or conv_obj.biases is None:
            ERROR(
                '[Parser]: Meets invalid Conv/ConvTranspose Op(%s) in rename_conv!' % conv)
            continue
        conv_attr = conv_obj.copied_attr()
        if conv_obj.type == 'Conv':
            if int(np.prod(conv_obj.weights.shape[0:2])) == conv_obj.group \
                    and conv_obj.group > 1:
                multiplier = conv_obj.weights.shape[0] // conv_obj.group
                new_weights = np.transpose(
                    conv_obj.weights, axes=ArmDepthwiseConvOp.perm_onnx_to_ir())
                conv_attr.update(
                    {'weights': new_weights, 'multiplier': multiplier})
                conv_node.replace_obj('ArmDepthwiseConv', conv_attr)
            else:
                is_3d = len(conv_obj.weights.shape) == 5
                new_weights = np.transpose(conv_obj.weights,
                                           axes=ArmConvolution3DOp.perm_onnx_to_ir(
                                           ) if is_3d else ArmConvolutionOp.perm_onnx_to_ir()
                                           )
                conv_attr.update({'weights': new_weights})
                conv_node.replace_obj(
                    'ArmConvolution3D' if is_3d else 'ArmConvolution', conv_attr)
        elif conv_obj.type == 'ConvTranspose':
            is_3d = len(conv_obj.weights.shape) == 5
            new_weights = np.transpose(conv_obj.weights,
                                       axes=ArmConvTranspose3DOp.perm_onnx_to_ir(
                                       ) if is_3d else ArmConvTransposeOp.perm_onnx_to_ir()
                                       )
            conv_attr.update({'weights': new_weights})
            conv_node.replace_obj(
                'ArmConvTranspose3D' if is_3d else 'ArmConvTranspose', conv_attr)
        elif conv_obj.type == 'ConvInteger':
            if str(conv_obj.x_zero_point.dtype) != 'uint8':
                WARN('[Parser]: Only support uint8 ConvInteger(%s) for now!' % conv)
                continue
            new_weights = np.transpose(conv_obj.weights,
                                       axes=ArmConvIntegerOp.perm_onnx_to_ir()
                                       )
            conv_attr.update({'weights': new_weights,
                              'x_zero_point': conv_obj.x_zero_point,
                              'w_zero_point': conv_obj.w_zero_point
                              })
            conv_node.replace_obj('ArmConvInteger', conv_attr)
        else:
            ERROR('[Parser]: Conv type %s is not implemented in rename_conv!' %
                  conv_obj.type)


def rename_gemm(graph):
    matches = single_node_matcher(graph, 'Gemm')
    for m in matches:
        gemm = m['target']
        gemm_obj = NodeWrap(graph, gemm)['object']
        in_edges = graph.sorted_in_edges(gemm, data=True)
        if gemm_obj is not None and len(in_edges) in (2, 3):
            output_shapes = gemm_obj.get_output_shapes()
            input_shapes = gemm_obj.get_input_shapes()
            if input_shapes and output_shapes and len(input_shapes) == len(in_edges) and len(output_shapes) >= 1 \
                    and output_shapes[0] is not None \
                    and all(d is not None for d in output_shapes[0]):
                if len(in_edges) == 2:
                    np_dtype_str = gemm_obj.get_inputs_info()[2][1]
                    insert_constant(
                        graph, gemm + '_C', np.zeros(output_shapes[0], dtype=getattr(np, np_dtype_str)), gemm, in_port=2)
                elif list(output_shapes[0]) != list(input_shapes[2]):
                    if input_shapes[2] is not None and all(d is not None for d in input_shapes[2]):
                        ret = OpNeedUniBroadcast.cal_reshape_and_tile([output_shapes[0], input_shapes[2]])
                        if ret:
                            if ret[0]['reshape']:
                                in_port = 2
                                insert_reshape(
                                    graph, in_edges[in_port][0], gemm, in_edges[in_port][2], ret[0]['reshape'],
                                    quantize=gemm_obj.quantize)
                                in_edges = graph.sorted_in_edges(gemm, data=True)
                            if ret[0]['tile']:
                                insert_tile(
                                    graph, in_edges[2][0], gemm, in_edges[2][2], ret[0]['tile'])
                            else:
                                ERROR(
                                    '[Parser]: Invalid pattern of Node(%s) for Gemm broadcasting in rename_gemm!' % gemm)
                        else:
                            ERROR(
                                '[Parser]: Cal reshape_tile error of Node(%s) for Gemm broadcasting in rename_gemm!' % gemm)
                    else:
                        ERROR(
                            '[Parser]: Get input_shape[2] error of Node(%s) for Gemm broadcasting in rename_gemm!' % gemm)
            else:
                ERROR(
                    '[Parser]: Get input/output shape error of Node(%s) for Gemm broadcasting in rename_gemm!' % gemm)
            gemm_attr = gemm_obj.copied_attr()
            gemm_attr.update({'trans_a': gemm_obj.transA,
                              'trans_b': gemm_obj.transB,
                              'alpha': gemm_obj.alpha,
                              'beta': gemm_obj.beta,
                              })
            NodeWrap(graph, gemm).replace_obj('ArmGemm', gemm_attr)
        else:
            ERROR(
                '[Parser]: Meets Invalid Gemm Op (%s) that cannot be converted in rename_gemm!' % gemm)


def rename_generate_proposals(graph):
    matches = single_node_matcher(graph, 'GenerateProposals')
    for m in matches:
        gp = m['target']
        gp_obj = NodeWrap(graph, gp)['object']
        if gp_obj is not None:
            gp_attr = gp_obj.copied_attr()
            gp_attr.update({'iou_threshold': gp_attr['nms_threshold']})
            NodeWrap(graph, gp).replace_obj('ArmGenerateProposals', gp_attr)


def rename_gridsample(graph):
    matches = single_node_matcher(graph, 'GridSample')
    for m in matches:
        gridsample = m['target']
        gridsample_obj = NodeWrap(graph, gridsample)['object']
        input_shapes = gridsample_obj.get_input_shapes()
        if len(input_shapes) < 1 or input_shapes[0] is None:
            ERROR('[Parser]: Meets invalid inputs of GridSample Node(%s) in rename_gridsample!' % gridsample)
            continue
        mode = gridsample_obj.mode
        if len(input_shapes[0]) == 4 and mode == 'linear':
            mode = 'bilinear'
        gridsample_attr = gridsample_obj.copied_attr()
        gridsample_attr.update({'method': mode.upper()})
        NodeWrap(graph, gridsample).replace_obj(
            'ArmGridSample', gridsample_attr)


def rename_layernorm(graph):
    matched = False
    matches = single_node_matcher(graph, 'LayerNormalization')
    for m in matches:
        layernorm = m['target']
        layernorm_obj = NodeWrap(graph, layernorm)['object']
        in_edges = graph.sorted_in_edges(layernorm, data=True)
        out_edges = graph.sorted_out_edges(layernorm, data=True)
        if layernorm_obj is None or len(in_edges) < 3 or len(out_edges) < 1:
            ERROR('[Parser]: Meets invalid LayerNormalization Node(%s) in rename_layernorm!' % layernorm)
            continue
        scale_in_attr = in_edges[1][2]
        bias_in_attr = in_edges[2][2]
        if scale_in_attr['tensor'] is None or not scale_in_attr['tensor'].is_const \
                or bias_in_attr['tensor'] is None or not bias_in_attr['tensor'].is_const:
            WARN('[Parser]: Meets unsupported non-constant scale and bias of LayerNormalization Node(%s) in rename_layernorm!' % layernorm)
            continue
        in_shapes = layernorm_obj.get_input_shapes()
        if len(in_shapes) < 1 or in_shapes[0] is None:
            ERROR('[Parser]: Meets invalid input shape of LayerNormalization Node(%s) in rename_layernorm!' % layernorm)
            continue
        in_shape_len = len(in_shapes[0])
        ln_axes = OpHasAxis.make_axes_non_negative(layernorm_obj.axes, in_shape_len)
        non_ln_axes = [axis for axis in range(in_shape_len) if axis not in ln_axes]
        pre_perm = non_ln_axes + ln_axes
        if pre_perm != list(range(in_shape_len)):
            updated_axes = list(range(len(non_ln_axes), in_shape_len))
            src, _, in_attr = in_edges[0]
            insert_transpose(graph, src, layernorm, in_attr, pre_perm, type='ArmTranspose')
            post_perm = Op.cal_inverse_perm(pre_perm)
            post_trans = insert_transpose_after(graph, layernorm, post_perm, type='ArmTranspose')
            if layernorm in graph._attr['output_names']:
                index = graph._attr['output_names'].index(layernorm)
                graph._attr['output_names'][index] = post_trans
        else:
            updated_axes = ln_axes
        matched = True
        graph.remove_edges_from(in_edges[1:])

        layernorm_attr = layernorm_obj.copied_attr()
        scale = scale_in_attr['tensor'].value
        biases = bias_in_attr['tensor'].value
        layernorm_attr.update({'weights': scale, 'biases': biases, 'axes': updated_axes})
        if layernorm_obj.quantize:
            if scale_in_attr['tensor'].scale_zp:
                layernorm_attr.update({'weights_scale_zp': list(scale_in_attr['tensor'].scale_zp)})
            if bias_in_attr['tensor'].scale_zp:
                layernorm_attr.update({'biases_scale_zp': list(bias_in_attr['tensor'].scale_zp)})
        NodeWrap(graph, layernorm).replace_obj('ArmLayerNorm', layernorm_attr)
    if matched:
        clear_redundant_nodes(graph)


def rename_groupnorm(graph):
    matches = single_node_matcher(graph, 'GroupNormalization')
    for m in matches:
        groupnorm = m['target']
        groupnorm_obj = NodeWrap(graph, groupnorm)['object']
        in_edges = graph.sorted_in_edges(groupnorm, data=True)
        if groupnorm_obj is None or len(in_edges) < 3:
            ERROR('[Parser]: Meet invalid GroupNormalization Node(%s) in rename_groupnorm!' % groupnorm)
            continue
        input_shapes = groupnorm_obj.get_input_shapes()
        if len(input_shapes) < 1 or input_shapes[0] is None or None in input_shapes[0]:
            ERROR('[Parser]: Meet invalid input shape of GroupNormalization Node(%s) in rename_groupnorm!' % groupnorm)
            continue
        scale_in_attr = in_edges[1][2]
        bias_in_attr = in_edges[2][2]
        if scale_in_attr['tensor'] is None or not scale_in_attr['tensor'].is_const \
                or bias_in_attr['tensor'] is None or not bias_in_attr['tensor'].is_const:
            WARN('[Parser]: Meets unsupported non-constant scale and bias of GroupNormalization Node(%s) in rename_groupnorm!' % groupnorm)
            continue
        num_groups = groupnorm_obj.num_groups
        channels = input_shapes[0][-1]
        channel_num_per_group = channels // num_groups
        scale = scale_in_attr['tensor'].value
        bias = bias_in_attr['tensor'].value
        if list(scale.shape) == [num_groups]:  # maybe onnx bug, from SPEC, scale/bias should be [c]
            scale = np.repeat(scale, channel_num_per_group, axis=0)
        if list(bias.shape) == [num_groups]:
            bias = np.repeat(bias, channel_num_per_group, axis=0)
        if list(scale.shape) != [channels] \
                or list(bias.shape) != [channels]:
            ERROR('[Parser]: Meet invalid weights/biases of GroupNormalization Node(%s) in rename_groupnorm!' % groupnorm)
            continue
        weights = scale
        biases = bias
        node_attr = groupnorm_obj.copied_attr()
        if num_groups == 1:
            tile_reps = input_shapes[0][1:-1] + [1]
            weights = np.tile(weights, tile_reps)
            biases = np.tile(biases, tile_reps)
            node_attr.update({'axis': None, 'axes': list(range(1, len(input_shapes[0]))),
                              'weights': weights, 'biases': biases})
            NodeWrap(graph, groupnorm).replace_obj('ArmLayerNorm', node_attr)
        elif num_groups == channels:
            node_attr.update({'non_channel_axes': None, 'weights': weights, 'biases': biases})
            NodeWrap(graph, groupnorm).replace_obj('ArmInstanceNorm', node_attr)
        else:
            node_attr.update({'group': num_groups, 'axis': -1, 'axes': None, 'weights': weights, 'biases': biases})
            NodeWrap(graph, groupnorm).replace_obj('ArmGroupNorm', node_attr)


def rename_logical(graph):
    logical_map = {'And': 'AND',
                   'Equal': 'EQUAL',
                   'Greater': 'GREATER',
                   'GreaterOrEqual': 'GREATER_EQUAL',
                   'Less': 'LESS',
                   'LessOrEqual': 'LESS_EQUAL',
                   'Not': 'NOT',
                   'Or': 'OR',
                   'Xor': 'XOR'}
    matches = extend_lists([single_node_matcher(graph, op)
                            for op in logical_map.keys()])
    for m in matches:
        logical = m['target']
        logical_obj = NodeWrap(graph, logical)['object']
        in_edges = graph.sorted_in_edges(logical, data=True)
        if logical_obj is not None \
                and ((logical_obj.type == 'Not' and len(in_edges) == 1) or len(in_edges) == 2):
            meta_ret = True
            in_types = [NodeWrap(graph, e[0])['object'].type for e in in_edges]
            in_tensors = [e[2]['tensor'] for e in in_edges]
            if (logical_obj.type == 'Not' and in_types.count('Constant') == 1) \
                    or in_types.count('Constant') == 2:
                meta_ret = False
                WARN(
                    '[Parser]: Logical Op (%s) with Constant inputs should be fused in rename_logical!' % logical)
            elif logical_obj.type == 'Not':
                pass
            elif len(in_tensors) == 2:
                if in_tensors[0].get_dtype() is None \
                        or in_tensors[1].get_dtype() is None \
                        or in_tensors[0].get_dtype() != in_tensors[1].get_dtype():
                    meta_ret = False
                    ERROR(
                        '[Parser]: Invalid inputs of Node(%s) for broadcasting in rename_logical!' % logical)
            if meta_ret:
                method = logical_map[logical_obj.type]
                logical_attr = logical_obj.copied_attr()
                logical_attr.update({'method': method})
                NodeWrap(graph, logical).replace_obj(
                    'ArmLogical', logical_attr)
        else:
            ERROR(
                '[Parser]: Meets invalid Logical Op (%s) that cannot be converted in rename_logical!' % logical)


def rename_matmulinteger(graph):
    matches = single_node_matcher(graph, 'MatMulInteger')
    for m in matches:
        matmul = m['target']
        matmul_obj = NodeWrap(graph, matmul)['object']
        in_edges = graph.sorted_in_edges(matmul)
        if matmul_obj is not None and 2 <= len(in_edges) <= 4:
            input_dtypes = matmul_obj.get_input_dtypes()
            if len(input_dtypes) >= 2 and all(dtype in ('int8', 'uint8') for dtype in input_dtypes[:2]):
                a_zp = matmul_obj.a_zero_point
                b_zp = matmul_obj.b_zero_point
                graph.remove_edges_from(in_edges[2:])
                matmul_attr = matmul_obj.copied_attr()
                matmul_attr.update(
                    {'a_zero_point': a_zp, 'b_zero_point': b_zp})
                NodeWrap(graph, matmul).replace_obj(
                    'ArmMatMulInteger', matmul_attr)
            else:
                ERROR(
                    '[Parser]: Meets invalid dtype of MatMulInteger Op (%s) in rename_matmulinteger!' % matmul)
        else:
            ERROR(
                '[Parser]: Meets invalid MatMulInteger Op (%s) in rename_matmulinteger!' % matmul)


def rename_maxunpool(graph):
    matched = False
    matches = single_node_matcher(graph, 'MaxUnpool')
    for m in matches:
        mup = m['target']
        mup_obj = NodeWrap(graph, mup)['object']
        in_edges = graph.sorted_in_edges(mup, data=True)
        if mup_obj is not None and len(in_edges) == 3:
            _, _, out_shape_in_attr = in_edges[2]
            if out_shape_in_attr['tensor'] is None \
                    or out_shape_in_attr['tensor'].value is None:
                ERROR(
                    '[Parser]: Meets MaxUnpool Node(%s) with invalid output shape in rename_maxunpool!' % mup)
                continue
            if not out_shape_in_attr['tensor'].is_const:
                WARN(
                    '[Parser]: Meets unsupported non-constant output_shape of MaxUnpool Node(%s) in rename_maxunpool!' % mup)
                continue
            matched = True
            output_shape = out_shape_in_attr['tensor'].value.tolist()
            graph.remove_edges_from(in_edges[2:])
            mup_attr = mup_obj.copied_attr()
            mup_attr.update(
                {'output_shape': output_shape, 'flatten_dim': 'NCHW'})
            NodeWrap(graph, mup).replace_obj('ArmMaxUnpool', mup_attr)
        else:
            ERROR('[Parser]: Meets invalid MaxUnpool Node(%s) in rename_maxunpool!' % mup)
    if matched:
        clear_redundant_nodes(graph)


def rename_moments(graph):
    matches = single_node_matcher(graph, 'Moments')
    for m in matches:
        moments = m['target']
        moments_obj = NodeWrap(graph, moments)['object']
        if moments_obj is None:
            ERROR('[Parser]: Meets invalid Moments Node(%s) in rename_moments!' % moments)
            continue
        moments_attr = moments_obj.copied_attr()
        moments_attr.update({'keepdims': True})
        NodeWrap(graph, moments).replace_obj('ArmMoments', moments_attr)
        if not moments_obj.keepdims:
            if len(moments_obj.get_input_shapes()) >= 1 \
                    and all([d is not None for d in moments_obj.get_input_shapes()[0]]):
                input_shape = list(moments_obj.get_input_shapes()[0])
                axes = sorted(OpHasAxis.make_axes_non_negative(
                    moments_obj.axes, len(input_shape)))
                out_shape = [d for (i, d) in enumerate(
                    input_shape) if i not in axes]
                post_reshapes = []
                for out_port in moments_obj.get_out_ports():
                    reshape = insert_reshape_after(graph,
                                                   moments,
                                                   out_shape,
                                                   out_port=out_port,
                                                   quantize=moments_obj.quantize)
                    post_reshapes.append(reshape)
                if moments in graph._attr['output_names'] and post_reshapes:
                    index = graph._attr['output_names'].index(moments)
                    graph._attr['output_names'].pop(index)
                    for reshape in post_reshapes:
                        graph._attr['output_names'].insert(index, reshape)
                        index += 1
            else:
                ERROR(
                    '[Parser]: Meets invalid Moments Node(%s) in rename_moments!' % moments)


def rename_mul_add_max_min(graph):
    for op_type in ['Mul', 'Add', 'Sub', 'Max', 'Min']:
        matches = single_node_matcher(graph, op_type)
        for m in matches:
            eltwise = m['target']
            eltwise_node = NodeWrap(graph, eltwise)
            eltwise_obj = eltwise_node['object']
            in_edges = graph.sorted_in_edges(eltwise, data=True)
            if len(in_edges) == 2:
                meta_ret = True
                in_types = [NodeWrap(graph, e[0])[
                    'object'].type for e in in_edges]
                in_shapes = [e[2]['tensor'].get_shape() for e in in_edges]
                if in_types.count('Constant') == 2:
                    meta_ret = False
                    WARN(
                        '[Parser]: Mul/Add/Sub/Max/Min (%s) with two Constant inputs should have been fused before rename_mul_add_max_min!' % eltwise)
                elif in_types.count('Constant') <= 1 \
                        and in_shapes[0] is not None \
                        and None not in in_shapes[0] \
                        and in_shapes[1] is not None \
                        and None not in in_shapes[1] \
                        and list(in_shapes[0]) == list(in_shapes[1]):
                    pass
                elif in_shapes[0] is not None \
                        and None not in in_shapes[0] \
                        and in_shapes[1] is not None \
                        and None not in in_shapes[1]:
                    pass
                else:
                    meta_ret = False
                    ERROR(
                        '[Parser]: Invalid pattern of Node(%s) to convert into Eltwise in rename_mul_add_max_min!' % eltwise)
                if meta_ret:
                    eltwise_attr = eltwise_obj.copied_attr()
                    method = op_type.upper()
                    eltwise_attr.update({'method': method})
                    eltwise_node.replace_obj('ArmEltwise', eltwise_attr)


def rename_normalization(graph):
    norm_types = ['LpNormalization']
    for op_type in norm_types:
        matches = single_node_matcher(graph, op_type)
        for m in matches:
            norm = m['target']
            norm_obj = NodeWrap(graph, norm)['object']
            method = 'L1' if norm_obj.p == 1 else 'L2'
            norm_attr = norm_obj.copied_attr()
            norm_attr.update({'method': method})
            NodeWrap(graph, norm).replace_obj(
                'ArmNormalization', norm_attr)


def rename_onehot(graph):
    matched = False
    matches = single_node_matcher(graph, 'OneHot')
    for m in matches:
        onehot = m['target']
        onehot_obj = NodeWrap(graph, onehot)['object']
        in_edges = graph.sorted_in_edges(onehot, data=True)
        if onehot_obj is None \
                or len(in_edges) != 3:
            ERROR(
                '[Parser]: invalid Onehot Node(%s) in rename_onehot!' % onehot)
            continue

        is_const_node = np.all(list(map(lambda node: node.type == 'Constant', list(
            map(lambda in_edge: NodeWrap(graph, in_edge[0])['object'], in_edges[1:])))))

        if in_edges[1][2]['tensor'].is_const \
                and in_edges[1][2]['tensor'].value is not None \
                and in_edges[2][2]['tensor'].is_const \
                and in_edges[2][2]['tensor'].value is not None \
                and in_edges[2][2]['tensor'].value.size == 2 \
                and is_const_node:
            matched = True
            graph.remove_edges_from(in_edges[1:])
            depth = int(in_edges[1][2]['tensor'].value)
            values = in_edges[2][2]['tensor'].value
            if values.dtype == 'float64':
                values = np.array(values).astype(np.float32)
            elif values.dtype == 'int64':
                values = np.array(values).astype(np.int32)

            onehot_attr = onehot_obj.copied_attr()
            onehot_attr.update({'depth': depth, 'values': values})
            NodeWrap(graph, onehot).replace_obj('ArmOneHot', onehot_attr)
        else:
            WARN(
                '[Parser]: Meets unsupported non-constant depth/values of Onehot Node(%s) in rename_onehot!' % onehot)
    if matched:
        clear_redundant_nodes(graph)


def rename_pad(graph):
    need_clear = False
    matches = single_node_matcher(graph, 'Pad')
    for m in matches:
        pad = m['target']
        pad_obj = NodeWrap(graph, pad)['object']
        in_edges = graph.sorted_in_edges(pad, data=True)
        if pad_obj is not None and len(in_edges) >= 1:
            input_shapes = pad_obj.get_input_shapes()
            if len(input_shapes) < 1 or input_shapes[0] is None or None in input_shapes[0]:
                ERROR('[Parser]: Meets invalid input shape of Pad Node(%s) in rename_pad!' % pad)
                continue
            pads_value = pad_obj.pads
            negative_pads = [val if val < 0 else 0 for val in pads_value]
            if any(val != 0 for val in negative_pads):
                input_length = len(input_shapes[0])
                if len(pads_value) != input_length * 2:
                    ERROR('[Parser]: Meets invalid pads of Pad Node(%s) in rename_pad!' % pad)
                    continue
                src, _, in_attr = in_edges[0]
                begins = (-1 * np.array(negative_pads[:input_length])).tolist()
                sizes = (np.array(input_shapes[0]) +
                         np.array(negative_pads[:input_length]) +
                         np.array(negative_pads[input_length:])).tolist()
                insert_slice(graph, src, pad, in_attr, begins, sizes, type='ArmSlice')
                pads_value = [val if val >= 0 else 0 for val in pads_value]
            if pad_obj.mode == 'wrap':
                max_pads_value = np.repeat(input_shapes[0], 2).tolist()
                if any(pad > max_pad for pad, max_pad in zip(pads_value, max_pads_value)):
                    WARN('[Parser]: Meets unsupported pads of Pad Node(%s) in rename_pad; '
                         'pads can not be larger than input shape!' % pad)
            pad_attr = pad_obj.copied_attr()
            pad_attr.update({'pads': pads_value})
            pad_attr.update({'constant_value': float(pad_obj.value)})
            NodeWrap(graph, pad).replace_obj('ArmPad', pad_attr)
            if len(in_edges) > 1:
                need_clear = True
                graph.remove_edges_from(in_edges[1:])
        else:
            ERROR('[Parser]: Meets invalid Pad Node(%s) in rename_pad!' % pad)
    if need_clear:
        clear_redundant_nodes(graph)


def rename_pool(graph):
    for op_type in ['AveragePool', 'LpPool', 'MaxPool']:
        matches = single_node_matcher(graph, op_type)
        for m in matches:
            pool = m['target']
            pool_node = NodeWrap(graph, pool)
            pool_obj = pool_node['object']
            pool_attr = pool_obj.copied_attr()
            if len(pool_obj.get_out_ports()) == 1:
                if op_type == 'LpPool':
                    method = 'L1' if int(pool_obj.p) == 1 else 'L2'
                else:
                    method = 'AVG' if op_type == 'AveragePool' else 'MAX'
                pool_attr.update({'method': method})
                pool_node.replace_obj('ArmPooling3D' if len(
                    pool_obj.kernel_shape) == 3 else 'ArmPooling', pool_attr)
            else:
                input_shapes = pool_obj.get_input_shapes()
                if len(input_shapes) < 1:
                    ERROR('[Parser]: Meets invalid MaxPool Node (%s) in rename_pool!' % pool)
                    continue
                if len(input_shapes[0]) != 4:
                    WARN(
                        '[Parser]: Only 4D MaxPool (%s) can be converted to MaxPoolingWithArgMax in rename_pool!' % pool)
                    continue
                pool_attr.update({'flatten_dim': 'NHWC'})
                pool_node.replace_obj('ArmMaxPoolingWithArgMax', pool_attr)


def rename_reduce(graph):
    reduce_methods = {'ReduceAll': 'ALL',
                      'ReduceAny': 'ANY',
                      'ReduceMean': 'MEAN',
                      'ReduceMin': 'MIN',
                      'ReduceMax': 'MAX',
                      'ReduceProd': 'PROD',
                      'ReduceSum': 'SUM',
                      'ReduceL1': 'L1',
                      'ReduceL2': 'L2',
                      'ReduceVariance': 'VARIANCE',
                      }
    for op_type in reduce_methods:
        matches = single_node_matcher(graph, op_type)
        for m in matches:
            reduce = m['target']
            reduce_obj = NodeWrap(graph, reduce)['object']
            reduce_attr = reduce_obj.copied_attr()
            method = reduce_methods[op_type]
            if method == 'VARIANCE' and reduce_obj.unbiased:
                method = 'UNBIASED_VARIANCE'
            reduce_attr.update({'method': method, 'keepdims': True})
            NodeWrap(graph, reduce).replace_obj('ArmReduce', reduce_attr)
            if reduce_obj.keepdims:
                continue
            if len(reduce_obj.get_input_shapes()) >= 1 \
                    and reduce_obj.get_input_shapes()[0] is not None \
                    and len(reduce_obj.get_output_shapes()) >= 1 \
                    and reduce_obj.get_output_shapes()[0] is not None:
                out_shape = reduce_obj.get_output_shapes()[0]
                if not out_shape:
                    out_shape = []
                reshape = insert_reshape_after(
                    graph, reduce, out_shape, type='Reshape',
                    quantize=reduce_obj.quantize)
                if reduce in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(reduce)
                    graph._attr['output_names'][index] = reshape
            else:
                ERROR(
                    '[Parser]: Meets invalid Reduce Node(%s) in rename_reduce!' % reduce)


def rename_reshape(graph):
    matches = single_node_matcher(graph, 'Reshape')
    for m in matches:
        reshape = m['target']
        reshape_obj = NodeWrap(graph, reshape)['object']
        if reshape_obj is not None:
            dim = reshape_obj.shape[:]
            out_shapes = reshape_obj.get_output_shapes()
            if len(out_shapes) > 0 \
                    and out_shapes[0] is not None:
                if len(out_shapes[0]) == len(dim) \
                        and all([(s == d or d == -1) for (s, d) in zip(out_shapes[0], dim)]):
                    dim = out_shapes[0][:]
                else:
                    ERROR(
                        '[Parser]: Dim of Reshape (%s) does not equal to output shape!' % reshape)
            cur_ver = reshape_obj.cur_version
            if cur_ver >= 5:
                in_edges = graph.sorted_in_edges(reshape)
                if len(in_edges) == 2:
                    graph.remove_edges_from(in_edges[1:])
            reshape_attr = reshape_obj.copied_attr()
            reshape_attr.update({'dim': dim})
            NodeWrap(graph, reshape).replace_obj('ArmReshape', reshape_attr)
        else:
            ERROR('[Parser]: Meets invalid Reshape Node(%s) in rename_reshape!' % reshape)


def rename_resize(graph):
    matches = single_node_matcher(graph, 'Resize')
    for m in matches:
        resize = m['target']
        resize_obj = NodeWrap(graph, resize)['object']
        in_edges = graph.sorted_in_edges(resize)
        if resize_obj is not None and len(in_edges) > 1:
            input_shape = resize_obj.get_input_shapes()[0]
            output_shape = resize_obj.get_output_shapes()[0]
            if not input_shape \
                    or len(input_shape) not in (4, 5) \
                    or not output_shape \
                    or len(output_shape) not in (4, 5) \
                    or resize_obj.scales.size not in (4, 5):
                WARN(
                    '[Parser]: Can only support Resize Op (%s) with 4D/5D inputs in rename_resize!' % resize)
                continue
            if np.any(resize_obj.scales[np.array([True] + [False] * (resize_obj.scales.size - 2) + [True])] != 1):
                WARN(
                    '[Parser]: Can not support Resize Op (%s) with none-1 scales in batch or channel dimension in rename_resize!' % resize)
                continue
            if resize_obj.coordinate_transformation_mode == 'tf_crop_and_resize':
                if resize_obj.roi is None \
                        or resize_obj.roi.size != 8 \
                        or np.any(np.reshape(resize_obj.roi, (2, -1))[:, 0] != np.array([0, 1], np.float32)) \
                        or np.any(np.reshape(resize_obj.roi, (2, -1))[:, 3] != np.array([0, 1], np.float32)):
                    ERROR(
                        '[Parser]: Meets invalid ROI for Resize Op (%s) in rename_resize!' % resize)
                    continue
                else:
                    batch = input_shape[0]
                    box_num = batch
                    boxes = np.tile(np.reshape(np.reshape(
                        resize_obj.roi, (2, -1))[:, 1:3].flatten(), (1, -1)), (box_num, 1))
                    box_indices = np.array(list(range(box_num)), np.int32)
                    crop_size = output_shape[1:3]
                    method = 'NEAREST' if resize_obj.mode == 'nearest' else 'BILINEAR'
                    graph.remove_edges_from(in_edges[1:])
                    insert_constant(graph, resize + '_boxes', boxes,
                                    resize, in_port=1, data_format='NHWC')
                    insert_constant(graph, resize + '_boxes_indices',
                                    box_indices, resize, in_port=2, data_format='NHWC')
                    crop_attr = resize_obj.copied_attr()
                    crop_attr.update({'crop_size': crop_size,
                                      'method': method
                                      })
                    NodeWrap(graph, resize).replace_obj(
                        'ArmCropAndResize', crop_attr)
            else:
                factors = resize_obj.scales.tolist()[1:-1]
                try:
                    sizes = resize_obj.sizes.tolist()[1:-1]
                except:
                    sizes = None
                graph.remove_edges_from(in_edges[1:])
                mode = resize_obj.coordinate_transformation_mode
                method = 'NEAREST' if resize_obj.mode == 'nearest' else 'BILINEAR'
                interp_attr = resize_obj.copied_attr()
                if hasattr(resize_obj, 'ori_keep_aspect_ratio_policy') and \
                        resize_obj.ori_keep_aspect_ratio_policy in ('not_larger', 'not_smaller'):
                    sizes = None
                    interp_attr.update({'keep_aspect_ratio_policy': resize_obj.ori_keep_aspect_ratio_policy})
                interp_attr.update({'factors': factors,
                                    'sizes': sizes,
                                    'method': method,
                                    'mode': mode})
                NodeWrap(graph, resize).replace_obj('ArmResize', interp_attr)


def rename_roipool(graph):
    matches = single_node_matcher(graph, 'MaxRoiPool')
    for m in matches:
        max_roipool = m['target']
        max_roipool_obj = NodeWrap(graph, max_roipool)['object']
        in_edges = graph.sorted_in_edges(max_roipool, keys=True, data=True)
        if max_roipool_obj is not None and len(in_edges) == 2:
            rois_src, _, k, in_attr = in_edges[1]
            insert_gather(graph, rois_src, max_roipool, np.array(
                [0, 2, 1, 4, 3], np.int32), axis=1, edge_attr=in_attr, key=k)
            roipool_attr = max_roipool_obj.copied_attr()
            roipool_attr.update(
                {'spatial': 2 * [max_roipool_obj.spatial_scale]})
            NodeWrap(graph, max_roipool).replace_obj(
                'ArmMaxRoiPool', roipool_attr)


def rename_roialign(graph):
    matches = single_node_matcher(graph, 'RoiAlign')
    for m in matches:
        node_name = m['target']
        node_obj = NodeWrap(graph, node_name)['object']

        in_edges = graph.sorted_in_edges(node_name, data=True)
        input_shapes = node_obj.get_input_shapes()
        if node_obj is not None \
                and len(in_edges) == 3 \
                and len(input_shapes) == 3 \
                and input_shapes[1] is not None \
                and input_shapes[2] is not None \
                and len(input_shapes[1]) == 2 \
                and len(input_shapes[2]) == 1:
            # concat
            concat = get_valid_node_name(graph, node_name + '_concat')
            roi_dtype = None
            for index in [1, 2]:
                src, dst, attr = in_edges[index]
                if index == 1:
                    roi_dtype = attr['tensor'].dtype
                graph.remove_edge(src, dst)
                attr['dst_in_port'] = 1 if index == 1 else 0
                graph.add_edge(src, concat, **attr)
            concat_attr = node_obj.copied_attr()
            concat_attr.update({'name': concat, 'axis': 1})
            NodeWrap(graph, concat).replace_obj('ArmConcat', concat_attr)

            if in_edges[2][-1]['tensor'].dtype != roi_dtype:
                insert_cast(graph, in_edges[2][0], concat, roi_dtype,
                            in_attr=in_edges[2][-1], type='ArmCast')

            # gather, reshape
            indices = np.array([1, 0, 3, 2], np.int32)
            roi_inp, _, roi_attr = in_edges[1]
            insert_gather(graph, roi_inp, concat, indices,
                          axis=1, edge_attr=roi_attr)

            index_inp, _, index_attr = graph.sorted_in_edges(concat, data=True)[0]
            reshape_dim = list(input_shapes[2]) + [1]
            insert_reshape(graph, index_inp, concat, index_attr,
                           reshape_dim, type='ArmReshape',
                           quantize=node_obj.quantize)

            # concat->roialign
            graph.add_edge(concat, node_name, **
                           {'src_out_port': 0, 'dst_in_port': 1})

            # update roialign attrs
            roialign_attr = node_obj.copied_attr()
            roialign_attr.update({
                                 'pooled_shape': [node_obj.output_height, node_obj.output_width],
                                 'spatial_scale': 2 * [node_obj.spatial_scale],
                                 'method': node_obj.mode.upper(),
                                 'sample_ratio': 2 * [node_obj.sampling_ratio],
                                 })
            NodeWrap(graph, node_name).replace_obj(
                'ArmRoiAlign', roialign_attr)


def rename_scatternd(graph):
    matches = single_node_matcher(graph, 'ScatterND')
    for m in matches:
        scatter = m['target']
        scatter_obj = NodeWrap(graph, scatter)['object']
        if scatter_obj is not None:
            scatter_attr = scatter_obj.copied_attr()
            reduction = scatter_obj.reduction
            scatter_attr.update({'reduction': reduction.upper()})
            NodeWrap(graph, scatter).replace_obj('ArmScatterND', scatter_attr)
        else:
            ERROR(
                '[Parser]: Meets invalid ScatterND Op (%s) in rename_scatternd!' % scatter)


def rename_scatterel(graph):
    matches = single_node_matcher(graph, ['ScatterElements', 'Scatter'])
    for m in matches:
        scatterel = m['target']
        scatterel_obj = NodeWrap(graph, scatterel)['object']
        if scatterel_obj is not None:
            scatterel_attr = scatterel_obj.copied_attr()
            if scatterel_obj.type == 'ScatterElements':
                reduction = scatterel_obj.reduction
                scatterel_attr.update({'reduction': reduction.upper()})
            NodeWrap(graph, scatterel).replace_obj(
                'ArmScatterElements', scatterel_attr)
        else:
            ERROR(
                '[Parser]: Meets invalid ScatterElements/Scatter Op (%s) in rename_scatterel!' % scatterel)


def rename_slice(graph):
    matches = single_node_matcher(graph, 'Slice')
    for m in matches:
        slice = m['target']
        slice_obj = NodeWrap(graph, slice)['object']
        in_edges = graph.sorted_in_edges(slice, data=True)
        if slice_obj is not None \
                and ((slice_obj.cur_version == 1 and len(in_edges) == 1) or (slice_obj.cur_version > 1 and 3 <= len(in_edges) <= 5)):
            if len(in_edges) > 1 and any(not in_attr['tensor'].is_const for _, _, in_attr in in_edges[1:]):
                ERROR('Meets unsupported non-const starts/ends/axes/steps of Slice Node(%s) in rename_slice!' % slice)
                continue
            graph.remove_edges_from(in_edges[1:])
            slice_attr = slice_obj.copied_attr()
            ends = np.array(slice_obj.ends, np.int64)
            ends_mask = np.logical_and(
                ends < -1, np.array(slice_obj.steps, np.int64) < 0)
            ends[ends_mask] = -1
            slice_attr.update({'ends': ends.tolist()})
            if 'steps' not in slice_attr:
                slice_attr.update({'steps': slice_obj.steps})
            NodeWrap(graph, slice).replace_obj('ArmSlice', slice_attr)
        else:
            ERROR('[Parser]: Meets invalid Slice Op (%s) in rename_slice!' % slice)


def rename_tile(graph):
    matches = single_node_matcher(graph, 'Tile')
    for m in matches:
        tile = m['target']
        in_edges = graph.sorted_in_edges(tile, data=True)
        if len(in_edges) == 2:
            reps = in_edges[1][0]
            reps_value = in_edges[1][2]['tensor'].value
            graph.remove_edge(reps, tile)
            tile_attr = NodeWrap(graph, tile)['object'].copied_attr()
            tile_attr.update({'reps': reps_value.tolist()})
            NodeWrap(graph, tile).replace_obj('ArmTile', tile_attr)

            _, _, in_attr = in_edges[0]
            if in_attr['tensor'] is not None \
                    and (in_attr['tensor'].dtype is not None or len(in_attr['tensor'].scale_zp) > 0):
                for _, _, out_attr in graph.sorted_out_edges(tile, data=True):
                    if out_attr['tensor'] is not None:
                        out_attr['tensor'].dtype = in_attr['tensor'].dtype
                        out_attr['tensor'].scale_zp = in_attr['tensor'].scale_zp
                        out_attr['tensor'].activation_quantization_axis = in_attr['tensor'].activation_quantization_axis


def rename_topk(graph):
    matches = single_node_matcher(graph, 'TopK')
    for m in matches:
        topk = m['target']
        topk_obj = NodeWrap(graph, topk)['object']
        if topk_obj is not None:
            ver = topk_obj.cur_version
            in_edges = graph.sorted_in_edges(topk)
            if (ver == 1 and len(in_edges) == 1) \
                    or (ver > 1 and len(in_edges) == 2):
                in_consts = topk_obj.sorted_in_consts()
                k = topk_obj.k if ver == 1 else int(in_consts[0][2])
                need_sorted = topk_obj.sorted if ver >= 11 else True
                largest = topk_obj.largest if ver >= 11 else True
                topk_attr = topk_obj.copied_attr()
                topk_attr.update(
                    {'k': k, 'sorted': need_sorted, 'largest': largest})
                NodeWrap(graph, topk).replace_obj('ArmTopK', topk_attr)
                if ver > 1:
                    graph.remove_edges_from(in_edges[1:])


def rename_where(graph):
    matches = single_node_matcher(graph, 'Where')
    for m in matches:
        where = m['target']
        where_obj = NodeWrap(graph, where)['object']
        if where_obj is not None:
            in_consts = where_obj.sorted_in_consts()
            for c, _, v in in_consts:
                if v is not None:
                    NodeWrap(graph, c)['object'].value = v.astype(np.float32)
            select_attr = where_obj.copied_attr()
            NodeWrap(graph, where).replace_obj('ArmWhere', select_attr)


def split_crd_d2s(graph):
    matches = single_node_matcher(graph, 'DepthToSpace')
    for m in matches:
        d2s = m['target']
        d2s_obj = NodeWrap(graph, d2s)['object']
        if d2s_obj is not None and d2s_obj.data_format == 'NHWC' and d2s_obj.mode == 'CRD':
            in_edges = graph.sorted_in_edges(d2s, data=True)
            out_edges = graph.sorted_out_edges(d2s, data=True)
            if len(in_edges) == 1 and len(out_edges) >= 1:
                input_shapes = d2s_obj.get_input_shapes()
                n, h, w, c = input_shapes[0]
                dim = [n, h, w, c // d2s_obj.blocksize **
                       2, d2s_obj.blocksize, d2s_obj.blocksize]
                out_dim = [n, h * d2s_obj.blocksize, w *
                           d2s_obj.blocksize, c // d2s_obj.blocksize ** 2]
                perm = [0, 1, 4, 2, 5, 3]
                src, _, in_attr = in_edges[0]
                reshape = insert_reshape(graph, src, d2s, in_attr, dim, quantize=d2s_obj.quantize)
                in_edges = graph.sorted_in_edges(d2s, data=True)
                _, _, in_attr2 = in_edges[0]
                insert_transpose(graph, reshape, d2s, in_attr2, perm)
                insert_constant(graph, d2s + '_shape', np.array(out_dim,
                                                                np.int64), d2s, in_port=1, data_format='NHWC')
                out_reshape_attr = d2s_obj.copied_attr()
                out_reshape_attr.update({'name': d2s, 'opset_version': 5})
                NodeWrap(graph, d2s).replace_obj('Reshape', out_reshape_attr)
            else:
                ERROR(
                    '[Parser]: Invalid DepthToSpace Node (%s) to split in split_crd_d2s!' % d2s)


def split_expand(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('expand', {'op': 'Expand'}),
                                      ('shape', {'op': 'Constant', 'unique': False})
                                      ],
                               edges=[
                                   ('shape', 'expand', {'src_out_port': 0, 'dst_in_port': 1})]
                               )
    for m in matches:
        expand, shape = m['expand'], m['shape']
        expand_obj = NodeWrap(graph, expand)['object']
        shape_obj = NodeWrap(graph, shape)['object']
        in_edges = graph.sorted_in_edges(expand, data=True)
        if expand_obj is not None \
                and shape_obj is not None \
                and len(in_edges) == 2:
            input_shapes = expand_obj.get_input_shapes()
            if len(input_shapes) == 2 \
                    and input_shapes[0] is not None \
                    and all(d is not None for d in input_shapes[0]):
                matched = True
                graph.remove_edges_from(in_edges[1:])
                input_shape = input_shapes[0]
                shape_value = shape_obj.value.astype(np.int32)
                quantize = expand_obj.quantize
                # Dimensions are right alignment
                if shape_value.size > len(input_shape):
                    diff_size = shape_value.size - len(input_shape)
                    input_shape = [1] * diff_size + input_shape
                    src, _, in_attr = in_edges[0]
                    insert_reshape(graph, src, expand, in_attr, input_shape, quantize=quantize)
                elif shape_value.size < len(input_shape):
                    diff_size = len(input_shape) - shape_value.size
                    shape_value = np.concatenate(
                        [np.array([1] * diff_size, np.int32), shape_value], axis=0)
                input_shape = np.array(input_shape, np.int32)
                ones_mask = shape_value == 1
                shape_value[ones_mask] = input_shape[ones_mask]
                tile_reps = shape_value // input_shape
                insert_constant(graph, expand + '_reps', tile_reps,
                                expand, in_port=1, data_format='NHWC')
                NodeWrap(graph, expand).replace_obj(
                    'Tile', {'name': expand, 'opset_version': 6, 'quantize': quantize})
            else:
                ERROR(
                    '[Parser]: Meets invalid shape of Expand Node (%s) in split_expand!' % expand)
        else:
            ERROR('[Parser]: Meets invalid Expand Node (%s) or Constant Node (%s) in split_expand!' % (
                expand, shape))
    if matched:
        clear_redundant_nodes(graph)


def fuse_clip(graph):
    supportted_op_clusters = Op.framework_to_op(
        graph._attr['framework']) + [ArmOp]
    possible_op_types = [op_type.get_concrete_subclass_names()
                         for op_type in supportted_op_clusters]
    possible_op_types = extend_lists(possible_op_types)
    ops_has_clip_list = list(set(BaseActivationOp.get_concrete_subclass_names())
                             .difference(BaseReluOp.get_concrete_subclass_names())
                             .intersection(possible_op_types))
    clip_ops_list = ['Clip', 'ArmClip']
    fuse_clip_combinations = itertools.product(
        ops_has_clip_list, clip_ops_list)
    for op_type, clip_type in fuse_clip_combinations:
        matches = two_nodes_matcher(graph, op_type, clip_type)
        for m in matches:
            op_has_clip, clip = m['begin'], m['end']
            op_has_clip_obj = NodeWrap(graph, op_has_clip)['object']
            clip_obj = NodeWrap(graph, clip)['object']
            if op_has_clip_obj is not None and op_has_clip_obj.activations == 'NONE':
                op_has_clip_obj.clip_min = clip_obj.min
                op_has_clip_obj.clip_max = clip_obj.max
                remove_node_safely(graph, clip)


def fuse_quant_op(graph, op_list):
    '''Merge ArmDequantize(one or multiple)+float op+ArmQuantize into one quant op.
    '''
    if not graph._attr.get('quantize', False) or not op_list:
        return
    op_list = [op_list] if not isinstance(op_list, (list, tuple)) else list(op_list)

    matched = False
    matches = single_node_matcher(graph, op_list)
    for m in matches:
        float_op = m['target']
        in_edges = graph.sorted_in_edges(float_op, data=True)
        if len(in_edges) < 1:
            ERROR('[Parser]: Meets invalid float Op(%s) in fuse_quant_op!' % float_op)
            continue
        out_edges = graph.sorted_out_edges(float_op, data=True)
        if len(out_edges) < 1:
            continue

        op_in_names = [e[0] for e in in_edges]
        op_out_names = [e[1] for e in out_edges]
        names = op_in_names + [float_op] + op_out_names
        obj_dict = {n: NodeWrap(graph, n)['object'] for n in names}
        if any(v is None for v in obj_dict.values()):
            ERROR('[Parser]: Meets invalid Op in fuse_quant_op!')
            continue
        if any(obj_dict[n].type != 'ArmDeQuantize' for n in op_in_names):
            continue
        if any(len(graph.sorted_in_edges(n)) != 1 for n in op_in_names):
            ERROR('[Parser]: Meets invalid ArmDeQuantize Op in fuse_quant_op!')
            continue
        if any(obj_dict[n].type != 'ArmQuantize' for n in op_out_names):
            continue

        matched = True
        graph.remove_edges_from(in_edges)

        for i, dequant in enumerate(op_in_names):
            dequant_in_edges = graph.sorted_in_edges(dequant, data=True)
            src, _, in_attr = dequant_in_edges[0]
            new_in_attr = copy.deepcopy(in_attr)
            new_in_attr['dst_in_port'] = i
            x_scale, x_zp = obj_dict[dequant].scale, obj_dict[dequant].zero_point
            new_in_attr['tensor'].dtype = str(x_zp.dtype)
            new_in_attr['tensor'].scale_zp = (x_scale, x_zp)
            new_in_attr['tensor'].activation_quantization_axis = obj_dict[dequant].axis
            graph.add_edge(src, float_op, **new_in_attr)

        for quant in op_out_names:
            src_out_port = graph.sorted_in_edges(quant, data=True)[0][2]['src_out_port']
            y_scale, y_zp = obj_dict[quant].scale, obj_dict[quant].zero_point
            quant_out_edges = graph.sorted_out_edges(quant, data=True)
            for _, dst, out_attr in quant_out_edges:
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


def fuse_relu(graph):
    supportted_op_clusters = Op.framework_to_op(
        graph._attr['framework']) + [ArmOp]
    possible_op_types = [op_type.get_concrete_subclass_names()
                         for op_type in supportted_op_clusters]
    possible_op_types = extend_lists(possible_op_types)
    ops_has_relu_list = list(set(BaseActivationOp.get_concrete_subclass_names())
                             .difference(BaseReluOp.get_concrete_subclass_names())
                             .intersection(possible_op_types))
    relu_ops_list = list(set(BaseReluOp.get_concrete_subclass_names()).intersection(
        possible_op_types)) + ['Clip', 'ArmClip']
    relu_ops_list = list(set(relu_ops_list).difference(['PRelu']))
    ops_has_relu_list = list(set(ops_has_relu_list).difference(relu_ops_list))
    fuse_relu_combinations = itertools.product(
        ops_has_relu_list, relu_ops_list)
    for op_type, relu_type in fuse_relu_combinations:
        matched = False
        matches1 = matched_patterns(graph,
                                    nodes=[
                                        ('linear', {'op': op_type}),
                                        ('relu', {'op': relu_type})
                                    ],
                                    edges=[
                                        ('linear', 'relu'),
                                    ])
        matches2 = matched_patterns(graph,
                                    nodes=[
                                        ('linear', {'op': op_type}),
                                        ('transpose', {'op': 'ArmTranspose'}),
                                        ('relu', {'op': relu_type})
                                    ],
                                    edges=[
                                        ('linear', 'transpose'),
                                        ('transpose', 'relu')
                                    ])
        matches = matches1 + matches2
        for m in matches:
            op_has_relu, relu, transpose = m['linear'], m['relu'], m.get(
                'transpose', None)
            op_has_relu_node = NodeWrap(graph, op_has_relu)
            op_has_relu_obj = op_has_relu_node['object']
            relu_obj = NodeWrap(graph, relu)['object']
            transpose_obj = NodeWrap(graph, transpose)[
                'object'] if transpose is not None else None
            if op_has_relu_obj is None \
                    or relu_obj is None \
                    or (not isinstance(op_has_relu_obj, BaseActivationOp)) \
                    or op_has_relu_obj.activations != 'NONE' \
                    or len(graph.sorted_out_edges(op_has_relu)) != 1 \
                    or (transpose and transpose_obj is None) \
                    or (transpose and len(graph.sorted_out_edges(transpose)) != 1):
                continue
            relu_attr = relu_obj.copied_attr()
            if relu_obj.type in ('Clip', 'ArmClip'):
                if FLOAT_EQUAL(relu_obj.min, 0) and FLOAT_EQUAL(relu_obj.max, 6):
                    relu_attr.update({'activations': 'RELU6'})
                else:
                    continue
            matched = True
            op_has_relu_obj.update_activation(relu_attr)
            node_before_relu = op_has_relu if (not transpose) else transpose
            graph.remove_edge(node_before_relu, relu)
            for _, dst, out_attr in graph.sorted_out_edges(relu, data=True):
                graph.remove_edge(relu, dst)
                graph.add_edge(node_before_relu, dst, **out_attr)
            if relu in graph._attr['output_names']:
                out_index = graph._attr['output_names'].index(relu)
                graph._attr['output_names'][out_index] = node_before_relu
        if matched:
            clear_redundant_nodes(graph)


def detection_post_process(graph, params):
    if params.get('detection_postprocess', '').upper() in ('SSD', 'SSD_RESNET'):
        if len(graph._attr['output_names']) == 2:
            out1, out2 = graph._attr['output_names']
            out1_obj, out2_obj = NodeWrap(
                graph, out1)['object'], NodeWrap(graph, out2)['object']
            if out1_obj is None or out2_obj is None:
                ERROR('[Parser]: Invalid output nodes (%s or %s) for detection_post_process!' % (
                    out1, out2))
                return
            out1_out_shapes = out1_obj.get_output_shapes()
            out2_out_shapes = out2_obj.get_output_shapes()
            if not out1_out_shapes \
                    or out1_out_shapes[0] is None \
                    or any((shape is None for shape in out1_out_shapes[0])) \
                    or not out2_out_shapes \
                    or out2_out_shapes[0] is None \
                    or any((shape is None for shape in out2_out_shapes[0])):
                ERROR('[Parser]: Invalid params of output nodes (%s or %s) for detection_post_process!' % (
                    out1, out2))
                return
            if len(out1_out_shapes[0]) not in (3, 4):
                ERROR('[Parser]: The length of output shape of output node (%s) should be 3 or 4 but got %d for detection_post_process!' % (
                    out1, len(out1_out_shapes[0])))
                return
            if len(out2_out_shapes[0]) not in (3, 4):
                ERROR('[Parser]: The length of output shape of output node (%s) should be 3 or 4 but got %d for detection_post_process!' % (
                    out2, len(out2_out_shapes[0])))
                return

            out1_new_dim, out2_new_dim = None, None
            origin_out1_out_edges = graph.sorted_out_edges(out1, data=True)
            origin_out2_out_edges = graph.sorted_out_edges(out2, data=True)
            if len(out1_out_shapes[0]) == 3 and len(out2_out_shapes[0]) == 3 \
                    and out1_out_shapes[0][1] == 1 and out2_out_shapes[0][1] == 1:
                if out1_out_shapes[0][2] % 4 == 0 and out2_out_shapes[0][2] % (out1_out_shapes[0][2] // 4) == 0:
                    pred_box_num = out1_out_shapes[0][2] // 4
                    total_classes_num = out2_out_shapes[0][2] // pred_box_num
                    out1_new_dim = [out1_out_shapes[0][0], pred_box_num, 4]
                    out2_new_dim = [out2_out_shapes[0][0],
                                    pred_box_num, total_classes_num]
                else:
                    pred_box_num = out2_out_shapes[0][2] // 4
                    total_classes_num = out1_out_shapes[0][2] // pred_box_num
                    out1_new_dim = [out1_out_shapes[0][0],
                                    pred_box_num, total_classes_num]
                    out2_new_dim = [out2_out_shapes[0][0], pred_box_num, 4]
            elif (len(out1_out_shapes[0]) == 4 or len(out2_out_shapes[0]) == 4):
                if out1_out_shapes[0][0] != out2_out_shapes[0][0] \
                        or int(np.prod(out1_out_shapes[0][1:-1])) != int(np.prod(out2_out_shapes[0][1:-1])):
                    ERROR('[Parser]: Meets unexpected output shape of output nodes (%s, %s) for detection_post_process!' % (
                        out1, out2))
                    return
                if len(out1_out_shapes[0]) == 4:
                    out1_new_dim = [out1_out_shapes[0][0], int(
                        np.prod(out1_out_shapes[0][1:-1])), out1_out_shapes[0][-1]]
                else:  # len(out2_out_shapes[0]) == 4
                    out2_new_dim = [out2_out_shapes[0][0], int(
                        np.prod(out2_out_shapes[0][1:-1])), out2_out_shapes[0][-1]]

            if out1_new_dim is not None:
                out1_out_edges = graph.sorted_out_edges(out1, data=True)
                _, out_name1, out_attr1 = out1_out_edges[0]
                reshape1 = insert_reshape(
                    graph, out1, out_name1, out_attr1, out1_new_dim, type='ArmReshape')
                out1 = reshape1
                graph._attr['output_names'][0] = reshape1

            if out2_new_dim is not None:
                out2_out_edges = graph.sorted_out_edges(out2, data=True)
                _, out_name2, out_attr2 = out2_out_edges[0]
                reshape2 = insert_reshape(
                    graph, out2, out_name2, out_attr2, out2_new_dim, type='ArmReshape')
                out2 = reshape2
                graph._attr['output_names'][1] = reshape2

            vaild_box_num = None

            if out1_out_shapes[0][1] != out2_out_shapes[0][1]:
                out1_tensors = NodeWrap(graph, out1)['object'].get_output_tensors()
                out2_tensors = NodeWrap(graph, out2)['object'].get_output_tensors()
                trans1 = get_valid_node_name(graph, out1 + '_post_transpose')
                trans2 = get_valid_node_name(graph, out2 + '_post_transpose')
                perm = [0, 2, 1]

                out1_attr = None
                for _, out, out_attr in graph.sorted_out_edges(out1, data=True):
                    # graph.remove_edge(out1, out)
                    if out1_attr is None:
                        out1_attr = copy.deepcopy(out_attr)
                        out1_attr['dst_in_port'] = 0
                    new_out_attr = copy.deepcopy(out_attr)
                    if out1_tensors[0] is not None:
                        new_out_attr['tensor'].value = np.transpose(out1_tensors[0], perm)
                        out1_out_shapes = [new_out_attr['tensor'].value.shape]
                    # graph.add_edge(trans1, out, **new_out_attr)
                graph.add_edge(out1, trans1, **out1_attr)
                NodeWrap(graph, trans1).replace_obj('ArmTranspose', {'name': trans1, 'perm': perm})

                out2_attr = None
                for _, out, out_attr in graph.sorted_out_edges(out2, data=True):
                    # graph.remove_edge(out2, out)
                    if out2_attr is None:
                        out2_attr = copy.deepcopy(out_attr)
                        out2_attr['dst_in_port'] = 0
                    new_out_attr = copy.deepcopy(out_attr)
                    if out2_tensors[0] is not None:
                        new_out_attr['tensor'].value = np.transpose(out2_tensors[0], perm)
                        out2_out_shapes = [new_out_attr['tensor'].value.shape]
                    # graph.add_edge(trans2, out, **new_out_attr)
                graph.add_edge(out2, trans2, **out2_attr)
                NodeWrap(graph, trans2).replace_obj('ArmTranspose', {'name': trans2, 'perm': perm})

                out1, out2 = trans1, trans2
                # graph._attr['output_names'] = [trans1, trans2]
                # out1_out_shapes = NodeWrap(graph, trans1)['object'].get_output_shapes()
                # out2_out_shapes = NodeWrap(graph, trans2)['object'].get_output_shapes()

            if out2_out_shapes and out2_out_shapes[0][-1] == 4:
                class_predict, box_predict = out1, out2
                origin_box_pred_out_edges = origin_out2_out_edges
                origin_class_pred_out_edges = origin_out1_out_edges
                class_num = out1_out_shapes[0][-1]
                vaild_box_num = out2_out_shapes[0][1]
            else:
                class_predict, box_predict = out2, out1
                origin_box_pred_out_edges = origin_out1_out_edges
                origin_class_pred_out_edges = origin_out2_out_edges
                class_num = out2_out_shapes[0][-1]
                vaild_box_num = out1_out_shapes[0][1]

            class_predict_in_edges = graph.sorted_in_edges(class_predict)
            class_predict_out_edges = graph.sorted_out_edges(class_predict, data=True)
            box_predict_out_edges = graph.sorted_out_edges(box_predict, data=True)

            if not class_predict_out_edges:
                class_predict_out_edges = origin_class_pred_out_edges
            if not box_predict_out_edges:
                box_predict_out_edges = origin_box_pred_out_edges
            anchors = None
            is_xy_box_format = (params.get('box_format', '').upper() == 'XY')
            if graph._attr['framework'].name == 'CAFFE' \
                    or params.get('detection_postprocess', '').upper() == 'SSD_RESNET' \
                    or is_xy_box_format:
                if len(box_predict_out_edges) == 1:
                    split = get_valid_node_name(
                        graph, box_predict + '_post_split')
                    concat = get_valid_node_name(
                        graph, box_predict + '_post_concat')
                    graph.add_edge(split, concat, **
                                   {'src_out_port': 0, 'dst_in_port': 1})
                    graph.add_edge(split, concat, **
                                   {'src_out_port': 1, 'dst_in_port': 0})
                    graph.add_edge(split, concat, **
                                   {'src_out_port': 2, 'dst_in_port': 3})
                    graph.add_edge(split, concat, **
                                   {'src_out_port': 3, 'dst_in_port': 2})
                    for _, dst, out_attr in box_predict_out_edges:
                        # graph.remove_edge(box_predict, dst)
                        graph.add_edge(box_predict, split, **out_attr)
                        # graph.add_edge(concat, dst)
                    # if box_predict in graph._attr['output_names']:
                    #     out_index = graph._attr['output_names'].index(
                    #         box_predict)
                    #     graph._attr['output_names'][out_index] = concat

                    split_attr = NodeWrap(graph, box_predict)[
                        'object'].copied_attr()
                    split_attr.update(
                        {'name': split, 'axis': 2, 'split': [1] * 4})
                    NodeWrap(graph, split).replace_obj('ArmSplit', split_attr)
                    concat_attr = NodeWrap(graph, box_predict)[
                        'object'].copied_attr()
                    concat_attr.update({'name': concat, 'axis': 2})
                    NodeWrap(graph, concat).replace_obj(
                        'ArmConcat', concat_attr)
                    box_predict = concat

            class_pred_obj = NodeWrap(graph, class_predict)['object']
            class_pred_parent = class_predict_in_edges[0][0]
            class_pred_parent_obj = NodeWrap(
                graph, class_pred_parent)['object']
            class_predict_tensor = copy.deepcopy(
                class_predict_out_edges[0][2]['tensor'])
            box_predict_tensor = copy.deepcopy(
                box_predict_out_edges[0][2]['tensor'])
            # Add Sigmoid/Softmax if class_predict is not sigmoid or softmax
            if class_pred_obj.type not in ('ArmActivation', 'ArmSoftmax') and \
                    class_pred_parent_obj.type not in ('ArmActivation', 'ArmSoftmax'):
                act = params.get('ssd_activation', 'softmax').lower()
                if act not in ('softmax', 'sigmoid'):
                    ERROR('[Parser]: Meet invalid value of activation (%s) in detection_post_process! '
                          'Supported values are softmax, sigmoid!' % act)
                act_node = get_valid_node_name(
                    graph, class_predict + '_' + act)
                graph.add_edge(class_predict, act_node, **
                               {'tensor': class_predict_out_edges[0][2]['tensor']})
                if act == 'softmax':
                    NodeWrap(graph, act_node).replace_obj(
                        'ArmSoftmax', {'name': act_node})
                else:
                    NodeWrap(graph, act_node).replace_obj(
                        'ArmActivation', {'name': act_node, 'method': 'SIGMOID'})
                class_predict = act_node

            decode_box = get_valid_node_name(
                graph, params['model_name'] + '_decode_box')
            class_predict_out_attr = {'src_out_port': 0,
                                      'dst_in_port': 0,
                                      'tensor': class_predict_tensor}
            box_predict_out_attr = {'src_out_port': 0,
                                    'dst_in_port': 1,
                                    'tensor': box_predict_tensor}

            if graph._attr.get('quantize', False):
                if not class_predict_tensor.scale_zp \
                        or not box_predict_tensor.scale_zp \
                        or len(class_predict_tensor.scale_zp) != 2 \
                        or len(box_predict_tensor.scale_zp) != 2:
                    ERROR(
                        '[Parser]: Meet invalid scale/zp for SSD model in detection_post_process.')
                class_predict_scale = np.array(
                    class_predict_tensor.scale_zp[0]).astype(np.float32)
                class_predict_zp = class_predict_tensor.scale_zp[1]
                box_predict_scale = np.array(
                    box_predict_tensor.scale_zp[0]).astype(np.float32)
                box_predict_zp = box_predict_tensor.scale_zp[1]

                class_predict_dequant = get_valid_node_name(
                    graph, class_predict + '_dequant_class_predict')
                graph.add_edge(class_predict, class_predict_dequant,
                               **class_predict_out_attr)

                box_predict_dequant = get_valid_node_name(
                    graph, box_predict + '_dequant_box_predict')
                box_predict_dequant_in_attr = copy.deepcopy(
                    box_predict_out_attr)
                box_predict_dequant_in_attr.update({'dst_in_port': 0})
                graph.add_edge(box_predict, box_predict_dequant,
                               **box_predict_dequant_in_attr)

                NodeWrap(graph, class_predict_dequant).replace_obj('ArmDeQuantize', {'name': class_predict_dequant,
                                                                                     'from_dtype': class_predict_tensor.dtype,
                                                                                     'scale': class_predict_scale,
                                                                                     'zero_point': class_predict_zp})
                NodeWrap(graph, box_predict_dequant).replace_obj('ArmDeQuantize', {'name': box_predict_dequant,
                                                                                   'from_dtype': box_predict_tensor.dtype,
                                                                                   'scale': box_predict_scale,
                                                                                   'zero_point': box_predict_zp})

                class_predict_float_tensor = ((class_predict_tensor.value.astype(
                    np.int32) - class_predict_zp) * class_predict_scale).astype(np.float32)
                box_predict_float_tensor = ((box_predict_tensor.value.astype(
                    np.int32) - box_predict_zp) * box_predict_scale).astype(np.float32)
                class_predict_out_attr = {'src_out_port': 0,
                                          'dst_in_port': 0,
                                          'tensor': Tensor(value=class_predict_float_tensor)}

                box_predict_out_attr = {'src_out_port': 0,
                                        'dst_in_port': 1,
                                        'tensor': Tensor(value=box_predict_float_tensor)}
                class_predict = class_predict_dequant
                box_predict = box_predict_dequant

            graph.add_edge(class_predict, decode_box, **class_predict_out_attr)
            graph.add_edge(box_predict, decode_box, **box_predict_out_attr)

            feature_map = []
            if params.get('feature_map', []):
                feature_map = ArmDecodeBoxOp.convert_to_nested_list(extend_lists(
                    list_string_to_list(params['feature_map'])))
            else:
                WARN('[Parser]: feature_map is required by SSD post process but not provided in config file! ' +
                     'Will try to infer feature_map but it could be incorrect!')
                # TODO: Some models may rely on it now. Remove it in the future.
                concat_before_sigmoid = class_predict_in_edges[0][0]
                for u, _ in graph.sorted_in_edges(concat_before_sigmoid):
                    u_obj = NodeWrap(graph, u)['object']
                    u_input_shapes = u_obj.get_input_shapes()
                    feature_map.append(u_input_shapes[0][1:3])

            image_width = int(params.get('image_width', 300))
            image_height = int(params.get('image_height', 300))
            if graph._attr['framework'].name == 'CAFFE':
                anchors = graph._attr.get('anchors', None)
            elif graph._attr['framework'].name in ['ONNX', 'TFLITE', 'TENSORFLOW']:
                if graph._attr.get('anchors') is not None:
                    anchors = graph._attr['anchors']
                elif params.get('detection_postprocess', '').upper() == 'SSD_RESNET':
                    anchors = ArmDecodeBoxOp.generate_anchors_for_resnet(
                        [image_width, image_height], feature_map)
                else:
                    anchors = ArmDecodeBoxOp.generate_anchors(feature_map)

            if anchors is not None:
                if is_xy_box_format and anchors.shape[-1] == 4:
                    DEBUG('Change anchor format from xywh/xyxy to yxhw/yxyx!')
                    split0, split1, split2, split3 = np.split(anchors, 4, axis=-1)
                    anchors = np.concatenate([split1, split0, split3, split2], axis=-1)
                anchor_tensor_format = params.get(
                    'anchor_tensor_format', 'center').upper()
                supported_anchor_format = ['CENTER', 'CORNER']
                if anchor_tensor_format not in supported_anchor_format:
                    ERROR('[Parser]: Meet invalid value of anchor_tensor_format! Supported values are %s!' %
                          str(supported_anchor_format)[1:-1])
                elif anchor_tensor_format == 'CORNER':
                    DEBUG('Change anchor format from yxyx to yxhw!')
                    anchors = OpHasAnchors.convert_to_center_coordinate(
                        anchors)

            max_box_num = max(
                int(params.get('max_box_num', 5000)), vaild_box_num)
            decodebox_attr = {'name': decode_box,
                              'feature_map': feature_map,
                              'image_width': image_width,
                              'image_height': image_height,
                              'max_box_num': max_box_num,
                              'class_num': int(params.get('class_num', class_num)),
                              'score_threshold': float(params.get('score_threshold', 0.5)),
                              'proposal_normalized': params.get('proposal_normalized', True),
                              }
            if params.get('variance', ''):
                decodebox_attr.update(
                    {'variance': float_string_to_list(params['variance'])})
            if params.get('firstbox_scale', ''):
                decodebox_attr.update(
                    {'firstbox_scale': float_string_to_list(params['firstbox_scale'])})
            if anchors is not None:
                decodebox_attr.update({'anchors': anchors})
            NodeWrap(graph, decode_box).replace_obj(
                'ArmDecodeBox', decodebox_attr)
            decodebox_out = get_valid_node_name(graph, decode_box + '_out')
            graph.add_edge(decode_box, decodebox_out, **
                           {'src_out_port': 4, 'dst_in_port': 0})
            NodeWrap(graph, decodebox_out).replace_obj(
                'Out', {'name': decodebox_out})

            nms = get_valid_node_name(graph, params['model_name'] + '_nms')
            for i in range(4):
                graph.add_edge(decode_box, nms, **
                               {'src_out_port': i, 'dst_in_port': i})

            nms_attr = {'name': nms,
                        'image_width': image_width,
                        'image_height': image_height,
                        'max_box_num': max_box_num,
                        'iou_threshold': float(params.get('iou_threshold', 0.6)),
                        'center_point_box': 0
                        }
            NodeWrap(graph, nms).replace_obj('ArmNMS', nms_attr)
            for i in range(4):
                nms_out = get_valid_node_name(graph, nms + '_out_' + str(i))
                graph.add_edge(nms, nms_out, **
                               {'src_out_port': i, 'dst_in_port': 0})
                nms_out_attr = NodeWrap(graph, nms)['object'].copied_attr()
                nms_out_attr.update({'name': nms_out})
                NodeWrap(graph, nms_out).replace_obj('Out', nms_out_attr)

            graph._attr['output_names'].clear()
            graph._attr['output_nodes'].clear()
            graph._attr['output_names'] = [decode_box, nms]
        else:
            ERROR('[Parser]: Invalid outputs number (%d) before post process in detection_post_process!' %
                  len(graph._attr['output_names']))

    elif params.get('detection_postprocess', '').upper() == 'YOLO2':
        if len(graph._attr['output_names']) == 1:
            anchors_num = 5
            net_out = graph._attr['output_names'][0]
            net_out_obj = NodeWrap(graph, net_out)['object']
            net_out_shapes = net_out_obj.get_output_shapes()
            net_out_edges = graph.sorted_out_edges(net_out, data=True)
            if net_out_shapes \
                    and net_out_shapes[0] is not None \
                    and (net_out_shapes[0][1:3] in ([13, 13], [12, 12]) or net_out_shapes[0][2:4] in ([13, 13], [12, 12]))\
                    and (int(np.prod(net_out_shapes[0][3:])) % anchors_num == 0 or net_out_shapes[0][1] % anchors_num == 0) \
                    and len(net_out_edges) == 1:

                if net_out_shapes[0][2:4] in ([13, 13], [12, 12]):
                    net_out_shapes = [np.take(np.array(s, np.int64), [
                                              0, 2, 3, 1]).tolist() for s in net_out_shapes]
                    _, dst, out_attr = net_out_edges[0]
                    transpose = get_valid_node_name(
                        graph, net_out + '_transpose_to_nhwc')
                    graph.remove_edge(net_out, dst)
                    graph.add_edge(net_out, transpose, **out_attr)
                    graph.add_edge(transpose, dst)
                    NodeWrap(graph, transpose).replace_obj(
                        'ArmTranspose', {'name': transpose, 'perm': [0, 2, 3, 1]})
                    if net_out in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(net_out)
                        graph._attr['output_names'][index] = transpose
                    net_out = transpose
                    net_out_edges = graph.sorted_out_edges(
                        transpose, data=True)

                obj_num = int(np.prod(net_out_shapes[0][3:])) // anchors_num
                class_num = obj_num - 5

                if len(net_out_shapes[0]) == 4:
                    dim = net_out_shapes[0][0:3] + [anchors_num, obj_num]
                    out = net_out_edges[0][1]
                    out_in_edges = graph.sorted_in_edges(out, data=True)
                    net_out = insert_reshape(
                        graph, net_out, out, out_in_edges[0][2], dim, type='ArmReshape')

                net_out_edges = graph.sorted_out_edges(net_out, data=True)
                region = get_valid_node_name(
                    graph, params['model_name'] + '_region')
                for _, dst, out_attr in net_out_edges:
                    graph.remove_edge(net_out, dst)
                    graph.add_edge(net_out, region, **out_attr)
                region_out = get_valid_node_name(graph, region + '_out')
                graph.add_edge(region, region_out, **
                               {'src_out_port': 3, 'dst_in_port': 0})

                grid_height, grid_width, box_per_grid = net_out_shapes[0][1:3] + [
                    anchors_num]
                region_attr = {'name': region,
                               'grid_width': grid_width,
                               'grid_height': grid_height,
                               'box_per_grid': box_per_grid,
                               'max_box_num': int(params.get('max_box_num', 5000)),
                               'class_num': class_num,
                               'obj_threshold': float(params.get('obj_threshold', 0.3)),
                               'grid_compensate': True if params.get('grid_compensate', 'true').lower() == 'true' else False
                               }
                if params.get('anchors', ''):
                    region_attr.update(
                        {'anchors': float_string_to_list(params['anchors'])})
                NodeWrap(graph, region).replace_obj('ArmRegion', region_attr)
                NodeWrap(graph, region_out).replace_obj(
                    'Out', {'name': region_out})

                nms = get_valid_node_name(graph, params['model_name'] + '_nms')
                graph.add_edge(
                    region, nms, **{'src_out_port': 0, 'dst_in_port': 3})
                graph.add_edge(
                    region, nms, **{'src_out_port': 1, 'dst_in_port': 0})
                graph.add_edge(
                    region, nms, **{'src_out_port': 2, 'dst_in_port': 1})
                graph.add_edge(
                    region, nms, **{'src_out_port': 4, 'dst_in_port': 2})

                nms_attr = {'name': nms,
                            'image_width': int(params.get('image_width', 416)),
                            'image_height': int(params.get('image_width', 416)),
                            'max_box_num': int(params.get('max_box_num', 5000)),
                            'iou_threshold': float(params.get('iou_threshold', 0.5)),
                            'center_point_box': 0
                            }
                NodeWrap(graph, nms).replace_obj('ArmNMS', nms_attr)
                for i in range(4):
                    nms_out = get_valid_node_name(
                        graph, nms + '_out_' + str(i))
                    graph.add_edge(nms, nms_out, **
                                   {'src_out_port': i, 'dst_in_port': 0})
                    nms_out_attr = NodeWrap(graph, nms)['object'].copied_attr()
                    nms_out_attr.update({'name': nms_out})
                    NodeWrap(graph, nms_out).replace_obj('Out', nms_out_attr)
                graph._attr['output_names'].clear()
                graph._attr['output_names'] = [region, nms]
            else:
                ERROR(
                    '[Parser]: Yolo2 preprocess output shape error in detection_post_process!')
        else:
            ERROR('[Parser]: Invalid outputs number (%d) before post process in detection_post_process!' %
                  len(graph._attr['output_names']))

    elif params.get('detection_postprocess', '').upper() in ('YOLO3_TINY', 'YOLO3_FULL'):
        mode = params.get('detection_postprocess', '').upper()
        if (len(graph._attr['output_names']) == 2 and mode == 'YOLO3_TINY') \
                or (len(graph._attr['output_names']) == 3 and mode == 'YOLO3_FULL'):
            outputs = [NodeWrap(graph, out_name)['object']
                       for out_name in graph._attr['output_names']]
            out_shapes = [out.get_output_shapes() for out in outputs]
            if len(out_shapes) in (2, 3) \
                    and all([len(out_s) == 1 for out_s in out_shapes]) \
                    and (len(out_shapes[0][0]) == 4 and out_shapes[0][0][1:] in ([13, 13, 255], [19, 19, 255], [255, 13, 13], [255, 19, 19])) \
                    and (len(out_shapes[1][0]) == 4 and out_shapes[1][0][1:] in ([26, 26, 255], [38, 38, 255], [255, 26, 26], [255, 38, 38])) \
                    and (len(out_shapes) == 2
                         or (len(out_shapes) == 3 and len(out_shapes[2][0]) == 4 and out_shapes[2][0][1:] in ([52, 52, 255], [76, 76, 255], [255, 52, 52], [255, 76, 76]))):

                if (out_shapes[0][0][1:] == [255, 13, 13] and out_shapes[1][0][1:] == [255, 26, 26] and (len(out_shapes) == 2 or out_shapes[2][0][1:] == [255, 52, 52])) \
                        or (out_shapes[0][0][1:] == [255, 19, 19] and out_shapes[1][0][1:] == [255, 38, 38] and (len(out_shapes) == 2 or out_shapes[2][0][1:] == [255, 76, 76])):
                    out_shapes = [
                        [np.take(np.array(s[0], np.int64), [0, 2, 3, 1]).tolist()] for s in out_shapes]
                    for out in outputs:
                        out_edges = graph.sorted_out_edges(out.name, data=True)
                        _, dst, out_attr = out_edges[0]
                        transpose = get_valid_node_name(
                            graph, out.name + '_transpose_to_nhwc')
                        graph.remove_edge(out.name, dst)
                        graph.add_edge(out.name, transpose, **out_attr)
                        graph.add_edge(transpose, dst)
                        NodeWrap(graph, transpose).replace_obj(
                            'ArmTranspose', {'name': transpose, 'perm': [0, 2, 3, 1]})
                        if out.name in graph._attr['output_names']:
                            index = graph._attr['output_names'].index(out.name)
                            graph._attr['output_names'][index] = transpose

                meta_ret = True
                anchors_num = 3
                anchors = params.get('anchors', None)
                if anchors is not None:
                    try:
                        anchors = float_string_to_list(params['anchors'])
                        anchors = np.reshape(anchors, [len(out_shapes), -1]).tolist()
                    except:
                        anchors = None
                        WARN('[Parser]: Meets invalid anchors of detection postprocess(%s)! Default anchors will be used!' % mode)
                if anchors is None:
                    if mode == 'YOLO3_TINY':
                        anchors = [[2.53125, 2.5625, 4.21875, 5.28125, 11.65625, 9.96875], [
                            0.625, 0.875, 1.4375, 1.6875, 2.3125, 3.625]]
                    else:
                        anchors = (np.array([[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [
                            10, 13, 16, 30, 33, 23]]) / np.array([[32.0], [16.0], [8.0]])).tolist()

                region_fuse = get_valid_node_name(
                    graph, params['model_name'] + '_region_fuse')
                region_list = []
                for n in range(len(out_shapes)):
                    src = graph._attr['output_names'][n]
                    obj_num = out_shapes[n][0][3] // anchors_num
                    out_edges = graph.sorted_out_edges(src, data=True)
                    if len(out_edges) == 1:
                        dim = out_shapes[n][0][0:3] + [anchors_num, obj_num]
                        dst = out_edges[0][1]
                        out_attr = copy.deepcopy(out_edges[0][2])
                        out_attr.update({'dst_in_port': 0})
                        reshape = insert_reshape(
                            graph, src, dst, out_attr, dim, type='ArmReshape')
                        if reshape:
                            reshape_out_edges = graph.sorted_out_edges(
                                reshape, data=True)
                            region = get_valid_node_name(
                                graph, params['model_name'] + '_region_' + str(n + 1))
                            region_list.append(region)
                            for _, dst, out_attr in reshape_out_edges:
                                graph.remove_edge(reshape, dst)
                                graph.add_edge(reshape, region, **out_attr)

                            if n < 2:
                                for ro in range(5):
                                    graph.add_edge(
                                        region, region_fuse, **{'src_out_port': ro, 'dst_in_port': 2 * ro + n})

                            grid_height, grid_width, box_per_grid = out_shapes[n][0][1:3] + [
                                anchors_num]
                            region_attr = {'name': region,
                                           'grid_width': grid_width,
                                           'grid_height': grid_height,
                                           'box_per_grid': box_per_grid,
                                           'max_box_num': int(params.get('max_box_num', 5000)),
                                           'class_num': obj_num - 5,
                                           'obj_threshold': float(params.get('obj_threshold', 0.5)),
                                           'grid_compensate': 1 if params.get('grid_compensate', 'true').lower() == 'true' else 0
                                           }
                            region_attr.update({'anchors': anchors[n]})
                            NodeWrap(graph, region).replace_obj(
                                'ArmRegion', region_attr)
                        else:
                            ERROR(
                                '[Parser]: invalid Reshape was inserted for Yolo3-tiny in detection_post_process!')
                            meta_ret = False
                    else:
                        ERROR('[Parser]: Invalid out edges for output name: %s in detection_post_process' %
                              graph._attr['output_names'][n])
                        meta_ret = False

                if meta_ret and len(region_list) in (2, 3):
                    region_fuse_attr = NodeWrap(graph, region_list[0])[
                        'object'].copied_attr()
                    region_fuse_attr.update(
                        {'name': region_fuse, 'class_num': int(params.get('class_num', 80))})
                    NodeWrap(graph, region_fuse).replace_obj(
                        'ArmRegionFuse', region_fuse_attr)

                    if len(region_list) == 2:
                        region_fuse_out = get_valid_node_name(
                            graph, region_fuse + '_out')
                        graph.add_edge(region_fuse, region_fuse_out,
                                       **{'src_out_port': 3, 'dst_in_port': 0})
                        NodeWrap(graph, region_fuse_out).replace_obj(
                            'Out', {'name': region_fuse_out})
                        final_region_fuse = region_fuse
                    else:
                        region_fuse2 = get_valid_node_name(
                            graph, params['model_name'] + '_region_fuse2')
                        for ro in range(5):
                            graph.add_edge(
                                region_fuse, region_fuse2, **{'src_out_port': ro, 'dst_in_port': 2 * ro})
                            graph.add_edge(
                                region_list[2], region_fuse2, **{'src_out_port': ro, 'dst_in_port': 2 * ro + 1})
                        region_fuse2_attr = NodeWrap(
                            graph, region_list[-1])['object'].copied_attr()
                        region_fuse2_attr.update(
                            {'name': region_fuse2, 'class_num': int(params.get('class_num', 80))})
                        NodeWrap(graph, region_fuse2).replace_obj(
                            'ArmRegionFuse', region_fuse2_attr)
                        region_fuse2_out = get_valid_node_name(
                            graph, region_fuse2 + '_out')
                        graph.add_edge(region_fuse2, region_fuse2_out,
                                       **{'src_out_port': 3, 'dst_in_port': 0})
                        NodeWrap(graph, region_fuse2_out).replace_obj(
                            'Out', {'name': region_fuse2_out})
                        final_region_fuse = region_fuse2

                    nms = get_valid_node_name(
                        graph, params['model_name'] + '_nms')
                    graph.add_node(nms)
                    nms_attr = {'name': nms,
                                'image_width': int(params.get('image_width', 416)),
                                'image_height': int(params.get('image_width', 416)),
                                'max_box_num': int(params.get('max_box_num', 10000)),
                                'iou_threshold': float(params.get('iou_threshold', 0.5)),
                                'center_point_box': 0
                                }
                    NodeWrap(graph, nms).replace_obj('ArmNMS', nms_attr)
                    for ni in range(4):
                        nms_out = get_valid_node_name(
                            graph, nms + '_out_' + str(ni))
                        graph.add_edge(nms, nms_out, **
                                       {'src_out_port': ni, 'dst_in_port': 0})
                        nms_out_attr = NodeWrap(graph, nms)[
                            'object'].copied_attr()
                        nms_out_attr.update({'name': nms_out})
                        NodeWrap(graph, nms_out).replace_obj(
                            'Out', nms_out_attr)

                    graph.add_edge(final_region_fuse, nms, **
                                   {'src_out_port': 0, 'dst_in_port': 3})
                    graph.add_edge(final_region_fuse, nms, **
                                   {'src_out_port': 1, 'dst_in_port': 0})
                    graph.add_edge(final_region_fuse, nms, **
                                   {'src_out_port': 2, 'dst_in_port': 1})
                    graph.add_edge(final_region_fuse, nms, **
                                   {'src_out_port': 4, 'dst_in_port': 2})

                    graph._attr['output_names'].clear()
                    graph._attr['output_names'] = [final_region_fuse, nms]
                else:
                    ERROR(
                        '[Parser]: %s post-process cannot proceed in detection_post_process!' % graph.name)
            else:
                ERROR(
                    '[Parser]: %s post-process output shape error in detection_post_process!' % graph.name)
        else:
            ERROR('[Parser]: Invalid outputs number (%d) before post-process in detection_post_process!' %
                  len(graph._attr['output_names']))

    elif params.get('detection_postprocess', '').upper() == 'CAFFE_FASTERRCNN':
        if len(graph._attr['output_names']) == 2:
            class_pred, box_pred = graph._attr['output_names']
            class_pred_obj, box_pred_obj = [NodeWrap(graph, name)['object'] for name in [
                class_pred, box_pred]]
            roipool_matches = single_node_matcher(graph, 'ArmMaxRoiPool')
            if class_pred_obj is not None and box_pred is not None and len(roipool_matches) == 1:
                roi_pool = roipool_matches[0]['target']
                roi_pool_in_edges = graph.sorted_in_edges(roi_pool, data=True)
                if len(roi_pool_in_edges) == 2 and NodeWrap(graph, roi_pool_in_edges[1][0])['object'] is not None:
                    roi = roi_pool_in_edges[1][0]
                    roi_obj = NodeWrap(graph, roi)['object']
                    class_shape = class_pred_obj.get_output_shapes()[0]
                    box_shape = box_pred_obj.get_output_shapes()[0]
                    roi_shape = roi_obj.get_output_shapes()[0]

                    class_reshape_dim = [
                        1, *[s for s in class_shape if s != 1]]
                    box_num = class_reshape_dim[1]
                    class_num = int(np.prod(box_shape)) // box_num // 4
                    box_reshape_dim = [1, box_num, class_num, 4]
                    roi_reshape_dim = [1] + roi_shape

                    class_predict = insert_reshape_after(
                        graph, class_pred, class_reshape_dim, type='ArmReshape')
                    box_predict = insert_reshape_after(
                        graph, box_pred, box_reshape_dim, type='ArmReshape')

                    class_predict_out_edges = graph.sorted_out_edges(
                        class_predict, data=True)
                    box_predict_out_edges = graph.sorted_out_edges(
                        box_predict, data=True)
                    roi_out_edges = graph.sorted_out_edges(roi, data=True)
                    _, _, roi_out_attr = roi_out_edges[0]
                    graph.remove_edges_from(
                        class_predict_out_edges + box_predict_out_edges)

                    detection = get_valid_node_name(
                        graph, params['model_name'] + '_detection')
                    graph.add_edge(class_predict, detection, **{'src_out_port': 0,
                                                                'dst_in_port': 0,
                                                                'tensor': copy.deepcopy(class_predict_out_edges[0][2]['tensor'])})
                    graph.add_edge(box_predict, detection, **{'src_out_port': 0,
                                                              'dst_in_port': 1,
                                                              'tensor': copy.deepcopy(box_predict_out_edges[0][2]['tensor'])})
                    graph.add_edge(roi, detection, **
                                   {'src_out_port': 0, 'dst_in_port': 2})

                    new_roi_out_attr = copy.deepcopy(roi_out_attr)
                    new_roi_out_attr.update({'dst_in_port': 2})
                    roi_reshape = insert_reshape(
                        graph, roi, detection, new_roi_out_attr, roi_reshape_dim, type='ArmReshape')
                    _, _, roi_reshape_out_attr = graph.sorted_out_edges(
                        roi_reshape, data=True)[0]
                    begin = [0, 0, 1]
                    size = roi_reshape_dim[:-1] + [roi_reshape_dim[-1] - 1]
                    insert_slice(graph, roi_reshape, detection,
                                 roi_reshape_out_attr, begin, size, type='ArmSlice')

                    detection_in_edges = graph.sorted_in_edges(
                        detection, data=True)
                    roi_reshape_slice, _, roi_reshape_slice_out_attr = detection_in_edges[2]
                    indices = np.array([1, 0, 3, 2], np.int32)
                    insert_gather(graph, roi_reshape_slice, detection, indices,
                                  axis=2, edge_attr=roi_reshape_slice_out_attr, type='ArmGather')

                    score_threshold = float(params.get('score_threshold', 0.7))
                    img_width = int(params.get('image_width', 224))
                    img_height = int(params.get('image_height', 224))
                    detection_attr = {'name': detection,
                                      'image_width': img_width,
                                      'image_height': img_height,
                                      'class_num': class_num,
                                      'score_threshold': score_threshold,
                                      'anchor_mode': 'caffe_detection',
                                      'variance': [1.0, 1.0, 1.0, 1.0]}
                    NodeWrap(graph, detection).replace_obj(
                        'ArmDetectionOutput', detection_attr)

                    detection_out = get_valid_node_name(
                        graph, detection + '_out')
                    graph.add_edge(detection, detection_out, **
                                   {'src_out_port': 3, 'dst_in_port': 0})
                    NodeWrap(graph, detection_out).replace_obj(
                        'Out', {'name': detection_out})

                    nms = get_valid_node_name(
                        graph, params['model_name'] + '_nms')
                    graph.add_edge(detection, nms, **
                                   {'src_out_port': 1, 'dst_in_port': 0})
                    graph.add_edge(detection, nms, **
                                   {'src_out_port': 2, 'dst_in_port': 1})
                    graph.add_edge(detection, nms, **
                                   {'src_out_port': 4, 'dst_in_port': 2})
                    graph.add_edge(detection, nms, **
                                   {'src_out_port': 0, 'dst_in_port': 3})

                    nms_attr = {'name': nms,
                                'image_width': int(params.get('image_width', 224)),
                                'image_height': int(params.get('image_width', 224)),
                                'max_box_num': int(params.get('max_box_num', 10000)),
                                'iou_threshold': float(params.get('nms_threshold', 0.3)),
                                'center_point_box': 0
                                }
                    NodeWrap(graph, nms).replace_obj('ArmNMS', nms_attr)
                    for ni in range(4):
                        nms_out = get_valid_node_name(
                            graph, nms + '_out_' + str(ni))
                        graph.add_edge(nms, nms_out, **
                                       {'src_out_port': ni, 'dst_in_port': 0})
                        nms_out_attr = NodeWrap(graph, nms)[
                            'object'].copied_attr()
                        nms_out_attr.update({'name': nms_out})
                        NodeWrap(graph, nms_out).replace_obj(
                            'Out', nms_out_attr)

                    graph._attr['output_names'] = [detection, nms]

                    detection_in_edges = graph.sorted_in_edges(
                        detection, data=True)
                    box_predict, _, box_predict_in_attr = detection_in_edges[1]
                    insert_slice(graph,
                                 box_predict,
                                 detection,
                                 box_predict_in_attr,
                                 [0, 0, 1, 0],
                                 [1, box_num, class_num - 1, 4],
                                 type='ArmSlice')

                else:
                    ERROR(
                        '[Parser]: Invalid detection_postprocess parameters in detection_post_process!')
            else:
                ERROR(
                    '[Parser]: Invalid detection_postprocess parameters in detection_post_process!')
        else:
            ERROR('[Parser]: Invalid outputs number (%d) before post process in detection_post_process!' %
                  len(graph._attr['output_names']))


def remove_redundant_reshape_transpose(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('reshape_1', {'op': 'ArmReshape'}),
                                   ('transpose', {'op': 'ArmTranspose'}),
                                   ('reshape_2', {'op': 'ArmReshape'})],
                               edges=[('reshape_1', 'transpose'),
                                      ('transpose', 'reshape_2')]
                               )
    matched = False
    for m in matches:
        obj_dict = {name: NodeWrap(graph, m[name])['object']
                    for name in ['reshape_1', 'transpose', 'reshape_2']}
        reshape_1_in_edges = graph.sorted_in_edges(m['reshape_1'], data=True)
        if any(obj is None for obj in obj_dict.values()) \
                or len(obj_dict['reshape_1'].get_input_shapes()) < 1 \
                or len(reshape_1_in_edges) < 1:
            ERROR('[Parser]: Meets invalid Node in remove_redundant_reshape_transpose!')
            continue
        reshape_1_out_edges = graph.sorted_out_edges(m['reshape_1'], data=True)
        transpose_out_edges = graph.sorted_out_edges(m['transpose'], data=True)

        if len(reshape_1_out_edges) != 1 or len(transpose_out_edges) != 1:
            continue
        inp, _, in_attr = reshape_1_in_edges[0]
        in_shape = obj_dict['reshape_1'].get_input_shapes()[0]
        if in_shape is None \
                or any([s is None for s in in_shape]) \
                or in_shape != obj_dict['reshape_2'].dim:
            continue

        in_tensor = np.array(np.random.ranf(in_shape)).astype('float32')
        in_reshape1 = np.reshape(in_tensor, obj_dict['reshape_1'].dim)
        in_trans = np.transpose(in_reshape1, obj_dict['transpose'].perm)
        in_reshape2 = np.reshape(in_trans, obj_dict['reshape_2'].dim)
        if (in_tensor != in_reshape2).any():
            continue

        matched = True
        graph.remove_edge(inp, m['reshape_1'])
        for _, dst, out_attr in graph.sorted_out_edges(m['reshape_2'], data=True):
            graph.remove_edge(m['reshape_2'], dst)
            new_out_attr = copy.deepcopy(out_attr)
            new_out_attr.update({'src_out_port': in_attr['src_out_port']})
            graph.add_edge(inp, dst, **new_out_attr)

        if m['reshape_2'] in graph._attr['output_names']:
            if inp not in graph._attr['output_names']:
                index = graph._attr['output_names'].index(m['reshape_2'])
                graph._attr['output_names'][index] = inp
            else:
                graph._attr['output_names'].remove(m['reshape_2'])

    if matched:
        clear_redundant_nodes(graph)


def remove_const(graph):
    removing_const = []
    for node_name in graph.nodes:
        node = NodeWrap(graph, node_name)
        node_obj = node['object']
        if node_obj is not None and node_obj.type in ('Constant', 'Dummy'):
            const_out_edges = graph.sorted_out_edges(node_name, data=True)
            if len(const_out_edges) >= 1 and node_obj.type == 'Constant':
                const_child = const_out_edges[0][1]
                const_child_obj = NodeWrap(graph, const_child)['object']
                if const_child_obj is not None:
                    if isinstance(const_child_obj, (ArmOp, )):
                        in_port = const_out_edges[0][2]['dst_in_port']
                        op_in_ports_num = type(const_child_obj).num_in_ports()
                        if op_in_ports_num < 0 or (op_in_ports_num >= 0 and in_port < op_in_ports_num):
                            const_attr = node_obj.copied_attr()
                            const_attr.update({'weights': node_obj.value})
                            if node_obj.quantize and const_out_edges[0][2]['tensor'].scale_zp:
                                const_attr.update({'weights_scale_zp': list(const_out_edges[0][2]['tensor'].scale_zp)})
                            NodeWrap(graph, node_name).replace_obj(
                                'ArmConstant', const_attr)
                        else:
                            removing_const.append(node_name)
                    elif len(const_out_edges) == 1 and const_child_obj.type == 'Out' \
                            and (node_name not in graph._attr['output_names']
                                 or len(graph._attr['output_names']) > 1):
                        removing_const.append(node_name)
                        if node_name in graph._attr['output_names']:
                            WARN('[Parser]: Remove isolated Constant Node(%s) from output in remove_const!' % node_name)
                            graph._attr['output_names'].remove(node_name)
                        if const_child_obj.name in graph._attr['output_nodes']:
                            graph._attr['output_nodes'].remove(const_child_obj.name)
                    else:
                        const_attr = node_obj.copied_attr()
                        const_attr.update({'weights': node_obj.value})
                        if node_obj.quantize and const_out_edges[0][2]['tensor'].scale_zp:
                            const_attr.update({'weights_scale_zp': list(const_out_edges[0][2]['tensor'].scale_zp)})
                        NodeWrap(graph, node_name).replace_obj(
                            'ArmConstant', const_attr)
                else:
                    ERROR('[Parser]: Meets invalid Constant Node(%s) in remove_const!' %
                          const_child)
            else:
                removing_const.append(node_name)
    graph.remove_nodes_from(removing_const)


def remove_special_where(graph):
    '''Remove special Where whose first input(condition) is const and all the value are same(True or False).
    This pass should be done after pass multidirectional_broadcasting because Where op supports multidirectional
    broadcasting.
    '''
    matched = False
    matches = single_node_matcher(graph, 'Where')
    for m in matches:
        where = m['target']
        where_in_edges = graph.sorted_in_edges(where, data=True)
        if len(where_in_edges) != 3:
            ERROR('[Parser]: Meets invalid inputs of Where Node(%s) in remove_special_where!' % where)
            continue
        cond_in_attr = where_in_edges[0][2]
        if cond_in_attr['tensor'] is None or not cond_in_attr['tensor'].is_const \
                or cond_in_attr['tensor'].value is None:
            continue
        cond_value = cond_in_attr['tensor'].value
        all_true = (np.bool_(cond_value) == True).all()
        all_false = (np.bool_(cond_value) == False).all()
        if not all_true and not all_false:
            continue
        matched = True
        where_out_edges = graph.sorted_out_edges(where, data=True)
        graph.remove_edges_from(where_out_edges)
        src, _, in_attr = where_in_edges[1] if all_true else where_in_edges[2]
        src_out_port = in_attr['src_out_port']
        for _, dst, out_attr in where_out_edges:
            out_attr['src_out_port'] = src_out_port
            graph.add_edge(src, dst, **out_attr)
        if where in graph._attr['output_names']:
            index = graph._attr['output_names'].index(where)
            graph._attr['output_names'][index] = src
    if matched:
        clear_redundant_nodes(graph)


def trim_weights(graph):
    def _data_in_supported_dtype(np_data, attr_name, node_name):
        if not isinstance(np_data, np.ndarray):
            np_data = np.array(np_data)
        data_dtype = str(np_data.dtype)
        if data_dtype in ArmCastOp.attributes()['to_dtype']['options']:
            return np_data

        to_supported_dtype = get_converted_dtype(data_dtype)
        if to_supported_dtype is None:
            ERROR('[Parser]: Meets invalid dtype %s in %s of Node (%s) in trim_weights!' %
                  (data_dtype, attr_name, node_name))
            return np_data

        WARN('[Parser]: Convert unsupported dtype %s to %s for %s of Node (%s)' %
             (data_dtype, to_supported_dtype.__name__, attr_name, node_name))
        return np_data.astype(to_supported_dtype)

    nodes_list = determined_sort(graph, graph._attr['output_names'], sort_input=True)

    offset = 0
    for node_name in nodes_list:
        node = NodeWrap(graph, node_name)
        node_obj = node['object']
        if node_obj is not None:
            out_ports = node_obj.get_out_ports()
            if len(out_ports) > 0 \
                    and len(out_ports) == len(node_obj.top_ranges) \
                    and len(out_ports) == len(node_obj.top_ranges_offset):
                for pi in range(len(out_ports)):
                    if node_obj.top_ranges[pi] is not None:
                        node_obj.top_ranges[pi] = _data_in_supported_dtype(
                            node_obj.top_ranges[pi], 'top_range', node_name)
                        node_obj.top_ranges_offset[pi] = offset
                        offset += node_obj.top_ranges[pi].size * node_obj.top_ranges[pi].dtype.itemsize
            if graph._attr.get('quantize', False):
                if len(out_ports) > 0 \
                        and len(out_ports) == len(node_obj.top_scales) \
                        and len(out_ports) == len(node_obj.top_scales_offset) \
                        and len(out_ports) == len(node_obj.top_zps) \
                        and len(out_ports) == len(node_obj.top_zps_offset):
                    for i, scales_or_zps in enumerate([node_obj.top_scales, node_obj.top_zps]):
                        for pi, _ in enumerate(out_ports):
                            if scales_or_zps[pi] is not None:
                                scales_or_zps[pi] = _data_in_supported_dtype(
                                    scales_or_zps[pi], 'top_scale_or_zp', node_name)
                                if i == 0:
                                    node_obj.top_scales[pi] = scales_or_zps[pi]
                                    node_obj.top_scales_offset[pi] = offset
                                else:
                                    node_obj.top_zps[pi] = scales_or_zps[pi]
                                    node_obj.top_zps_offset[pi] = offset
                                offset += scales_or_zps[pi].size * scales_or_zps[pi].dtype.itemsize

            if isinstance(node_obj, OpHasWeights):
                if node_obj.weights is not None:
                    node_obj.weights = _data_in_supported_dtype(
                        node_obj.weights, 'weights', node_name)
                    node_obj.weights_offset = offset
                    offset += node_obj.weights.size * node_obj.weights.dtype.itemsize
                    if node_obj.weights_range is not None and len(node_obj.weights_range) == 2:
                        weights_range_value = _data_in_supported_dtype(
                            node_obj.weights_range, 'weights_range', node_name)
                        if weights_range_value.size == 2:
                            node_obj.weights_range = None
                            node_obj.weights_range_list = [weights_range_value.item(0), weights_range_value.item(1)]
                        else:
                            node_obj.weights_range = weights_range_value
                            node_obj.weights_range_offset = offset
                            offset += node_obj.weights_range.size * node_obj.weights_range.dtype.itemsize
                    if graph._attr.get('quantize', False) \
                            and np.issubdtype(node_obj.weights.dtype, np.integer) \
                            and len(node_obj.weights_scale_zp) == 2:
                        weights_scales_value = _data_in_supported_dtype(
                            node_obj.weights_scale_zp[0], 'weights_scale', node_name)
                        weights_zps_value = _data_in_supported_dtype(
                            node_obj.weights_scale_zp[1], 'weights_zp', node_name)
                        if weights_scales_value.size == 1 and weights_zps_value.size == 1:
                            node_obj.weights_scale_zp = []
                            node_obj.weights_scale_zp_list = [weights_scales_value.item(), weights_zps_value.item()]
                        else:
                            node_obj.weights_scale_zp[0] = weights_scales_value
                            node_obj.weights_scale_offset = offset
                            offset += node_obj.weights_scale_zp[0].size * node_obj.weights_scale_zp[0].dtype.itemsize
                            node_obj.weights_scale_zp[1] = weights_zps_value
                            node_obj.weights_zp_offset = offset
                            offset += node_obj.weights_scale_zp[1].size * node_obj.weights_scale_zp[1].dtype.itemsize
                else:
                    ERROR('[Parser]: Meets invalid weights for Node %s in trim_weights!' %
                          node_name)
            if isinstance(node_obj, OpHasBiases):
                if node_obj.biases is not None:
                    node_obj.biases = _data_in_supported_dtype(
                        node_obj.biases, 'biases', node_name)
                    node_obj.biases_offset = offset
                    offset += node_obj.biases.size * node_obj.biases.dtype.itemsize
                    if node_obj.biases_range is not None and len(node_obj.biases_range) == 2:
                        biases_range_value = _data_in_supported_dtype(
                            node_obj.biases_range, 'biases_range', node_name)
                        if biases_range_value.size == 2:
                            node_obj.biases_range = None
                            node_obj.biases_range_list = [biases_range_value.item(0), biases_range_value.item(1)]
                        else:
                            node_obj.biases_range = biases_range_value
                            node_obj.biases_range_offset = offset
                            offset += node_obj.biases_range.size * node_obj.biases_range.dtype.itemsize
                    if graph._attr.get('quantize', False) \
                            and np.issubdtype(node_obj.biases.dtype, np.integer) \
                            and len(node_obj.biases_scale_zp) == 2:
                        biases_scales_value = _data_in_supported_dtype(
                            node_obj.biases_scale_zp[0], 'biases_scale', node_name)
                        biases_zps_value = _data_in_supported_dtype(
                            node_obj.biases_scale_zp[1], 'biases_zp', node_name)
                        if biases_scales_value.size == 1 and biases_zps_value.size == 1:
                            node_obj.biases_scale_zp = []
                            node_obj.biases_scale_zp_list = [biases_scales_value.item(), biases_zps_value.item()]
                        else:
                            node_obj.biases_scale_zp[0] = biases_scales_value
                            node_obj.biases_scale_offset = offset
                            offset += node_obj.biases_scale_zp[0].size * node_obj.biases_scale_zp[0].dtype.itemsize
                            node_obj.biases_scale_zp[1] = biases_zps_value
                            node_obj.biases_zp_offset = offset
                            offset += node_obj.biases_scale_zp[1].size * node_obj.biases_scale_zp[1].dtype.itemsize
                else:
                    ERROR('[Parser]: Meets invalid biases for Node %s in trim_weights!' %
                          node_name)
            if isinstance(node_obj, OpHasAnchors):
                if node_obj.anchors is not None:
                    anchors = _data_in_supported_dtype(node_obj.anchors, 'anchors', node_name)
                    # The anchors should be in the format of center coordinate.
                    if len(anchors.shape) != 2 or anchors.shape[1] != 4:
                        ERROR('[Parser]: Meet invalid anchor shape in trim_weights!')
                        continue
                    y_center, x_center, height, width = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
                    node_obj.anchors = None
                    # x center
                    node_obj.xcenter = x_center
                    node_obj.xcenter_offset = offset
                    offset += x_center.size * x_center.dtype.itemsize
                    # y center
                    node_obj.ycenter = y_center
                    node_obj.ycenter_offset = offset
                    offset += y_center.size * y_center.dtype.itemsize
                    # height
                    node_obj.ha = height
                    node_obj.ha_offset = offset
                    offset += height.size * height.dtype.itemsize
                    # width
                    node_obj.wa = width
                    node_obj.wa_offset = offset
                    offset += width.size * width.dtype.itemsize
                else:
                    ERROR('[Parser]: Meets invalid anchors for Node %s in trim_weights!' % node_name)
            if isinstance(node_obj, (BaseActivationOp, ArmActivationOp)) \
                    and hasattr(node_obj, 'negative_slope') \
                    and hasattr(node_obj, 'negative_slope_offset') \
                    and node_obj.negative_slope is not None \
                    and np.ndim(node_obj.negative_slope) > 0:
                node_obj.negative_slope = _data_in_supported_dtype(
                    node_obj.negative_slope, 'negative_slope', node_name)
                node_obj.negative_slope_offset = offset
                offset += node_obj.negative_slope.size * node_obj.negative_slope.dtype.itemsize
                if node_obj.negative_slope_range is not None and len(node_obj.negative_slope_range) == 2:
                    negative_slope_range_value = _data_in_supported_dtype(
                        node_obj.negative_slope_range, 'negative_slope_range', node_name)
                    if negative_slope_range_value.size == 2:
                        node_obj.negative_slope_range = None
                        node_obj.negative_slope_range_list = [
                            negative_slope_range_value.item(0), negative_slope_range_value.item(1)]
                    else:
                        node_obj.negative_slope_range = negative_slope_range_value
                        node_obj.negative_slope_range_offset = offset
                        offset += node_obj.negative_slope_range.size * node_obj.negative_slope_range.dtype.itemsize
                if node_obj.quantize \
                        and node_obj.negative_slope_scale is not None \
                        and node_obj.negative_slope_zp is not None:
                    negative_slope_scales_value = _data_in_supported_dtype(
                        node_obj.negative_slope_scale, 'negative_slope_scale', node_name)
                    negative_slope_zps_value = _data_in_supported_dtype(
                        node_obj.negative_slope_zp, 'negative_slope_zp', node_name)
                    if negative_slope_scales_value.size == 1 and negative_slope_zps_value.size == 1:
                        node_obj.negative_slope_scale = None
                        node_obj.negative_slope_zp = None
                        node_obj.negative_slope_scale_zp_list = [
                            negative_slope_scales_value.item(), negative_slope_zps_value.item()]
                    else:
                        node_obj.negative_slope_scale = negative_slope_scales_value
                        node_obj.negative_slope_scale_offset = offset
                        offset += node_obj.negative_slope_scale.size * node_obj.negative_slope_scale.dtype.itemsize
                        node_obj.negative_slope_zp = negative_slope_zps_value
                        node_obj.negative_slope_zp_offset = offset
                        offset += node_obj.negative_slope_zp.size * node_obj.negative_slope_zp.dtype.itemsize
            if isinstance(node_obj, BaseQuantizeDequantizeOp):
                node_obj.scale = _data_in_supported_dtype(node_obj.scale, 'scale', node_name)
                node_obj.scale_offset = offset
                offset += node_obj.scale.size * node_obj.scale.dtype.itemsize
                node_obj.zero_point = _data_in_supported_dtype(node_obj.zero_point, 'zero_point', node_name)
                node_obj.zero_point_offset = offset
                offset += node_obj.zero_point.size * node_obj.zero_point.dtype.itemsize
            if isinstance(node_obj, BaseRnnOp) \
                    and node_obj.activations_scale is not None \
                    and node_obj.activations_zp is not None:
                activations_scales_value = _data_in_supported_dtype(
                    node_obj.activations_scale, 'activations_scale', node_name)
                activations_zps_value = _data_in_supported_dtype(
                    node_obj.activations_zp, 'activations_zp', node_name)
                if activations_scales_value.size == 1 and activations_zps_value.size == 1:
                    node_obj.activations_scale = None
                    node_obj.activations_zp = None
                    node_obj.activations_scale_zp_list = [activations_scales_value.item(), activations_zps_value.item()]
                else:
                    node_obj.activations_scale = activations_scales_value
                    node_obj.activations_scale_offset = offset
                    offset += node_obj.activations_scale.size * node_obj.activations_scale.dtype.itemsize
                    node_obj.activations_zp = activations_zps_value
                    node_obj.activations_zp_offset = offset
                    offset += node_obj.activations_zp.size * node_obj.activations_zp.dtype.itemsize
            if isinstance(node_obj, PluginOp) \
                    and node_obj.constants:
                # Keep the original dtype for constants in Plugin
                for key, np_array in node_obj.constants.items():
                    node_obj.constants_offset_dict.update({key: offset})
                    offset += np_array.size * np_array.dtype.itemsize
        else:
            ERROR(
                '[Parser]: Meets invalid Op object for Node %s in trim_weights!' % node_name)


def insert_preprocess(graph):
    if PARSER_OP_DICT and 'Preprocess' in PARSER_OP_DICT:
        if '.preprocess.Preprocess' in PARSER_OP_DICT \
                and PARSER_OP_DICT['Preprocess'] == PARSER_OP_DICT['.preprocess.Preprocess']:
            # handled in function apply_preprocess_plugin
            return
        ds = determined_sort(graph, graph._attr['output_names'])
        matches = extend_lists([single_node_matcher(graph, op)
                                for op in ('Input', 'ArmInput')])
        input_names = [m['target'] for m in matches]
        if ds and input_names:
            input_names = sorted(input_names, key=lambda x: ds.index(x))
            input_objs = [NodeWrap(graph, n)['object'] for n in input_names]
            if all([o is not None for o in input_objs]):

                preprocess = get_valid_node_name(
                    graph, graph._attr['name'] + '_preprocess')
                graph.add_node(preprocess)
                NodeWrap(graph, preprocess).replace_obj(
                    'Preprocess', {'name': preprocess})

                cur_base_port = 0
                for i, in_obj in enumerate(input_objs):
                    out_edges = graph.sorted_out_edges(in_obj.name, data=True)
                    graph.remove_edges_from(out_edges)
                    input_out_attr = copy.deepcopy(out_edges[0][2])
                    input_out_attr.update({'dst_in_port': i})
                    graph.add_edge(in_obj.name, preprocess, **input_out_attr)

                    for _, dst, out_attr in out_edges:
                        preprocess_out_attr = copy.deepcopy(out_attr)
                        preprocess_out_attr.update(
                            {'src_out_port': cur_base_port})
                        graph.add_edge(preprocess, dst, **preprocess_out_attr)

                    cur_base_port += len(in_obj.get_out_ports())
            else:
                ERROR('[Parser]: Not all Input nodes is valid in insert_preprocess!')
        else:
            ERROR('[Parser]: Invalid parameters for insert_preprocess !')


def insert_cast_if_must(graph):
    happened = False
    if graph:
        ds = determined_sort(graph, graph._attr['output_names'])
        for n in ds:
            obj = NodeWrap(graph, n)['object']
            if obj is not None:
                quantize = obj.quantize
                if isinstance(obj, ArmOp) and len(obj.cast_in_ports()) > 0:
                    in_edges = graph.sorted_in_edges(n, data=True, keys=True)
                    cast_in_ports = {}
                    for key, value in obj.cast_in_ports().items():
                        v = [value] if isinstance(value, str) else value
                        cast_in_ports_i = {
                            (key + len(in_edges)) if (key is not None and key < 0) else key: v}
                        cast_in_ports.update(cast_in_ports_i)
                    for src, _, k, in_attr in in_edges:
                        for v in cast_in_ports.values():
                            if all([vi in ArmCastOp.attributes()['to_dtype']['options'] for vi in v]):
                                in_port = in_attr.get('dst_in_port', 0)
                                if in_port in cast_in_ports or None in cast_in_ports:
                                    cast_key = in_port if in_port in cast_in_ports else None
                                    cast_type_all = cast_in_ports[cast_key]
                                    if in_attr.get('tensor', None) is not None:
                                        dtype = in_attr['tensor'].get_dtype()
                                        if quantize and dtype is not None \
                                                and dtype in ('uint8', 'int8', 'int16', 'int32'):
                                            break
                                        if dtype is not None and all(dtype != cast_type for cast_type in cast_type_all):
                                            happened = True
                                            cast_type = get_converted_dtype(dtype, return_type_string=True)
                                            if cast_type not in cast_type_all:
                                                if graph._attr.get('quantize', False) and obj.quantize:
                                                    cast_type = get_closest_dtype(cast_type, cast_type_all)
                                                else:
                                                    cast_type = cast_type_all[0]
                                            insert_cast(graph, src, n, cast_type,
                                                        in_attr=in_attr, key=k, type='ArmCast')
                                            break
                                        elif dtype is None and len(cast_type_all) == 1:
                                            happened = True
                                            cast_type = cast_type_all[0]
                                            insert_cast(graph, src, n, cast_type,
                                                        in_attr=in_attr, key=k, type='ArmCast')
                                            break

            else:
                ERROR('[Parser]: Meets invalid Node (%s) in insert_cast_if_must!' % n)
    return happened


def sink_single_transpose(graph):
    unaware_types = set(ArmOp.get_concrete_subclass_names()).intersection(
        LayoutUnawareOp.get_concrete_subclass_names() + ['ArmQuantize', 'ArmDeQuantize'])
    unaware_types = sorted(list(unaware_types))
    matches = matched_patterns(graph,
                               nodes=[('transpose', {'op': 'ArmTranspose'}),
                                      ('unaware', {'op': unaware_types})
                                      ],
                               edges=[('transpose', 'unaware')]
                               )
    for m in matches:
        transpose, unaware = m['transpose'], m['unaware']
        transpose_obj = NodeWrap(graph, transpose)['object']
        unaware_obj = NodeWrap(graph, unaware)['object']
        if transpose_obj is not None and unaware_obj is not None:
            trans_out_edges = graph.sorted_out_edges(transpose, data=True)
            if len(trans_out_edges) == 1 and unaware_obj.num_in_ports() == 1 and len(unaware_obj.get_out_ports()) == 1:
                unaware_input_shape = unaware_obj.get_input_shapes()[0]
                if unaware_input_shape is None \
                        or any([s is None for s in unaware_input_shape]):
                    ERROR(
                        '[Parser]: Meets invalid input shape of Node(%s) in sink_single_transpose!' % unaware)
                    continue
                unaware_out_edges = graph.sorted_out_edges(unaware, data=True)
                if len(unaware_out_edges) < 1:
                    continue
                if unaware_obj.type in ['ArmQuantize', 'ArmDeQuantize'] \
                        and unaware_obj.axis is not None:
                    continue
                trans_in_edges = graph.sorted_in_edges(transpose, data=True)
                src, _, trans_in_attr = trans_in_edges[0]
                _, _, trans_out_attr = trans_out_edges[0]
                graph.remove_edge(src, transpose)
                graph.remove_edges_from(trans_out_edges)
                for _, dst, out_attr in unaware_out_edges:
                    graph.remove_edge(unaware, dst)
                    new_out_attr = copy.deepcopy(out_attr)
                    new_out_attr['src_out_port'] = 0
                    graph.add_edge(transpose, dst, **new_out_attr)
                unaware_out_tensor = copy.deepcopy(unaware_out_edges[0][2]['tensor'])
                if unaware_out_tensor is not None:
                    new_perm = [transpose_obj.perm.index(i) for i in range(len(transpose_obj.perm))]
                    if unaware_out_tensor.value is not None:
                        unaware_out_tensor.value = np.transpose(unaware_out_tensor.value, new_perm)
                    else:
                        unware_out_shape = unaware_out_tensor.get_shape()
                        if unware_out_shape is not None and None not in unware_out_shape:
                            unaware_out_tensor.shape = tuple([unware_out_shape[idx] for idx in new_perm])
                new_in_attr = copy.deepcopy(trans_in_attr)
                new_in_attr['dst_in_port'] = trans_out_attr['dst_in_port']
                graph.add_edge(src, unaware, **new_in_attr)
                graph.add_edge(unaware, transpose, **{
                               'src_out_port': 0, 'dst_in_port': 0, 'tensor': unaware_out_tensor})
                if unaware_obj.type == 'ArmActivation' and unaware_obj.method == 'PRELU':
                    if len(unaware_obj.negative_slope.shape) < len(unaware_input_shape):
                        new_slope_shape = [1] * (len(unaware_input_shape) - len(unaware_obj.negative_slope.shape)) + \
                            list(unaware_obj.negative_slope.shape)
                        unaware_obj.negative_slope = np.reshape(
                            unaware_obj.negative_slope, new_slope_shape)
                    slope_perm = Op.cal_inverse_perm(transpose_obj.perm)
                    unaware_obj.negative_slope = np.transpose(
                        unaware_obj.negative_slope, slope_perm)
                if unaware_obj.type == 'ArmQuantize':
                    transpose_obj.quantize = True
                if unaware in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(unaware)
                    graph._attr['output_names'][index] = transpose
        else:
            ERROR('[Parser]: Meets invalid Node(%s) or Node(%s) in sink_single_transpose!' % (
                transpose, unaware))


def sink_double_transpose(graph):
    unaware_types = set(ArmOp.get_concrete_subclass_names()).intersection(
        LayoutUnawareOp.get_concrete_subclass_names() + ['ArmQuantize', 'ArmDeQuantize'])
    unaware_types = sorted(list(unaware_types))
    matches = matched_patterns(graph,
                               nodes=[
                                   ('trans1', {'op': 'ArmTranspose', 'unique': False}),
                                   ('trans2', {'op': 'ArmTranspose', 'unique': False}),
                                   ('unaware', {'op': unaware_types})
                               ],
                               edges=[
                                   ('trans1', 'unaware', {
                                       'src_out_port': 0, 'dst_in_port': 0}),
                                   ('trans2', 'unaware', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                               ]
                               )
    for m in matches:
        trans1, trans2, unaware = m['trans1'], m['trans2'], m['unaware']
        if any(not graph.has_node(name) for name in [trans1, trans2, unaware]):
            DEBUG('[Parser]: Meets invalid name that does not exist in graph in sink_double_transpose!')
            continue
        trans1_obj = NodeWrap(graph, trans1)['object']
        trans2_obj = NodeWrap(graph, trans2)['object']
        unaware_obj = NodeWrap(graph, unaware)['object']
        if trans1_obj is not None and trans2_obj is not None and unaware_obj is not None:
            if trans1_obj.perm == trans2_obj.perm \
                    and unaware_obj.num_in_ports() == 2 \
                    and len(unaware_obj.get_out_ports()) == 1:
                trans1_in_edges = graph.sorted_in_edges(trans1, data=True)
                trans2_in_edges = graph.sorted_in_edges(trans2, data=True)
                if len(trans1_in_edges) < 1 or len(trans2_in_edges) < 1:
                    ERROR('[Parser]: Meets invalid Node(%s) or Node(%s) in sink_double_transposes!' % (
                        trans1, trans2))
                    continue
                if unaware_obj.type in ['ArmQuantize', 'ArmDeQuantize'] \
                        and unaware_obj.axis is not None:
                    continue
                src1, _, in_attr1 = trans1_in_edges[0]
                src2, _, in_attr2 = trans2_in_edges[0]

                new_in_attr1 = copy.deepcopy(in_attr1)
                new_in_attr2 = copy.deepcopy(in_attr2)
                new_in_attr2['dst_in_port'] = 1
                graph.remove_edge(trans1, unaware)
                graph.remove_edge(trans2, unaware)
                graph.add_edge(src1, unaware, **new_in_attr1)
                graph.add_edge(src2, unaware, **new_in_attr2)

                trans1_out_edges = graph.sorted_out_edges(trans1)
                trans2_out_edges = graph.sorted_out_edges(trans2)
                if len(trans1_out_edges) == 0:
                    trans1_in_edges = graph.sorted_in_edges(trans1)
                    graph.remove_edges_from(trans1_in_edges)
                if len(trans2_out_edges) == 0:
                    trans2_in_edges = graph.sorted_in_edges(trans2)
                    graph.remove_edges_from(trans2_in_edges)

                post_trans = insert_transpose_after(graph, unaware, trans1_obj.perm,
                                                    port=unaware_obj.get_out_ports()[0],
                                                    type='ArmTranspose', quantize=unaware_obj.quantize)

                if unaware in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(unaware)
                    graph._attr['output_names'][index] = post_trans

                clear_redundant_nodes(graph)

        else:
            ERROR('[Parser]: Meets invalid Node(%s) or Node(%s) or Node(%s) in sink_double_transposes!' % (
                trans1, trans2, unaware))


def sink_double_reshape(graph):
    unaware_types = set(ArmOp.get_concrete_subclass_names()).intersection(
        LayoutUnawareOp.get_concrete_subclass_names() + ['ArmQuantize', 'ArmDeQuantize'])
    unaware_types = sorted(list(unaware_types))
    matches = matched_patterns(graph,
                               nodes=[
                                   ('reshape1', {'op': 'ArmReshape', 'unique': False}),
                                   ('reshape2', {'op': 'ArmReshape', 'unique': False}),
                                   ('unaware', {'op': unaware_types})
                               ],
                               edges=[
                                   ('reshape1', 'unaware', {
                                       'src_out_port': 0, 'dst_in_port': 0}),
                                   ('reshape2', 'unaware', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                               ]
                               )
    for m in matches:
        reshape1, reshape2, unaware = m['reshape1'], m['reshape2'], m['unaware']
        if any(not graph.has_node(name) for name in [reshape1, reshape2, unaware]):
            DEBUG('[Parser]: Meets invalid name that does not exist in graph in sink_double_reshape!')
            continue
        reshape1_obj = NodeWrap(graph, reshape1)['object']
        reshape2_obj = NodeWrap(graph, reshape2)['object']
        unaware_obj = NodeWrap(graph, unaware)['object']
        if reshape1_obj is not None and reshape2_obj is not None and unaware_obj is not None:
            if reshape1_obj.dim == reshape2_obj.dim \
                    and unaware_obj.num_in_ports() == 2 \
                    and len(unaware_obj.get_out_ports()) == 1:
                reshape1_in_edges = graph.sorted_in_edges(reshape1, data=True)
                reshape2_in_edges = graph.sorted_in_edges(reshape2, data=True)
                if len(reshape1_in_edges) < 1 or len(reshape2_in_edges) < 1:
                    ERROR('[Parser]: Meets invalid Node(%s) or Node(%s) in sink_double_reshape!' % (
                        reshape1, reshape2))
                    continue
                if unaware_obj.type in ['ArmQuantize', 'ArmDeQuantize'] \
                        and unaware_obj.axis is not None:
                    continue
                src1, _, in_attr1 = reshape1_in_edges[0]
                src2, _, in_attr2 = reshape2_in_edges[0]

                if in_attr1['tensor'].shape is None or \
                        in_attr2['tensor'].shape is None or \
                        in_attr1['tensor'].shape != in_attr2['tensor'].shape:
                    continue

                new_in_attr1 = copy.deepcopy(in_attr1)
                new_in_attr2 = copy.deepcopy(in_attr2)
                new_in_attr2['dst_in_port'] = 1
                graph.remove_edge(reshape1, unaware)
                graph.remove_edge(reshape2, unaware)
                graph.add_edge(src1, unaware, **new_in_attr1)
                graph.add_edge(src2, unaware, **new_in_attr2)

                reshape1_out_edges = graph.sorted_out_edges(reshape1)
                reshape2_out_edges = graph.sorted_out_edges(reshape2)
                if len(reshape1_out_edges) == 0:
                    reshape1_in_edges = graph.sorted_in_edges(reshape1)
                    graph.remove_edges_from(reshape1_in_edges)
                if len(reshape2_out_edges) == 0:
                    reshape2_in_edges = graph.sorted_in_edges(reshape2)
                    graph.remove_edges_from(reshape2_in_edges)

                post_reshape = insert_reshape_after(graph, unaware, reshape1_obj.dim,
                                                    old_dim=in_attr1['tensor'].shape,
                                                    type='ArmReshape',
                                                    quantize=unaware_obj.quantize)

                if unaware in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(unaware)
                    graph._attr['output_names'][index] = post_reshape

                clear_redundant_nodes(graph)

        else:
            ERROR('[Parser]: Meets invalid Node(%s) or Node(%s) or Node(%s) in sink_double_reshape!' % (
                reshape1, reshape2, unaware))


def sink_transpose_through_argminmax(graph):
    matches = two_nodes_matcher(graph, 'ArmTranspose', 'ArmArgMinMax')
    for m in matches:
        transpose, argminmax = m['begin'], m['end']
        transpose_obj = NodeWrap(graph, transpose)['object']
        argminmax_obj = NodeWrap(graph, argminmax)['object']
        if transpose_obj is None or argminmax_obj is None:
            ERROR('[Parser]: Meets invalid Transpose (%s) or ArgMinMax (%s) in sink_transpose_through_argminmax!'
                  % (transpose, argminmax))
            continue
        transpose_in_edges = graph.sorted_in_edges(transpose, data=True)
        if len(transpose_in_edges) != 1:
            continue
        src, _, in_attr = transpose_in_edges[0]
        graph.remove_edge(transpose, argminmax)
        graph.add_edge(src, argminmax, **in_attr)
        insert_transpose_after(graph, argminmax, transpose_obj.perm, type='ArmTranspose')
        argminmax_obj.axis = Op.cal_inverse_perm(transpose_obj.perm).index(argminmax_obj.axis)


def sink_transpose_through_norm(graph):
    matches = two_nodes_matcher(graph, 'ArmTranspose', 'ArmNormalization')
    for m in matches:
        transpose, norm = m['begin'], m['end']
        transpose_obj = NodeWrap(graph, transpose)['object']
        norm_obj = NodeWrap(graph, norm)['object']
        if transpose_obj is None or norm_obj is None:
            ERROR('[Parser]: Meets invalid Transpose (%s) or Normalization (%s) in sink_transpose_through_norm!'
                  % (transpose, norm))
            continue
        transpose_in_edges = graph.sorted_in_edges(transpose, data=True)
        if len(transpose_in_edges) != 1:
            continue
        if len(norm_obj.axes) != 1:
            continue
        src, _, in_attr = transpose_in_edges[0]
        graph.remove_edge(transpose, norm)
        graph.add_edge(src, norm, **in_attr)
        insert_transpose_after(graph, norm, transpose_obj.perm, type='ArmTranspose')
        norm_obj.axes = [Op.cal_inverse_perm(transpose_obj.perm).index(norm_obj.axes[0])]


def sink_reshape_through_cast(graph):
    matches = two_nodes_matcher(graph, 'ArmReshape', 'ArmCast')
    for m in matches:
        reshape, cast = m['begin'], m['end']
        cast_obj = NodeWrap(graph, cast)['object']
        reshape_in_edges = graph.sorted_in_edges(reshape, data=True)
        cast_out_edges = graph.sorted_out_edges(cast, data=True)
        if cast_obj is None or len(reshape_in_edges) < 1 or len(cast_out_edges) < 1:
            ERROR('[Parser]: Meets invalid Node object in sink_reshape_through_cast!')
            continue
        reshape_out_edges = graph.sorted_out_edges(reshape)
        if len(reshape_out_edges) != 1:
            continue
        graph.remove_edges_from(reshape_in_edges + cast_out_edges)
        graph.remove_edge(reshape, cast)
        src, _, in_attr = reshape_in_edges[0]
        graph.add_edge(src, cast, **in_attr)
        reshape_in_attr = copy.deepcopy(in_attr)
        reshape_in_attr.update({'src_out_port': 0})
        if in_attr['tensor'] is not None:
            if in_attr['tensor'].value is not None:
                reshape_in_attr['tensor'].value = in_attr['tensor'].value.astype(np.dtype(cast_obj.to_dtype))
            cast_out_tensor = cast_out_edges[0][2]['tensor']
            if cast_out_tensor is not None and len(cast_out_tensor.scale_zp) > 0:
                reshape_in_attr['tensor'].scale_zp = cast_out_tensor.scale_zp
                reshape_in_attr['tensor'].dtype = cast_out_tensor.dtype
                reshape_in_attr['tensor'].activation_quantization_axis = cast_out_tensor.activation_quantization_axis
        graph.add_edge(cast, reshape, **reshape_in_attr)
        for _, dst, out_attr in cast_out_edges:
            graph.add_edge(reshape, dst, **out_attr)

        if cast in graph._attr['output_names']:
            index = graph._attr['output_names'].index(cast)
            graph._attr['output_names'][index] = reshape


def sink_transpose_with_const(graph):
    unaware_types = set(ArmOp.get_concrete_subclass_names()).intersection(
        LayoutUnawareOp.get_concrete_subclass_names())
    unaware_types = sorted(list(unaware_types))
    matches = [matched_patterns(graph,
                                nodes=[
                                    ('trans', {'op': 'ArmTranspose'}),
                                    ('const', {'op': 'ArmConstant'}),
                                    ('unaware', {'op': uw})
                                ],
                                edges=[
                                    ('trans', 'unaware', {
                                     'src_out_port': 0, 'dst_in_port': tr_in_port}),
                                    ('const', 'unaware', {
                                     'src_out_port': 0, 'dst_in_port': 1 - tr_in_port}),
                                ]
                                ) for uw in unaware_types for tr_in_port in [0, 1]]
    matches = extend_lists(matches)
    matched = False
    for m in matches:
        trans, const, unaware = m['trans'], m['const'], m['unaware']
        trans_obj = NodeWrap(graph, trans)['object']
        const_obj = NodeWrap(graph, const)['object']
        unaware_obj = NodeWrap(graph, unaware)['object']
        if trans_obj is not None and const_obj is not None and unaware_obj is not None:
            const_out_edges = graph.sorted_out_edges(const, data=True)
            unaware_in_edges = graph.sorted_in_edges(unaware)
            unaware_out_edges = graph.sorted_out_edges(unaware, data=True)
            if not has_path(graph, trans, const) \
                    and not has_path(graph, const, trans) \
                    and len(const_out_edges) == 1 \
                    and (len(trans_obj.perm) == len(const_obj.weights.shape)
                         or (unaware_obj.type == 'ArmPow' and len(const_obj.weights.shape) == 1)) \
                    and unaware_obj.num_in_ports() == 2 \
                    and len(unaware_in_edges) == 2 \
                    and len(unaware_obj.get_out_ports()) == 1 \
                    and len(unaware_out_edges) >= 1:
                matched = True
                inverse_perm = Op.cal_inverse_perm(trans_obj.perm)
                trans_in_edges = graph.sorted_in_edges(
                    trans, keys=True, data=True)
                src, _, k, trans_in_attr = trans_in_edges[0]

                graph.remove_edge(trans, unaware)
                new_src_out_attr = copy.deepcopy(trans_in_attr)
                new_src_out_attr['dst_in_port'] = 0 if unaware_in_edges[0][0] == trans else 1
                graph.add_edge(src, unaware, **new_src_out_attr)

                post_trans = get_valid_node_name(
                    graph, unaware + '_post_transpose')
                unaware_out_tensor = Tensor() if unaware_out_edges[0][2]['tensor'] is None \
                    else copy.deepcopy(unaware_out_edges[0][2]['tensor'])
                for _, dst, out_attr in unaware_out_edges:
                    graph.remove_edge(unaware, dst)
                    graph.add_edge(post_trans, dst, **out_attr)
                    if out_attr['tensor'].value is not None:
                        unaware_out_tensor.value = np.transpose(out_attr['tensor'].value,
                                                                inverse_perm)
                    out_shape = out_attr['tensor'].get_shape()
                    if out_shape is not None and None not in out_shape:
                        unaware_out_tensor.shape = tuple([out_shape[idx]
                                                          for idx in inverse_perm])
                graph.add_edge(unaware, post_trans, **{'tensor': unaware_out_tensor})

                trans_out_edges = graph.sorted_out_edges(trans)
                if len(trans_out_edges) == 0:
                    trans_in_edges = graph.sorted_in_edges(trans)
                    graph.remove_edges_from(trans_in_edges)

                if len(const_obj.weights.shape) == len(inverse_perm):
                    const_obj.weights = np.transpose(
                        const_obj.weights, inverse_perm)
                    if const_out_edges[0][2]['tensor'] is not None \
                            and const_out_edges[0][2]['tensor'].value is not None:
                        const_out_edges[0][2]['tensor'].value = const_obj.weights
                        const_out_edges[0][2]['tensor'].shape = const_obj.weights.shape
                post_trans_attr = unaware_obj.copied_attr()
                post_trans_attr.update(
                    {'name': post_trans, 'perm': trans_obj.perm})
                NodeWrap(graph, post_trans).replace_obj(
                    'ArmTranspose', post_trans_attr)

                if unaware in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(unaware)
                    graph._attr['output_names'][index] = post_trans

        else:
            ERROR('[Parser]: Meets invalid Node(%s) or Node(%s) or Node(%s) in sink_transpose_with_const!' % (
                trans, const, unaware))
    if matched:
        clear_redundant_nodes(graph)


def sink_transpose_through_concat(graph):
    matches = single_node_matcher(graph, 'ArmConcat')
    for m in matches:
        concat = m['target']
        concat_obj = NodeWrap(graph, concat)['object']
        concat_in_edges = graph.sorted_in_edges(concat, data=True)
        if concat_obj is None or len(concat_in_edges) < 1:
            ERROR(
                '[Parser]: Meets invalid Concat Node(%s) in sink_transpose_through_concat!' % concat)
            continue
        concat_out_edges = graph.sorted_out_edges(concat, data=True)
        if len(concat_out_edges) < 1:
            continue
        concat_in_shapes = concat_obj.get_input_shapes()
        if len(concat_in_shapes) != len(concat_in_edges) \
                or None in concat_in_shapes \
                or any(d is None for s in concat_in_shapes for d in s):
            continue
        all_src_are_trans = True
        in_trans_names = []
        in_trans_objs = {}
        for src, _, _ in concat_in_edges:
            src_obj = NodeWrap(graph, src)['object']
            if src_obj is None or src_obj.type != 'ArmTranspose':
                all_src_are_trans = False
                break
            in_trans_names.append(src)
            in_trans_objs[src] = src_obj
        if not all_src_are_trans:
            continue
        if any([not graph.has_node(name) for name in [concat] + in_trans_names]):
            DEBUG(
                '[Parser]: Meets invalid name that does not exist in graph in sink_transpose_through_concat!')
            continue

        if any([in_trans_objs[in_trans_names[0]].perm != in_trans_objs[name].perm for name in in_trans_names]):
            continue
        perm = in_trans_objs[in_trans_names[0]].perm
        inverse_perm = Op.cal_inverse_perm(perm)
        for i, (trans, _, concat_in_attr) in enumerate(concat_in_edges):
            insert_transpose(graph, trans, concat, concat_in_attr, inverse_perm, type='ArmTranspose')
        post_trans = get_valid_node_name(graph, concat + '_post_transpose')
        for _, dst, out_attr in concat_out_edges:
            graph.remove_edge(concat, dst)
            graph.add_edge(post_trans, dst, **out_attr)
        axis = concat_obj.axis
        if axis < 0:
            axis += len(perm)
        concat_obj.axis = perm[axis]
        concat_in_edges = graph.sorted_in_edges(concat, data=True)
        if all([edge_attr['tensor'].value is not None for _, _, edge_attr in concat_in_edges]):
            concat_inputs = [edge_attr['tensor'].value for _,
                             _, edge_attr in concat_in_edges]
            concat_output = np.concatenate(concat_inputs, concat_obj.axis)
            concat_output = np.transpose(concat_output, inverse_perm)
            concat_shape = concat_output.shape
        else:
            concat_output = None
            concat_dim = np.sum(s[axis] for s in concat_in_shapes)
            concat_shape = tuple(concat_dim if i == axis else d
                                 for i, d in enumerate(concat_in_shapes[0]))
            concat_shape = tuple(np.array(concat_shape)[inverse_perm].tolist())
        concat_out_attr = copy.deepcopy(concat_out_edges[0][2])
        concat_out_attr.update({'dst_in_port': 0})
        if concat_out_attr['tensor'] is not None:
            concat_out_attr['tensor'].value = concat_output
            concat_out_attr['tensor'].shape = concat_shape
        else:
            concat_out_attr['tensor'] = Tensor(value=concat_output,
                                               shape=concat_shape)
        graph.add_edge(concat, post_trans, **concat_out_attr)
        if concat in graph._attr['output_names']:
            index = graph._attr['output_names'].index(concat)
            graph._attr['output_names'][index] = post_trans

        post_trans_attr = concat_obj.copied_attr()
        post_trans_attr.update({'name': post_trans, 'perm': perm})
        NodeWrap(graph, post_trans).replace_obj(
            'ArmTranspose', post_trans_attr)


def sink_transpose_through_group_norm(graph):
    '''Lower Transpose node from Transpose(perm=P)+GN(axis=C) to GN(axis=C')+Transpose(perm=P),
    in which perm of Transpose keeps unchanged, axis of GN needs changed.
    '''
    matched = False
    matches = two_nodes_matcher(graph, 'ArmTranspose', 'ArmGroupNorm')
    for m in matches:
        trans, norm = m['begin'], m['end']
        trans_obj = NodeWrap(graph, trans)['object']
        norm_obj = NodeWrap(graph, norm)['object']
        if trans_obj is None or norm_obj is None:
            ERROR('[Parser]: Meets invalid Node object in sink_transpose_through_norm!')
            continue
        trans_in_edges = graph.sorted_in_edges(trans, data=True)
        norm_in_edges = graph.sorted_in_edges(norm, data=True)
        if len(trans_in_edges) < 1 or len(norm_in_edges) < 1:
            continue
        trans_perm = trans_obj.perm
        input_length = len(trans_perm)
        nchw_to_nhwc = [0] + list(range(2, input_length)) + [1]
        nhwc_to_nchw = Op.cal_inverse_perm(nchw_to_nhwc)
        norm_axis = OpHasAxis.make_axes_non_negative(norm_obj.axis, input_length)
        if norm_axis not in (1, input_length - 1):
            continue
        if norm_axis == 1:
            if trans_perm != nhwc_to_nchw:
                continue
            norm_new_axis = input_length - 1
        else:
            if trans_perm != nchw_to_nhwc:
                continue
            norm_new_axis = 1
        matched = True
        norm_obj.axis = norm_new_axis
        inverse_perm = Op.cal_inverse_perm(trans_perm)
        src, _, in_attr = trans_in_edges[0]
        graph.remove_edges_from(norm_in_edges)
        graph.add_edge(src, norm, **in_attr)
        post_trans = insert_transpose_after(graph, norm, trans_perm, type='ArmTranspose')
        if norm in graph._attr['output_names']:
            index = graph._attr['output_names'].index(norm)
            graph._attr['output_names'][index] = post_trans
    if matched:
        clear_redundant_nodes(graph)


def sink_transpose_through_special_reshape(graph):
    matches = single_node_matcher(graph, 'ArmTranspose')
    matched = False
    for m in matches:
        trans = m['target']
        trans_obj = NodeWrap(graph, trans)['object']
        if trans_obj is None:
            ERROR(
                '[Parser]: Meets invalid ArmTranspose Node(%s) in sink_transpose_through_special_reshape!' % trans)
            continue
        trans_out_names = graph.children(trans)
        if not trans_out_names:
            continue
        out_reshape_objs = {}
        for trans_out in trans_out_names:
            trans_out_obj = NodeWrap(graph, trans_out)['object']
            if trans_out_obj is not None and trans_out_obj.type == 'ArmReshape':
                out_reshape_objs[trans_out] = trans_out_obj
        if not out_reshape_objs:
            continue
        if any([not graph.has_node(name) for name in [trans] + trans_out_names]):
            ERROR(
                '[Parser]: Meets invalid name that does not exist in graph in sink_transpose_through_special_reshape!')
            continue

        trans_in_shape = trans_obj.get_input_shapes()[0]
        trans_in_edges = graph.sorted_in_edges(trans, data=True)
        for reshape, reshape_obj in out_reshape_objs.items():
            reshape_out_edges = graph.sorted_out_edges(reshape, data=True)
            if len(reshape_out_edges) != 1:
                continue
            reshape_in_shape = reshape_obj.get_input_shapes()[0]
            reshape_out_shape = reshape_obj.get_output_shapes()[0]
            if reshape_in_shape is None or reshape_out_shape is None \
                    or (1 not in reshape_in_shape and 1 not in reshape_out_shape):
                continue
            shape_len_diff = len(reshape_in_shape) - len(reshape_out_shape)
            if abs(shape_len_diff) != 1:
                continue
            none_one_in_shape = [d for d in reshape_in_shape if d != 1]
            none_one_out_shape = [d for d in reshape_out_shape if d != 1]
            if none_one_in_shape != none_one_out_shape:
                continue
            if len(none_one_in_shape) != len(set(none_one_in_shape)) or len(none_one_out_shape) != len(set(none_one_out_shape)):
                continue

            diff_axis = 0
            for in_dim, out_dim in zip(reshape_in_shape, reshape_out_shape):
                if in_dim != out_dim:
                    break
                diff_axis += 1
            if shape_len_diff < 0:
                if diff_axis in trans_obj.perm:
                    change_pos = trans_obj.perm.index(diff_axis)
                else:
                    change_pos = len(trans_obj.perm)
                new_dim = trans_in_shape[:]
                new_dim.insert(change_pos, 1)
            else:
                new_dim = trans_in_shape[:]
                for idx in range(diff_axis, len(trans_in_shape)):
                    change_pos = trans_obj.perm[idx]
                    if new_dim[change_pos] == 1:
                        new_dim.pop(change_pos)
                        break

            if int(np.prod(new_dim)) != int(np.prod(reshape_in_shape)) or len(new_dim) == len(trans_in_shape):
                WARN(
                    '[Parser]: Meets invalid Reshape(%s) dim in sink_transpose_through_special_reshape!' % reshape)
                continue

            matched = True

            new_perm = []
            for d in reshape_out_shape:
                for new_i, new_d in enumerate(new_dim):
                    if new_d == d and new_i not in new_perm:
                        new_perm.append(new_i)
                        break

            src, _, trans_in_attr = trans_in_edges[0]
            graph.remove_edge(trans, reshape)
            graph.add_edge(src, reshape, **trans_in_attr)
            new_transpose = get_valid_node_name(
                graph, reshape + '_post_trans')
            for _, dst, out_attr in reshape_out_edges:
                graph.remove_edge(reshape, dst)
                graph.add_edge(new_transpose, dst, **out_attr)
            reshape_obj.dim = new_dim
            reshape_in_edges = graph.sorted_in_edges(reshape, data=True)
            reshape_out_tensor = copy.deepcopy(reshape_out_edges[0][2]['tensor'])
            if reshape_in_edges[0][2]['tensor'].value is not None:
                reshape_out_tensor.value = np.reshape(
                    reshape_in_edges[0][2]['tensor'].value, reshape_obj.dim)
            else:
                reshape_out_tensor.shape = reshape_obj.dim
            graph.add_edge(reshape, new_transpose, **{'tensor': reshape_out_tensor})

            new_transpose_attr = reshape_obj.copied_attr()
            new_transpose_attr.update(
                {'name': new_transpose, 'perm': new_perm})
            NodeWrap(graph, new_transpose).replace_obj(
                'ArmTranspose', new_transpose_attr)

            if reshape in graph._attr['output_names']:
                index = graph._attr['output_names'].index(reshape)
                graph._attr['output_names'][index] = new_transpose

        trans_out_edges = graph.sorted_out_edges(trans)
        if len(trans_out_edges) == 0:
            trans_in_edges = graph.sorted_in_edges(trans)
            graph.remove_edges_from(trans_in_edges)

    if matched:
        clear_redundant_nodes(graph)


def sink_transpose_through_split(graph):
    matches = two_nodes_matcher(graph, 'ArmTranspose', 'ArmSplit')
    for m in matches:
        need_clear = False
        trans, split = m['begin'], m['end']
        trans_obj, split_obj = [
            NodeWrap(graph, name)['object'] for name in [trans, split]]
        if trans_obj is not None and split_obj is not None:
            inverse_perm = [trans_obj.perm.index(
                i) for i in range(len(trans_obj.perm))]
            trans_in_edges = graph.sorted_in_edges(trans, data=True)
            trans_out_edges = graph.sorted_out_edges(trans, data=True)
            if len(trans_out_edges) > 1:
                continue
            split_out_edges = graph.sorted_out_edges(
                split, keys=True, data=True)
            src, _, in_attr = trans_in_edges[0]
            graph.remove_edge(trans, split)
            graph.add_edge(src, split, **in_attr)
            split_obj.axis = trans_obj.perm[split_obj.axis]

            out_ports = split_obj.get_out_ports()
            last_names = []
            for p in out_ports:
                post_trans = insert_transpose_after(graph, split, trans_obj.perm,
                                                    port=p, type='ArmTranspose',
                                                    quantize=split_obj.quantize)
                last_names.append(post_trans)
            if len(graph.sorted_out_edges(trans)) == 0:
                trans_in_edges = graph.sorted_in_edges(trans)
                graph.remove_edges_from(trans_in_edges)
                need_clear = True

            if split in graph._attr['output_names'] and last_names:
                index = graph._attr['output_names'].index(split)
                graph._attr['output_names'].remove(split)
                for name in last_names:
                    graph._attr['output_names'].insert(index, name)
                    index += 1

            if need_clear:
                clear_redundant_nodes(graph)
        else:
            ERROR(
                '[Parser]: Meets invalid Node(%s) object in sink_transpose_through_split!' % (trans))


def sink_transpose_through_tile(graph):
    matches = two_nodes_matcher(graph, 'ArmTranspose', 'ArmTile')
    for m in matches:
        tr, tile = m['begin'], m['end']
        tr_obj = NodeWrap(graph, tr)['object']
        tile_obj = NodeWrap(graph, tile)['object']
        if tr_obj is not None and tile_obj is not None:
            tr_out_edges = graph.sorted_out_edges(tr, data=True)
            if len(tr_out_edges) == 1:
                inverse_perm = Op.cal_inverse_perm(tr_obj.perm)
                tile_obj.reps = np.array(tile_obj.reps)[
                    np.array(inverse_perm)].tolist()

                tr_in_edges = graph.sorted_in_edges(tr, data=True)
                tr_out_edges = graph.sorted_out_edges(tr, data=True)
                src, _, tr_in_attr = tr_in_edges[0]
                graph.remove_edges_from(tr_in_edges + tr_out_edges)

                graph.add_edge(src, tile, **tr_in_attr)
                for _, dst, out_attr in graph.sorted_out_edges(tile, data=True):
                    graph.remove_edge(tile, dst)
                    graph.add_edge(tr, dst, **out_attr)
                tile_out_tensor = copy.deepcopy(tr_in_attr['tensor']) if tr_in_attr['tensor'] is not None else Tensor()
                if tile_out_tensor.value is not None:
                    tile_out_tensor.value = np.tile(
                        tile_out_tensor.value, tile_obj.reps)
                else:
                    tr_in_shape = tile_out_tensor.get_shape()
                    if tr_in_shape is not None and None not in tr_in_shape:
                        tile_out_tensor.shape = tuple([int(shape * rep) for shape, rep
                                                       in zip(tr_in_shape, tile_obj.reps)])
                graph.add_edge(
                    tile, tr, **{'tensor': tile_out_tensor})

                if tile in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(tile)
                    graph._attr['output_names'][index] = tr
        else:
            ERROR('[Parser]: Meets invalid Node object in sink_transpose_through_tile!')


def sink_transpose_through_slice_pair(graph):
    matched = False
    unaware_types = set(ArmOp.get_concrete_subclass_names()).intersection(
        LayoutUnawareOp.get_concrete_subclass_names())
    unaware_types = sorted(list(unaware_types))
    matches = matched_patterns(graph,
                               nodes=[
                                   ('transpose', {'op': 'ArmTranspose'}),
                                   ('slice1', {'op': 'ArmSlice'}),
                                   ('slice2', {'op': 'ArmSlice'}),
                                   ('unaware', {'op': unaware_types})
                               ],
                               edges=[
                                   ('transpose', 'slice1', {'dst_in_port': 0}),
                                   ('transpose', 'slice2', {'dst_in_port': 0}),
                                   ('slice1', 'unaware', {
                                       'src_out_port': 0, 'dst_in_port': 0}),
                                   ('slice2', 'unaware', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                               ]
                               )
    for m in matches:
        names = ['transpose', 'slice1', 'slice2', 'unaware']
        if any(not graph.has_node(m[n]) for n in names):
            DEBUG('[Parser]: Meets invalid name that does not exist in graph in sink_transpose_through_slice_pair!')
            continue
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in obj_dict.values()]):
            ERROR('[Parser]: Meets invalid Op in sink_transpose_through_slice_pair!')
            continue
        if obj_dict['unaware'].num_in_ports() != 2:
            continue
        unaware_in_edges = graph.sorted_in_edges(m['unaware'], data=True)
        if len(unaware_in_edges) != 2:
            continue
        slice1_out_edges = graph.sorted_out_edges(m['slice1'])
        slice2_out_edges = graph.sorted_out_edges(m['slice2'])
        if len(slice1_out_edges) != 1 or len(slice2_out_edges) != 1:
            continue
        transpose_in_edges = graph.sorted_in_edges(m['transpose'], data=True)
        if len(transpose_in_edges) < 1:
            continue
        slice1_in_edges = graph.sorted_in_edges(m['slice1'], keys=True, data=True)
        slice2_in_edges = graph.sorted_in_edges(m['slice2'], keys=True, data=True)
        if len(slice1_in_edges) < 1 or len(slice2_in_edges) < 1:
            continue

        matched = True

        src, _, trans_in_attr = transpose_in_edges[0]
        _, _, k1, slice1_in_attr = slice1_in_edges[0]
        _, _, k2, slice2_in_attr = slice2_in_edges[0]

        graph.remove_edge(m['transpose'], m['slice1'], key=k1)
        graph.remove_edge(m['transpose'], m['slice2'], key=k2)
        graph.add_edge(src, m['slice1'], **trans_in_attr)
        graph.add_edge(src, m['slice2'], **trans_in_attr)
        post_transpose = insert_transpose_after(
            graph, m['unaware'], obj_dict['transpose'].perm, port=0, type='ArmTranspose')

        inv_perm = Op.cal_inverse_perm(obj_dict['transpose'].perm)
        np_inv_perm = np.array(inv_perm)
        obj_dict['slice1'].starts = np.array(obj_dict['slice1'].starts)[np_inv_perm].tolist()
        obj_dict['slice1'].ends = np.array(obj_dict['slice1'].ends)[np_inv_perm].tolist()
        obj_dict['slice1'].steps = np.array(obj_dict['slice1'].steps)[np_inv_perm].tolist()
        obj_dict['slice2'].starts = np.array(obj_dict['slice2'].starts)[np_inv_perm].tolist()
        obj_dict['slice2'].ends = np.array(obj_dict['slice2'].ends)[np_inv_perm].tolist()
        obj_dict['slice2'].steps = np.array(obj_dict['slice2'].steps)[np_inv_perm].tolist()

        if m['unaware'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['unaware'])
            graph._attr['output_names'][index] = post_transpose

    if matched:
        clear_redundant_nodes(graph)


def sink_quantize_through_concat(graph):
    if not graph._attr.get('quantize', False):
        return
    matched = False
    matches = single_node_matcher(graph, 'ArmConcat')
    for m in matches:
        concat = m['target']
        concat_obj = NodeWrap(graph, concat)['object']
        concat_in_edges = graph.sorted_in_edges(concat, data=True)
        if concat_obj is None or len(concat_in_edges) < 1:
            ERROR(
                '[Parser]: Meets invalid Concat Node(%s) in sink_quantize_through_concat!' % concat)
            continue
        concat_out_edges = graph.sorted_out_edges(concat, data=True)
        if len(concat_out_edges) < 1:
            continue
        all_src_are_quant = True
        in_quant_names = []
        in_ports = []
        in_quant_objs = {}
        for src, _, in_attr in concat_in_edges:
            src_obj = NodeWrap(graph, src)['object']
            if src_obj is None or src_obj.type != 'ArmQuantize' \
                    or len(graph.sorted_in_edges(src)) != 1:
                all_src_are_quant = False
                break
            in_quant_names.append(src)
            in_ports.append(in_attr['dst_in_port'])
            in_quant_objs[src] = src_obj
        if not all_src_are_quant:
            continue
        quant_obj = in_quant_objs[in_quant_names[0]]
        quant_scale = quant_obj.scale
        quant_zp = quant_obj.zero_point
        quant_axis = quant_obj.axis
        if any((not FLOAT_EQUAL(quant_scale, in_quant_objs[name].scale)
                or quant_zp != in_quant_objs[name].zero_point
                or quant_axis != in_quant_objs[name].axis) for name in in_quant_names[1:]):
            continue
        matched = True
        quant_attr = quant_obj.copied_attr()
        graph.remove_edges_from(concat_in_edges + concat_out_edges)
        for quant, dst_in_port in zip(in_quant_names, in_ports):
            quant_src, _, in_attr = graph.sorted_in_edges(quant, data=True)[0]
            concat_in_attr = copy.deepcopy(in_attr)
            concat_in_attr.update({'dst_in_port': dst_in_port})
            graph.add_edge(quant_src, concat, **concat_in_attr)
        post_quant = get_valid_node_name(graph, concat + '_post_quant')
        concat_out_attr = copy.deepcopy(concat_out_edges[0][2])
        concat_out_attr.update({'dst_in_port': 0})
        graph.add_edge(concat, post_quant, **concat_out_attr)
        for _, dst, out_attr in concat_out_edges:
            graph.add_edge(post_quant, dst, **out_attr)

        concat_obj.quantize = False
        if concat in graph._attr['output_names']:
            index = graph._attr['output_names'].index(concat)
            graph._attr['output_names'][index] = post_quant

        quant_attr.update({'name': post_quant})
        NodeWrap(graph, post_quant).replace_obj('ArmQuantize', quant_attr)
    if matched:
        clear_redundant_nodes(graph)


def assign_top_range_scale_zp(graph):
    def _type_map(dtype):
        if str(dtype) not in ['int8', 'uint8', 'int16', 'int32']:
            WARN('[Parser]: Meets unsupported zp type(%s)!' % str(dtype))
        return dtype

    has_quantize_nodes = False
    for n in graph.nodes:
        n_obj = NodeWrap(graph, n)['object']
        if n_obj is not None:
            top_info = n_obj.get_outputs_info()
            if len(top_info) < 4:
                continue
            if any('min_max' in info for info in top_info[3]):
                out_ports = n_obj.get_out_ports()
                if len(out_ports) != len(top_info[3]):
                    ERROR('[Parser]: Length of top info of Node (%s) should be equal to length of out ports!' % n)
                    continue
                top_ranges_value = [t_info['min_max'] if 'min_max' in t_info else None for t_info in top_info[3]]
                if all(val is None for val in top_ranges_value):
                    pass
                elif all((val is None or (np.array(val).size == 2 and len(val) == 2)) for val in top_ranges_value):
                    n_obj.top_ranges_list = [None if val is None else [
                        float(np.array(val).item(0)), float(np.array(val).item(1))] for val in top_ranges_value]
                else:
                    n_obj.top_ranges = [None if s is None else np.array(s, dtype=np.float32) for s in top_ranges_value]
                    n_obj.top_ranges_offset = [-1] * len(n_obj.top_ranges)
            if graph._attr.get('quantize', False):
                if n_obj.quantize:
                    has_quantize_nodes = True
                else:
                    if n_obj.type == 'ArmDeQuantize':
                        has_quantize_nodes = True
                    continue
                if not any('scale_zp' in info for info in top_info[3]):
                    continue
                out_ports = n_obj.get_out_ports()
                if len(out_ports) != len(top_info[3]):
                    ERROR('[Parser]: Length of top info of Node (%s) should be equal to length of out ports!' % n)
                    continue
                top_scales_value = [t_info['scale_zp'][0] if 'scale_zp' in t_info else None for t_info in top_info[3]]
                if all(val is None for val in top_scales_value):
                    pass
                elif all((val is None or np.array(val).size == 1) for val in top_scales_value):
                    n_obj.top_scales_list = [None if val is None else float(
                        np.array(val).item()) for val in top_scales_value]
                else:
                    n_obj.top_scales = [np.array(s).astype(
                        np.float32) if s is not None else None for s in top_scales_value]
                    n_obj.top_scales_offset = [-1] * len(n_obj.top_scales)
                top_zps_value = [t_info['scale_zp'][1] if 'scale_zp' in t_info else None for t_info in top_info[3]]
                if all(val is None for val in top_zps_value):
                    pass
                elif all((val is None or np.array(val).size == 1) for val in top_zps_value):
                    n_obj.top_zps_list = [None if val is None else np.array(val).item() for val in top_zps_value]
                else:
                    n_obj.top_zps = [np.array(z).astype(_type_map(z.dtype))
                                     if z is not None else None for z in top_zps_value]
                    n_obj.top_zps_offset = [-1] * len(n_obj.top_zps)
                assert len(n_obj.top_scales_list) == len(n_obj.top_zps_list) \
                    and len(n_obj.top_scales) == len(n_obj.top_zps), \
                    'top_scales and top_zps should have the same size!'
                top_activation_quantization_axis = [t_info.get(
                    'activation_quantization_axis', None) for t_info in top_info[3]]
                if any(axis is not None for axis in top_activation_quantization_axis):
                    n_obj.activation_quantization_axis = top_activation_quantization_axis
        else:
            ERROR('[Parser]: Meets invalid Node(%s) in assign_top_range_scale_zp!' % n)
    if graph._attr.get('quantize', False) and not has_quantize_nodes:
        graph._attr['quantize'] = False


def back_passes(graph, params):
    '''
    Pass is an optimization based on IR to remove redundant operators and perform hardware-friendly operator transformation.
    Among them, middle_pass focuses on operator splitting and merging, 
    while back_pass focuses on converting onnx operators into Arm operators defined in IR def.
    '''

    from .middle_passes import broadcast_prelu, multidirectional_broadcasting, split_mean, split_sum_or_max_or_min
    broadcast_prelu(graph)
    merge_squared_diff(graph)
    merge_square(graph)
    merge_square2(graph)
    multidirectional_broadcasting(graph)
    remove_special_where(graph)
    split_mean(graph)
    split_sum_or_max_or_min(graph)

    convert_uni_gru(graph)
    convert_bi_gru(graph)
    convert_uni_lstm(graph)
    convert_bi_lstm(graph)

    merge_b2s(graph)
    merge_b2s_nd(graph)
    merge_s2b(graph)
    merge_s2b_nd(graph)
    merge_rsqrt(graph)
    merge_not_equal(graph)

    merge_greater_less_equal_or(graph)
    # split_crd_d2s(graph)
    split_expand(graph)

    rename_argminmax(graph)
    rename_bitwise(graph)
    rename_cum(graph)
    rename_bn(graph)
    rename_cast(graph)
    rename_col2im(graph)
    rename_compress(graph)
    rename_conv(graph)
    rename_dilation_erosion(graph)
    rename_div(graph)
    rename_gemm(graph)
    rename_generate_proposals(graph)
    rename_gridsample(graph)
    rename_layernorm(graph)
    rename_groupnorm(graph)
    rename_logical(graph)
    rename_matmulinteger(graph)
    rename_maxunpool(graph)
    rename_moments(graph)
    rename_mul_add_max_min(graph)
    rename_normalization(graph)
    rename_onehot(graph)
    rename_pad(graph)
    rename_pool(graph)
    rename_quantize_dequantize(graph)
    rename_reduce(graph)

    rename_reshape(graph)
    rename_resize(graph)
    rename_roipool(graph)
    rename_roialign(graph)
    rename_scatternd(graph)
    rename_scatterel(graph)
    rename_slice(graph)
    rename_tile(graph)
    rename_topk(graph)
    rename_where(graph)

    simple_rename_list = [
        'Abs', 'AccidentalHits', 'Acos', 'Acosh', 'AdaptivePool', 'AffineGrid', 'Asin',
        'Asinh', 'Atan', 'Atanh', {'BatchGather': 'ArmGather'}, 'BitShift', 'BNLL',
        'Ceil', 'ChannelShuffle', 'Concat', {'Cos': 'ArmCosine'}, 'Cosh', 'CropAndResize',
        'CTCGreedyDecoder', 'DepthToSpace', 'DivMod', 'Erf', 'Exp', 'Filter', 'Floor',
        'FractionalPool', 'FullyConnected', 'Gather', 'GatherND', 'GatherElements', 'If', 'Input',
        {'InstanceNormalization': 'ArmInstanceNorm'}, 'InTopK', 'IsInf', 'IsNaN', 'Log', 'LogSoftmax', 'LRN',
        'MatMul', {'MeanVarianceNormalization': 'ArmMVN'}, 'Meshgrid', 'Mod', {'Neg': 'ArmNegative'},
        'NonZero', 'NormalizedMoments', 'OverlapAdd', 'Pow', 'QueryRebatch', 'Reciprocal', 'Repeat',
        'ReverseSequence', 'Round', 'SegmentReduce', 'Sign', {'Sin': 'ArmSine'}, 'Sinh', 'SlotUpdate',
        'Softmax', 'SpaceToDepth', 'SufficientStatistics', 'Split', 'Sqrt', 'Tan', 'Transpose',
        'Trunc', 'Unique', 'ZeroFraction'
    ]

    for op in simple_rename_list:
        simple_rename(graph, op)

    fuse_relu(graph)
    rename_activations(graph)

    merge_group_conv(graph)

    detection_post_process(graph, params)
    adjust_5d_to_4d(graph)
    adjust_pow(graph)
    merge_nhwc_maxpoolargmax(graph)
    merge_hwc_maxpoolargmax(graph)
    merge_hw_maxpoolargmax(graph)
    merge_hw_maxunpool(graph)
    merge_s2b_pool_b2s(graph)

    remove_redundant_bn(graph)
    sink_transpose_through_special_reshape(graph)
    sink_quantize_through_concat(graph)
    remove_redundant_reshape_transpose(graph)
    remove_redundant_reshape(graph, 'ArmReshape')
    remove_redundant_transpose(graph)
    remove_redundant_transpose2(graph)
    remove_useless_op(graph, ['ArmReduce', 'ArmReshape', 'ArmTranspose', 'ArmTile', 'ArmCast'])

    fuse_const(graph)
    remove_const(graph)
    fuse_quant_op(graph, ['ArmEltwise', 'ArmSlice'])

    if graph._attr['framework'] in (Framework.ONNX, Framework.CAFFE):
        iter_times = min(max(len(graph) // 15, 15), 20)
        nodes_num_list = [len(graph.nodes)]
        for i in range(iter_times):
            try:
                for f in [sink_transpose_through_argminmax,
                          sink_transpose_through_split,
                          sink_transpose_through_concat,
                          sink_transpose_through_special_reshape,
                          sink_transpose_through_tile,
                          sink_transpose_through_slice_pair,
                          sink_transpose_through_group_norm,
                          sink_transpose_through_norm,
                          sink_reshape_through_cast,
                          sink_single_transpose,
                          sink_double_transpose,
                          sink_double_reshape,
                          sink_transpose_with_const,
                          merge_same_op_at_out_port
                          ]:
                    f(graph)
                    remove_redundant_transpose(graph)
                    remove_redundant_transpose_unaware(graph)
                    remove_redundant_reshape(graph, 'ArmReshape')
                    remove_useless_op(graph, ['ArmReshape', 'ArmTranspose'])
                if len(nodes_num_list) < 4:
                    nodes_num_list.append(len(graph.nodes))
                else:
                    if all(num == nodes_num_list[0] for num in nodes_num_list[1:]):
                        break
                    nodes_num_list.pop(0)
                    nodes_num_list.append(len(graph.nodes))
            except Exception as e:
                WARN(
                    '[Parser]: Meets exception (%s) in remove redundant Transpose! But will proceed!', str(e))
                infer(graph)

    merge_transpose_matmul_gemm(graph)
    insert_cast_if_must(graph)
    remove_redundant_cast(graph)
    insert_preprocess(graph)
