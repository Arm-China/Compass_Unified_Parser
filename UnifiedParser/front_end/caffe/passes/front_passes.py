# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import re
import copy
from collections import OrderedDict
from ....ops.op import BaseActivationOp, BaseReluOp, OpHasWeights, CaffeOp, OpHasPaddingStrides
from ....graph.node_wrap import NodeWrap
from ....graph.graph_algo import get_valid_node_name, clear_redundant_nodes
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from ...onnx.passes.common_passes import insert_constant, insert_reshape, insert_tile, insert_transpose, remove_node_safely, insert_reshape_after
from ....common.defs import Tensor, FLOAT_EQUAL
from ....common.utils import extend_lists
from ....common.errors import *


def adjust_filter(graph):
    matches = single_node_matcher(graph, 'CaffeFILTER')
    for m in matches:
        filter_name = m['target']
        filter_obj = NodeWrap(graph, filter_name)['object']
        in_edges = graph.sorted_in_edges(filter_name, data=True)
        if filter_obj is not None and len(in_edges) >= 2:
            selector, _, in_attr = in_edges[-1]
            in_shapes = filter_obj.get_input_shapes()
            batch_size = in_shapes[-1][0]
            insert_reshape(graph, selector, filter_name, in_attr, [batch_size])
            valid_num_out = get_valid_node_name(
                graph, filter_name + '_valid_num')
            graph.add_node(valid_num_out)
            NodeWrap(graph, valid_num_out).replace_obj(
                'Out', {'name': valid_num_out})
            out_edge_attr = {'src_out_port': 2, 'dst_in_port': 0}
            graph.add_edge(filter_name, valid_num_out, **out_edge_attr)
        else:
            WARN(
                '[Parser]: Meets invalid CaffeFILTER Op (%s) in adjust_filter!' % filter_name)


def convert_proposal_roipooling(graph, params):
    matches = two_nodes_matcher(graph, 'CaffePROPOSAL', 'CaffeROIPOOLING')
    for m in matches:
        proposal, roipooling = m['begin'], m['end']
        proposal_obj, roipooling_obj = [NodeWrap(graph, name)['object'] for name in [
            proposal, roipooling]]
        if proposal_obj is not None and proposal_obj.anchors is not None and roipooling_obj is not None:
            proposal_in_edges = graph.sorted_in_edges(proposal, data=True)
            roipooling_in_edges = graph.sorted_in_edges(roipooling, data=True)
            if len(proposal_in_edges) == 3 and len(roipooling_in_edges) == 2:
                anchors, pre_nms_topn = proposal_obj.anchors, proposal_obj.pre_nms_topn
                anchor_num = anchors.shape[0]
                _, _, box_attr = roipooling_in_edges[1]
                box_num = roipooling_obj.get_input_shapes()[1][0]
                score_shape = proposal_obj.get_input_shapes()[0]

                graph.remove_edges_from(roipooling_in_edges[1:])
                score_slice = get_valid_node_name(
                    graph, proposal + '_score_slice')
                concat = get_valid_node_name(graph, proposal + '_concat')
                reshape = get_valid_node_name(graph, proposal + '_reshape')
                proposal_out_score = get_valid_node_name(
                    graph, proposal + '_out_score')
                proposal_out_num = get_valid_node_name(
                    graph, proposal + '_out_num')

                score, _, score_attr = proposal_in_edges[0]
                graph.remove_edge(score, proposal)
                graph.add_edge(score, score_slice, **score_attr)
                graph.add_edge(score_slice, proposal)

                graph.add_edge(proposal, proposal_out_score, **
                               {'src_out_port': 0, 'dst_in_port': 0})
                graph.add_edge(proposal, concat, **
                               {'src_out_port': 1, 'dst_in_port': 1})
                graph.add_edge(proposal, concat, **
                               {'src_out_port': 2, 'dst_in_port': 0})
                graph.add_edge(proposal, proposal_out_num, **
                               {'src_out_port': 3, 'dst_in_port': 0})
                graph.add_edge(concat, reshape)
                new_box_attr = copy.deepcopy(box_attr)
                new_box_attr.update({'src_out_port': 0})
                graph.add_edge(reshape, roipooling, **new_box_attr)
                insert_constant(graph, reshape + '_shape', np.array(
                    [box_num, 5], np.int64), reshape, in_port=1, data_format='NHWC')

                score_slice_attr = proposal_obj.copied_attr()
                score_slice_attr.update({'name': score_slice,
                                         'opset_version': 1,
                                         'axes': [1],
                                         'starts': [anchor_num],
                                         'ends': score_shape[1:2]
                                         })
                NodeWrap(graph, score_slice).replace_obj(
                    'Slice', score_slice_attr)

                proposal_attr = proposal_obj.copied_attr()
                proposal_attr.update({'weights': anchors,
                                      'nms_threshold': proposal_obj.nms_threshold,
                                      'feature_stride': proposal_obj.feat_stride,
                                      'image_width': int(params.get('image_width', 600)),
                                      'image_height': int(params.get('image_height', 600))
                                      })
                NodeWrap(graph, proposal).replace_obj(
                    'GenerateProposals', proposal_attr)
                NodeWrap(graph, proposal_out_score).replace_obj(
                    'Out', {'name': proposal_out_score})
                NodeWrap(graph, proposal_out_num).replace_obj(
                    'Out', {'name': proposal_out_num})

                concat_attr = proposal_obj.copied_attr()
                concat_attr.update(
                    {'name': concat, 'opset_version': 4, 'axis': 2})
                NodeWrap(graph, concat).replace_obj('Concat', concat_attr)

                reshape_attr = proposal_obj.copied_attr()
                reshape_attr.update({'name': reshape, 'opset_version': 5})
                NodeWrap(graph, reshape).replace_obj('Reshape', reshape_attr)

                roipooling_attr = roipooling_obj.copied_attr()
                roipooling_attr.update({'pooled_shape': [
                                       roipooling_obj.pooled_h, roipooling_obj.pooled_w], 'opset_version': 1})
                NodeWrap(graph, roipooling).replace_obj(
                    'MaxRoiPool', roipooling_attr)
            else:
                WARN('[Parser]: Invalid CaffePROPOSAL Op(%s) or CaffeROIPOOLING Op(%s) in convert_proposal_roipooling!' % (
                    proposal, roipooling))


def convert_bias(graph):
    matches = single_node_matcher(graph, 'CaffeBIAS')
    for m in matches:
        bias = m['target']
        bias_obj = NodeWrap(graph, bias)['object']
        in_edges = graph.sorted_in_edges(bias, data=True)
        if bias_obj is not None and len(in_edges) == 2:
            in_shapes = bias_obj.get_input_shapes()
            bottom1, _, bottom1_attr = in_edges[1]
            if bias_obj.bias_reshape_dim:
                bottom1_obj = NodeWrap(graph, bottom1)['object']
                if bottom1_obj.type == 'Constant' and len(bottom1_obj.get_output_shapes()) == 1:
                    bottom1_obj.value = np.reshape(
                        bottom1_obj.value, bias_obj.bias_reshape_dim)
                elif len(in_shapes[1]) == len(in_shapes[0]):
                    continue
                else:
                    reshape = insert_reshape(
                        graph, bottom1, bias, bottom1_attr, bias_obj.bias_reshape_dim, data_format='NCHW')
            NodeWrap(graph, bias).replace_obj(
                'Add', {'name': bias, 'opset_version': 7})
        else:
            WARN('[Parser]: Meets invalid CaffeBIAS Op (%s) in convert_bias!' % bias)


def convert_channel_shuffle(graph):
    matches = single_node_matcher(graph, 'CaffeSHUFFLECHANNEL')
    for m in matches:
        sc = m['target']
        sc_obj = NodeWrap(graph, sc)['object']
        if sc_obj is not None:
            sc_attr = sc_obj.copied_attr()
            sc_attr.update({'splits': 1})
            NodeWrap(graph, sc).replace_obj('ChannelShuffle', sc_attr)
        else:
            WARN(
                '[Parser]: Meets invalid CaffeSHUFFLECHANNEL Op (%s) in convert_channel_shuffle!' % sc)


def convert_lstm(graph):
    matched = False
    matches = single_node_matcher(graph, 'CaffeLSTM')
    for m in matches:
        lstm = m['target']
        lstm_obj = NodeWrap(graph, lstm)['object']
        in_edges = graph.sorted_in_edges(lstm, data=True)
        out_edges = graph.sorted_out_edges(lstm, data=True)
        if lstm_obj is not None \
                and ((not lstm_obj.expose_hidden and len(in_edges) == 2 and len(out_edges) == 1)
                     or (lstm_obj.expose_hidden and len(in_edges) == 4 and len(out_edges) == 3)):
            matched = True
            input_shapes = lstm_obj.get_input_shapes()
            hidden_size = lstm_obj.num_output
            src, _, in_attr = in_edges[0]
            _, dst, out_attr = out_edges[0]
            graph.remove_edges_from(in_edges[1:])

            if len(input_shapes[0]) > 3:
                dim = input_shapes[0][:2]+[int(np.prod(input_shapes[0][2:]))]
                src = insert_reshape(graph, src, lstm, in_attr, dim)
                time_steps, batch_size, input_size = dim
            else:
                time_steps, batch_size, input_size = input_shapes[0]

            w = lstm_obj.weights_list[0]
            w_stack = np.stack(np.split(w, 4, axis=0))
            reordered_w = w_stack[np.array([0, 2, 1, 3])]
            reordered_w = np.reshape(reordered_w, newshape=(1, -1, input_size))

            r = lstm_obj.weights_list[2]
            r_stack = np.stack(np.split(r, 4, axis=0))
            reordered_r = r_stack[np.array([0, 2, 1, 3])]
            reordered_r = np.reshape(
                reordered_r, newshape=(1, -1, hidden_size))

            wb = lstm_obj.weights_list[1]
            wb_stack = np.stack(np.split(wb, 4, axis=0))
            reordered_wb = wb_stack[np.array([0, 2, 1, 3])]
            rb = np.zeros_like(reordered_wb)
            reordered_b = np.concatenate([reordered_wb, rb], axis=0)
            reordered_b = np.reshape(reordered_b, newshape=(1, -1))

            insert_constant(graph, lstm + '_W', reordered_w,
                            lstm, in_port=1, data_format='NHWC')
            insert_constant(graph, lstm + '_R', reordered_r,
                            lstm, in_port=2, data_format='NHWC')
            insert_constant(graph, lstm + '_B', reordered_b,
                            lstm, in_port=3, data_format='NHWC')
            if lstm_obj.expose_hidden:
                sequence_lens = np.array([time_steps] * batch_size, np.int32)
                insert_constant(graph, lstm + '_seq_len', sequence_lens,
                                lstm, in_port=4, data_format='NHWC')
                init_h, _, init_h_attr = in_edges[2]
                init_c, _, init_c_attr = in_edges[3]
                new_init_h_attr = copy.deepcopy(init_h_attr)
                new_init_h_attr['dst_in_port'] = 5
                graph.add_edge(init_h, lstm, **new_init_h_attr)
                new_init_c_attr = copy.deepcopy(init_c_attr)
                new_init_c_attr['dst_in_port'] = 6
                graph.add_edge(init_c, lstm, **new_init_c_attr)

            post_reshape = insert_reshape(graph, lstm, dst, out_attr, dim=[
                                          time_steps, batch_size, hidden_size], data_format='NHWC')
            if lstm in graph._attr['output_names']:
                index = graph._attr['output_names'].index(lstm)
                if not lstm_obj.expose_hidden:
                    graph._attr['output_names'][index] = post_reshape
                else:
                    graph._attr['output_names'].insert(index, post_reshape)

            lstm_attr = lstm_obj.copied_attr()
            lstm_attr.update({'opset_version': 7,
                              'input_size': input_size,
                              'time_steps': time_steps,
                              'hidden_size': hidden_size,
                              'direction': 'forward',
                              'activations': ['SIGMOID', 'TANH', 'TANH'],
                              })
            NodeWrap(graph, lstm).replace_obj('LSTM', lstm_attr)
        else:
            WARN('[Parser]: Invalid CaffeLSTM(%s) to convert in convert_lstm!' % lstm)
    if matched:
        clear_redundant_nodes(graph)


def convert_pool(graph):
    matches = single_node_matcher(graph, 'CaffePOOLING')
    for m in matches:
        pool = m['target']
        pool_obj = NodeWrap(graph, pool)['object']
        if pool_obj is not None and pool_obj.method in ('AVE', 'MAX'):
            pool_tpye = 'AveragePool' if pool_obj.method == 'AVE' else 'MaxPool'
            new_node_attr = pool_obj.copied_attr()
            new_node_attr.update(
                {'opset_version': 10, 'ceil_mode': 1 if pool_obj.round_mode == 'CEIL' else 0})
            if pool_obj.method == 'AVE':
                if all(int(p) == 0 for p in pool_obj.pads) \
                        and pool_obj.pad_h == 0 \
                        and pool_obj.pad_w == 0:
                    count_include_pad = 0
                else:
                    count_include_pad = 1
                new_node_attr.update({'count_include_pad': count_include_pad})
            elif len(pool_obj.get_out_ports()) == 2:
                input_shapes = pool_obj.get_input_shapes()
                output_shapes = pool_obj.get_output_shapes()
                if len(input_shapes) == 1 \
                        and len(output_shapes) == 2 \
                        and input_shapes[0] \
                        and output_shapes[0] \
                        and output_shapes[1] \
                        and len(input_shapes[0]) == 4 \
                        and len(output_shapes[0]) == 4 \
                        and list(output_shapes[0]) == list(output_shapes[1]):
                    n, c, h, w = input_shapes[0]
                    sub_oprand = np.reshape(np.arange(
                        0, n), (n, 1, 1, 1)) * c * h * w + np.reshape(np.arange(0, c), (c, 1, 1)) * h * w
                    sub_oprand = sub_oprand.astype(np.float32)
                    cast_to_float = get_valid_node_name(
                        graph, pool + '_indices_to_float')
                    sub = get_valid_node_name(graph, pool + '_indices_sub')

                    out_edges = graph.sorted_out_edges(
                        pool, keys=True, data=True)
                    for _, dst, k, out_attr in out_edges:
                        if out_attr['src_out_port'] == 1:
                            graph.remove_edge(pool, dst, key=k)
                            new_out_attr = copy.deepcopy(out_attr)
                            new_out_attr['src_out_port'] = 0
                            graph.add_edge(sub, dst, **new_out_attr)
                    graph.add_edge(pool, cast_to_float, **
                                   {'src_out_port': 1, 'dst_in_port': 0})
                    graph.add_edge(cast_to_float, sub)
                    insert_constant(graph, sub + '_oprand',
                                    sub_oprand, sub, in_port=1)
                    NodeWrap(graph, cast_to_float).replace_obj(
                        'Cast', {'name': cast_to_float, 'opset_version': 13, 'to': 1})
                    NodeWrap(graph, sub).replace_obj(
                        'Sub', {'name': sub, 'opset_version': 7})
                    if pool in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(pool)
                        graph._attr['output_names'].insert(index+1, sub)
                else:
                    WARN(
                        '[Parser]: Invalid CaffePOOLING (%s) to convert to MaxPoolWithArgmax in convert_pool!' % pool)
            NodeWrap(graph, pool).replace_obj(pool_tpye, new_node_attr)
        else:
            WARN('[Parser]: Invalid CaffePOOLING (%s) to convert in convert_pool!' % pool)


def convert_scale(graph):
    matches = single_node_matcher(graph, 'CaffeSCALE')
    for m in matches:
        scale = m['target']
        scale_obj = NodeWrap(graph, scale)['object']
        in_edges = graph.sorted_in_edges(scale, data=True)
        if scale_obj is not None:
            if len(in_edges) in (2, 3):
                in_shapes = scale_obj.get_input_shapes()
                bottom1, _, bottom1_attr = in_edges[1]
                if scale_obj.scale_reshape_dim:
                    bottom1_obj = NodeWrap(graph, bottom1)['object']
                    if bottom1_obj.type == 'Constant' and len(bottom1_obj.get_output_shapes()) == 1:
                        bottom1_obj.value = np.reshape(
                            bottom1_obj.value, scale_obj.scale_reshape_dim)
                    elif len(in_shapes[1]) == len(in_shapes[0]):
                        continue
                    else:
                        insert_reshape(graph, bottom1, scale, bottom1_attr,
                                       scale_obj.scale_reshape_dim, data_format='NCHW')

                graph.remove_edges_from(in_edges[2:])

                if scale_obj.bias_term and len(in_edges) == 3:
                    out_edges = graph.sorted_out_edges(scale, data=True)
                    bottom2, _, bottom2_attr = in_edges[2]
                    add = get_valid_node_name(graph, scale + '_post_add')
                    new_bottom2_attr = copy.deepcopy(bottom2_attr)
                    new_bottom2_attr.update({'dst_in_port': 1})
                    graph.add_edge(bottom2, add, **new_bottom2_attr)
                    for _, dst, out_attr in out_edges:
                        graph.remove_edge(scale, dst)
                        graph.add_edge(add, dst, **out_attr)
                    graph.add_edge(scale, add)
                    NodeWrap(graph, add).replace_obj(
                        'Add', {'name': add, 'opset_version': 7})
                    if scale in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(scale)
                        graph._attr['output_names'][index] = add
                    if scale_obj.bias_reshape_dim:
                        bottom2_obj = NodeWrap(graph, bottom2)['object']
                        if bottom2_obj.type == 'Constant' and len(bottom2_obj.get_output_shapes()) == 1:
                            bottom2_obj.value = np.reshape(
                                bottom2_obj.value, scale_obj.bias_reshape_dim)
                        elif len(in_shapes[2]) == len(in_shapes[0]):
                            continue
                        else:
                            insert_reshape(
                                graph, bottom2, add, bottom2_attr, scale_obj.bias_reshape_dim, data_format='NCHW')

                NodeWrap(graph, scale).replace_obj(
                    'Mul', {'name': scale, 'opset_version': 7})
        else:
            WARN('[Parser]: Meets invalid CaffeSCALE Op (%s) in convert_scale!' % scale)


def convert_scale_to_bn(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('const', {'op': 'Constant'}),
                                   ('scale', {'op': 'CaffeSCALE'})
                               ],
                               edges=[
                                   ('const', 'scale', {
                                    'src_out_port': 0, 'dst_in_port': 1})
                               ]
                               )
    matched = False
    for m in matches:
        const, scale = m['const'], m['scale']
        const_obj, scale_obj = [
            NodeWrap(graph, name)['object'] for name in [const, scale]]
        in_edges = graph.sorted_in_edges(scale, data=True)
        if all([obj is not None for obj in [const_obj, scale_obj]]) \
                and scale_obj.axis == 1 \
                and len(const_obj.get_output_shapes()) == 1 \
                and np.ndim(const_obj.value) in (0, 1) \
                and len(in_edges) in (2, 3):
            matched = True
            scale_in_shape = scale_obj.get_input_shapes()[0]
            if np.ndim(const_obj.value) == 0:
                const_obj.value = np.tile(
                    const_obj.value, (scale_in_shape[scale_obj.axis],))
            num_output = const_obj.value.size
            new_scale = const_obj.value
            new_shift = np.zeros_like(const_obj.value)
            if scale_obj.bias_term and len(scale_obj.sorted_in_consts()) == 2:
                new_shift += scale_obj.sorted_in_consts()[1][2]
            mean_value = np.zeros((num_output,), np.float32)
            var_value = np.ones((num_output,), np.float32)

            graph.remove_edges_from(in_edges[1:])
            insert_constant(graph, scale + '_gamma',
                            new_scale, scale, in_port=1)
            insert_constant(graph, scale + '_beta',
                            new_shift, scale, in_port=2)
            insert_constant(graph, scale + '_mean',
                            mean_value, scale, in_port=3)
            insert_constant(graph, scale + '_var', var_value, scale, in_port=4)

            bn_attr = scale_obj.copied_attr()
            bn_attr.update({'opset_version': 9, 'epsilon': 0})
            NodeWrap(graph, scale).replace_obj('BatchNormalization', bn_attr)

    if matched:
        clear_redundant_nodes(graph)


def convert_slice(graph):
    matches = single_node_matcher(graph, 'CaffeSLICE')
    for m in matches:
        slice = m['target']
        slice_obj = NodeWrap(graph, slice)['object']
        if slice_obj is not None:
            input_shapes = slice_obj.get_input_shapes()
            if input_shapes is not None \
                    and len(input_shapes) == 1 \
                    and input_shapes[0] is not None:
                if input_shapes[0][slice_obj.axis] % 2 == 0 \
                        and len(slice_obj.slice_point) == 1 \
                        and input_shapes[0][slice_obj.axis] == 2 * slice_obj.slice_point[0]:
                    splits = 2 * slice_obj.slice_point
                    split_attr = slice_obj.copied_attr()
                    split_attr.update({'opset_version': 2, 'split': splits})
                    NodeWrap(graph, slice).replace_obj('Split', split_attr)
                else:
                    axes = np.array([slice_obj.axis], np.int64)
                    starts = np.array([0] * len(input_shapes[0]), np.int64)
                    ends = np.array(input_shapes[0], np.int64)

                    slice_num = len(slice_obj.slice_point) + 1
                    in_edges = graph.sorted_in_edges(slice, data=True)
                    out_edges = graph.sorted_out_edges(slice, data=True)
                    if len(in_edges) == 1 and len(out_edges) > 1:
                        src, _, in_attr = in_edges[0]
                        src_out_ports = NodeWrap(graph, src)[
                            'object'].get_out_ports()
                        graph.remove_edge(src, slice)
                        new_slice_names = []
                        for i, (_, dst, out_attr) in enumerate(out_edges):
                            new_slice = get_valid_node_name(
                                graph, slice + '_new_' + str(i+1))
                            new_in_attr = copy.deepcopy(in_attr)
                            graph.add_edge(src, new_slice, **new_in_attr)

                            graph.remove_edge(slice, dst)
                            new_out_attr = copy.deepcopy(out_attr)
                            new_out_attr.update({'src_out_port': 0})
                            graph.add_edge(new_slice, dst, **new_out_attr)

                            cur_start, cur_end = starts.copy()[slice_obj.axis], ends.copy()[
                                slice_obj.axis]
                            if i > 0:
                                cur_start = slice_obj.slice_point[i-1]
                            if i < slice_num - 1:
                                cur_end = slice_obj.slice_point[i]
                            insert_constant(
                                graph, new_slice + '_starts', np.array([cur_start], np.int64), new_slice, in_port=1)
                            insert_constant(
                                graph, new_slice + '_ends', np.array([cur_end], np.int64), new_slice, in_port=2)
                            insert_constant(graph, new_slice +
                                            '_axes', axes, new_slice, in_port=3)
                            insert_constant(
                                graph, new_slice + '_steps', np.array([1], np.int64), new_slice, in_port=4)
                            new_slice_attr = slice_obj.copied_attr()
                            new_slice_attr.update(
                                {'opset_version': 10, 'name': new_slice})
                            NodeWrap(graph, new_slice).replace_obj(
                                'Slice', new_slice_attr)
                            new_slice_names.append(new_slice)

                        if slice in graph._attr['output_names'] and new_slice_names:
                            index = graph._attr['output_names'].index(slice)
                            graph._attr['output_names'][index] = new_slice_names[0]
                            for i, name in enumerate(new_slice_names[1:]):
                                index += 1
                                graph._attr['output_names'].insert(
                                    index, new_slice_names[1+i])

                        remove_node_safely(graph, slice)
                    else:
                        WARN(
                            '[Parser]: Meets invalid Caffe Slice Op (%s) edges in convert_slice!' % slice)
            else:
                WARN(
                    '[Parser]: Meets invalid Caffe Slice Op (%s) shapes in convert_slice!' % slice)
        else:
            WARN('[Parser]: Meets invalid Caffe Slice Op (%s) in convert_slice!' % slice)


def convert_upsample(graph):
    matches = single_node_matcher(graph, 'CaffeUPSAMPLE')
    for m in matches:
        upsample = m['target']
        upsample_obj = NodeWrap(graph, upsample)['object']
        in_edges = graph.sorted_in_edges(upsample, keys=True, data=True)
        if upsample_obj is not None and len(in_edges) == 2:
            input_shapes = upsample_obj.get_input_shapes()
            if len(input_shapes[0]) != 4:
                WARN(
                    '[Parser]: Meets invalid input length for CaffeUPSAMPLE Node(%s) in convert_upsample!' % upsample)
                continue

            # Convert indices from HW to NCHW
            indice_src, _, k, in_attr = in_edges[1]
            graph.remove_edge(indice_src, upsample, key=k)
            add = get_valid_node_name(graph, upsample + '_indices_add')
            cast_to_int = get_valid_node_name(
                graph, upsample + '_indices_to_int')
            graph.add_edge(indice_src, add, **in_attr)
            graph.add_edge(add, cast_to_int)
            graph.add_edge(cast_to_int, upsample, **
                           {'src_out_port': 0, 'dst_in_port': 1})

            n, c, h, w = input_shapes[0]
            offset = np.reshape(np.arange(0, n), (n, 1, 1, 1)) * c * h * w + \
                np.reshape(np.arange(0, c), (c, 1, 1)) * h * w
            offset = np.tile(offset, [1, 1, h, w]).astype(np.float32)
            insert_constant(graph, add + '_oprand',
                            offset, add, in_port=1)

            NodeWrap(graph, add).replace_obj(
                'Add', {'name': add, 'opset_version': 7})
            NodeWrap(graph, cast_to_int).replace_obj(
                'Cast', {'name': cast_to_int, 'opset_version': 1, 'to': 'int32'})

            # Rename upsample to maxunpool in onnx
            if upsample_obj.upsample_h and upsample_obj.upsample_w:
                output_shape = np.array([n, c, upsample_obj.upsample_h,
                                         upsample_obj.upsample_w]).astype(np.int64)
            else:
                output_shape = np.array([n, c, h * upsample_obj.scale,
                                         w * upsample_obj.scale]).astype(np.int64)
            insert_constant(graph, upsample + '_output_shape',
                            output_shape, upsample, in_port=2)

            maxunpool_attr = upsample_obj.copied_attr()
            maxunpool_attr.update({'opset_version': 9,
                                   'kernel_shape': [1, 1]})
            NodeWrap(graph, upsample).replace_obj('MaxUnpool', maxunpool_attr)
        else:
            WARN(
                '[Parser]: Meets invalid CaffeUPSAMPLE Node(%s) in convert_upsample!' % upsample)


def merge_bn_scale(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('batchnorm', {'op': 'CaffeBATCHNORM'}),
                                   ('const', {'op': 'Constant'}),
                                   ('scale', {'op': 'CaffeSCALE'})
                               ],
                               edges=[
                                   ('batchnorm', 'scale'),
                                   ('const', 'scale', {
                                    'src_out_port': 0, 'dst_in_port': 1})
                               ]
                               )
    matched = False
    for m in matches:
        bn, const, scale = m['batchnorm'], m['const'], m['scale']
        bn_obj, const_obj, scale_obj = [
            NodeWrap(graph, name)['object'] for name in [bn, const, scale]]
        if all([obj is not None for obj in [bn_obj, const_obj, scale_obj]]):
            if len(bn_obj.get_output_shapes()) == 1 \
                    and bn_obj.weights is not None \
                    and bn_obj.biases is not None \
                    and const_obj.value is not None \
                    and list(bn_obj.weights.shape) == list(const_obj.value.shape) \
                    and scale_obj.axis == 1 \
                    and bn_obj.get_output_shapes()[0][scale_obj.axis] == const_obj.value.size:
                matched = True
                num_output = bn_obj.weights.size
                new_scale = bn_obj.weights * const_obj.value
                new_shift = bn_obj.biases * const_obj.value
                if scale_obj.bias_term and len(scale_obj.sorted_in_consts()) == 2:
                    new_shift += scale_obj.sorted_in_consts()[1][2]
                mean_value = np.zeros((num_output,), np.float32)
                var_value = np.ones((num_output,), np.float32)

                remove_node_safely(graph, scale)
                insert_constant(graph, bn + '_gamma', new_scale, bn, in_port=1)
                insert_constant(graph, bn + '_beta', new_shift, bn, in_port=2)
                insert_constant(graph, bn + '_mean', mean_value, bn, in_port=3)
                insert_constant(graph, bn + '_var', var_value, bn, in_port=4)

                bn_attr = bn_obj.copied_attr()
                bn_attr.update({'opset_version': 9, 'epsilon': 0})
                NodeWrap(graph, bn).replace_obj('BatchNormalization', bn_attr)
        else:
            WARN(
                '[Parser]: Meets invalid CaffeBATCHNORM (%s) /CaffeSCALE (%s) in merge_bn_scale!' % (bn, scale))

    if matched:
        clear_redundant_nodes(graph)


def merge_reduce_reshape(graph):
    matches = two_nodes_matcher(graph, 'CaffeREDUCTION', 'CaffeRESHAPE')
    for m in matches:
        reduce, reshape = m['begin'], m['end']
        reduce_obj, reshape_obj = [
            NodeWrap(graph, name)['object'] for name in [reduce, reshape]]
        if reduce_obj is not None \
                and FLOAT_EQUAL(reduce_obj.coeff, 1.0) \
                and reshape_obj is not None \
                and len(reduce_obj.get_output_shapes()) == 1 \
                and len(reduce_obj.get_input_shapes()) == 1 \
                and reduce_obj.get_input_shapes()[0] \
                and len(reduce_obj.get_input_shapes()[0]) == len(reshape_obj.shape):
            if reduce_obj.method == 'SUM':
                reduce_type = 'ReduceSum'
            elif reduce_obj.method == 'MEAN':
                reduce_type = 'ReduceMean'
            else:
                WARN('[Parser]: Reduce type %s is not supported yet in merge_reduce_reshape!' %
                     reduce_obj.method)
                continue
            axes = list(range(reduce_obj.axis, len(
                reduce_obj.get_input_shapes()[0])))
            keepdims = True
            out_edges = graph.sorted_out_edges(reshape, data=True)
            for _, dst, out_attr in out_edges:
                graph.remove_edge(reshape, dst)
                graph.add_edge(reduce, dst, **out_attr)
            remove_node_safely(graph, reshape)
            reduce_attr = reduce_obj.copied_attr()
            reduce_attr.update(
                {'opset_version': 13, 'axes': axes, 'keepdims': keepdims})
            NodeWrap(graph, reduce).replace_obj(reduce_type, reduce_attr)


def split_argmax(graph):
    matches = single_node_matcher(graph, 'CaffeARGMAX')
    for m in matches:
        argmax = m['target']
        argmax_obj = NodeWrap(graph, argmax)['object']
        if argmax_obj is not None:
            input_shapes = argmax_obj.get_input_shapes()
            output_shapes = argmax_obj.get_output_shapes()
            if len(input_shapes) == 1 and input_shapes[0]\
                    and len(output_shapes) >= 1\
                    and output_shapes[0] is not None:
                in_shape = input_shapes[0]
                new_attr = argmax_obj.copied_attr()
                in_edges = graph.sorted_in_edges(argmax, data=True)
                out_edges = graph.sorted_out_edges(argmax, data=True)

                src, _, in_attr = in_edges[0]
                last, out_port = argmax, 0

                if argmax_obj.axis is None:
                    new_attr.update({'axis': 1})
                    reshape_dim = [in_shape[0], int(np.prod(in_shape[1:])), 1]
                    _ = insert_reshape(graph, src, argmax,
                                       in_attr, reshape_dim)
                post_shape = output_shapes[0]

                if argmax_obj.out_max_val:
                    if argmax_obj.axis is not None:
                        out = get_valid_node_name(
                            graph, argmax + '_indices_out')
                        graph.add_edge(
                            argmax, out, **{'src_out_port': 1, 'dst_in_port': 0})
                        NodeWrap(graph, out).replace_obj('Out', {'name': out})
                        last, out_port = argmax, 0
                    else:
                        need_insert_unsqueeze = len(input_shapes[0]) <= 3
                        reshape1 = get_valid_node_name(
                            graph, argmax + '_value_reshape')
                        reshape2 = get_valid_node_name(
                            graph, argmax + '_indices_reshape')
                        cast = get_valid_node_name(
                            graph, argmax + '_indices_cast')
                        concat = get_valid_node_name(graph, argmax + '_concat')
                        for _, dst, out_attr in out_edges:
                            graph.remove_edge(argmax, dst)
                            graph.add_edge(concat, dst, **out_attr)
                        if need_insert_unsqueeze:
                            graph.add_edge(argmax, reshape1)
                            graph.add_edge(argmax, reshape2, **
                                           {'src_out_port': 1, 'dst_in_port': 0})
                            graph.add_edge(reshape2, cast)
                            graph.add_edge(reshape1, concat, **
                                           {'src_out_port': 0, 'dst_in_port': 1})
                            axes = list(range(3, len(input_shapes[0])))
                            NodeWrap(graph, reshape1).replace_obj('Unsqueeze', {
                                'name': reshape1, 'opset_version': 11, 'axes': axes})
                            NodeWrap(graph, reshape2).replace_obj('Unsqueeze', {
                                'name': reshape2, 'opset_version': 11, 'axes': axes})
                        else:
                            graph.add_edge(
                                argmax, concat, **{'src_out_port': 0, 'dst_in_port': 1})
                            graph.add_edge(
                                argmax, cast, **{'src_out_port': 1, 'dst_in_port': 0})

                        graph.add_edge(cast, concat)
                        input_tensors = argmax_obj.get_input_tensors()
                        dtype = str(input_tensors[0].dtype)
                        cast_attr = {'name': cast,
                                     'opset_version': 1, 'to': dtype}
                        NodeWrap(graph, cast).replace_obj('Cast', cast_attr)
                        concat_attr = {'name': concat,
                                       'opset_version': 11, 'axis': 1}
                        NodeWrap(graph, concat).replace_obj(
                            'Concat', concat_attr)
                        last, out_port = concat, 0
                else:
                    for _, dst, out_attr in out_edges:
                        out_attr['src_out_port'] = 1
                    out = get_valid_node_name(graph, argmax + '_value_out')
                    graph.add_edge(
                        argmax, out, **{'src_out_port': 0, 'dst_in_port': 0})
                    NodeWrap(graph, out).replace_obj('Out', {'name': out})
                    if argmax_obj.axis is not None:
                        last, out_port = argmax, 1
                    else:
                        indices_reshape = get_valid_node_name(
                            graph, argmax + '_indices_reshape')
                        for _, dst, out_attr in out_edges:
                            if out_attr['src_out_port'] == 1:
                                graph.remove_edge(argmax, dst)
                                graph.add_edge(indices_reshape,
                                               dst, **out_attr)
                        graph.add_edge(argmax, indices_reshape,
                                       **{'src_out_port': 1, 'dst_in_port': 0})
                        NodeWrap(graph, indices_reshape).replace_obj('Unsqueeze', {
                            'name': indices_reshape, 'opset_version': 11, 'axes': [1]})
                        last, out_port = indices_reshape, 1
                post_reshape = insert_reshape_after(
                    graph, last, post_shape, out_port=out_port)
                last = post_reshape
                new_attr.update(
                    {'opset_version': 11, 'enable_transform': False})
                NodeWrap(graph, argmax).replace_obj('TopK', new_attr)
                k_value = np.array([argmax_obj.k], np.int64)
                insert_constant(graph, argmax + '_k',
                                k_value, argmax, in_port=1)

                if argmax in graph._attr['output_names'] and argmax != last:
                    index = graph._attr['output_names'].index(argmax)
                    graph._attr['output_names'][index] = last
            else:
                WARN(
                    '[Parser]: Meets invalid input shapes of CaffeARGMAX Node(%s) in split_argmax!' % argmax)
        else:
            WARN('[Parser]: Meets invalid CaffeARGMAX Node(%s) in split_argmax!' % argmax)


def split_axpy(graph):
    matches = single_node_matcher(graph, 'CaffeAXPY')
    for m in matches:
        axpy = m['target']
        axpy_obj = NodeWrap(graph, axpy)['object']
        in_edges = graph.sorted_in_edges(axpy, data=True)
        out_edges = graph.sorted_out_edges(axpy, data=True)
        if axpy_obj is not None and len(in_edges) == 3 and axpy_obj.get_input_shapes()[0][-2:] == [1, 1]:
            a_shape, x_shape = axpy_obj.get_input_shapes()[0:2]
            a, _, a_in_attr = in_edges[0]
            y, _, y_in_attr = in_edges[2]
            graph.remove_edges_from(in_edges[2:])
            add = get_valid_node_name(graph, axpy + '_post_add')
            new_y_in_attr = copy.deepcopy(y_in_attr)
            new_y_in_attr.update({'dst_in_port': 1})
            graph.add_edge(y, add, **new_y_in_attr)
            for _, dst, out_attr in out_edges:
                graph.remove_edge(axpy, dst)
                graph.add_edge(add, dst, **out_attr)
            graph.add_edge(axpy, add)
            reps = np.array(x_shape, np.int64) // np.array(a_shape, np.int64)
            insert_tile(graph, a, axpy, a_in_attr, reps, data_format='NCHW')
            NodeWrap(graph, axpy).replace_obj(
                'Mul', {'name': axpy, 'opset_version': 7})
            NodeWrap(graph, add).replace_obj(
                'Add', {'name': add, 'opset_version': 7})
            if axpy in graph._attr['output_names']:
                index = graph._attr['output_names'].index(axpy)
                graph._attr['output_names'][index] = add
        else:
            WARN(
                '[Parser]: Meets invalid Caffe AXPY (%s) for splitting in split_axpy!' % axpy)


def split_exp(graph):
    matches = single_node_matcher(graph, 'CaffeEXP')
    for m in matches:
        exp = m['target']
        exp_obj = NodeWrap(graph, exp)['object']
        in_edges = graph.sorted_in_edges(exp, data=True)
        if exp_obj is not None and len(in_edges) == 1:
            base, scale, shift = exp_obj.base, exp_obj.scale, exp_obj.shift
            log_base = 1 if FLOAT_EQUAL(base, -1) else np.log(base)
            inner_scale = log_base * scale
            outer_scale = 1 if FLOAT_EQUAL(shift, 0) else (
                np.power(base, shift) if not FLOAT_EQUAL(base, -1) else np.exp(shift))
            if not FLOAT_EQUAL(inner_scale, 1):
                pre_mul = get_valid_node_name(graph, exp + '_pre_mul')
                src, _, in_attr = in_edges[0]
                graph.remove_edge(src, exp)
                graph.add_edge(src, pre_mul, **in_attr)
                insert_constant(graph, pre_mul + '_multiplier',
                                np.array(inner_scale, np.float32), pre_mul, in_port=1)
                graph.add_edge(pre_mul, exp)
                mul_attr = {'name': pre_mul, 'opset_version': 7}
                NodeWrap(graph, pre_mul).replace_obj('Mul', mul_attr)
            if not FLOAT_EQUAL(outer_scale, 1):
                post_mul = get_valid_node_name(graph, exp + '_post_mul')
                out_edges = graph.sorted_out_edges(exp, data=True)
                for _, dst, out_attr in out_edges:
                    graph.remove_edge(exp, dst)
                    graph.add_edge(post_mul, dst, **out_attr)
                graph.add_edge(exp, post_mul)
                insert_constant(graph, post_mul + '_multiplier',
                                np.array(outer_scale, np.float32), post_mul, in_port=1)
                mul_attr = {'name': post_mul, 'opset_version': 7}
                NodeWrap(graph, post_mul).replace_obj('Mul', mul_attr)
                if exp in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(exp)
                    graph._attr['output_names'][index] = post_mul
            exp_attr = {'name': exp, 'opset_version': 6}
            NodeWrap(graph, exp).replace_obj('Exp', exp_attr)
        else:
            WARN('[Parser]: Meets invalid CaffeEXP node(%s) in split_exp!' % exp)


def split_inner_product(graph):
    matches = single_node_matcher(graph, 'CaffeINNER_PRODUCT')
    for m in matches:
        ip = m['target']
        ip_obj = NodeWrap(graph, ip)['object']
        input_shapes = ip_obj.get_input_shapes()
        output_shapes = ip_obj.get_output_shapes()
        in_edges = graph.sorted_in_edges(ip, data=True)
        if len(in_edges) == 1 \
                and input_shapes is not None \
                and output_shapes is not None \
                and len(input_shapes) == 1 \
                and input_shapes[0] is not None \
                and len(output_shapes) >= 1 \
                and output_shapes[0] is not None:
            inp_shape = input_shapes[0]
            axis = ip_obj.axis
            if axis < 0:
                axis += len(inp_shape)
            assert(axis >= 0 and axis <= len(
                inp_shape)), 'The axis value of the CaffeINNER_PRODUCT op is invalid in split_inner_product.'
            sqz = [int(np.prod(inp_shape[:axis])),
                   int(np.prod(inp_shape[axis:]))]
            src, _, in_attr = in_edges[0]
            insert_reshape(graph, src, ip, in_attr, sqz, data_format='NCHW')
            out_reshape = insert_reshape_after(
                graph, ip, list(output_shapes[0]))
            if ip in graph._attr['output_names']:
                index = graph._attr['output_names'].index(ip)
                graph._attr['output_names'][index] = out_reshape
            matmul_attr = ip_obj.copied_attr()
            if not ip_obj.bias_term and ip_obj.biases is None:
                matmul_attr.update(
                    {'biases': np.zeros((ip_obj.num_output,), np.float32)})
            NodeWrap(graph, ip).replace_obj('FullyConnected', matmul_attr)
        else:
            WARN(
                '[Parser]: Meets invalid INNER_PRODUCT(%s) to convert to Onnx in split_inner_product!' % ip)


def split_log(graph):
    '''
  // LogLayer computes outputs y = log_base(shift + scale * x), for base > 0.
  // Or if base is set to the default (-1), base is set to e,
  // so y = ln(shift + scale * x) = log_e(shift + scale * x)
    '''
    matches = single_node_matcher(graph, 'CaffeLOG')
    for m in matches:
        log = m['target']
        log_obj = NodeWrap(graph, log)['object']
        if log_obj is not None:
            in_edges = graph.sorted_in_edges(log, data=True)
            if len(in_edges) == 1:
                if not FLOAT_EQUAL(log_obj.scale, 1.0) or not FLOAT_EQUAL(log_obj.shift, 0):
                    src, _, in_attr = in_edges[0]
                    mul = get_valid_node_name(graph, log + '_pre_mul')
                    add = get_valid_node_name(graph, log + '_pre_add')
                    graph.add_edge(mul, add)
                    graph.remove_edge(src, log)
                    graph.add_edge(src, mul, **in_attr)
                    graph.add_edge(add, log)

                    mul_attr = log_obj.copied_attr()
                    mul_attr.update({'name': mul, 'opset_version': 7})
                    NodeWrap(graph, mul).replace_obj('Mul', mul_attr)
                    insert_constant(
                        graph, mul + '_multiplier', np.array(log_obj.scale, np.float32), mul, in_port=1)

                    add_attr = log_obj.copied_attr()
                    add_attr.update({'name': add, 'opset_version': 7})
                    NodeWrap(graph, add).replace_obj('Add', add_attr)
                    insert_constant(
                        graph, add + '_adder', np.array(log_obj.shift, np.float32), add, in_port=1)

                    in_edges = graph.sorted_in_edges(log, data=True)

                new_attr = log_obj.copied_attr()
                new_attr.update({'name': log, 'opset_version': 6})
                NodeWrap(graph, log).replace_obj('Log', new_attr)
                if log_obj.base >= 0:
                    src, _, in_attr = in_edges[0]
                    base_log = get_valid_node_name(graph, log + '_base_log')
                    div = get_valid_node_name(graph, log + '_post_div')
                    graph.add_edge(base_log, div, **
                                   {'src_out_port': 0, 'dst_in_port': 1})

                    base_log_attr = log_obj.copied_attr()
                    base_log_attr.update(
                        {'name': base_log, 'opset_version': 6})
                    NodeWrap(graph, base_log).replace_obj('Log', base_log_attr)
                    insert_constant(graph, base_log + '_base',
                                    np.array(log_obj.base, np.float32), base_log)

                    out_edges = graph.sorted_out_edges(log, data=True)
                    for _, dst, out_attr in out_edges:
                        graph.remove_edge(log, dst)
                        graph.add_edge(div, dst, **out_attr)
                    graph.add_edge(log, div)

                    div_attr = log_obj.copied_attr()
                    div_attr.update({'name': div, 'opset_version': 7})
                    NodeWrap(graph, div).replace_obj('Div', div_attr)

                    if log in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(log)
                        graph._attr['output_names'][index] = div
            else:
                WARN(
                    '[Parser]: Dose not support splitting of CaffeLOG node(%s) in split_log!' % log)
        else:
            WARN('[Parser]: Meets invalid CaffeLOG node(%s) in split_log!' % log)


def split_mvn_special(graph):
    matches = single_node_matcher(graph, 'CaffeMVN')
    for m in matches:
        mvn = m['target']
        mvn_obj = NodeWrap(graph, mvn)['object']
        in_edges = graph.sorted_in_edges(mvn, data=True)
        out_edges = graph.sorted_out_edges(mvn, data=True)
        input_shapes = mvn_obj.get_input_shapes()
        if mvn_obj is not None and len(in_edges) == 1 and len(out_edges) >= 1 and len(input_shapes) == 1:
            in_dim = input_shapes[0]
            if len(in_dim) < 4:
                reshape_dim = list(input_shapes[0]) + [1] * (4 - len(in_dim))
                src, _, in_attr = in_edges[0]
                _ = insert_reshape(graph, src, mvn, in_attr, reshape_dim)
                in_edges = graph.sorted_in_edges(mvn, data=True)
            if not mvn_obj.normalize_variance:
                src, _, in_attr = in_edges[0]
                sub = get_valid_node_name(graph, mvn + '_sub')
                for _, dst, out_attr in out_edges:
                    graph.remove_edge(mvn, dst)
                    graph.add_edge(sub, dst, **out_attr)
                graph.add_edge(
                    mvn, sub, **{'src_out_port': 0, 'dst_in_port': 1})
                graph.add_edge(src, sub, **copy.deepcopy(in_attr))
                reduce_attr = mvn_obj.copied_attr()
                reduce_attr.update(
                    {'name': mvn, 'opset_version': 11, 'axes': mvn_obj.axes, 'keepdims': True})
                NodeWrap(graph, mvn).replace_obj('ReduceMean', reduce_attr)
                sub_attr = mvn_obj.copied_attr()
                sub_attr.update({'name': sub, 'opset_version': 7})
                NodeWrap(graph, sub).replace_obj('Sub', sub_attr)
                if mvn in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(mvn)
                    graph._attr['output_names'][index] = sub
        else:
            WARN('[Parser]: Invalid CaffeMVN (%s) to split in split_mvn_special!' % mvn)


def split_normalize(graph):
    matches = single_node_matcher(graph, 'CaffeNORMALIZE')
    for m in matches:
        normalize = m['target']
        normalize_obj = NodeWrap(graph, normalize)['object']
        if normalize_obj is not None:
            if not getattr(normalize_obj, 'across_spatial', 'True'):
                normalize_out_edges = graph.sorted_out_edges(
                    normalize, data=True)
                bn = get_valid_node_name(graph, normalize + '_post_bn')
                for _, dst, out_attr in normalize_out_edges:
                    graph.remove_edge(normalize, dst)
                    graph.add_edge(bn, dst, **out_attr)
                graph.add_edge(normalize, bn)

                l2norm_attr = normalize_obj.copied_attr()
                l2norm_attr.update({'opset_version': 1, 'p': 2, 'axis': 1})
                NodeWrap(graph, normalize).replace_obj(
                    'LpNormalization', l2norm_attr)

                num_output = normalize_obj.weights.size
                new_scale = normalize_obj.weights
                new_shift = np.zeros((num_output,), np.float32)
                mean_value = np.zeros((num_output,), np.float32)
                var_value = np.ones((num_output,), np.float32)
                insert_constant(graph, bn + '_gamma', new_scale, bn, in_port=1)
                insert_constant(graph, bn + '_beta', new_shift, bn, in_port=2)
                insert_constant(graph, bn + '_mean', mean_value, bn, in_port=3)
                insert_constant(graph, bn + '_var', var_value, bn, in_port=4)
                bn_attr = normalize_obj.copied_attr()
                bn_attr.update(
                    {'name': bn, 'opset_version': 9, 'epsilon': 0.0})
                NodeWrap(graph, bn).replace_obj('BatchNormalization', bn_attr)
            else:
                WARN(
                    '[Parser]: Dose not support splitting of CaffeNORMALIZE (across_spatial=true) in split_normalize!')
        else:
            WARN(
                '[Parser]: Meets invalid CaffeNORMALIZE node(%s) in split_normalize!' % normalize)


def split_power(graph):
    matches = single_node_matcher(graph, 'CaffePOWER')
    for m in matches:
        pow = m['target']
        pow_obj = NodeWrap(graph, pow)['object']
        if pow_obj is not None:
            in_edges = graph.sorted_in_edges(pow, data=True)
            if len(in_edges) == 1:
                src, _, in_attr = in_edges[0]
                if not FLOAT_EQUAL(pow_obj.scale, 1.0) or not FLOAT_EQUAL(pow_obj.shift, 0):
                    mul = get_valid_node_name(graph, pow + '_pre_mul')
                    add = get_valid_node_name(graph, pow + '_pre_add')
                    graph.add_edge(mul, add)
                    graph.remove_edge(src, pow)
                    graph.add_edge(src, mul, **in_attr)
                    graph.add_edge(add, pow)

                    mul_attr = pow_obj.copied_attr()
                    mul_attr.update({'name': mul, 'opset_version': 7})
                    NodeWrap(graph, mul).replace_obj('Mul', mul_attr)
                    insert_constant(
                        graph, mul + '_multiplier', np.array(pow_obj.scale, np.float32), mul, in_port=1)

                    add_attr = pow_obj.copied_attr()
                    add_attr.update({'name': add, 'opset_version': 7})
                    NodeWrap(graph, add).replace_obj('Add', add_attr)
                    insert_constant(
                        graph, add + '_adder', np.array(pow_obj.shift, np.float32), add, in_port=1)

                insert_constant(
                    graph, pow + '_pow', np.array(pow_obj.power, np.float32), pow, in_port=1)
                new_attr = pow_obj.copied_attr()
                new_attr.update({'name': pow, 'opset_version': 7})
                NodeWrap(graph, pow).replace_obj('Pow', new_attr)
            else:
                WARN('[Parser]:Meets invalid CaffePOWER node(%s) in split_power!' % pow)
        else:
            WARN('[Parser]: Meets invalid CaffePOWER node(%s) in split_power!' % pow)


def split_reduce_asum(graph):
    matches = single_node_matcher(graph, 'CaffeREDUCTION')
    for m in matches:
        reduce = m['target']
        reduce_obj = NodeWrap(graph, reduce)['object']
        in_edges = graph.sorted_in_edges(reduce, data=True)
        if reduce_obj is not None and reduce_obj.method == 'ASUM' and len(in_edges) == 1:
            abs = get_valid_node_name(graph, reduce + '_pre_abs')
            src, _, in_attr = in_edges[0]
            new_in_attr = copy.deepcopy(in_attr)
            graph.remove_edge(src, reduce)
            graph.add_edge(src, abs, **in_attr)
            new_in_attr.update({'src_out_port': 0})
            if new_in_attr.get('tensor', None) is not None and new_in_attr['tensor'].value is not None:
                new_in_attr['tensor'].value = np.abs(
                    new_in_attr['tensor'].value)
            graph.add_edge(abs, reduce, **new_in_attr)
            abs_attr = reduce_obj.copied_attr()
            abs_attr.update({'name': abs, 'opset_version': 6})
            NodeWrap(graph, abs).replace_obj('Abs', abs_attr)


def split_spp(graph):
    matches = single_node_matcher(graph, 'CaffeSPP')
    for m in matches:
        spp = m['target']
        spp_obj = NodeWrap(graph, spp)['object']
        in_edges = graph.sorted_in_edges(spp, data=True)
        if spp_obj is not None and spp_obj.method in ('AVE', 'MAX') and len(in_edges) == 1:
            input_shape = spp_obj.get_input_shapes()[0]
            batch, channel = input_shape[0:2]
            src, _, in_attr = in_edges[0]
            pools_num = spp_obj.pyramid_height
            pool_type = 'AveragePool' if spp_obj.method == 'AVE' else 'MaxPool'
            graph.remove_edges_from(in_edges)

            for i in range(pools_num):
                pooling = get_valid_node_name(graph, spp + '_pool_' + str(i))
                reshape = get_valid_node_name(graph, pooling + '_post_reshape')
                pool_in_attr = copy.deepcopy(in_attr)
                graph.add_edge(src, pooling, **pool_in_attr)
                graph.add_edge(pooling, reshape)
                graph.add_edge(reshape, spp, **
                               {'src_out_port': 0, 'dst_in_port': i})

                num_bins = 2 ** i
                in_shape = np.array(input_shape[2:], np.float32)
                kernel = np.ceil(
                    in_shape / np.array([num_bins, num_bins], np.float32)).astype(np.int32)
                remainder = (
                    kernel * np.array([num_bins, num_bins], np.float32) - in_shape).astype(np.int32)
                pad = (remainder + 1) // 2

                out_shape = np.array([num_bins, num_bins], np.int64)
                strides = kernel
                kernel_shape = kernel
                dilations = np.array([1, 1], np.int64)
                new_pads = (out_shape - 1) * strides + \
                    (kernel_shape - 1) * dilations + 1 - in_shape
                new_pads_list = new_pads.tolist()
                new_pads = pad.tolist() + \
                    [new_pads_list[0] - pad[0], new_pads_list[1] - pad[1]]

                pool_attr = spp_obj.copied_attr()
                pool_attr.update({'opset_version': 10,
                                  'name': pooling,
                                  'kernel_shape': kernel.tolist(),
                                  'strides': kernel.tolist(),
                                  'pads': new_pads,
                                  'ceil_mode': True
                                  })
                NodeWrap(graph, pooling).replace_obj(pool_type, pool_attr)

                dim = [batch, channel * num_bins * num_bins]
                reshape_attr = spp_obj.copied_attr()
                reshape_attr.update({'opset_version': 5, 'name': reshape})
                NodeWrap(graph, reshape).replace_obj('Reshape', reshape_attr)
                insert_constant(graph, reshape + '_shape',
                                np.array(dim, np.int64), reshape, in_port=1)

            concat_attr = spp_obj.copied_attr()
            concat_attr.update({'opset_version': 4, 'axis': 1})
            NodeWrap(graph, spp).replace_obj('Concat', concat_attr)
        else:
            WARN('[Parser]: Invalid CaffeSPP (%s) to split in split_spp!' % spp)


def remove_detection_postprocess(graph):
    matches = single_node_matcher(graph, 'CaffeDETECTIONOUTPUT')
    for m in matches:
        detection = m['target']
        detection_obj = NodeWrap(graph, detection)['object']
        input_shapes = detection_obj.get_input_shapes()
        in_edges = graph.sorted_in_edges(detection, data=True)
        if len(input_shapes) == 3 \
                and all([s is not None for s in input_shapes]) \
                and len(input_shapes[0]) >= 2 \
                and len(input_shapes[1]) >= 2 \
                and input_shapes[0][0] == input_shapes[1][0] \
                and len(in_edges) == len(input_shapes):
            inputs = detection_obj.get_input_tensors()
            batch = input_shapes[0][0]
            class_num = detection_obj.num_classes
            in_shape1 = int(np.prod(input_shapes[0][1:]))
            in_shape2 = int(np.prod(input_shapes[1][1:]))
            if in_shape1 // 4 == in_shape2 // class_num:
                box_predict, conf_predict = in_edges[0][0], in_edges[1][0]
                box_in_port, conf_in_port = 0, 1
                box_num = in_shape2 // class_num
            else:
                box_predict, conf_predict = in_edges[1][0], in_edges[0][0]
                box_in_port, conf_in_port = 1, 0
                box_num = in_shape2 // class_num

            anchors = np.reshape(inputs[2][0, 0, ...], (box_num, 4))
            anchors_center_x = (anchors[..., 0] + anchors[..., 2]) / 2.0
            anchors_center_y = (anchors[..., 1] + anchors[..., 3]) / 2.0
            anchors_width = anchors[..., 2] - anchors[..., 0]
            anchors_height = anchors[..., 3] - anchors[..., 1]
            anchors = np.stack(
                [anchors_center_y, anchors_center_x, anchors_height, anchors_width], axis=-1).astype(np.float32)
            graph._attr['anchors'] = anchors

            if len(input_shapes[box_in_port]) != 3:
                box_out = insert_reshape(
                    graph, box_predict, detection, in_edges[box_in_port][2], [batch, box_num, 4])
            else:
                box_out = box_predict
            if len(input_shapes[conf_in_port]) != 3:
                conf_out = insert_reshape(graph, conf_predict, detection, in_edges[conf_in_port][2], [
                                          batch, box_num, class_num])
            else:
                conf_out = conf_predict

            in_edges = graph.sorted_in_edges(detection)
            out_edges = graph.sorted_out_edges(detection)
            graph.remove_edges_from(in_edges + out_edges)

            for net_out in [conf_out, box_out]:
                out = get_valid_node_name(graph, net_out + '_out')
                graph.add_edge(net_out, out, **
                               {'src_out_port': 0, 'dst_in_port': 0})
                NodeWrap(graph, out).replace_obj('Out', {'name': out})

            if detection in graph._attr['output_names']:
                index = graph._attr['output_names'].index(detection)
                graph._attr['output_names'][index] = conf_out
                graph._attr['output_names'].insert(index+1, box_out)

        clear_redundant_nodes(graph)


def refinedet_postprocess(graph, params):
    matches = single_node_matcher(graph, 'CaffeREFINEDETECTIONOUTPUT')
    for m in matches:
        detection = m['target']
        detection_obj = NodeWrap(graph, detection)['object']
        if detection_obj is not None:
            input_shapes = detection_obj.get_input_shapes()
            inputs = detection_obj.get_input_tensors()
            in_edges = graph.sorted_in_edges(detection, data=True)
            if len(input_shapes) == 5 \
                    and all([s for s in input_shapes]) \
                    and len(input_shapes[0]) >= 2 \
                    and len(input_shapes[1]) >= 2 \
                    and input_shapes[0][0] == input_shapes[1][0] \
                    and len(in_edges) == len(input_shapes):
                objectness_threshold = 0.01
                confidence_threshold = 0.01
                iou_threshold = 0.45
                image_width = int(params['image_width']) if 'image_width' in params else list(
                    graph._attr['input_tensors'].values())[0].shape[3]
                image_height = int(params['image_height']) if 'image_height' in params else list(
                    graph._attr['input_tensors'].values())[0].shape[2]

                batch = input_shapes[0][0]
                class_num = detection_obj.num_classes
                in_shape2 = int(np.prod(input_shapes[1][1:]))
                odm_loc, odm_conf = in_edges[0][0], in_edges[1][0]
                arm_loc, arm_conf = in_edges[4][0], in_edges[3][0]
                box_num = in_shape2 // class_num
                anchors = np.reshape(inputs[2][0, 0, ...], (1, box_num, 4))

                # topk_out_box_num = 1000

                # odm_loc: [1, 16320, 1, 4]
                # odm_conf: [1, 16320, 1, 60]
                # anchors: [1, 16320, 4]
                # arm_loc:  [1, 16320, 1, 4]
                # arm_conf:  [1, 16320, 1, 2]
                graph.remove_edges_from(in_edges)

                post_reshape_shapes = OrderedDict()
                for i, (src, _, in_attr) in enumerate(in_edges):
                    if i == 2:
                        continue
                    post_reshape = get_valid_node_name(
                        graph, src + '_post_reshape')
                    reshape_in_attr = copy.deepcopy(in_attr)
                    reshape_in_attr['dst_in_port'] = 0
                    graph.add_edge(src, post_reshape, **reshape_in_attr)
                    dim = input_shapes[i][0:2] + input_shapes[i][3:]
                    dim_const = np.array(dim, np.int64)
                    insert_constant(graph, post_reshape + '_shape', dim_const,
                                    post_reshape, in_port=1, data_format='NHWC')
                    NodeWrap(graph, post_reshape).replace_obj(
                        'Reshape', {'name': post_reshape, 'opset_version': 5})
                    post_reshape_shapes[post_reshape] = dim
                odm_loc_post_reshape, odm_conf_post_reshape, arm_conf_post_reshape, arm_loc_post_reshape = post_reshape_shapes.keys()

                graph.add_edge(arm_conf_post_reshape, detection)
                graph.add_edge(odm_conf_post_reshape, detection,
                               **{'src_out_port': 0, 'dst_in_port': 1})
                graph.add_edge(arm_loc_post_reshape, detection,
                               **{'src_out_port': 0, 'dst_in_port': 2})
                graph.add_edge(odm_loc_post_reshape, detection,
                               **{'src_out_port': 0, 'dst_in_port': 3})
                insert_constant(graph, 'anchors', anchors,
                                detection, in_port=4)

                refinedet_attr = detection_obj.copied_attr()
                refinedet_attr.update({'obj_thresh': objectness_threshold,
                                       'conf_thresh': confidence_threshold,
                                       'pre_nms_topk': 1000,
                                       'post_nms_topk': 200
                                       })
                NodeWrap(graph, detection).replace_obj(
                    'ArmRefineDetDetection', refinedet_attr)
                clear_redundant_nodes(graph)
        else:
            WARN(
                '[Parser]: Meets invalid CaffeREFINEDETECTIONOUTPUT Node in refinedet_postprocess (%s)!' % detection)


def remove_useless_reshape(graph):
    matches = two_nodes_matcher(graph, '', 'CaffeRESHAPE')
    for m in matches:
        inp, reshape = m['begin'], m['end']
        inp_obj = NodeWrap(graph, inp)['object']
        reshape_obj = NodeWrap(graph, reshape)['object']
        in_edges = graph.sorted_in_edges(reshape, data=True)
        if inp_obj is not None and reshape_obj is not None and len(in_edges) == 1:
            in_shape = reshape_obj.get_input_shapes()[0]
            out_shape = reshape_obj.get_output_shapes()[0]
            if in_shape and in_shape == out_shape and in_shape == reshape_obj.shape:
                _, _, in_attr = in_edges[0]
                for _, dst, out_attr in graph.sorted_out_edges(reshape, data=True):
                    new_out_attr = copy.deepcopy(out_attr)
                    new_out_attr['src_out_port'] = in_attr['src_out_port']
                    graph.remove_edge(reshape, dst)
                    graph.add_edge(inp, dst, **new_out_attr)
                remove_node_safely(graph, reshape)


def convert_to_onnx(graph):
    '''Convert the model to the onnx version.'''
    caffe_ops = CaffeOp.get_concrete_subclass_names()
    matches = extend_lists([single_node_matcher(graph, op_type)
                            for op_type in caffe_ops])
    for m in matches:
        node_name = m['target']
        node_obj = NodeWrap(graph, node_name)['object']
        if node_obj is not None:
            new_node_attr = node_obj.copied_attr()
            pure_type = re.sub(r'^Caffe', '', node_obj.type)
            if getattr(node_obj, 'correspond_onnx_op', None) is not None:
                if isinstance(node_obj, OpHasWeights):
                    if node_obj.weights is None:
                        FATAL(
                            '[Parser]: Meets invalid weights for Node(%s) in convert_to_onnx!' % node_name)
                    new_weights = np.transpose(node_obj.weights, axes=type(
                        node_obj).perm_caffe_to_onnx(len(node_obj.weights.shape)))
                    new_node_attr.update({'weights': new_weights})
                if pure_type in ('BATCHNORM', 'BN'):
                    in_edges = graph.sorted_in_edges(node_name)
                    graph.remove_edges_from(in_edges[1:])
                    num_output = node_obj.weights.size
                    new_scale = node_obj.weights
                    new_shift = node_obj.biases
                    mean_value = np.zeros((num_output,), np.float32)
                    var_value = np.ones((num_output,), np.float32)
                    insert_constant(graph, node_name + '_gamma',
                                    new_scale, node_name, in_port=1)
                    insert_constant(graph, node_name + '_beta',
                                    new_shift, node_name, in_port=2)
                    insert_constant(graph, node_name + '_mean',
                                    mean_value, node_name, in_port=3)
                    insert_constant(graph, node_name + '_var',
                                    var_value, node_name, in_port=4)
                    new_node_attr.update({'epsilon': 0})
                elif pure_type == 'BATCHREINDEX':
                    in_edges = graph.sorted_in_edges(node_name, data=True)
                    if len(in_edges) == 2:
                        src, _, in_attr = in_edges[1]
                        new_in_attr = copy.deepcopy(in_attr)
                        if new_in_attr.get('tensor', None) is not None \
                                and new_in_attr['tensor'].value is not None \
                                and str(new_in_attr['tensor'].value.dtype) != 'int32':
                            new_in_attr['tensor'].value = new_in_attr['tensor'].value.astype(
                                np.int32)
                            graph.add_edge(src, node_name, **new_in_attr)
                            if src in graph._attr['input_tensors']:
                                graph._attr['input_tensors'][src].value = graph._attr['input_tensors'][src].value.astype(
                                    np.int32)
                elif pure_type == 'CROP':
                    in_edges = graph.sorted_in_edges(node_name)
                    graph.remove_edges_from(in_edges[1:])
                    starts = np.array(node_obj.offset, np.int64)
                    ends = starts + np.array(node_obj.size, np.int64)
                    insert_constant(graph, node_name + '_starts',
                                    starts, node_name, in_port=1)
                    insert_constant(graph, node_name + '_ends',
                                    ends, node_name, in_port=2)
                elif pure_type == 'ELTWISE':
                    if any([c != 1.0 for c in node_obj.coeff]):
                        in_edges = graph.sorted_in_edges(
                            node_name, data=True, keys=True)
                        if len(in_edges) == len(node_obj.coeff):
                            for i, (src, _, k, in_attr) in enumerate(in_edges):
                                if node_obj.coeff[i] != 1.0:
                                    mul = get_valid_node_name(
                                        graph, node_name + '_pre_mul' + str(i+1))
                                    graph.remove_edge(src, node_name, key=k)

                                    mul_in_attr = copy.deepcopy(in_attr)
                                    mul_in_attr.update({'dst_in_port': 0})
                                    graph.add_edge(src, mul, **mul_in_attr)

                                    elt_in_attr = copy.deepcopy(in_attr)
                                    elt_in_attr.update(
                                        {'src_out_port': 0, 'tensor': Tensor()})
                                    graph.add_edge(
                                        mul, node_name, **elt_in_attr)

                                    insert_constant(
                                        graph, mul+'_multiplier', np.array(node_obj.coeff[i], np.float32), mul, in_port=1)
                                    mul_attr = node_obj.copied_attr()
                                    mul_attr.update(
                                        {'name': mul, 'opset_version': 7})
                                    NodeWrap(graph, mul).replace_obj(
                                        'Mul', mul_attr)
                        else:
                            WARN(
                                '[Parser]: Converting ELTWISE Op %s meets error in convert_to_onnx.' % node_name)
                            continue
                elif pure_type == 'FLATTEN':
                    try:
                        input_shape = node_obj.get_input_shapes()[0]
                        dim = type(node_obj).cal_dim(
                            input_shape, node_obj.axis, node_obj.end_axis)
                        insert_constant(
                            graph, node_name + '_shape', np.array(dim, np.int64), node_name, in_port=1)
                    except Exception as e:
                        WARN('[Parser]: Converting Flatten Op %s meets error: %s!' % (
                            node_name, str(e)))
                        continue
                elif pure_type == 'INTERP':
                    insert_constant(graph, node_name + '_roi',
                                    np.array([], np.float32), node_name, in_port=1)
                    insert_constant(graph, node_name + '_scales',
                                    np.array([], np.float32), node_name, in_port=2)
                    input_shape = node_obj.get_input_shapes()[0]
                    insert_constant(graph, node_name + '_sizes',
                                    np.array(input_shape[:-2] + [node_obj.height, node_obj.width], np.int64), node_name, in_port=3)
                elif pure_type == 'LRN':
                    if 'k' in new_node_attr:
                        new_node_attr.update({'bias': new_node_attr['k']})
                    new_node_attr.update({'method': node_obj.norm_region})
                elif pure_type == 'PERMUTE':
                    new_node_attr.update({'perm': node_obj.order})
                elif pure_type == 'POOLING':
                    pass
                elif pure_type == 'PRELU':
                    input_shape = node_obj.get_input_shapes()[0]
                    slope = np.reshape(node_obj.weights, list(
                        node_obj.weights.shape) + [1] * (len(input_shape)-2))
                    insert_constant(graph, node_name + '_slope',
                                    slope, node_name, in_port=1)
                elif pure_type == 'REDUCTION':
                    input_shape = node_obj.get_input_shapes()[0]
                    axes = list(range(node_obj.axis, len(input_shape)))
                    if not FLOAT_EQUAL(node_obj.coeff, 1.0):
                        out_edges = graph.sorted_out_edges(
                            node_name, data=True)
                        mul = get_valid_node_name(
                            graph, node_name + '_post_mul')
                        for _, dst, out_attr in out_edges:
                            graph.remove_edge(node_name, dst)
                            graph.add_edge(mul, dst, **out_attr)
                        graph.add_edge(node_name, mul)
                        insert_constant(
                            graph, mul + '_multiplier', np.array(node_obj.coeff, np.float32), mul, in_port=1)
                        mul_attr = node_obj.copied_attr()
                        mul_attr.update({'name': mul, 'opset_version': 7})
                        NodeWrap(graph, mul).replace_obj('Mul', mul_attr)
                        if node_name in graph._attr['output_names']:
                            index = graph._attr['output_names'].index(
                                node_name)
                            graph._attr['output_names'][index] = mul
                    new_node_attr.update({'axes': axes})
                elif pure_type == 'RESHAPE':
                    insert_constant(
                        graph, node_name + '_shape', np.array(node_obj.shape, np.int64), node_name, in_port=1)
                elif pure_type == 'ROIPOOLING':
                    new_node_attr.update(
                        {'pooled_shape': [node_obj.pooled_h, node_obj.pooled_w]})
                elif pure_type == 'THRESHOLD':
                    try:
                        input_shape = node_obj.get_input_shapes()[0]
                        # TODO!
                        new_node_attr.update({'broadcast': True})
                        insert_constant(graph, node_name + '_threshold',
                                        np.array([node_obj.threshold]), node_name, in_port=1)
                    except Exception as e:
                        WARN('[Parser]: Converting Threshold Op %s meets error: %s!' % (
                            node_name, str(e)))
                        continue
                elif pure_type == 'UPSAMPLEDARKNET':
                    new_node_attr.update({'mode': 'linear'})
                    insert_constant(graph, node_name + '_roi',
                                    np.array([], np.float32), node_name, in_port=1)
                    scales = np.array([1, 1] + node_obj.strides, np.float32)
                    insert_constant(graph, node_name + '_scale',
                                    scales, node_name, in_port=2)
                new_node_attr.update(
                    {'opset_version': node_obj.correspond_onnx_op['version']})
                NodeWrap(graph, node_name).replace_obj(
                    node_obj.correspond_onnx_op['type'], new_node_attr)
            else:
                WARN('[Parser]: Caffe Op %s (%s) cannot be converted to Onnx' % (
                    pure_type, node_name))
        else:
            WARN(
                '[Parser]: Meets invalid Caffe op for Node(%s) in convert_to_onnx!' % node_name)
