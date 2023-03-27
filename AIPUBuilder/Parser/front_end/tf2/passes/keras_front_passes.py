# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import re
import copy
import torch
from ....ops.op import BaseActivationOp, Op, OpHasAxis, OpHasWeights, KerasOp, KerasGlobalPoolingOp, KerasNeedBroadcast, Tf2Op
from ....graph.node_wrap import NodeWrap
from ....graph.graph_algo import get_valid_node_name, clear_redundant_nodes
from ....graph.pattern_match import matched_patterns, single_node_matcher
from ...onnx.passes.common_passes import insert_constant, insert_reshape, insert_reshape_after, \
    insert_slice, insert_tile, insert_transpose, insert_transpose_after, remove_useless_op
from ....common.defs import Tensor, FLOAT_EQUAL
from ....common.utils import extend_lists
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL


def convert_activations(graph):
    matches = single_node_matcher(graph, 'TfKerasActivation')
    for m in matches:
        act = m['target']
        act_obj = NodeWrap(graph, act)['object']
        if act_obj is None:
            ERROR(
                '[Parser]: Meets invalid TfKerasActivation Op (%s) in convert_activations!' % act)
            continue
        act_name = act_obj.activations
        onnx_op_dict = BaseActivationOp.activation_to_onnx_op(act_name)
        onnx_op_type = onnx_op_dict.get('type', None)
        opset_version = onnx_op_dict.get('opset_version', None)
        node_attr = act_obj.copied_attr()
        if onnx_op_type is None or opset_version is None:
            tf_activations_optype_map = {
                # activations: (onnx op type, onnx opset version, attr_dict)
                'ABS': ('Abs', 6, {}),
                'EXP': ('Exp', 13, {}),
                'REDUCE_LOGSUMEXP': ('ReduceLogSumExp', 13, {'keepdims': 0}),
            }
            if act_name in tf_activations_optype_map:
                onnx_op_type, opset_version, attr_dict = tf_activations_optype_map[act_name]
                node_attr.update(attr_dict)
                node_attr.update({'type': onnx_op_type, 'opset_version': opset_version})
            else:
                WARN('[Parser]: Meet unsupported activation (%s) in TfKerasActivation Op (%s) in convert_activations!' %
                     (act_name, act))
                continue
        else:
            node_attr.update(onnx_op_dict)
        NodeWrap(graph, act).replace_obj(onnx_op_type, node_attr)


def convert_batchnorm(graph):
    matches = single_node_matcher(graph, 'TfKerasBatchNormalization')
    for m in matches:
        batchnorm = m['target']
        batchnorm_obj = NodeWrap(graph, batchnorm)['object']
        if batchnorm_obj is None:
            ERROR(
                '[Parser]: Meets invalid TfKerasBatchNormalization Op (%s) in convert_batchnorm!' % batchnorm)
            continue
        input_shapes = batchnorm_obj.get_input_shapes()
        in_edges = graph.sorted_in_edges(batchnorm, data=True)
        if len(in_edges) < 1 \
                or len(input_shapes) < 1 \
                or input_shapes[0] is None \
                or None in input_shapes[0]:
            continue

        m_v_index = int(batchnorm_obj.scale)+int(batchnorm_obj.center)
        mean_value = batchnorm_obj.weights_list[m_v_index] if m_v_index < len(
            batchnorm_obj.weights_list) else np.zeros_like(batchnorm_obj.weights, np.float32)
        var_value = batchnorm_obj.weights_list[m_v_index+1] if m_v_index < (len(
            batchnorm_obj.weights_list)-1) else np.ones_like(batchnorm_obj.biases, np.float32)

        graph.remove_edges_from(in_edges[1:])
        insert_constant(graph, batchnorm + '_scale',
                        batchnorm_obj.weights, batchnorm, in_port=1, data_format='NHWC')
        insert_constant(graph, batchnorm + '_bias',
                        batchnorm_obj.biases, batchnorm, in_port=2, data_format='NHWC')
        insert_constant(graph, batchnorm + '_mean',
                        mean_value, batchnorm, in_port=3, data_format='NHWC')
        insert_constant(graph, batchnorm + '_var',
                        var_value, batchnorm, in_port=4, data_format='NHWC')
        if batchnorm_obj.axis != len(input_shapes[0]) - 1:
            src, _, in_attr = in_edges[0]
            pre_perm = [idx for idx in range(len(input_shapes[0])) if idx != batchnorm_obj.axis] + [batchnorm_obj.axis]
            insert_transpose(graph, src, batchnorm, in_attr, pre_perm)
            post_perm = Op.cal_inverse_perm(pre_perm)
            post_trans = insert_transpose_after(graph, batchnorm, post_perm)
            if batchnorm in graph._attr['output_names']:
                index = graph._attr['output_names'].index(batchnorm)
                graph._attr['output_names'][index] = post_trans
        node_attr = batchnorm_obj.copied_attr()
        node_attr.update({'opset_version': 14, 'data_format': 'NHWC'})
        NodeWrap(graph, batchnorm).replace_obj('BatchNormalization', node_attr)


def convert_bidirectional(graph):
    matched = False
    matches = single_node_matcher(graph, 'TfKerasBidirectional')
    for m in matches:
        bidir = m['target']
        bidir_obj = NodeWrap(graph, bidir)['object']
        in_edges = graph.sorted_in_edges(bidir, data=True)
        out_edges = graph.sorted_out_edges(bidir, data=True)
        if bidir_obj is None or len(in_edges) < 1:
            ERROR('[Parser]: Meets invalid Op (%s) in convert_bidirectional!' % bidir)
            continue
        forward_node_obj = bidir_obj.create_node('forward')
        if forward_node_obj is None:
            ERROR('[Parser]: Meet invalid layer of Op (%s) in convert_bidirectional!' % bidir)
            continue
        backward_node_obj = bidir_obj.create_node('backward')
        if backward_node_obj is None:
            ERROR('[Parser]: Meet invalid backward_layer of Op (%s) in convert_bidirectional!' % bidir)
            continue
        in_shapes = bidir_obj.get_input_shapes()
        if len(in_shapes) < 1 or in_shapes[0] is None or len(in_shapes[0]) != 3:
            ERROR('[Parser]: Meets invalid input shapes of Op (%s) in convert_bidirectional!' % bidir)
            continue
        matched = True
        time_dim = 0 if forward_node_obj.time_major else 1
        if time_dim == 0:
            time_steps, batch_size, _ = in_shapes[0]
        else:
            batch_size, time_steps, _ = in_shapes[0]
        forward_hidden_size = forward_node_obj.units
        backward_hidden_size = backward_node_obj.units
        forward_node = forward_node_obj.name
        backward_node = backward_node_obj.name
        # TODO: Consider initial state inputs
        src, _, in_attr = in_edges[0]
        graph.remove_edges_from(in_edges)
        graph.add_edge(src, forward_node, **in_attr)
        graph.add_edge(src, backward_node, **in_attr)
        forward_type_is_lstm = (forward_node_obj.type == 'TfKerasLSTM')
        backward_type_is_lstm = (backward_node_obj.type == 'TfKerasLSTM')
        return_sequences = forward_node_obj.return_sequences
        backward_rev = backward_node
        if return_sequences:
            backward_rev = get_valid_node_name(graph, backward_node + '_rev')
            graph.add_edge(backward_node, backward_rev)
            seq_len = np.array([time_steps] * batch_size, np.int32)
            insert_constant(graph, backward_rev + '_seq_len', seq_len, backward_rev, in_port=1)
            rev_seq_attr = {'name': backward_rev, 'time_axis': time_dim, 'batch_axis': 1-time_dim,
                            'opset_version': 10}
            NodeWrap(graph, backward_rev).replace_obj('ReverseSequence', rev_seq_attr)
        # output_info_list save new output node and new src_out_port
        output_info_list = []
        merge_mode = bidir_obj.merge_mode
        mode_type_map = {'sum': 'Add', 'mul': 'Mul', 'concat': 'Concat', 'ave': 'Mean'}
        if merge_mode in mode_type_map:
            forward_out_attr = {'src_out_port': 0, 'dst_in_port': 0}
            graph.add_edge(forward_node, bidir, **forward_out_attr)
            backward_out_attr = {'src_out_port': 0, 'dst_in_port': 1}
            graph.add_edge(backward_rev, bidir, **backward_out_attr)

            opset_version = 13
            merge_node_attr = bidir_obj.copied_attr()
            merge_node_attr.update({'opset_version': opset_version})
            if merge_mode == 'concat':
                merge_node_attr.update({'axis': -1})
            NodeWrap(graph, bidir).replace_obj(mode_type_map[merge_mode], merge_node_attr)
            output_info_list.append([bidir, 0])
        else:  # merge_mode is 'none'
            concat_y = get_valid_node_name(graph, bidir + '_concat_y')
            graph.add_edge(forward_node, concat_y, **{'dst_in_port': 0})
            graph.add_edge(backward_rev, concat_y, **{'dst_in_port': 1})
            graph.add_edge(concat_y, bidir)
            NodeWrap(graph, concat_y).replace_obj('Concat', {'name': concat_y, 'axis': -1, 'opset_version': 13})
            split_y_attr = bidir_obj.copied_attr()
            split_y_attr.update({'axis': -1, 'split': [forward_hidden_size, backward_hidden_size],
                                 'opset_version': 11})
            NodeWrap(graph, bidir).replace_obj('Split', split_y_attr)
            output_info_list.extend([[bidir, 0], [bidir, 1]])
        split_state = None
        return_state = forward_node_obj.return_state
        if return_state:
            concat_state = get_valid_node_name(graph, bidir + '_concat_state')
            split_state = get_valid_node_name(graph, bidir + '_split_state')
            dst_in_port = 0
            split = []
            for direction in ('forward', 'backward'):
                from_node, split_size, is_lstm = (forward_node, forward_hidden_size, forward_type_is_lstm) if direction == 'forward' \
                    else (backward_node, backward_hidden_size, backward_type_is_lstm)
                for out_port in (1, 2):
                    if out_port == 2 and not is_lstm:
                        continue
                    graph.add_edge(from_node, concat_state, **{'src_out_port': out_port, 'dst_in_port': dst_in_port})
                    output_info_list.append([split_state, dst_in_port])
                    dst_in_port = dst_in_port + 1
                    split.append(split_size)
            graph.add_edge(concat_state, split_state)
            NodeWrap(graph, concat_state).replace_obj(
                'Concat', {'name': concat_state, 'axis': -1, 'opset_version': 13})
            split_state_attr = {'name': split_state, 'axis': -1,
                                'split': split, 'opset_version': 11}
            NodeWrap(graph, split_state).replace_obj('Split', split_state_attr)
        for _, dst, out_attr in out_edges:
            graph.remove_edge(bidir, dst)
            src_out_port = out_attr['src_out_port']
            if src_out_port >= len(output_info_list):
                ERROR('[Parser]: Meet invalid src_out_port (%d) of Op (%s) in convert_bidirectional!' % (src_out_port, bidir))
                continue
            new_src_node, new_src_out_port = output_info_list[src_out_port]
            out_attr.update({'src_out_port': new_src_out_port})
            graph.add_edge(new_src_node, dst, **out_attr)
        if bidir in graph._attr['output_names']:
            index = graph._attr['output_names'].index(bidir)
            # For GRU, if return_sequences is False, the first output is state output. No matter
            # return_state is True or False, there is only 1 output in tf2 GRU.
            if split_state is not None \
                    and (return_sequences or forward_type_is_lstm or backward_type_is_lstm):
                graph._attr['output_names'].insert(index + 1, split_state)
        if matched:
            clear_redundant_nodes(graph)


def convert_centercrop(graph):
    '''
    Convert TfKerasCenterCrop op to onnx Slice.
    '''
    matches = single_node_matcher(graph, 'TfKerasCenterCrop')
    for m in matches:
        crop = m['target']
        crop_obj = NodeWrap(graph, crop)['object']
        in_edges = graph.sorted_in_edges(crop, data=True)
        if crop_obj is None or len(in_edges) < 1 or len(crop_obj.get_input_shapes()) < 1:
            ERROR('[Parser]: Meets invalid TfKerasCenterCrop Op (%s) in convert_centercrop!' % crop)
            continue
        input_shape = crop_obj.get_input_shapes()[0]
        if input_shape is None or len(input_shape) < 3 or None in input_shape:
            continue
        input_rank = len(input_shape)
        input_height = input_shape[-3]
        input_width = input_shape[-2]
        target_height = crop_obj.height
        target_width = crop_obj.width
        diff_height = input_height - target_height
        diff_width = input_width - target_width
        new_node_attr = crop_obj.copied_attr()
        if diff_height < 0 or diff_width < 0:
            ERROR('[Parser]: The crop height(%d)/width(%d) should not be greater than input height(%d)/width(%d)!' %
                  (target_height, target_width, input_height, input_width))
            continue
        starts = [0] * (input_rank - 3) + [int(diff_height / 2), int(diff_width / 2), 0]
        sizes = input_shape[:-3] + [target_height, target_width] + input_shape[-1:]
        ends = (np.array(starts) + np.array(sizes)).tolist()
        new_node_attr.update({'starts': starts, 'ends': ends, 'opset_version': 1})
        NodeWrap(graph, crop).replace_obj('Slice', new_node_attr)


def convert_global_pooling(graph):
    op_types = KerasGlobalPoolingOp.get_concrete_subclass_names()
    matches = single_node_matcher(graph, op_types)
    for m in matches:
        global_pool = m['target']
        global_pool_obj = NodeWrap(graph, global_pool)['object']
        if global_pool_obj is None:
            ERROR(
                '[Parser]: Meets invalid Op (%s) in convert_global_pooling!' % global_pool)
            continue
        if getattr(global_pool_obj, 'correspond_onnx_op', None) is None:
            continue
        in_edges = graph.sorted_in_edges(global_pool)
        graph.remove_edges_from(in_edges[1:])
        node_data_format = 'NCHW' if global_pool_obj.data_format.startswith('NC') else 'NHWC'
        if not global_pool_obj.keepdims:
            input_shapes = global_pool_obj.get_input_shapes()
            if len(input_shapes) < 1 \
                    or input_shapes[0] is None \
                    or len(input_shapes[0]) not in (3, 4, 5) \
                    or None in input_shapes[0]:
                continue
            if node_data_format == 'NCHW':
                reshape_dim = input_shapes[0][:2]
                old_dim = input_shapes[0][:2] + [1] * (len(input_shapes[0]) - 2)
            else:
                reshape_dim = [input_shapes[0][0], input_shapes[0][-1]]
                old_dim = [input_shapes[0][0]] + [1] * (len(input_shapes[0]) - 2) + [input_shapes[0][-1]]
            post_reshape = insert_reshape_after(graph, global_pool, reshape_dim, old_dim)
            if global_pool in graph._attr['output_names']:
                index = graph._attr['output_names'].index(global_pool)
                graph._attr['output_names'][index] = post_reshape
        new_node_attr = global_pool_obj.copied_attr()
        new_node_attr.update(
            {'opset_version': global_pool_obj.correspond_onnx_op['version'],
             'data_format': node_data_format})
        NodeWrap(graph, global_pool).replace_obj(
            global_pool_obj.correspond_onnx_op['type'], new_node_attr)


def convert_gru_lstm(graph):
    # TODO: Consider mask and initial_state
    matches = extend_lists([single_node_matcher(graph, op_type)
                            for op_type in ['TfKerasGRU', 'TfKerasLSTM']])
    for m in matches:
        rnn = m['target']
        rnn_obj = NodeWrap(graph, rnn)['object']
        in_edges = graph.sorted_in_edges(rnn, data=True)
        out_edges = graph.sorted_out_edges(rnn, data=True)
        if rnn_obj is None \
                or len(rnn_obj.get_input_shapes()) < 1 \
                or rnn_obj.get_input_shapes()[0] is None \
                or len(rnn_obj.get_input_shapes()[0]) != 3 \
                or len(in_edges) < 1:
            ERROR(
                '[Parser]: Meets invalid Op (%s) in convert_gru_lstm!' % rnn)
            continue
        rnn_type = rnn_obj.type
        hidden_size = rnn_obj.units
        input_shape = rnn_obj.get_input_shapes()[0]
        if rnn_obj.time_major:
            seq_length, batch_size, _ = input_shape
            seq_output_shape = [seq_length, batch_size, hidden_size]
            seq_output_shape_with_dir = [
                seq_length, 1, batch_size, hidden_size]
            state_output_shape_with_dir = [1, batch_size, hidden_size]
        else:
            batch_size, seq_length, _ = input_shape
            seq_output_shape = [batch_size, seq_length, hidden_size]
            seq_output_shape_with_dir = [
                batch_size, seq_length, 1, hidden_size]
            state_output_shape_with_dir = [batch_size, 1, hidden_size]
        state_output_shape = [batch_size, hidden_size]
        # weights and biases
        kernel = rnn_obj.kernel
        recurrent_kernel = rnn_obj.recurrent_kernel
        bias = rnn_obj.bias
        B_value = None
        if rnn_type == 'TfKerasGRU':
            W_value = np.expand_dims(np.transpose(kernel), axis=0)
            R_value = np.expand_dims(np.transpose(recurrent_kernel), axis=0)
            if bias is not None:
                if bias.size == 3 * hidden_size:
                    bias = np.concatenate([bias, np.zeros_like(bias)])
                if bias.size == 6 * hidden_size:
                    B_value = np.reshape(bias, [1, -1])
        else:
            # Note that tf kernel_w and bias are in format ifco, while onnx is iofc.
            input_w, forget_w, cell_w, output_w = np.split(kernel, 4, axis=1)
            input_r, forget_r, cell_r, output_r = np.split(
                recurrent_kernel, 4, axis=1)
            W_value = np.stack([np.transpose(np.concatenate(
                [input_w, output_w, forget_w, cell_w], axis=1))])
            R_value = np.stack([np.transpose(np.concatenate(
                [input_r, output_r, forget_r, cell_r], axis=1))])
            if bias is not None:
                input_wb, forget_wb, cell_wb, output_wb = np.split(
                    bias, 4, axis=0)
                bias_w = np.concatenate(
                    [input_wb, output_wb, forget_wb, cell_wb])
                bias_r = np.zeros_like(bias_w)
                B_value = np.stack([np.concatenate([bias_w, bias_r])])

        graph.remove_edges_from(in_edges[1:])

        if rnn_obj.go_backwards:
            # tf2 GRU/LSTM reverse is different with onnx GRU/LSTM reverse:
            # tf2 return the reversed sequence while onnx doesn't. Thus, convert
            # to forward in this pass so that no need to reverse output twice.
            inp, _, inp_in_attr = in_edges[0]
            rev = get_valid_node_name(graph, rnn + '_reverse')
            graph.remove_edge(inp, rnn)
            graph.add_edge(inp, rev, **inp_in_attr)
            rnn_in_attr = copy.deepcopy(inp_in_attr)
            if inp_in_attr['tensor'] is not None and inp_in_attr['tensor'].value is not None:
                rnn_in_attr['tensor'].value = np.flip(inp_in_attr['tensor'].value, 1)
            rnn_in_attr.update({'src_out_port': 0})
            graph.add_edge(rev, rnn, **rnn_in_attr)
            seq_len = np.array([seq_length] * batch_size, np.int32)
            insert_constant(graph, rev + '_seq_len', seq_len, rev, in_port=1)
            time_axis = 0 if rnn_obj.time_major else 1
            rev_seq_attr = {'name': rev, 'time_axis': time_axis, 'batch_axis': 1-time_axis,
                            'opset_version': 10}
            NodeWrap(graph, rev).replace_obj('ReverseSequence', rev_seq_attr)

        insert_constant(graph, get_valid_node_name(graph, rnn + '_W'),
                        W_value, rnn, in_port=1)
        insert_constant(graph, get_valid_node_name(graph, rnn + '_R'),
                        R_value, rnn, in_port=2)
        if B_value is not None:
            insert_constant(graph, get_valid_node_name(graph, rnn + '_B'),
                            B_value, rnn, in_port=3)
        if not rnn_obj.return_sequences:
            for _, dst, out_attr in out_edges:
                if out_attr['src_out_port'] == 0:
                    graph.remove_edge(rnn, dst)
                    new_out_attr = copy.deepcopy(out_attr)
                    new_out_attr.update({'src_out_port': 1})
                    graph.add_edge(rnn, dst, **new_out_attr)
        Y_reshape_after = insert_reshape_after(
            graph, rnn, seq_output_shape, seq_output_shape_with_dir, out_port=0)
        Y_h_reshape_after = insert_reshape_after(graph, rnn, state_output_shape,
                                                 state_output_shape_with_dir, out_port=1)
        Y_c_reshape_after = None
        if rnn_type == 'TfKerasLSTM':
            Y_c_reshape_after = insert_reshape_after(graph, rnn, state_output_shape,
                                                     state_output_shape_with_dir, out_port=2)
        if rnn in graph._attr['output_names']:
            index = graph._attr['output_names'].index(rnn)
            if rnn_obj.return_sequences and rnn_obj.return_state:
                graph._attr['output_names'][index] = Y_reshape_after
                graph._attr['output_names'].insert(index, Y_h_reshape_after)
                if Y_c_reshape_after is not None:
                    graph._attr['output_names'].insert(
                        index + 1, Y_c_reshape_after)
            elif not rnn_obj.return_sequences:
                graph._attr['output_names'][index] = Y_h_reshape_after
                if rnn_obj.return_state and Y_c_reshape_after is not None:
                    graph._attr['output_names'].insert(
                        index, Y_c_reshape_after)
            else:
                graph._attr['output_names'][index] = Y_reshape_after

        rnn_attr = rnn_obj.copied_attr()
        rnn_attr.update({'direction': 'forward',
                         'hidden_size': hidden_size,
                         'layout': 0 if rnn_obj.time_major else 1,
                         'opset_version': 14})
        if rnn_type == 'TfKerasGRU':
            rnn_attr.update({'activations': [rnn_obj.recurrent_activation.upper(), rnn_obj.activation.upper()],
                             'linear_before_reset': 1 if rnn_obj.reset_after else 0})
            dst_onnx_type = 'GRU'
        else:
            rnn_attr.update({'activations': [rnn_obj.recurrent_activation.upper(),
                                             rnn_obj.activation.upper(), rnn_obj.activation.upper()]})
            dst_onnx_type = 'LSTM'
        NodeWrap(graph, rnn).replace_obj(dst_onnx_type, rnn_attr)


def convert_normalization(graph):
    '''
    Convert TfKerasNormalization op to sub(x-mean) + mul(*1/sqrt(var)); if mean and var are None or
    mean is 0 and var is 1, convert to Identity.
    '''
    matches = single_node_matcher(graph, 'TfKerasNormalization')
    for m in matches:
        norm = m['target']
        norm_obj = NodeWrap(graph, norm)['object']
        in_edges = graph.sorted_in_edges(norm, data=True)
        if norm_obj is None or len(in_edges) < 1:
            ERROR('[Parser]: Meets invalid Op (%s) in convert_normalization!' % norm)
            continue
        new_node_attr = norm_obj.copied_attr()
        mean = norm_obj.mean
        variance = norm_obj.variance
        if (mean is None and variance is None) or (FLOAT_EQUAL(mean, 0) and FLOAT_EQUAL(variance, 1)):
            new_node_attr.update({'opset_version': 13})
            NodeWrap(graph, norm).replace_obj('Identity', new_node_attr)
        else:
            if mean is None:
                mean = np.zeros_like(variance, np.float32)
            if variance is None:
                variance = np.ones_like(mean, np.float32)
            sub = get_valid_node_name(graph, norm + '_sub')
            src, _, in_attr = in_edges[0]
            graph.remove_edges_from(in_edges)
            graph.add_edge(src, sub, **in_attr)
            insert_constant(graph, sub + '_mean', mean, sub, in_port=1)
            sub_out_attr = copy.deepcopy(in_attr)
            sub_out_attr.update({'src_out_port': 0})
            if in_attr['tensor'] is not None and in_attr['tensor'].value is not None:
                sub_out_attr['tensor'].value = in_attr['tensor'].value - mean
            graph.add_edge(sub, norm, **sub_out_attr)
            mul_value = np.array(1 / np.sqrt(variance))
            insert_constant(graph, norm + '_mul', mul_value, norm, in_port=1)

            NodeWrap(graph, sub).replace_obj('Sub', {'name': sub, 'opset_version': 13})
            new_node_attr.update({'opset_version': 13})
            NodeWrap(graph, norm).replace_obj('Mul', new_node_attr)


def convert_relu(graph):
    matched = False
    matches = single_node_matcher(graph, 'TfKerasReLU')
    for m in matches:
        relu = m['target']
        relu_obj = NodeWrap(graph, relu)['object']
        in_edges = graph.sorted_in_edges(relu, data=True)
        out_edges = graph.sorted_out_edges(relu, data=True)
        if relu_obj is None or len(in_edges) < 1:
            ERROR('[Parser]: Meets invalid Op (%s) in convert_relu!' % relu)
            continue
        matched = True
        threshold = np.array(relu_obj.threshold)
        negative_slope = np.array(relu_obj.negative_slope)
        negative_slope_is_zero = FLOAT_EQUAL(negative_slope, 0)
        max_value = relu_obj.max_value
        new_node_attr = relu_obj.copied_attr()

        if max_value is None and FLOAT_EQUAL(threshold, 0):
            if negative_slope_is_zero:
                new_node_type = 'Relu'
            else:
                new_node_type = 'LeakyRelu'
                new_node_attr.update({'alpha': negative_slope})
            new_node_attr.update({'opset_version': 6})
            NodeWrap(graph, relu).replace_obj(new_node_type, new_node_attr)
            continue

        graph.remove_edges_from(in_edges)
        src, _, in_attr = in_edges[0]

        # Step 1: For x > threshold, convert to thresholded_relu+min(which will make x<=threshold part becomes 0)
        thresholded_relu = get_valid_node_name(graph, relu + '_thresholded_relu')
        thresholded_relu_in_attr = copy.deepcopy(in_attr)
        graph.add_edge(src, thresholded_relu, **thresholded_relu_in_attr)
        thres_out_tensor = None if in_attr['tensor'].value is None else \
            torch.nn.Threshold(threshold.item(), 0)(torch.tensor(in_attr['tensor'].value)).detach().numpy()
        add_pos_in_tensor = thres_out_tensor
        pos_last_node = thresholded_relu
        if max_value is not None:
            min_after_thres = get_valid_node_name(graph, thresholded_relu + '_min')
            min_in_attr = {'tensor': Tensor(value=thres_out_tensor)}
            graph.add_edge(thresholded_relu, min_after_thres, **min_in_attr)
            insert_constant(graph, min_after_thres + '_operand', np.array(max_value),
                            min_after_thres, in_port=1, data_format='NHWC')
            add_pos_in_tensor = None if thres_out_tensor is None else \
                np.maximum(thres_out_tensor, max_value)

            min_attr = {'name': min_after_thres, 'opset_version': 12}
            NodeWrap(graph, min_after_thres).replace_obj('Min', min_attr)
            pos_last_node = min_after_thres

        thresholded_relu_attr = {'name': thresholded_relu, 'opset_version': 10, 'alpha': threshold.item()}
        NodeWrap(graph, thresholded_relu).replace_obj('ThresholdedRelu', thresholded_relu_attr)
        if negative_slope_is_zero:
            if relu in graph._attr['output_names']:
                index = graph._attr['output_names'].index(relu)
                graph._attr['output_names'][index] = pos_last_node
            for _, dst, out_attr in out_edges:
                graph.remove_edge(relu, dst)
                graph.add_edge(pos_last_node, dst, **out_attr)
            continue

        # Step 2: For x < threshold, convert to sub+min+mul(which will make x>=threshold part becomes 0)
        sub = get_valid_node_name(graph, relu + '_sub')
        graph.add_edge(src, sub, **in_attr)
        insert_constant(graph, relu + '_sub_operand', threshold, sub, in_port=1, data_format='NHWC')
        min_after_sub = get_valid_node_name(graph, sub + '_min')
        min_after_sub_in_tensor = None if in_attr['tensor'].value is None else (in_attr['tensor'].value - threshold)
        min_after_sub_in_attr = {'tensor': Tensor(value=min_after_sub_in_tensor)}
        graph.add_edge(sub, min_after_sub, **min_after_sub_in_attr)
        insert_constant(graph, min_after_sub + '_zero', np.array(0.), min_after_sub, in_port=1, data_format='NHWC')
        mul_after_min = get_valid_node_name(graph, relu + '_mul')
        mul_after_min_in_tensor = None if min_after_sub_in_tensor is None else np.minimum(min_after_sub_in_tensor, 0)
        mul_after_min_in_attr = {'tensor': Tensor(value=mul_after_min_in_tensor)}
        graph.add_edge(min_after_sub, mul_after_min, **mul_after_min_in_attr)
        insert_constant(graph, mul_after_min + '_mul_operand', negative_slope,
                        mul_after_min, in_port=1, data_format='NHWC')
        mul_after_min_out_tensor = None if mul_after_min_in_tensor is None else \
            mul_after_min_in_tensor * negative_slope

        sub_attr = {'name': sub, 'opset_version': 13}
        NodeWrap(graph, sub).replace_obj('Sub', sub_attr)
        min_after_sub_attr = {'name': min_after_sub, 'opset_version': 13}
        NodeWrap(graph, min_after_sub).replace_obj('Min', min_after_sub_attr)
        mul_after_min_attr = {'name': mul_after_min, 'opset_version': 13}
        NodeWrap(graph, mul_after_min).replace_obj('Mul', mul_after_min_attr)

        # Step 3: Add above 2 steps to get the final output if negative_slope is not 0
        add_neg_in_attr = {'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(value=mul_after_min_out_tensor)}
        graph.add_edge(mul_after_min, relu, **add_neg_in_attr)
        add_pos_in_attr = {'src_out_port': 0, 'dst_in_port': 1, 'tensor': Tensor(value=add_pos_in_tensor)}
        graph.add_edge(pos_last_node, relu, **add_pos_in_attr)

        new_node_attr.update({'opset_version': 13})
        NodeWrap(graph, relu).replace_obj('Add', new_node_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_rescaling(graph):
    matches = single_node_matcher(graph, 'TfKerasRescaling')
    for m in matches:
        rescaling = m['target']
        rescaling_obj = NodeWrap(graph, rescaling)['object']
        in_edges = graph.sorted_in_edges(rescaling, data=True)
        if rescaling_obj is None or len(in_edges) < 1:
            ERROR('[Parser]: Meets invalid Op (%s) in convert_rescaling!' % rescaling)
            continue
        mul = get_valid_node_name(graph, rescaling + '_mul')
        src, _, in_attr = in_edges[0]
        graph.remove_edges_from(in_edges)
        graph.add_edge(src, mul, **in_attr)
        insert_constant(graph, mul + '_scale', np.array(rescaling_obj.scale, np.float32), mul, in_port=1)

        mul_out_attr = copy.deepcopy(in_attr)
        mul_out_attr.update({'src_out_port': 0})
        if in_attr['tensor'] is not None and in_attr['tensor'].value is not None:
            mul_out_attr['tensor'].value = in_attr['tensor'].value * rescaling_obj.scale
        graph.add_edge(mul, rescaling, **mul_out_attr)
        insert_constant(graph, rescaling + '_offset', np.array(rescaling_obj.offset, np.float32), rescaling, in_port=1)

        NodeWrap(graph, mul).replace_obj('Mul', {'name': mul, 'opset_version': 13})
        add_attr = rescaling_obj.copied_attr()
        add_attr.update({'opset_version': 13})
        NodeWrap(graph, rescaling).replace_obj('Add', add_attr)


def convert_resizing(graph):
    matches = single_node_matcher(graph, ['TfKerasResizing', 'TfKerasUpSampling1D',
                                  'TfKerasUpSampling2D', 'TfKerasUpSampling3D'])
    for m in matches:
        resize = m['target']
        resize_obj = NodeWrap(graph, resize)['object']
        in_edges = graph.sorted_in_edges(resize, data=True)
        if resize_obj is None or len(in_edges) < 1:
            ERROR('[Parser]: Meets invalid Op (%s) in convert_resizing!' % resize)
            continue
        if resize_obj.interpolation not in ('bilinear', 'nearest', 'bicubic'):
            WARN('[Parser]: Meet unsupported interpolation method (%s) in convert_resizing!' % resize_obj.interpolation)
            continue
        mode = 'linear' if resize_obj.interpolation == 'bilinear' else (
            'cubic' if resize_obj.interpolation == 'bicubic' else 'nearest')
        input_shapes = resize_obj.get_input_shapes()
        if len(input_shapes) < 1 or input_shapes[0] is None \
                or None in input_shapes[0]:
            continue
        if resize_obj.type == 'TfKerasResizing':
            if len(input_shapes[0]) not in (3, 4):
                continue
            is_4d_input = len(input_shapes[0]) == 4
            target_shape = [resize_obj.height, resize_obj.width, input_shapes[0][-1]]
            if is_4d_input:
                input_height, input_width = input_shapes[0][1:3]
                target_shape = [input_shapes[0][0]] + target_shape
            else:
                input_height, input_width = input_shapes[0][0:2]
            if resize_obj.crop_to_aspect_ratio:
                crop_height = int(resize_obj.height * input_width / resize_obj.width)
                crop_height = min(crop_height, input_height)
                crop_width = int(resize_obj.width * input_height / resize_obj.height)
                crop_width = min(crop_width, input_width)
                crop_hstart = int((input_height - crop_height) / 2)
                crop_wstart = int((input_width - crop_width) / 2)
                if crop_hstart != 0 or crop_wstart != 0 \
                        or crop_height != input_height or crop_width != input_width:
                    if is_4d_input:
                        begin = [0, crop_hstart, crop_wstart, 0]
                        size = [input_shapes[0][0], crop_height, crop_width, input_shapes[0][-1]]
                    else:
                        begin = [crop_hstart, crop_wstart, 0]
                        size = [crop_height, crop_width, input_shapes[0][-1]]
                    src, _, in_attr = in_edges[0]
                    insert_slice(graph, src, resize, in_attr, begin, size)
        else:
            upsample_size = resize_obj.size
            if isinstance(upsample_size, int):
                upsample_size = [upsample_size]
            if len(input_shapes[0]) not in (3, 4, 5) and len(upsample_size) != len(input_shapes[0]) - 2:
                continue
            if resize_obj.data_format.startswith('NC'):
                target_shape = input_shapes[0][:2] + (np.array(input_shapes[0][2:]) * upsample_size).tolist()
            else:
                target_shape = [input_shapes[0][0]] + \
                    (np.array(input_shapes[0][1:-1]) * upsample_size).tolist() + \
                    [input_shapes[0][-1]]
        graph.remove_edges_from(in_edges[1:])
        # insert constant empty roi
        insert_constant(graph, resize + '_roi',
                        np.array([], np.float32), resize, in_port=1)
        # insert constant empty scale
        insert_constant(graph, resize + '_scale',
                        np.array([], np.float32), resize, in_port=2)
        # insert constant size
        insert_constant(graph, resize + '_size',
                        np.array(target_shape, np.int64), resize, in_port=3)
        resize_attr = resize_obj.copied_attr()
        resize_attr.update({'mode': mode, 'nearest_mode': 'round_prefer_ceil', 'opset_version': 13})
        NodeWrap(graph, resize).replace_obj('Resize', resize_attr)


def convert_dense(graph):
    matches = single_node_matcher(graph, 'TfKerasDense')
    for m in matches:
        dense = m['target']
        dense_obj = NodeWrap(graph, dense)['object']
        in_edges = graph.sorted_in_edges(dense, data=True)
        out_edges = graph.sorted_out_edges(dense, data=True)
        kernel_value = dense_obj.weights
        biases_value = dense_obj.biases

        if dense_obj is None:
            ERROR('[Parser]: Meets invalid Op (%s) in convert_dense!' % dense)
            continue

        in_shapes = dense_obj.get_input_shapes()[0]

        if len(in_edges) != 1 \
                or len(kernel_value.shape) != 2 \
                or len(in_shapes) < 2 \
                or in_shapes[-1] != kernel_value.shape[0]:
            ERROR('[Parser]: Meets invalid Op (%s) in convert_dense!' % dense)
            continue

        inp_value = dense_obj.get_input_tensors()[0]
        src, _, in_attr = in_edges[0]
        if biases_value is None:
            biases_value = np.zeros(dense_obj.units).astype(np.float32)

        if len(inp_value.shape) > 2:
            # calculate the shape
            old_shape = list(inp_value.shape)
            new_shape = [-1, inp_value.shape[-1]]
            dense_shape = copy.deepcopy(old_shape)
            dense_shape[-1] = dense_obj.units

            insert_reshape(graph, src, dense, in_attr, new_shape)
            last_reshape_name = insert_reshape_after(
                graph, dense, dense_shape)

            if dense in graph._attr['output_names']:
                index = graph._attr['output_names'].index(dense)
                graph._attr['output_names'][index] = last_reshape_name

        NodeWrap(graph, dense).replace_obj(
            'FullyConnected', {'name': dense, 'weights': np.transpose(kernel_value, [1, 0]),
                               'biases': biases_value,
                               'num_output': kernel_value.shape[1]})


def convert_seperable_conv(graph):
    '''
    Convert keras SeparableConv1D/2D op to depthwise Conv and pointwise Conv.
    '''
    matches = single_node_matcher(graph, ['TfKerasSeparableConv1D', 'TfKerasSeparableConv2D'])
    for m in matches:
        sep_conv = m['target']
        sep_conv_obj = NodeWrap(graph, sep_conv)['object']
        out_edges = graph.sorted_out_edges(sep_conv, data=True)
        if sep_conv_obj is None or len(out_edges) < 1 or len(sep_conv_obj.weights_list) < 2:
            ERROR('[Parser]: Meets invalid Op (%s) in convert_seperable_conv!' % sep_conv)
            continue
        graph.remove_edges_from(out_edges)
        data_format = 'NCHW' if sep_conv_obj.data_format.startswith('NC') else 'NHWC'
        pointwise_weights = sep_conv_obj.weights_list[1]

        pointwise_conv = get_valid_node_name(graph, sep_conv + '_pointwise_conv')
        depthwise_conv_out_attr = copy.deepcopy(out_edges[0][2])
        depthwise_conv_out_attr.update({'dst_in_port': 0})
        if depthwise_conv_out_attr['tensor'] is not None and depthwise_conv_out_attr['tensor'].value is not None:
            dtype = depthwise_conv_out_attr['tensor'].value.dtype
            shape = list(depthwise_conv_out_attr['tensor'].value.shape)
            if data_format == 'NHWC':
                depthwise_conv_out_shape = shape[:-1] + [pointwise_weights.shape[-2]]
            else:
                depthwise_conv_out_shape = [shape[0], pointwise_weights.shape[-2]] + shape[2:]
            depthwise_conv_out_attr['tensor'].value = np.random.ranf(depthwise_conv_out_shape).astype(dtype)
        graph.add_edge(sep_conv, pointwise_conv, **depthwise_conv_out_attr)
        for _, dst, out_attr in out_edges:
            graph.add_edge(pointwise_conv, dst, **out_attr)

        if sep_conv_obj.type == 'TfKerasSeparableConv1D':
            weights_perm = [2, 1, 0]
            kernel_shape = [1]
            strides = [1]
        else:
            weights_perm = [3, 2, 0, 1]
            kernel_shape = [1, 1]
            strides = [1, 1]
        new_pointwise_weights = np.transpose(pointwise_weights, weights_perm)
        pointwise_conv_attr = {'name': pointwise_conv, 'weights': new_pointwise_weights,
                               'auto_pad': 'VALID', 'kernel_shape': kernel_shape, 'strides': strides,
                               'biases': sep_conv_obj.biases, 'data_format': data_format, 'opset_version': 1}
        NodeWrap(graph, pointwise_conv).replace_obj('Conv', pointwise_conv_attr)

        depthwise_weights = sep_conv_obj.weights_list[0]
        new_depthwise_weights = np.reshape(depthwise_weights, list(
            depthwise_weights.shape[:-2]) + [-1, depthwise_weights.shape[-2]//sep_conv_obj.group])
        new_depthwise_weights = np.transpose(new_depthwise_weights, sep_conv_obj.perm_tf_to_onnx())
        depthwise_conv_attr = sep_conv_obj.copied_attr()
        depthwise_conv_attr.update({'weights': new_depthwise_weights, 'biases': None, 'opset_version': 1})
        NodeWrap(graph, sep_conv).replace_obj('Conv', depthwise_conv_attr)
        if sep_conv in graph._attr['output_names']:
            index = graph._attr['output_names'].index(sep_conv)
            graph._attr['output_names'][index] = pointwise_conv


def convert_softmax(graph):
    '''
    Keras Softmax support multiple axes. Convert to onnx Softmax for one axis, otherwise
    convert to transpose+reshape+Softmax(axis=0)+reshape+transpose.
    '''
    matches = single_node_matcher(graph, 'TfKerasSoftmax')
    for m in matches:
        softmax = m['target']
        softmax_obj = NodeWrap(graph, softmax)['object']
        in_edges = graph.sorted_in_edges(softmax, data=True)
        if softmax_obj is None or len(in_edges) < 1:
            ERROR('[Parser]: Meets invalid Op (%s) in convert_softmax!' % softmax)
            continue
        if len(in_edges) >= 2:
            _, _, mask_in_attr = in_edges[1]
            if mask_in_attr['tensor'] is None \
                    or not mask_in_attr['tensor'].is_const:
                WARN('[Parser]: Non-constant mask in Op (%s) is not yet supported!' % softmax)
                continue
            if mask_in_attr['tensor'].value.item(0) is not None \
                    and False in mask_in_attr['tensor'].value:
                add_operand = (1 - mask_in_attr['tensor'].value.astype(np.float32)) * np.finfo(np.float32).min
                add_node = get_valid_node_name(graph, softmax + '_masked')
                graph.remove_edges_from(in_edges)
                src, _, in_attr = in_edges[0]
                graph.add_edge(src, add_node, **in_attr)
                insert_constant(graph, add_node + '_operand', add_operand, add_node, in_port=1, data_format='NHWC')
                softmax_in_attr = copy.deepcopy(in_attr)
                softmax_in_attr.update({'src_out_port': 0})
                if in_attr['tensor'] is not None and in_attr['tensor'].value is not None:
                    softmax_in_attr['tensor'].value = in_attr['tensor'].value + add_operand
                graph.add_edge(add_node, softmax, **softmax_in_attr)
                NodeWrap(graph, add_node).replace_obj('Add', {'name': add_node, 'opset_version': 13})
                in_edges = graph.sorted_in_edges(softmax, data=True)
        new_node_attr = softmax_obj.copied_attr()
        if len(softmax_obj.axes) == 1:
            new_axis = softmax_obj.axes[0]
        else:
            in_shapes = softmax_obj.get_input_shapes()
            if len(in_shapes) < 1 or in_shapes[0] is None or None in in_shapes[0]:
                ERROR('[Parser]: Meets input shape for Op (%s) in convert_softmax!' % softmax)
                continue
            in_shape = in_shapes[0]
            in_length = len(in_shape)
            softmax_axes = OpHasAxis.make_axes_non_negative(softmax_obj.axes, in_length)
            softmax_axes.sort()
            pre_trans_perm = softmax_axes + [axis for axis in range(in_length) if axis not in softmax_axes]
            pre_reshape_dim = [-1] + [shape for axis, shape in enumerate(in_shape) if axis not in softmax_axes]
            src, _, in_attr = in_edges[0]
            pre_trans = insert_transpose(graph, src, softmax, in_attr, pre_trans_perm)
            pre_trans_out_attr = copy.deepcopy(in_attr)
            pre_trans_out_attr.update({'src_out_port': 0})
            if in_attr['tensor'] is not None and in_attr['tensor'].value is not None:
                pre_trans_out_attr['tensor'].value = np.transpose(in_attr['tensor'].value, pre_trans_perm)
            insert_reshape(graph, pre_trans, softmax, pre_trans_out_attr, pre_reshape_dim)

            post_reshape_dim = [in_shape[axis] for axis in pre_trans_perm]
            post_reshape_old_dim = [np.prod(post_reshape_dim[:len(softmax_axes)])] + \
                post_reshape_dim[len(softmax_axes):]
            post_trans_perm = Op.cal_inverse_perm(pre_trans_perm)
            post_reshape = insert_reshape_after(graph, softmax, post_reshape_dim, post_reshape_old_dim)
            post_trans = insert_transpose_after(graph, post_reshape, post_trans_perm)

            new_axis = 0
            if softmax in graph._attr['output_names']:
                index = graph._attr['output_names'].index(softmax)
                graph._attr['output_names'][index] = post_trans
        new_node_attr.update({'axes': None, 'axis': new_axis, 'opset_version': 13})
        NodeWrap(graph, softmax).replace_obj('Softmax', new_node_attr)
        graph.remove_edges_from(in_edges[1:])


def convert_to_onnx(graph):
    '''Convert keras op to the onnx version.'''
    def warn_invalid_node(op_type, node_name):
        ERROR('[Parser]: Meets invalid Keras op(%s) for Node(%s) in convert_to_onnx!' %
              (str(op_type), str(node_name)))

    def is_first_input_valid(in_edges, in_shapes, min_length=None):
        if len(in_edges) < 1 \
                or len(input_shapes) < 1 \
                or input_shapes[0] is None \
                or None in input_shapes[0]:
            warn_invalid_node(pure_type, node_name)
            return False
        if isinstance(min_length, int) \
                and len(input_shapes[0]) < min_length:
            return False
        return True

    keras_ops = KerasOp.get_concrete_subclass_names()
    matches = extend_lists([single_node_matcher(graph, op_type)
                            for op_type in keras_ops])
    for m in matches:
        node_name = m['target']
        node_obj = NodeWrap(graph, node_name)['object']
        if node_obj is None:
            warn_invalid_node(None, node_name)
            continue

        in_edges = graph.sorted_in_edges(node_name, data=True)
        input_shapes = node_obj.get_input_shapes()
        new_node_attr = node_obj.copied_attr()
        node_data_format = 'NCHW' if node_obj.data_format.startswith('NC') else 'NHWC'
        pure_type = re.sub(r'^TfKeras', '', node_obj.type)
        if getattr(node_obj, 'correspond_onnx_op', None) is None:
            continue
        if isinstance(node_obj, OpHasWeights) and type(node_obj).perm_tf_to_onnx():
            if node_obj.weights is None:
                ERROR('[Parser]: Node(%s) does not contain weights!' % node_name)
                continue
            if pure_type == 'DepthwiseConv2D':
                # Tf fliter:[f_h,f_w,in_channel,channel_multiper]
                # onnx fliter:[out_channel,in_channel/group,f_h,f_w]
                new_weights = np.reshape(node_obj.weights, list(
                    node_obj.weights.shape[:2]) + [-1, node_obj.weights.shape[2]//node_obj.group])
            else:
                new_weights = node_obj.weights

            new_weights = np.transpose(
                new_weights, axes=type(node_obj).perm_tf_to_onnx())
            new_node_attr.update({'weights': new_weights})

        if pure_type in ('Cropping1D', 'Cropping2D', 'Cropping3D'):
            if not is_first_input_valid(in_edges, input_shapes, 3):
                continue
            input_shape = input_shapes[0]
            spatial_rank = len(input_shape) - 2
            cropping = np.array(node_obj.cropping)
            if cropping.size != spatial_rank * 2:
                ERROR('[Parser]: Meets invalid Cropping op for Node(%s) in convert_to_onnx!' % node_name)
                continue
            cropping = np.reshape(cropping, [spatial_rank, 2])
            begin_crops = cropping[:, 0].tolist()
            end_crops = cropping[:, 1].tolist()
            if node_data_format == 'NCHW':
                slice_starts = [0, 0] + begin_crops
                end_crops = [input_shape[2+idx] if end == 0 else -end for idx, end in enumerate(end_crops)]
                slice_ends = input_shape[:2] + end_crops
            else:
                slice_starts = [0] + begin_crops + [0]
                end_crops = [input_shape[1+idx] if end == 0 else -end for idx, end in enumerate(end_crops)]
                slice_ends = input_shape[:1] + end_crops + input_shape[-1:]
            new_node_attr.update({'starts': slice_starts, 'ends': slice_ends})
        elif pure_type == 'Flatten':
            if node_data_format == 'NCHW':
                if not is_first_input_valid(in_edges, input_shapes):
                    continue
                if len(input_shapes[0]) > 2:
                    perm = [idx for idx in range(len(input_shapes[0])) if idx != 1] + [1]
                    src, _, in_attr = in_edges[0]
                    insert_transpose(graph, src, node_name, in_attr, perm)
                    node_data_format = 'NHWC'
            new_node_attr.update({'shape': [0, -1]})
        elif pure_type == 'Permute':
            perm = [0] + list(node_obj.dims)
            new_node_attr.update({'perm': perm})
        elif pure_type == 'PReLU':
            insert_constant(graph, node_name + '_slope', node_obj.alpha, node_name, in_port=1, data_format='NHWC')
        elif pure_type == 'Reshape':
            if not is_first_input_valid(in_edges, input_shapes, 1):
                continue
            batch_size = input_shapes[0][0]
            shape = [batch_size] + list(node_obj.target_shape)
            new_node_attr.update({'shape': shape})
        elif pure_type == 'ThresholdedReLU':
            new_node_attr.update({'alpha': node_obj.theta})
        elif pure_type in ('ZeroPadding1D', 'ZeroPadding2D', 'ZeroPadding3D'):
            if not is_first_input_valid(in_edges, input_shapes, 1):
                continue
            spatial_rank = len(input_shapes[0]) - 2
            padding = np.array(node_obj.padding)
            if padding.size != spatial_rank * 2:
                ERROR('[Parser]: Meets invalid ZeroPadding op for Node(%s) in convert_to_onnx!' % node_name)
                continue
            padding = np.reshape(padding, [spatial_rank, 2])
            begin_pads = padding[:, 0].tolist()
            end_pads = padding[:, 1].tolist()
            if node_data_format == 'NCHW':
                full_pads = [0, 0] + begin_pads + [0, 0] + end_pads
            else:
                full_pads = [0] + begin_pads + [0, 0] + end_pads + [0]
            new_node_attr.update({'paddings': full_pads})

        new_node_attr.update(
            {'opset_version': node_obj.correspond_onnx_op['version'],
             'data_format': node_data_format})
        NodeWrap(graph, node_name).replace_obj(
            node_obj.correspond_onnx_op['type'], new_node_attr)


def multidirectional_broadcasting(graph):
    op_type_list = KerasNeedBroadcast.get_concrete_subclass_names()
    matches = single_node_matcher(graph, op_type_list)
    for m in matches:
        broadcast = m['target']
        broadcast_obj = NodeWrap(graph, broadcast)['object']
        in_edges = graph.sorted_in_edges(broadcast, keys=True, data=True)
        if broadcast_obj is None or len(in_edges) < 2:
            ERROR(
                '[Parser]: Meets Invalid broadcast Op (%s) in multidirectional_broadcasting!' % broadcast)
            continue
        in_shapes = broadcast_obj.get_input_shapes()
        dims_and_reps = KerasNeedBroadcast.cal_reshape_and_tile(in_shapes)
        if len(dims_and_reps) not in (0, len(in_edges)):
            ERROR(
                '[Parser]: Failed to calculate broadcast for Broadcast op (%s) in multidirectional_broadcasting!' % broadcast)
            continue
        for i, dr in enumerate(dims_and_reps):
            if dr['reshape'] is not None:
                src, _, k, in_attr = in_edges[i]
                insert_reshape(graph, src, broadcast, in_attr, dr['reshape'], key=k)
                in_edges = graph.sorted_in_edges(broadcast, keys=True, data=True)
            if dr['tile'] is not None:
                src, _, k, in_attr = in_edges[i]
                insert_tile(graph, src, broadcast, in_attr, dr['tile'], key=k)
                in_edges = graph.sorted_in_edges(broadcast, keys=True, data=True)


def process_keras_op_before_infer(graph):
    if not graph._attr['is_keras_model']:
        return

    convert_bidirectional(graph)
    convert_gru_lstm(graph)
    convert_centercrop(graph)
    convert_normalization(graph)
    convert_relu(graph)
    convert_rescaling(graph)
    convert_resizing(graph)
    convert_activations(graph)

    from ...lite.passes.front_passes import split_op_has_activation
    split_op_has_activation(graph, is_tf_op=True)
    convert_dense(graph)
    multidirectional_broadcasting(graph)

    from ...onnx.passes.middle_passes import split_sum_or_max_or_min
    split_sum_or_max_or_min(graph, op_type_list=['TfKerasMultiply'])


def process_keras_op_after_infer(graph):
    if not graph._attr['is_keras_model']:
        return

    remove_useless_op(graph, ['TfKerasDropout'])

    convert_batchnorm(graph)
    convert_global_pooling(graph)
    convert_softmax(graph)
    convert_seperable_conv(graph)

    convert_to_onnx(graph)
