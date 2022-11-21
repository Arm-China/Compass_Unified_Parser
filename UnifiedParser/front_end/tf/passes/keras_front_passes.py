# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import re
import copy
from ....ops.op import Op, OpHasWeights, KerasOp, KerasGlobalPoolingOp
from ....graph.node_wrap import NodeWrap
from ....graph.graph_algo import get_valid_node_name, clear_redundant_nodes
from ....graph.pattern_match import matched_patterns, single_node_matcher
from ...onnx.passes.common_passes import insert_constant, insert_reshape_after, \
    insert_transpose, insert_transpose_after
from ....common.utils import extend_lists
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL


def convert_batchnorm(graph):
    matches = single_node_matcher(graph, 'TfKerasBatchNormalization')
    for m in matches:
        batchnorm = m['target']
        batchnorm_obj = NodeWrap(graph, batchnorm)['object']
        if batchnorm_obj is None:
            WARN(
                '[Parser]: Meets TfKerasBatchNormalization Op (%s) in convert_batchnorm!' % batchnorm)
            continue
        input_shapes = batchnorm_obj.get_input_shapes()
        in_edges = graph.sorted_in_edges(batchnorm, data=True)
        if len(in_edges) < 1 \
                or len(input_shapes) < 1 \
                or input_shapes[0] is None \
                or None in input_shapes[0]:
            continue
        if len(batchnorm_obj.weights_list) >= 4:
            mean_value = batchnorm_obj.weights_list[2]
            var_value = batchnorm_obj.weights_list[3]
        else:
            num_output = input_shapes[0][-1]
            mean_value = np.zeros(num_output, np.float32)
            var_value = np.ones(num_output, np.float32)
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


def convert_global_pooling(graph):
    op_types = KerasGlobalPoolingOp.get_concrete_subclass_names()
    matches = single_node_matcher(graph, op_types)
    for m in matches:
        global_pool = m['target']
        global_pool_obj = NodeWrap(graph, global_pool)['object']
        if global_pool_obj is None:
            WARN(
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
                or len(in_edges) < 1 \
                or len(rnn_obj.weights_list) < 2:
            WARN(
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
        kernel = rnn_obj.weights_list[0]
        recurrent_kernel = rnn_obj.weights_list[1]
        B_value, bias = None, None
        if rnn_obj.use_bias and len(rnn_obj.weights_list) >= 3:
            bias = rnn_obj.weights_list[2]
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
        rnn_attr.update({'direction': 'reverse' if rnn_obj.go_backwards else 'forward',
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


def convert_to_onnx(graph):
    '''Convert keras op to the onnx version.'''
    def warn_invalid_node(op_type, node_name):
        WARN('[Parser]: Meets invalid Keras op(%s) for Node(%s) in convert_to_onnx!' %
             (str(op_type), str(node_name)))

    def is_first_input_valid(in_edges, in_shapes):
        if len(in_edges) < 1 \
                or len(input_shapes) < 1 \
                or input_shapes[0] is None \
                or None in input_shapes[0]:
            warn_invalid_node(pure_type, node_name)
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
                WARN('[Parser]: Node(%s) does not contain weights!' % node_name)
                continue
            new_weights = np.transpose(
                node_obj.weights, axes=type(node_obj).perm_tf_to_onnx())
            new_node_attr.update({'weights': new_weights})

        if pure_type == 'Flatten':
            if node_data_format == 'NCHW':
                if not is_first_input_valid(in_edges, input_shapes):
                    continue
                if len(input_shapes[0]) > 2:
                    perm = [idx for idx in range(len(input_shapes[0])) if idx != 1] + [1]
                    src, _, in_attr = in_edges[0]
                    insert_transpose(graph, src, node_name, in_attr, perm)
                    node_data_format = 'NHWC'
            new_node_attr.update({'shape': [0, -1]})

        new_node_attr.update(
            {'opset_version': node_obj.correspond_onnx_op['version'],
             'data_format': node_data_format})
        NodeWrap(graph, node_name).replace_obj(
            node_obj.correspond_onnx_op['type'], new_node_attr)


def process_keras_op_before_infer(graph):
    if not graph._attr['is_keras_model']:
        return

    from ...lite.passes.front_passes import split_op_has_activation
    convert_gru_lstm(graph)
    split_op_has_activation(graph, is_tf_op=True)


def process_keras_op_after_infer(graph):
    if not graph._attr['is_keras_model']:
        return

    convert_batchnorm(graph)
    convert_global_pooling(graph)

    convert_to_onnx(graph)
