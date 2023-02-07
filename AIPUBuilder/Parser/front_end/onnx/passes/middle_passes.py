# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import itertools
import copy
from functools import reduce
from collections import OrderedDict
from ....common.defs import Tensor, FLOAT_EQUAL, TYPE_MAX
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL
from ....common.utils import extend_lists, get_converted_dtype
from ....graph.node_wrap import NodeWrap
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from ....graph.graph_algo import get_valid_node_name, clear_redundant_nodes, determined_sort, all_simple_paths, has_path
from ....ops.op import Op, BaseLinearOp, BaseConvOp, BaseDeconvOp, BaseOnnxPoolOp, OpHasOneOutPort, OpHasPaddingStrides, OpHasAxis, \
    OnnxOp, CommonOp, OpNeedBroadcast, OpNeedUniBroadcast
from ....ops.onnx_ops.array_ops import ReshapeOp
from .common_passes import fuse_const, remove_useless_op, remove_node_safely, insert_reshape, insert_reshape_after, \
    insert_cast, insert_constant, insert_slice, insert_slice_after, insert_tile, insert_transpose, insert_transpose_after, \
    remove_redundant_reshape, remove_redundant_transpose


def clear_useless_concat_input(graph):
    matched = False
    matches = single_node_matcher(graph, 'Concat')
    for m in matches:
        concat = m['target']
        concat_obj = NodeWrap(graph, concat)['object']
        if concat_obj is None:
            WARN(
                '[Parser]: Meets invalid Concat Op (%s) in clear_useless_concat_input!' % concat)
            continue
        in_edges = graph.sorted_in_edges(concat, keys=True)
        if len(in_edges) < 2:
            continue
        input_shapes = concat_obj.get_input_shapes()
        if len(input_shapes) != len(in_edges):
            WARN(
                '[Parser]: Meets invalid Concat Op (%s) in clear_useless_concat_input!' % concat)
            continue
        if any([s is None for s in input_shapes]):
            WARN('[Parser]: Meets invalid input shape for Concat Op (%s) in clear_useless_concat_input!' % concat)
            continue
        for i, (src, _, k) in enumerate(in_edges):
            if input_shapes[i][concat_obj.axis] == 0:
                graph.remove_edge(src, concat, key=k)
                matched = True
    if matched:
        clear_redundant_nodes(graph)


def convert_onnx_version(graph):
    possible_ops = OnnxOp.get_concrete_subclass_names()
    matches = [single_node_matcher(graph, onnx_type)
               for onnx_type in possible_ops]
    matches = extend_lists(matches)
    for m in matches:
        node = m['target']
        node_obj = NodeWrap(graph, node)['object']
        if node_obj is not None:
            if len(type(node_obj).attributes()) <= 1:
                continue
            node_obj.convert_version()
        else:
            WARN('[Parser]: Meets invalid node(%s) in convert_onnx_version!' % node)


def convert_1d_conv(graph):
    for conv_type in ['Conv', 'ConvTranspose']:
        matches = single_node_matcher(graph, conv_type)
        for m in matches:
            conv = m['target']
            conv_obj = NodeWrap(graph, conv)['object']
            if conv_obj is not None and len(conv_obj.weights.shape) == 3:
                in_edges = graph.sorted_in_edges(conv, data=True)
                out_edges = graph.sorted_out_edges(conv, data=True)
                if len(in_edges) >= 1 and len(out_edges) >= 1:
                    in_shape = conv_obj.get_input_shapes()[0]
                    out_shape = conv_obj.get_output_shapes()[0]
                    is_channels_last = (conv_obj.data_format == 'NHWC')
                    if is_channels_last:
                        reshape1_dim = [in_shape[0], 1,
                                        in_shape[1], in_shape[2]]
                    else:
                        reshape1_dim = [in_shape[0],
                                        in_shape[1], 1, in_shape[2]]
                    reshape2_dim = out_shape

                    src, _, in_attr = in_edges[0]
                    insert_reshape(graph, src, conv, in_attr, reshape1_dim)

                    reshape2 = get_valid_node_name(
                        graph, conv + '_post_reshape')
                    for _, dst, out_attr in out_edges:
                        graph.remove_edge(conv, dst)
                        graph.add_edge(reshape2, dst, **out_attr)
                    conv_out_tensor = None
                    conv_out_attr = out_edges[0][2]
                    if conv_out_attr['tensor'] is not None and conv_out_attr['tensor'].value is not None:
                        conv_out_tensor = np.expand_dims(conv_out_attr['tensor'].value, 1 if is_channels_last else 2)
                    graph.add_edge(conv, reshape2, **{'tensor': Tensor(value=conv_out_tensor)})

                    reshape2_shape_const = get_valid_node_name(
                        graph, reshape2 + '_shape')
                    insert_constant(graph, reshape2_shape_const, np.array(
                        reshape2_dim, np.int64), reshape2, in_port=1, data_format='NHWC')
                    reshape2_attr = conv_obj.copied_attr()
                    reshape2_attr.update(
                        {'name': reshape2, 'opset_version': 5})
                    NodeWrap(graph, reshape2).replace_obj(
                        'Reshape', reshape2_attr)

                    conv_attr = conv_obj.copied_attr()
                    conv_attr['weights'] = np.expand_dims(
                        conv_attr['weights'], axis=2)
                    conv_attr['kernel_shape'] = [1] + conv_attr['kernel_shape']
                    conv_attr['strides'] = [1] + conv_attr['strides']
                    conv_attr['dilations'] = [1] + conv_attr['dilations']
                    if 'pads' in conv_attr and conv_attr['pads']:
                        pad_len = len(conv_attr['pads'])
                        pads = copy.deepcopy(conv_attr['pads'])
                        pads.insert(pad_len // 2, 0)
                        pads.insert(0, 0)
                        conv_attr['pads'] = pads

                    if 'output_shape' in conv_attr and conv_attr['output_shape']:
                        conv_attr['output_shape'] = [
                            1] + conv_attr['output_shape']
                    if 'output_padding' in conv_attr and conv_attr['output_padding']:
                        conv_attr['output_padding'] = [
                            0] + conv_attr['output_padding']
                    NodeWrap(graph, conv).replace_obj(conv_type, conv_attr)
                    if conv in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(conv)
                        graph._attr['output_names'][index] = reshape2


def convert_1d_pooling(graph):
    for pool_type in ['AveragePool', 'MaxPool', 'LpPool']:
        matches = single_node_matcher(graph, pool_type)
        for m in matches:
            pool = m['target']
            pool_obj = NodeWrap(graph, pool)['object']
            if pool_obj is not None:
                in_shapes = pool_obj.get_input_shapes()
                out_shapes = pool_obj.get_output_shapes()
                if len(in_shapes) == 1 and in_shapes[0] and len(in_shapes[0]) == 3 \
                        and len(out_shapes) >= 1 and out_shapes[0] and len(out_shapes[0]) == 3 \
                        and len(pool_obj.pads) == 2 \
                        and len(pool_obj.get_out_ports()) == 1:
                    in_edges = graph.sorted_in_edges(pool, data=True)
                    out_edges = graph.sorted_out_edges(pool, data=True)
                    in_shape, out_shape = in_shapes[0], out_shapes[0]
                    pre_reshape_dim = (
                        [in_shape[0], 1] + in_shape[1:]) if pool_obj.data_format == 'NHWC' else (in_shape[0:2] + [1, in_shape[-1]])
                    post_reshape_dim = out_shape
                    pool_obj.pads = np.concatenate([np.array([[0], [0]]), np.transpose(
                        np.array([pool_obj.pads]))], axis=1).flatten().tolist()
                    if pool_obj.kernel_shape:
                        pool_obj.kernel_shape = [1] + pool_obj.kernel_shape
                    if pool_obj.strides:
                        pool_obj.strides = [1] + pool_obj.strides
                    if pool_obj.dilations:
                        pool_obj.dilations = [1] + pool_obj.dilations
                    src, _, in_attr = in_edges[0]
                    insert_reshape(
                        graph, src, pool, in_attr, pre_reshape_dim, data_format=pool_obj.data_format)

                    post_reshape = get_valid_node_name(
                        graph, pool + '_post_reshape')
                    for _, dst, out_attr in out_edges:
                        graph.remove_edge(pool, dst)
                        graph.add_edge(post_reshape, dst, **out_attr)
                    graph.add_edge(pool, post_reshape)
                    NodeWrap(graph, post_reshape).replace_obj(
                        'Reshape', {'name': post_reshape, 'opset_version': 5})
                    insert_constant(graph, post_reshape + '_shape', np.array(post_reshape_dim,
                                                                             np.int64), post_reshape, in_port=1, data_format=pool_obj.data_format)

                    if pool in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(pool)
                        graph._attr['output_names'][index] = post_reshape
            else:
                WARN(
                    '[Parser]: Meets invalid Pooling node(%s) in convert_1d_pooling!' % pool)


def convert_bn_train(graph):
    matched = False
    matches = single_node_matcher(graph, 'BatchNormalization')
    for m in matches:
        fusebnv3 = m['target']
        fusebnv3_obj = NodeWrap(graph, fusebnv3)['object']
        fusebnv3_in_edges = graph.sorted_in_edges(fusebnv3, data=True)
        fusebnv3_out_edges = graph.sorted_out_edges(fusebnv3, data=True)
        if fusebnv3_obj is not None \
                and len(fusebnv3_in_edges) == 5 \
                and len(fusebnv3_obj.get_input_tensors()) == 5:
            if fusebnv3_obj.training_mode is False:
                continue
            inputs = fusebnv3_obj.get_input_tensors()
            x = inputs[0]
            if x.shape is None:
                continue
            matched = True
            x_src, _, x_out_attr = fusebnv3_in_edges[0]
            scale, _, scale_out_attr = fusebnv3_in_edges[1]
            offset, _, offset_out_attr = fusebnv3_in_edges[2]
            input_mean, _, inmean_out_attr = fusebnv3_in_edges[3]
            input_var, _, invar_out_attr = fusebnv3_in_edges[4]
            inp_rank = len(x.shape)
            eps = fusebnv3_obj.epsilon
            momentum = fusebnv3_obj.momentum
            if fusebnv3_obj.data_format == 'NCHW':
                dims = [0] + list(range(2, inp_rank))
                reshape_dim = [-1] + [1] * (inp_rank - 2)
            else:
                dims = list(range(0, inp_rank - 1))
                reshape_dim = [1] * (inp_rank - 2) + [-1]
            # Step 1: Consider output Y
            # current_mean_value -> onnx input_mean; current_var_value -> onnx input_var
            inp_shape = np.array(x.shape, np.dtype(np.int32))
            reduce_dims = np.take(inp_shape, np.array(dims, np.int32), axis=0)
            cnt_float = np.prod(reduce_dims, axis=tuple(
                [0]), keepdims=False).astype(np.float32)
            current_mean_value = np.mean(x, axis=tuple(dims), keepdims=True)
            sub_value = np.subtract(x, current_mean_value)
            var_squeezed_value = np.sum(
                np.square(sub_value), axis=tuple(dims), keepdims=False)
            current_var_value = np.true_divide(var_squeezed_value, cnt_float)
            # weights_value -> onnx 1/sqrt(input_var+eps)=(input_var+eps)^(-0.5)
            current_var_eps_value = np.add(current_var_value, eps)
            sqrt_value = np.sqrt(current_var_eps_value)
            weights_value = np.true_divide(1, sqrt_value)
            reshaped_weights_value = np.reshape(weights_value, reshape_dim)
            # y = reshaped_weights_value * (x - input_mean) * scale + B
            #   = reshaped_weights_value * sub_value * scale + offset
            #   = reshaped_weights_value * sub_value * inputs[1] + inputs[2]
            mul_sub_value = reshaped_weights_value * sub_value

            current_mean = get_valid_node_name(graph, fusebnv3 + '_current_mean')
            sub = get_valid_node_name(graph, fusebnv3 + '_sub')
            var_squeezed = get_valid_node_name(
                graph, fusebnv3 + '_var_squeeze')
            current_var = get_valid_node_name(graph, fusebnv3 + '_current_var')
            current_var_eps = get_valid_node_name(graph, fusebnv3 + '_current_var_eps')
            weights = get_valid_node_name(graph, fusebnv3 + '_weights')
            mul_sub = get_valid_node_name(
                graph, fusebnv3 + '_mul_sub')
            mul_scale = get_valid_node_name(
                graph, fusebnv3 + '_mul_scale')
            y = get_valid_node_name(graph, fusebnv3 + '_add_bias')

            graph.remove_edges_from(fusebnv3_in_edges)

            graph.add_edge(x_src, current_mean, **x_out_attr)
            graph.add_edge(x_src, sub, **x_out_attr)
            current_mean_out_attr = {'src_out_port': 0, 'dst_in_port': 1, 'tensor': Tensor(value=current_mean_value)}
            graph.add_edge(current_mean, sub, **current_mean_out_attr)
            graph.add_edge(sub, var_squeezed, **{'tensor': Tensor(value=sub_value)})
            graph.add_edge(var_squeezed, current_var, **
                           {'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(value=var_squeezed_value)})
            current_var_out_attr = {'tensor': Tensor(value=current_var_value)}
            graph.add_edge(current_var, current_var_eps, **current_var_out_attr)
            graph.add_edge(current_var_eps, weights, **{'tensor': Tensor(value=current_var_eps_value)})

            # reshaped_weights_value * sub
            graph.add_edge(sub, mul_sub, **
                           {'src_out_port': 0, 'dst_in_port': 0})
            graph.add_edge(weights, mul_sub, **
                           {'src_out_port': 0, 'dst_in_port': 1, 'tensor': Tensor(value=weights_value)})
            # reshaped_weights_value * sub * scale
            graph.add_edge(mul_sub, mul_scale, **
                           {'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(value=mul_sub_value)})
            graph.add_edge(scale, mul_scale, **scale_out_attr)
            # reshaped_weights_value * sub * scale + offset
            offset_out_attr = copy.deepcopy(offset_out_attr)
            offset_out_attr.update({'dst_in_port': 1})
            graph.add_edge(mul_scale, y, **
                           {'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(value=mul_sub_value)})
            graph.add_edge(offset, y, **offset_out_attr)

            insert_constant(graph, current_var_eps + '_add', np.array(eps).astype(np.float32),
                            current_var_eps, in_port=1, data_format='NHWC')
            insert_constant(graph, weights + '_pow', np.array(-0.5).astype(np.float32),
                            weights, in_port=1, data_format='NHWC')
            insert_constant(graph, current_var + '_cnt_float', np.array(cnt_float),
                            current_var, in_port=1, data_format='NHWC')

            NodeWrap(graph, current_mean).replace_obj('ReduceMean', {
                'name': current_mean, 'axes': dims, 'keepdims': True, 'opset_version': 11})
            NodeWrap(graph, sub).replace_obj('Sub', {
                'name': sub, 'opset_version': 14})
            NodeWrap(graph, var_squeezed).replace_obj('ReduceSumSquare', {
                'name': var_squeezed, 'axes': dims, 'keepdims': False, 'opset_version': 13})
            NodeWrap(graph, current_var).replace_obj('Div', {
                'name': current_var, 'opset_version': 7})
            NodeWrap(graph, current_var_eps).replace_obj('Add', {
                'name': current_var_eps, 'opset_version': 7})
            NodeWrap(graph, weights).replace_obj('Pow', {
                'name': weights, 'opset_version': 7})
            NodeWrap(graph, mul_sub).replace_obj('Mul', {
                'name': mul_sub, 'opset_version': 7})
            NodeWrap(graph, mul_scale).replace_obj('Mul', {
                'name': mul_scale, 'opset_version': 7})
            NodeWrap(graph, y).replace_obj('Add', {
                'name': y, 'opset_version': 7})

            # Reshape is needed when data format is NCHW
            if fusebnv3_obj.data_format == 'NCHW':
                mul_sub_in_attr = {'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(value=weights_value)}
                insert_reshape(graph, weights, mul_sub, mul_sub_in_attr, reshape_dim)
                insert_reshape(graph, scale, mul_scale, scale_out_attr, reshape_dim)
                insert_reshape(graph, offset, y, offset_out_attr, reshape_dim)

            # Step 2: Consider output running_mean
            # running_mean = input_mean * momentum + reshaped_current_mean * (1 - momentum)
            mul_in_mean = get_valid_node_name(graph, fusebnv3 + '_mul_in_mean')
            mul_cur_mean = get_valid_node_name(graph, fusebnv3 + '_mul_cur_mean')
            running_mean = get_valid_node_name(graph, fusebnv3 + '_running_mean')

            # input_mean * momentum
            mul_in_mean_in_attr = copy.deepcopy(inmean_out_attr)
            mul_in_mean_in_attr.update({'dst_in_port': 0})
            graph.add_edge(input_mean, mul_in_mean, **mul_in_mean_in_attr)
            insert_constant(graph, mul_in_mean + '_momentum', np.array(momentum).astype(np.float32),
                            mul_in_mean, in_port=1, data_format='NHWC')
            # reshaped_current_mean * (1 - momentum)
            graph.add_node(mul_cur_mean)
            mul_cur_mean_in_attr = {'dst_in_port': 0, 'tensor': Tensor(value=current_mean_out_attr['tensor'].value)}
            reshaped_current_mean = insert_reshape(graph, current_mean, mul_cur_mean, mul_cur_mean_in_attr, [-1]) \
                if fusebnv3_obj.data_format == 'NCHW' else current_mean
            insert_constant(graph, mul_cur_mean + '_momentum', np.array(1 - momentum).astype(np.float32),
                            mul_cur_mean, in_port=1, data_format='NHWC')
            # input_mean * momentum + reshaped_current_mean * (1 - momentum)
            mul_in_mean_value = None if mul_in_mean_in_attr['tensor'].value is None \
                else (mul_in_mean_in_attr['tensor'].value * momentum)
            run_mean_in_attr = copy.deepcopy(inmean_out_attr)
            run_mean_in_attr.update({'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(value=mul_in_mean_value)})
            graph.add_edge(mul_in_mean, running_mean, **run_mean_in_attr)
            mul_cur_mean_value = None if mul_cur_mean_in_attr['tensor'].value is None \
                else np.reshape(mul_cur_mean_in_attr['tensor'].value, [-1]) * (1 - momentum)
            run_mean_in_attr1 = {'dst_in_port': 1, 'tensor': Tensor(value=mul_cur_mean_value)}
            graph.add_edge(mul_cur_mean, running_mean, **run_mean_in_attr1)

            NodeWrap(graph, mul_in_mean).replace_obj('Mul', {
                'name': mul_in_mean, 'opset_version': 7})
            NodeWrap(graph, mul_cur_mean).replace_obj('Mul', {
                'name': mul_cur_mean, 'opset_version': 7})
            NodeWrap(graph, running_mean).replace_obj('Add', {
                'name': running_mean, 'opset_version': 7})

            # Step 3: Consider output running_var
            # running_var = input_var * momentum + current_var * (1 - momentum)
            mul_in_var = get_valid_node_name(graph, fusebnv3 + '_mul_in_var')
            mul_cur_var = get_valid_node_name(graph, fusebnv3 + '_mul_cur_var')
            running_var = get_valid_node_name(graph, fusebnv3 + '_running_var')

            # input_var * momentum
            mul_in_var_in_attr = copy.deepcopy(invar_out_attr)
            mul_in_var_in_attr.update({'dst_in_port': 0})
            graph.add_edge(input_var, mul_in_var, **mul_in_var_in_attr)
            insert_constant(graph, mul_in_var + '_momentum', np.array(momentum).astype(np.float32),
                            mul_in_var, in_port=1, data_format='NHWC')
            # current_var * (1 - momentum)
            mul_cur_var_in_attr = {'dst_in_port': 0, 'tensor': Tensor(value=current_var_out_attr['tensor'].value)}
            graph.add_edge(current_var, mul_cur_var, **mul_cur_var_in_attr)
            insert_constant(graph, mul_cur_var + '_momentum', np.array(1 - momentum).astype(np.float32),
                            mul_cur_var, in_port=1, data_format='NHWC')
            # input_var * momentum + current_var * (1 - momentum)
            mul_in_var_value = None if mul_in_var_in_attr['tensor'].value is None \
                else mul_in_var_in_attr['tensor'].value * momentum
            run_var_in_attr = copy.deepcopy(invar_out_attr)
            run_var_in_attr.update({'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(value=mul_in_var_value)})
            graph.add_edge(mul_in_var, running_var, **run_var_in_attr)
            mul_cur_mean_value = None if mul_cur_var_in_attr['tensor'].value is None \
                else mul_cur_var_in_attr['tensor'].value * (1 - momentum)
            run_var_in_attr1 = {'dst_in_port': 1, 'tensor': Tensor(value=mul_cur_mean_value)}
            graph.add_edge(mul_cur_var, running_var, **run_var_in_attr1)

            NodeWrap(graph, mul_in_var).replace_obj('Mul', {
                'name': mul_in_var, 'opset_version': 7})
            NodeWrap(graph, mul_cur_var).replace_obj('Mul', {
                'name': mul_cur_var, 'opset_version': 7})
            NodeWrap(graph, running_var).replace_obj('Add', {
                'name': running_var, 'opset_version': 7})

            for _, dst, out_attr in fusebnv3_out_edges:
                graph.remove_edge(fusebnv3, dst)
                src_out_port = out_attr['src_out_port']
                new_out_attr = copy.deepcopy(out_attr)
                new_out_attr.update({'src_out_port': 0})
                if src_out_port == 0:
                    graph.add_edge(y, dst, **new_out_attr)
                elif src_out_port == 1:
                    graph.add_edge(running_mean, dst, **new_out_attr)
                else:
                    graph.add_edge(running_var, dst, **new_out_attr)

            if fusebnv3 in graph._attr['output_names']:
                index = graph._attr['output_names'].index(fusebnv3)
                graph._attr['output_names'][index] = y
                graph._attr['output_names'].insert(index + 1, running_mean)
                graph._attr['output_names'].insert(index + 2, running_var)

        else:
            WARN(
                '[Parser]: Meets invalid Node(%s) in convert_bn_training!'
                % (fusebnv3))
    if matched:
        clear_redundant_nodes(graph)


def convert_nms(graph):
    matches = single_node_matcher(graph, 'NonMaxSuppression')
    for m in matches:
        nms = m['target']
        nms_obj = NodeWrap(graph, nms)['object']
        if nms_obj is not None:
            in_edges = graph.sorted_in_edges(nms, data=True)
            if len(in_edges) == 5:

                # Calculate the required variables

                # Variables before NMS
                box_shape = nms_obj.get_input_shapes()[0]
                score_shape = nms_obj.get_input_shapes()[1]
                if box_shape is not None and score_shape is not None and len(box_shape) <= 2 or len(score_shape) <= 1:
                    WARN(
                        '[Parser]: box_shape or score_shape of node (%s) is None.' % nms)
                    continue

                onnx_batch = box_shape[0]
                box_num = box_shape[1]
                class_num = score_shape[1]
                max_output_boxes_per_class = nms_obj.max_output_boxes_per_class
                if class_num > 1 or onnx_batch > 1:
                    ERROR(
                        '[Parser]: Parser can not support NMS node (%s) with multi_batch or multi_class.' % nms)
                    continue
                per_class_boxes_num = box_num * np.ones(
                    (onnx_batch, class_num), dtype=np.int32)
                total_class = class_num * \
                    np.ones((onnx_batch, 1), dtype=np.int32)
                max_output_size = nms_obj.max_output_boxes_per_class
                center_box = nms_obj.center_point_box
                score_reshape_dim = [score_shape[0],
                                     score_shape[1] * score_shape[2]]
                tile_dim = [1, class_num, 1]
                add_num = np.array(
                    onnx_batch * [[-0.5] * box_num + [-0.5] * box_num + [0.5] * box_num + [0.5] * box_num])
                reshape_box_1_dim = [onnx_batch, box_num]
                reshapedim2 = [onnx_batch, box_shape[2], box_num]

                # Variables after NMS
                box_num = min(box_shape[1], max_output_boxes_per_class)
                out_edges = graph.sorted_out_edges(nms, data=True)

                post_slice_start = np.array([0, 0]).astype(np.int32).tolist()
                post_slice_end_num = box_num * class_num if onnx_batch * box_num * \
                    class_num < max_output_boxes_per_class else max_output_boxes_per_class // onnx_batch
                post_slice_end = np.array(
                    [onnx_batch, int(post_slice_end_num)]).tolist()
                post_slice_step = np.array([1, 1]).tolist()

                onnx_res_first_row = [box_num * class_num * [i]
                                      for i in range(onnx_batch)]
                onnx_res_second_row = [box_num * [j]
                                       for i in range(onnx_batch) for j in range(class_num)]
                class_num_list = np.array(onnx_res_second_row)  # class_num
                class_num_list = np.reshape(class_num_list, (-1, 1))
                class_num_list = class_num_list[:max_output_boxes_per_class, :]
                batch_list = np.array(onnx_res_first_row)  # batch
                batch_list = np.reshape(batch_list, (-1, 1))
                batch_list = batch_list[:max_output_boxes_per_class, :]
                need_insert_num1 = max_output_boxes_per_class - box_num * class_num * onnx_batch
                need_insert_num2 = max_output_boxes_per_class * onnx_batch * class_num - box_num
                complete_res = None
                if need_insert_num1 > 0 or need_insert_num2 > 0:
                    need_insert_num = min(need_insert_num1, need_insert_num2) if need_insert_num1 > 0 and need_insert_num2 > 0 else max(
                        need_insert_num1, need_insert_num2)
                    complete_res = np.zeros(need_insert_num)
                    complete_res = np.reshape(complete_res, (-1, 1))
                    batch_list = np.concatenate(
                        [batch_list, complete_res], axis=0)
                    class_num_list = np.concatenate(
                        [class_num_list, complete_res], axis=0)

                # Manipulate edges and nodes
                if center_box is True:
                    split = get_valid_node_name(graph, nms + '_split')
                    xcenter_reshape = get_valid_node_name(
                        graph, nms + '_reshape1')
                    ycenter_reshape = get_valid_node_name(
                        graph, nms + '_reshape2')
                    width_reshape = get_valid_node_name(
                        graph, nms + '_reshape3')
                    height_reshape = get_valid_node_name(
                        graph, nms + '_reshape4')
                    xyc_concat1 = get_valid_node_name(graph, nms + '_concat1')
                    hw_concat2 = get_valid_node_name(graph, nms + '_concat2')
                    mul_num = get_valid_node_name(graph, nms + '_mul')
                    xy_add = get_valid_node_name(graph, nms + '_add')
                    c_reshape = get_valid_node_name(graph, nms + '_reshape')
                    box_trans = get_valid_node_name(graph, nms + '_transpose')
                    box_tile = get_valid_node_name(graph, nms + '_tile')

                    graph.remove_edges_from(in_edges)
                    box_src, _, _ = in_edges[0]
                    score_src, _, in_attr = in_edges[1]
                    in_attr['dst_in_port'] = 3

                    graph.add_edge(box_src, split)
                    graph.add_edge(split, xcenter_reshape, **
                                   {'src_out_port': 0, 'dst_in_port': 0})
                    graph.add_edge(split, ycenter_reshape, **
                                   {'src_out_port': 1, 'dst_in_port': 0})
                    graph.add_edge(split, width_reshape, **
                                   {'src_out_port': 2, 'dst_in_port': 0})
                    graph.add_edge(split, height_reshape, **
                                   {'src_out_port': 3, 'dst_in_port': 0})
                    graph.add_edge(ycenter_reshape, xyc_concat1, **
                                   {'src_out_port': 0, 'dst_in_port': 0})
                    graph.add_edge(xcenter_reshape, xyc_concat1, **
                                   {'src_out_port': 0, 'dst_in_port': 1})
                    graph.add_edge(ycenter_reshape, xyc_concat1, **
                                   {'src_out_port': 0, 'dst_in_port': 2})
                    graph.add_edge(xcenter_reshape, xyc_concat1, **
                                   {'src_out_port': 0, 'dst_in_port': 3})
                    graph.add_edge(height_reshape, hw_concat2, **
                                   {'src_out_port': 0, 'dst_in_port': 0})
                    graph.add_edge(width_reshape, hw_concat2, **
                                   {'src_out_port': 0, 'dst_in_port': 1})
                    graph.add_edge(height_reshape, hw_concat2, **
                                   {'src_out_port': 0, 'dst_in_port': 2})
                    graph.add_edge(width_reshape, hw_concat2, **
                                   {'src_out_port': 0, 'dst_in_port': 3})
                    graph.add_edge(hw_concat2, mul_num)
                    graph.add_edge(xyc_concat1, xy_add)
                    graph.add_edge(mul_num, xy_add, **
                                   {'src_out_port': 0, 'dst_in_port': 1})
                    graph.add_edge(xy_add, c_reshape)
                    graph.add_edge(c_reshape, box_trans)
                    graph.add_edge(box_trans, box_tile)
                    graph.add_edge(box_tile, nms, **
                                   {'src_out_port': 0, 'dst_in_port': 0})
                    graph.add_edge(score_src, nms, **
                                   {'src_out_port': 0, 'dst_in_port': 3})
                    insert_reshape(
                        graph, score_src, nms, in_attr, score_reshape_dim, data_format='NCHW')
                    insert_constant(graph, xcenter_reshape + '_indices', np.array(
                        reshape_box_1_dim).astype(np.int32), xcenter_reshape, in_port=1, data_format='NHWC')
                    insert_constant(graph, ycenter_reshape + '_indices', np.array(
                        reshape_box_1_dim).astype(np.int32), ycenter_reshape, in_port=1, data_format='NHWC')
                    insert_constant(graph, width_reshape + '_indices', np.array(
                        reshape_box_1_dim).astype(np.int32), width_reshape, in_port=1, data_format='NHWC')
                    insert_constant(graph, height_reshape + '_indices', np.array(
                        reshape_box_1_dim).astype(np.int32), height_reshape, in_port=1, data_format='NHWC')
                    insert_constant(graph, box_tile + '_indices', np.array(tile_dim),
                                    box_tile, in_port=1, data_format='NHWC')
                    insert_constant(graph, mul_num + '_add_num',
                                    add_num, mul_num, in_port=1, data_format='NHWC')

                    NodeWrap(graph, split).replace_obj(
                        'Split', {'name': split, 'split': [1, 1, 1, 1], 'axis': 2, 'opset_version': 11})
                    NodeWrap(graph, xcenter_reshape).replace_obj('Reshape', {
                        'name': xcenter_reshape, 'shape': reshape_box_1_dim})
                    NodeWrap(graph, ycenter_reshape).replace_obj('Reshape', {
                        'name': ycenter_reshape, 'shape': reshape_box_1_dim})
                    NodeWrap(graph, width_reshape).replace_obj('Reshape', {
                        'name': width_reshape, 'shape': reshape_box_1_dim})
                    NodeWrap(graph, height_reshape).replace_obj('Reshape', {
                        'name': height_reshape, 'shape': reshape_box_1_dim})
                    NodeWrap(graph, xyc_concat1).replace_obj(
                        'Concat', {'name': xyc_concat1, 'opset_version': 11, 'axis': 1})
                    NodeWrap(graph, hw_concat2).replace_obj(
                        'Concat', {'name': hw_concat2, 'opset_version': 11, 'axis': 1})
                    NodeWrap(graph, mul_num).replace_obj(
                        'Mul', {'name': mul_num, 'opset_version': 7})
                    NodeWrap(graph, xy_add).replace_obj(
                        'Add', {'name': xy_add, 'opset_version': 7})
                    NodeWrap(graph, c_reshape).replace_obj('Reshape', {
                        'name': c_reshape, 'shape': reshapedim2, 'opset_version': 1})
                    NodeWrap(graph, box_trans).replace_obj('Transpose', {
                        'name': box_trans, 'opset_version': 13, 'perm': [0, 2, 1]})
                    NodeWrap(graph, box_tile).replace_obj('Tile', {
                        'name': box_tile, 'opset_version': 6})
                else:
                    box_tile = get_valid_node_name(graph, nms + '_tile')

                    graph.remove_edges_from(in_edges)
                    box_src, _, _ = in_edges[0]
                    score_src, _, in_attr = in_edges[1]
                    in_attr['dst_in_port'] = 3
                    graph.add_edge(score_src, nms, **
                                   {'src_out_port': 0, 'dst_in_port': 3})
                    graph.add_edge(box_src, box_tile, **
                                   {'src_out_port': 0, 'dst_in_port': 0})
                    graph.add_edge(box_tile, nms, **
                                   {'src_out_port': 0, 'dst_in_port': 0})
                    insert_reshape(
                        graph, score_src, nms, in_attr, score_reshape_dim, data_format='NCHW')
                    insert_constant(graph, box_tile + '_indices', np.array(tile_dim),
                                    box_tile, in_port=1, data_format='NHWC')

                    NodeWrap(graph, box_tile).replace_obj('Tile', {
                        'name': box_tile, 'opset_version': 6})

                post_slice = get_valid_node_name(graph, nms + '_slice')
                post_concatenate = get_valid_node_name(graph, nms + '_concat')
                post_reshape = get_valid_node_name(graph, nms + '_reshape')

                graph.remove_edges_from(out_edges)
                graph.add_edge(nms, post_slice, **
                               {'src_out_port': 3, 'dst_in_port': 0})
                graph.add_edge(post_slice, post_reshape, **
                               {'src_out_port': 0, 'dst_in_port': 0})
                graph.add_edge(post_reshape, post_concatenate, **
                               {'src_out_port': 0, 'dst_in_port': 2})
                insert_constant(graph, nms + '_outa', np.array(batch_list),
                                post_concatenate, in_port=0)
                insert_constant(graph, nms + '_outb', np.array(class_num_list),
                                post_concatenate, in_port=1)
                for _, dst, out_attr in out_edges:
                    new_out_attr = copy.deepcopy(out_attr)
                    new_out_attr.update({'src_out_port': 0})
                    graph.add_edge(post_concatenate, dst, **new_out_attr)

                insert_constant(graph, nms + '_per_class_boxes_num', np.array(
                    per_class_boxes_num).astype(np.int32), nms, in_port=1, data_format='NHWC')
                insert_constant(graph, nms + '_total_class', np.array(
                    total_class).astype(np.int32), nms, in_port=2, data_format='NHWC')

                if need_insert_num1 > 0 or need_insert_num2 > 0:
                    post_concatenate_0 = get_valid_node_name(
                        graph, nms + '_concat_0')

                    graph.add_edge(post_reshape, post_concatenate_0, **
                                   {'src_out_port': 0, 'dst_in_port': 1})
                    graph.add_edge(post_concatenate_0, post_concatenate, **
                                   {'src_out_port': 0, 'dst_in_port': 0})
                    graph.remove_edge(post_reshape, post_concatenate)
                    insert_constant(graph, nms + '_out0', np.array(complete_res),
                                    post_concatenate_0, in_port=0)

                    NodeWrap(graph, post_concatenate_0).replace_obj(
                        'Concat', {'name': post_concatenate_0, 'opset_version': 11, 'axis': 0})

                NodeWrap(graph, nms).replace_obj(
                    'ArmNMS', {'name': nms,
                               'iou_threshold': nms_obj.iou_threshold,
                               'max_box_num': max_output_size,
                               'score_threshold': nms_obj.score_threshold,
                               'center_point_box': 0,
                               'max_output_boxes_per_class': nms_obj.max_output_boxes_per_class
                               })
                NodeWrap(graph, post_slice).replace_obj(
                    'Slice', {'name': post_slice, 'opset_version': 1, 'starts': post_slice_start, 'ends': post_slice_end, 'steps': post_slice_step})
                NodeWrap(graph, post_reshape).replace_obj(
                    'Reshape', {'name': post_reshape, 'opset_version': 1, 'shape': [-1, 1]})
                NodeWrap(graph, post_concatenate).replace_obj(
                    'Concat', {'name': post_concatenate, 'opset_version': 11, 'axis': 1})

                if nms in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(nms)
                    graph._attr['output_names'].remove(nms)
                    graph._attr['output_names'].insert(index, post_concatenate)

            else:
                WARN('[Parser]: The in_edge length of the node (%s) is illegal.' % nms)
        else:
            WARN('[Parser]: Meets invalid node (%s) in NonMaxSuppression!' % nms)


def convert_sigmoid_mul_to_silu(graph):
    matched = False
    matches = [matched_patterns(graph,
                                nodes=[
                                    ('input', {}),
                                    ('sigmoid', {'op': 'Sigmoid'}),
                                    ('mul', {'op': 'Mul'}),

                                ],
                                edges=[
                                    ('input', 'sigmoid'),
                                    ('input', 'mul', {
                                     'dst_in_port': 1 - i}),
                                    ('sigmoid', 'mul', {
                                        'src_out_port': 0, 'dst_in_port': i}),
                                ]) for i in range(2)]
    matches = extend_lists(matches)
    for m in matches:
        inp, sigmoid, mul = m['input'], m['sigmoid'], m['mul']
        sigmoid_obj = NodeWrap(graph, sigmoid)['object']
        mul_obj = NodeWrap(graph, mul)['object']
        sigmoid_in_edges = graph.sorted_in_edges(sigmoid, data=True)
        sigmoid_out_edges = graph.sorted_out_edges(sigmoid)
        mul_in_edges = graph.sorted_in_edges(mul, data=True)
        if mul_obj is not None \
                and sigmoid_obj is not None \
                and len(sigmoid_in_edges) == 1 \
                and len(sigmoid_out_edges) == 1:
            matched = True
            _, _, in_attr1 = sigmoid_in_edges[0]
            in_attr2 = {'src_out_port': 0, 'dst_in_port': 0}
            for src, _, in_attr in mul_in_edges:
                if inp == src:
                    if in_attr['src_out_port'] != in_attr1['src_out_port']:
                        matched = False
                    else:
                        in_attr2 = copy.deepcopy(in_attr)
                    break
            if not matched:
                continue
            graph.remove_edges_from(sigmoid_in_edges + mul_in_edges)
            in_attr2['dst_in_port'] = 0
            graph.add_edge(inp, mul, **in_attr2)
            silu_attr = mul_obj.copied_attr()
            NodeWrap(graph, mul).replace_obj('Silu', silu_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_sigmoid_mul_to_swish(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('mul1', {'op': 'Mul'}),
                                   ('sigmoid', {'op': 'Sigmoid'}),
                                   ('mul2', {'op': 'Mul'}),
                               ],
                               edges=[
                                   ('mul1', 'sigmoid'),
                                   ('sigmoid', 'mul2', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                               ])
    for m in matches:
        obj_dict = {name: NodeWrap(graph, m[name])['object']
                    for name in ['mul1', 'sigmoid', 'mul2']}
        if any(obj is None for obj in obj_dict.values()):
            WARN('[Parser]: Meets invalid Node in convert_sigmoid_mul_to_swish!')
            continue

        mul1_in_edges = graph.sorted_in_edges(m['mul1'], data=True)
        mul1_out_edges = graph.sorted_out_edges(m['mul1'], data=True)
        mul2_in_edges = graph.sorted_in_edges(m['mul2'], data=True)
        sigmoid_out_edges = graph.sorted_out_edges(m['sigmoid'], data=True)
        if len(mul1_in_edges) == 2 \
                and len(mul2_in_edges) == 2 \
                and len(mul1_out_edges) == 1 \
                and len(sigmoid_out_edges) == 1 \
                and mul1_in_edges[1][2]['tensor'].is_const\
                and mul2_in_edges[0][0] == mul1_in_edges[0][0]:
            matched = True

            inp, _, in_attr1 = mul1_in_edges[0]
            in_attr2 = {'src_out_port': 0, 'dst_in_port': 0}
            for src, _, in_attr in mul2_in_edges:
                if inp == src:
                    if in_attr['src_out_port'] != in_attr1['src_out_port']:
                        matched = False
                    else:
                        in_attr2 = copy.deepcopy(in_attr)
                    break
            if not matched:
                continue

            graph.remove_edges_from(mul1_in_edges + mul2_in_edges)
            graph.add_edge(inp, m['mul2'], **in_attr2)
            silu_attr = obj_dict['mul2'].copied_attr()
            silu_attr.update({'alpha': float(mul1_in_edges[1][2]['tensor'].value)})
            NodeWrap(graph, m['mul2']).replace_obj('Swish', silu_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_fill(graph):
    matches = single_node_matcher(graph, 'Fill')
    matched = False
    for m in matches:
        fill = m['target']
        fill_obj = NodeWrap(graph, fill)['object']
        in_edges = graph.sorted_in_edges(fill, data=True)
        out_edges = graph.sorted_out_edges(fill, data=True)

        if fill_obj is not None \
                and len(in_edges) == 2 \
                and in_edges[0][2]['tensor'].is_const:

            matched = True
            dims = in_edges[0][2]['tensor'].value
            reshape_dim = [1] * len(dims)
            tile_dim = dims
            need_tile = np.any(np.array(dims) != 1)

            fill_reshape = get_valid_node_name(graph, fill + '_reshape')

            graph.remove_edges_from(in_edges + out_edges)
            src1, _, in_attr = in_edges[1]
            in_attr['dst_in_port'] = 0
            graph.add_edge(src1, fill_reshape, **in_attr)
            insert_constant(graph, fill_reshape + '_shape', np.array(reshape_dim),
                            fill_reshape, in_port=1, data_format='NHWC')
            name_add = fill_reshape
            if need_tile:
                fill_tile = get_valid_node_name(graph, fill + '_tile')
                name_add = fill_tile
                graph.add_edge(fill_reshape, fill_tile)
                insert_constant(graph, fill_tile + '_reps', tile_dim,
                                fill_tile, in_port=1, data_format='NHWC')
                NodeWrap(graph, fill_tile).replace_obj(
                    'Tile', {'name': fill_tile, 'opset_version': 6})

            for _, dst, out_attr in out_edges:
                graph.add_edge(name_add, dst, **out_attr)

            if fill in graph._attr['output_names']:
                index = graph._attr['output_names'].index(fill)
                graph._attr['output_names'].remove(fill)
                graph._attr['output_names'].insert(index, name_add)

            NodeWrap(graph, fill_reshape).replace_obj(
                'Reshape', {'name': fill_reshape, 'opset_version': 5})

    if matched:
        clear_redundant_nodes(graph)


def convert_gather_to_slice(graph):
    matches = single_node_matcher(graph, 'Gather')
    for m in matches:
        gather = m['target']
        gather_obj = NodeWrap(graph, gather)['object']
        if gather_obj is not None:
            in_edges = graph.sorted_in_edges(gather)
            input_shapes = gather_obj.get_input_shapes()
            in_consts = gather_obj.sorted_in_consts()
            if len(in_edges) == 2 \
                    and len(input_shapes) == 2 \
                    and input_shapes[0] is not None \
                    and len(in_consts) == 1 \
                    and in_consts[0][2] is not None \
                    and in_consts[0][2].size == 1:
                graph.remove_edges_from(in_edges[1:])
                in_shape = input_shapes[0]
                gather_axis = (gather_obj.axis + len(in_shape)) if gather_obj.axis < 0 else gather_obj.axis
                indices = np.array(in_consts[0][2]).item()
                indices = (indices + in_shape[gather_axis]) if indices < 0 else indices

                starts = [0] * len(in_shape)
                ends = in_shape
                starts[gather_axis] = int(indices)
                ends[gather_axis] = starts[gather_axis] + 1
                axes = list(range(len(in_shape)))
                slice_attr = gather_obj.copied_attr()
                slice_attr.update(
                    {'name': gather, 'opset_version': 1, 'axes': axes, 'starts': starts, 'ends': ends})
                NodeWrap(graph, gather).replace_obj('Slice', slice_attr)

                if np.ndim(in_consts[0][2]) == 0:
                    old_dim = (np.array(ends, np.int64) - np.array(starts, np.int64)).tolist()
                    reshape_dim = old_dim[:gather_axis] + old_dim[(gather_axis+1):]
                    reshape = insert_reshape_after(graph, gather, reshape_dim, old_dim)
                    if gather in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(gather)
                        graph._attr['output_names'][index] = reshape


def convert_gemm_to_fc(graph):
    matches = single_node_matcher(graph, 'Gemm')
    for m in matches:
        gemm = m['target']
        gemm_in_edges = graph.sorted_in_edges(gemm, data=True)
        if len(gemm_in_edges) in (2, 3) and len(gemm_in_edges[0][2]['tensor'].value.shape) == 2:
            input2 = gemm_in_edges[1][0]
            input3 = gemm_in_edges[2][0] if len(gemm_in_edges) == 3 else ''
            if NodeWrap(graph, input2)['object'].type == 'Constant' \
                    and (not input3 or NodeWrap(graph, input3)['object'].type == 'Constant'):
                gemm_obj = NodeWrap(graph, gemm)['object']
                W = NodeWrap(graph, input2)['object'].value * gemm_obj.alpha
                if bool(gemm_obj.transB):
                    W = np.transpose(W)
                num_output = W.shape[-1]
                b = NodeWrap(graph, input3)[
                    'object'].value * gemm_obj.beta if input3 else np.zeros((num_output,), np.float32)
                fc_attr = gemm_obj.copied_attr()
                fc_attr.update({'weights': np.transpose(W), 'biases': b})
                NodeWrap(graph, gemm).replace_obj('FullyConnected', fc_attr)
                graph.remove_nodes_from([input2, input3])
                if bool(gemm_obj.transA):
                    transpose = get_valid_node_name(
                        graph, gemm + '_pre_transpose')
                    graph.add_node(transpose)
                    for src, _, in_attr in gemm_in_edges:
                        graph.remove_edge(src, gemm)
                        graph.add_edge(src, transpose, **in_attr)
                    graph.add_edge(transpose, gemm)
                    perm = [1, 0]
                    transpose_attr = gemm_obj.copied_attr()
                    transpose_attr.update(
                        {'name': transpose, 'opset_version': 1, 'perm': perm})
                    NodeWrap(graph, transpose).replace_obj(
                        'Transpose', transpose_attr)


def convert_global_pool(graph):
    for global_pool_type in ('GlobalMaxPool', 'GlobalAveragePool', 'GlobalLpPool'):
        matches = single_node_matcher(graph, global_pool_type)
        opset_version = 11 if global_pool_type == 'GlobalLpPool' else 10
        for m in matches:
            global_pool = m['target']
            global_pool_obj = NodeWrap(graph, global_pool)['object']
            if global_pool_obj is not None \
                    and len(global_pool_obj.get_input_shapes()) == 1 \
                    and global_pool_obj.get_input_shapes()[0] is not None \
                    and len(global_pool_obj.get_input_shapes()[0]) > 2:
                input_shape = global_pool_obj.get_input_shapes()[0]
                kernel_shape = input_shape[1:-1] \
                    if global_pool_obj.data_format == 'NHWC' \
                    else input_shape[2:]
                global_pool_attr = global_pool_obj.copied_attr()
                global_pool_attr.update({'opset_version': opset_version,
                                         'kernel_shape': kernel_shape,
                                         'dilations': [1] * len(kernel_shape),
                                         'strides': [1] * len(kernel_shape)
                                         })
                pool_type = global_pool_type.replace('Global', '')
                NodeWrap(graph, global_pool).replace_obj(
                    pool_type, global_pool_attr)


def convert_einsum(graph):
    matches = single_node_matcher(graph, 'Einsum')
    for m in matches:
        einsum = m['target']
        einsum_obj = NodeWrap(graph, einsum)['object']
        if einsum_obj is not None and einsum_obj.equation is not None:
            in_edges = graph.sorted_in_edges(einsum, data=True)
            if len(in_edges) == 2:
                equation = einsum_obj.equation
                equ_list = equation.split('-> ')
                add_list = equ_list[0].split(', ')
                if len(add_list[1].strip()) >= 5 and len(add_list[0].strip()) >= 5:
                    if add_list[0][0] == add_list[1][0] \
                            and add_list[0][4] == add_list[1][4] \
                            and add_list[0][2] != add_list[1][2]:
                        ein_src, _, ein_in_attr = in_edges[1]
                        insert_transpose(graph, ein_src, einsum,
                                         ein_in_attr, [0, 2, 1])
                        matmul_attr = einsum_obj.copied_attr()
                        matmul_attr.update({'opset_version': 13})
                        NodeWrap(graph, einsum).replace_obj(
                            'MatMul', matmul_attr)
                    elif add_list[0][0] == add_list[1][0] \
                            and add_list[0][4] != add_list[1][4] \
                            and add_list[0][2] != add_list[1][2]:
                        matmul_attr = einsum_obj.copied_attr()
                        matmul_attr.update({'opset_version': 13})
                        NodeWrap(graph, einsum).replace_obj(
                            'MatMul', matmul_attr)
                    else:
                        WARN(
                            '[Parser]: This equation is currently not supported in convert_einsum.!')
                else:
                    WARN('[Parser]: The length of the string is illegal.')
        else:
            WARN('[Parser]: Meets invalid node in convert_einsum!')


def convert_special_clip_to_relu(graph):
    matches = single_node_matcher(graph, 'Clip')
    for m in matches:
        clip = m['target']
        clip_obj = NodeWrap(graph, clip)['object']
        if clip_obj is not None:
            inputs = clip_obj.get_input_tensors()
            if len(inputs) == 3 \
                    and inputs[1] is not None \
                    and inputs[2] is not None \
                    and FLOAT_EQUAL(inputs[1], 0) \
                    and inputs[2] >= TYPE_MAX(inputs[0].dtype):
                in_edges = graph.sorted_in_edges(clip)
                graph.remove_edges_from(in_edges[1:])
                NodeWrap(graph, clip).replace_obj(
                    'Relu', {'name': clip, 'opset_version': 6})
        else:
            WARN(
                '[Parser]: Meets invalid Clip Node (%s) in convert_special_clip_to_relu!' % clip)


def convert_special_matmul_to_fc(graph):
    matched = False
    matches = single_node_matcher(graph, 'MatMul')
    for m in matches:
        matmul = m['target']
        matmul_obj = NodeWrap(graph, matmul)['object']
        if matmul_obj is not None \
                and len(graph.sorted_in_edges(matmul)) == 2 \
                and all([t is not None and len(t.shape) == 2 for t in matmul_obj.get_input_tensors()]) \
                and len(matmul_obj.sorted_in_consts()) == 1:
            in_edges = graph.sorted_in_edges(matmul, data=True)
            const_index = 0 if NodeWrap(graph, in_edges[0][0])[
                'object'].type == 'Constant' else 1
            if const_index == 0:
                continue
            matched = True
            const = in_edges[const_index][0]
            weights = np.transpose(NodeWrap(graph, const)['object'].value)
            biases = np.zeros((weights.shape[0],), np.float32)
            graph.remove_edge(const, matmul)
            matmul_attr = matmul_obj.copied_attr()
            matmul_attr.update({'weights': weights, 'biases': biases})
            NodeWrap(graph, matmul).replace_obj('FullyConnected', matmul_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_upsample_to_resize(graph):
    matches = single_node_matcher(graph, 'Upsample')
    for m in matches:
        upsample = m['target']
        upsample_obj = NodeWrap(graph, upsample)['object']
        in_edges = graph.sorted_in_edges(upsample, data=True)
        if upsample_obj is not None and upsample_obj.cur_version == 9 and len(in_edges) == 2:
            scales_inp, _, scales_in_attr = in_edges[1]
            graph.remove_edges_from(in_edges[1:])
            insert_constant(graph, upsample + '_roi',
                            np.array([], np.float32), upsample, in_port=1)
            new_scales_in_attr = copy.deepcopy(scales_in_attr)
            new_scales_in_attr['dst_in_port'] = 2
            graph.add_edge(scales_inp, upsample, **new_scales_in_attr)
            insert_constant(graph, upsample + '_sizes',
                            np.array([], np.float32), upsample, in_port=3)
            resize_attr = upsample_obj.copied_attr()
            resize_attr.update({'opset_version': 13})
            resize_attr.update(
                {'coordinate_transformation_mode': 'asymmetric'})
            if upsample_obj.mode == 'nearest':
                resize_attr.update(
                    {'nearest_mode': 'simple'})
            NodeWrap(graph, upsample).replace_obj('Resize', resize_attr)
        else:
            WARN(
                '[Parser]: Meets invalid Upsample Op (%s) in convert_upsample_to_resize!' % upsample)


def convert_special_resize(graph):
    matches = single_node_matcher(graph, 'Resize')
    for m in matches:
        resize = m['target']
        resize_obj = NodeWrap(graph, resize)['object']
        if resize_obj is not None and getattr(resize_obj, 'scales', None) is not None:
            if resize_obj.mode == 'nearest' \
                    and resize_obj.scales.size > 4 \
                    and FLOAT_EQUAL(resize_obj.scales, np.round(resize_obj.scales)) \
                    and len(resize_obj.get_input_shapes()) >= 1 \
                    and len(resize_obj.get_input_shapes()[0]) > 4 \
                    and len(resize_obj.get_output_shapes()) >= 1 \
                    and len(resize_obj.get_output_shapes()[0]) > 4:
                in_shape = resize_obj.get_input_shapes()[0]
                out_shape = resize_obj.get_output_shapes()[0]
                pre_reshape_dim = [-1] + in_shape[2:] + [1]
                tile_reps = resize_obj.scales.astype(np.int32).tolist()
                post_reshape_dim = out_shape[:]

                pre_reshape = get_valid_node_name(
                    graph, resize + '_pre_reshape')

                in_edges = graph.sorted_in_edges(resize, data=True)
                src, _, in_attr = in_edges[0]
                graph.remove_edges_from(in_edges)
                graph.add_edge(src, pre_reshape, **in_attr)
                graph.add_edge(pre_reshape, resize)
                insert_tile(graph,
                            pre_reshape,
                            resize,
                            {'src_out_port': 0, 'dst_in_port': 0,
                                'tensor': Tensor(value=None)},
                            tile_reps,
                            data_format='NCHW')

                NodeWrap(graph, pre_reshape).replace_obj(
                    'Reshape', {'name': pre_reshape, 'opset_version': 5})
                insert_constant(graph,
                                pre_reshape + '_dim',
                                np.array(pre_reshape_dim, np.int32),
                                pre_reshape,
                                in_port=1)
                NodeWrap(graph, resize).replace_obj(
                    'Reshape', {'name': resize, 'opset_version': 5})
                insert_constant(graph,
                                resize + '_dim',
                                np.array(post_reshape_dim, np.int32),
                                resize,
                                in_port=1)
        else:
            WARN(
                '[Parser]: Meets invalid Resize Op (%s) in convert_special_resize!' % resize)


def convert_special_transpose(graph):
    matches = single_node_matcher(graph, 'Transpose')
    for m in matches:
        transpose = m['target']
        transpose_obj = NodeWrap(graph, transpose)['object']
        if transpose_obj is not None:
            if len(transpose_obj.perm) != 6 \
                    or transpose_obj.perm[0] != 0:
                continue
            in_edges = graph.sorted_in_edges(transpose, data=True)
            out_edges = graph.sorted_out_edges(transpose, data=True)
            if len(in_edges) < 1 or len(out_edges) < 1:
                continue
            input_shapes = transpose_obj.get_input_shapes()
            output_shapes = transpose_obj.get_output_shapes()
            if len(input_shapes) >= 1 \
                    and input_shapes[0] is not None \
                    and len(input_shapes[0]) == 6 \
                    and input_shapes[0][0] == 1 \
                    and len(output_shapes) >= 1 \
                    and output_shapes[0] is not None \
                    and len(output_shapes[0]) == 6:
                dim1 = input_shapes[0][1:]
                dim2 = output_shapes[0][:]
                src, _, in_attr = in_edges[0]
                transpose_obj.perm = (
                    np.array(transpose_obj.perm[1:], np.int32) - 1).tolist()
                insert_reshape(graph, src, transpose, in_attr, dim1)
                post_reshape = insert_reshape_after(graph, transpose, dim2)
                if transpose in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(transpose)
                    graph._attr['output_names'][index] = post_reshape

        else:
            WARN(
                '[Parser]: Meets invalid Transpose Op(%s) in convert_special_transpose!' % transpose)


def decompose_const_if(graph, params):
    matched = False
    matches = single_node_matcher(graph, 'If')
    for m in matches:
        if_name = m['target']
        if_obj = NodeWrap(graph, if_name)['object']
        if if_obj is not None \
                and len(if_obj.sorted_in_consts()) >= 1 \
                and if_obj.sorted_in_consts()[0][1] == 0:
            condition = if_obj.sorted_in_consts()[0][2]
            keep_branch = if_obj.then_branch if condition else if_obj.else_branch
            removing_branch = if_obj.else_branch if condition else if_obj.then_branch
            keep_in_ports = keep_branch._attr['root_in_ports']
            removing_in_ports = removing_branch._attr['root_in_ports']
            out_edges = graph.sorted_out_edges(if_name, keys=True, data=True)
            if len(keep_in_ports) == len(out_edges):
                matched = True
                in_edges = graph.sorted_in_edges(if_name, keys=True, data=True)
                for src, _, k, in_attr in in_edges:
                    if in_attr['dst_in_port'] == 0 or in_attr['dst_in_port'] in removing_in_ports:
                        graph.remove_edge(src, if_name, k)
                new_in_edges = graph.sorted_in_edges(
                    if_name, keys=True, data=True)
                if len(new_in_edges) == len(out_edges):
                    for (in_edge, out_edge) in zip(new_in_edges, out_edges):
                        in_k, out_k = in_edge[2], out_edge[2]
                        new_attr = copy.deepcopy(in_edge[3])
                        new_attr['dst_in_port'] = out_edge[3]['dst_in_port']
                        graph.remove_edge(in_edge[0], if_name, key=in_k)
                        graph.remove_edge(if_name, out_edge[1], key=out_k)
                        graph.add_edge(in_edge[0], out_edge[1], **new_attr)
                    if if_name in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(if_name)
                        graph._attr['output_names'][index] = new_in_edges[0][0]
                        for in_edge in new_in_edges[1:]:
                            index += 1
                            graph._attr['output_names'].insert(
                                index, in_edge[0])
                    if params.get('input_names', []):
                        for inp in params['input_names'][:]:
                            _has_path = False
                            for out in graph._attr['output_names']:
                                if has_path(graph, inp, out):
                                    _has_path = True
                                    break
                            if _has_path is False:
                                ind = params['input_names'].index(inp)
                                if len(params['input_names']) == len(list(graph._attr['input_tensors'])):
                                    key = list(graph._attr['input_tensors'])[
                                        ind]
                                    graph._attr['input_tensors'].pop(key)
                                params['input_shapes'].pop(inp)
                                params['input_names'].remove(inp)
                    if params.get('output_names', []):
                        params['output_names'] = graph._attr['output_names']
                else:
                    WARN('[Parser]: Meets invalid edges in decompose_const_if!')
    if matched:
        clear_redundant_nodes(graph)


def decompose_pack(graph):
    matches = single_node_matcher(graph, 'ConcatFromSequence')
    for m in matches:
        pack_or_concat = m['target']
        pack_or_concat_obj = NodeWrap(graph, pack_or_concat)['object']
        if pack_or_concat_obj is not None:
            if pack_or_concat_obj.new_axis:
                in_edges = graph.sorted_in_edges(
                    pack_or_concat, keys=True, data=True)
                input_shapes = pack_or_concat_obj.get_input_shapes()
                if len(in_edges) == len(input_shapes) and all([in_shape != [] for in_shape in input_shapes]):
                    reshape_dim = list(input_shapes[0])
                    pos = pack_or_concat_obj.axis if pack_or_concat_obj.axis >= 0 else (
                        len(reshape_dim) + 1 + pack_or_concat_obj.axis)
                    reshape_dim.insert(pos, 1)
                    for src, _, k, in_attr in in_edges:
                        insert_reshape(
                            graph, src, pack_or_concat, in_attr, reshape_dim, key=k, data_format='NCHW')
                else:
                    WARN(
                        '[Parser]: Meets invalid input shapes for Node (%s) in decompose_pack!' % pack_or_concat)
            node_attr = pack_or_concat_obj.copied_attr()
            node_attr.update({'opset_version': 4})
            NodeWrap(graph, pack_or_concat).replace_obj('Concat', node_attr)
        else:
            WARN(
                '[Parser]: Meets invalid ConcatFromSequence Op (%s) in decompose_pack!' % pack_or_concat)


def fuse_bias(graph):
    linear_op_list = list(set(BaseLinearOp.get_concrete_subclass_names()).intersection(
        OnnxOp.get_concrete_subclass_names())) + ['FullyConnected']
    matches1 = [matched_patterns(graph,
                                 nodes=[('linear', {'op': linear_type}),
                                        ('add', {'op': 'Add'})
                                        ],
                                 edges=[('linear', 'add')]
                                 ) for linear_type in linear_op_list]
    matches2 = [matched_patterns(graph,
                                 nodes=[('linear', {'op': linear_type}),
                                        ('transpose', {'op': 'Transpose'}),
                                        ('add', {'op': 'Add'})
                                        ],
                                 edges=[('linear', 'transpose'),
                                        ('transpose', 'add'),
                                        ]
                                 ) for linear_type in linear_op_list]
    matches1 = extend_lists(matches1)
    matches2 = extend_lists(matches2)
    matches = matches1 + matches2
    for m in matches:
        linear, bias = m['linear'], m['add']
        transpose = m.get('transpose', None)
        linear_obj = NodeWrap(graph, linear)['object']
        bias_obj = NodeWrap(graph, bias)['object']
        transpose_obj = NodeWrap(graph, transpose)['object']
        if linear_obj is not None and bias_obj is not None and (transpose is None or transpose_obj is not None):
            if transpose_obj is not None and (len(graph.sorted_out_edges(transpose)) != 1 or transpose_obj.perm != [0, 2, 3, 1]):
                continue
            bias_inputs = bias_obj.get_input_tensors()
            if len(bias_inputs) == 2 and np.ndim(bias_inputs[1]) == 1 and linear_obj.num_output == bias_inputs[1].size:
                new_biases = bias_inputs[1] if linear_obj.biases is None else (
                    linear_obj.biases + bias_inputs[1])
                linear_obj.biases = new_biases
                if transpose is None:
                    remove_node_safely(graph, bias)
                else:
                    bias_out_edges = graph.sorted_out_edges(bias, data=True)
                    graph.remove_edge(transpose, bias)
                    for _, dst, out_attr in bias_out_edges:
                        graph.remove_edge(bias, dst)
                        graph.add_edge(transpose, dst, **out_attr)
                    if bias in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(bias)
                        graph._attr['output_names'][index] = transpose
        else:
            WARN('[Parser]: Meets invalid Op in fuse_bias!')


def fuse_gather_const_mul(graph):
    init_matches = matched_patterns(graph,
                                    nodes=[('const', {'op': 'Constant'}),
                                           ('gather', {'op': 'Gather'})
                                           ],
                                    edges=[('const', 'gather', {'src_out_port': 0, 'dst_in_port': 0})
                                           ]
                                    )
    for init_m in init_matches:
        const, gather1 = init_m['const'], init_m['gather']
        if not graph.has_node(const) or not graph.has_node(gather1):
            continue
        const_obj, gather1_obj = [
            NodeWrap(graph, name)['object'] for name in [const, gather1]]
        if const_obj is not None and gather1_obj is not None:
            const_out_edges = graph.sorted_out_edges(const)
            if len(const_out_edges) >= 1:
                matched = False
                gather_num = len(const_out_edges)
                nodes = [('const', {'op': 'Constant'})] \
                    + [('gather_%s' % str(i), {'op': 'Gather'}) for i in range(gather_num)] \
                    + [('multiplier_%s' % str(i), {'op': 'Constant'}) for i in range(gather_num)] \
                    + [('mul_%s' % str(i), {'op': 'Mul'})
                        for i in range(gather_num)]
                edges = [('const', 'gather_%s' % str(i), {'src_out_port': 0, 'dst_in_port': 0}) for i in range(gather_num)] \
                    + [('gather_%s' % str(i), 'mul_%s' % str(i)) for i in range(gather_num)] \
                    + [('multiplier_%s' % str(i), 'mul_%s' % str(i),
                        {'src_out_port': 0, 'dst_in_port': 1}) for i in range(gather_num)]
                matches = matched_patterns(graph, nodes=nodes, edges=edges)
                for m in matches:
                    const = m['const']
                    const_obj = NodeWrap(graph, const)['object']
                    if const_obj is None:
                        continue
                    multiplier_names = ['multiplier_%s' %
                                        str(i) for i in range(gather_num)]
                    multiplier_objs = [
                        NodeWrap(graph, m[n])['object'] for n in multiplier_names]
                    if any([(obj is None or obj.value is None) for obj in multiplier_objs]):
                        continue
                    if multiplier_objs[0].value.size != 1:
                        continue
                    multiplier_value = multiplier_objs[0].value
                    if len(multiplier_objs) > 1 \
                            and any([not FLOAT_EQUAL(multiplier_value, obj.value) for obj in multiplier_objs[1:]]):
                        continue
                    matched = True
                    const_obj.value *= multiplier_value
                    gather_names = [m['gather_%s' %
                                      str(i)] for i in range(gather_num)]
                    mul_names = [m['mul_%s' % str(i)]
                                 for i in range(gather_num)]
                    for gather, mul in zip(gather_names, mul_names):
                        out_edges = graph.sorted_out_edges(mul, data=True)
                        graph.remove_edge(gather, mul)
                        for _, dst, out_attr in out_edges:
                            graph.remove_edge(mul, dst)
                            graph.add_edge(gather, dst, **out_attr)
                        if mul in graph._attr['output_names']:
                            index = graph._attr['output_names'].index(mul)
                            graph._attr['output_names'][index] = gather
                if matched:
                    clear_redundant_nodes(graph)
        else:
            WARN('[Parser]: Meets invalid Operator object in fuse_gather_const_mul!')


def fuse_linear_bn(graph):
    matched = False
    linear_op_list = list(set(BaseLinearOp.get_concrete_subclass_names()).intersection(
        OnnxOp.get_concrete_subclass_names())) + ['FullyConnected']
    matches1 = [matched_patterns(graph,
                                 nodes=[('linear', {'op': linear_type}),
                                        ('bn', {'op': 'BatchNormalization'})
                                        ],
                                 edges=[('linear', 'bn')]
                                 ) for linear_type in linear_op_list]
    matches2 = [matched_patterns(graph,
                                 nodes=[('linear', {'op': linear_type}),
                                        ('transpose', {'op': 'Transpose'}),
                                        ('bn', {'op': 'BatchNormalization'})
                                        ],
                                 edges=[('linear', 'transpose'),
                                        ('transpose', 'bn'),
                                        ]
                                 ) for linear_type in linear_op_list]
    matches1 = extend_lists(matches1)
    matches2 = extend_lists(matches2)
    matches = matches1 + matches2
    for m in matches:
        linear, bn = m['linear'], m['bn']
        transpose = m.get('transpose', None)
        linear_obj = NodeWrap(graph, linear)['object']
        bn_obj = NodeWrap(graph, bn)['object']
        transpose_obj = NodeWrap(graph, transpose)['object']
        if linear_obj is not None and bn_obj is not None and (transpose is None or transpose_obj is not None):
            if bn_obj.training_mode:
                continue
            if len(graph.sorted_out_edges(linear)) != 1:
                continue
            if transpose_obj is not None and (len(graph.sorted_out_edges(transpose)) != 1 or transpose_obj.perm != [0, 2, 3, 1]):
                continue
            bn_inputs = bn_obj.get_input_tensors()
            if len(bn_inputs) != 5:
                continue
            if len(bn_inputs[1].shape) > 1 \
                    or len(bn_inputs[2].shape) > 1 \
                    or len(bn_inputs[3].shape) > 1 \
                    or len(bn_inputs[4].shape) > 1 \
                    or bn_inputs[1].size != linear_obj.num_output:
                continue
            matched = True
            eps = bn_obj.epsilon
            scale, shift, mean, var = bn_inputs[1:]
            new_scale = scale / np.sqrt(var + eps)
            if linear_obj.type == 'ConvTranspose':
                new_scale_shape = (1,) + new_scale.shape + \
                    (1,) * (len(linear_obj.weights.shape) - 2)
            else:
                new_scale_shape = new_scale.shape + \
                    (1,) * (len(linear_obj.weights.shape) - 1)
            new_weights = new_scale.reshape(
                new_scale_shape) * linear_obj.weights
            new_biases = (linear_obj.biases - mean) * new_scale + shift
            linear_obj.weights = new_weights
            linear_obj.biases = new_biases
            if transpose is None:
                remove_node_safely(graph, bn)
            else:
                bn_out_edges = graph.sorted_out_edges(bn, data=True)
                graph.remove_edge(transpose, bn)
                for _, dst, out_attr in bn_out_edges:
                    graph.remove_edge(bn, dst)
                    graph.add_edge(transpose, dst, **out_attr)
                if bn in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(bn)
                    graph._attr['output_names'][index] = transpose
        else:
            WARN('[Parser]: Meets invalid node in fuse_linear_bn!')
    if matched:
        clear_redundant_nodes(graph)


def fuse_mul_add_or_sub(graph):
    matched = False
    add_sub = ['Add', 'Sub']
    bn_matches = [matched_patterns(graph,
                                   nodes=[
                                       ('inp', {}),
                                       ('mul', {'op': 'Mul'}),
                                       ('const_1', {'op': 'Constant'}),
                                       ('add_sub', {'op': op}),
                                       ('const_2', {'op': 'Constant'})
                                   ],
                                   edges=[
                                       ('inp', 'mul'),
                                       ('const_1', 'mul'),
                                       ('mul', 'add_sub'),
                                       ('const_2', 'add_sub')
                                   ]) for op in add_sub]
    bn_matches = extend_lists(bn_matches)
    for m in bn_matches:
        inp, mul, add_sub = m['inp'], m['mul'], m['add_sub']
        const_1, const_2 = m['const_1'], m['const_2']
        input_obj = NodeWrap(graph, inp)['object']
        mul_obj = NodeWrap(graph, mul)['object']
        add_sub_obj = NodeWrap(graph, add_sub)['object']

        if input_obj is not None and mul_obj is not None and add_sub_obj is not None:
            if mul_obj.quantize \
                    or add_sub_obj.quantize:
                continue

            mul_out_edges = graph.sorted_out_edges(mul)
            if len(mul_out_edges) > 1:
                continue
            input_shapes = input_obj.get_output_shapes()
            if len(input_shapes) < 1:
                continue

            input_out_shape = input_shapes[0]
            if input_out_shape is None or input_out_shape == []:
                continue

            num_output = input_out_shape[-1]

            weights = NodeWrap(graph, const_1)['object'].value
            biases = NodeWrap(graph, const_2)['object'].value
            if weights is None \
                    or biases is None \
                    or (weights.size != num_output and weights.size != 1) \
                    or (biases.size != num_output and biases.size != 1):
                continue

            if len(input_out_shape) > 1 \
                    and len(weights.shape) > 1 \
                    and input_out_shape[-1] != weights.shape[-1]:
                continue

            if len(input_out_shape) > 1 \
                    and len(biases.shape) > 1 \
                    and input_out_shape[-1] != biases.shape[-1]:
                continue

            matched = True

            if weights.size < num_output and weights.size == 1:
                weights = np.tile(weights, num_output)
            if biases.size < num_output and biases.size == 1:
                biases = np.tile(biases, num_output)
            if add_sub_obj.type == 'Sub':
                biases = (-1.0 * biases).astype(np.float32)
            if np.ndim(weights) > 1:
                weights = np.squeeze(weights)
            if np.ndim(biases) > 1:
                biases = np.squeeze(biases)
            mean_value = np.zeros((num_output,), np.float32)
            var_value = np.ones((num_output,), np.float32)

            graph.remove_edge(const_1, mul)
            add_sub_out_edges = graph.sorted_out_edges(add_sub, data=True)
            for _, out, out_attr in add_sub_out_edges:
                graph.remove_edge(add_sub, out)
                graph.add_edge(mul, out, **out_attr)
            if add_sub in graph._attr['output_names']:
                index = graph._attr['output_names'].index(add_sub)
                graph._attr['output_names'].pop(index)
                if mul not in graph._attr['output_names']:
                    graph._attr['output_names'].insert(index, mul)
            graph.remove_node(add_sub)

            gamma = get_valid_node_name(graph, mul + '_bn_gamma')
            beta = get_valid_node_name(graph, mul + '_bn_beta')
            mean = get_valid_node_name(graph, mul + '_bn_mean')
            var = get_valid_node_name(graph, mul + '_bn_var')

            graph.add_nodes_from([gamma, beta, mean, var])
            gamma_attr = {'name': gamma, 'value': weights, 'data_format': 'NHWC',
                          'opset_version': 9}
            beta_attr = {'name': beta, 'value': biases, 'data_format': 'NHWC',
                         'opset_version': 9}
            mean_attr = {'name': mean, 'value': mean_value, 'data_format': 'NHWC',
                         'opset_version': 9}
            var_attr = {'name': var, 'value': var_value, 'data_format': 'NHWC',
                        'opset_version': 9}
            NodeWrap(graph, gamma).replace_obj('Constant', gamma_attr)
            NodeWrap(graph, beta).replace_obj('Constant', beta_attr)
            NodeWrap(graph, mean).replace_obj('Constant', mean_attr)
            NodeWrap(graph, var).replace_obj('Constant', var_attr)

            bn_attr = mul_obj.copied_attr()
            bn_attr.update({'epsilon': 0, 'data_format': 'NHWC'})
            NodeWrap(graph, mul).replace_obj('BatchNormalization', bn_attr)

            graph.add_edge(
                gamma, mul, **{'src_out_port': 0, 'dst_in_port': 1, 'tensor': Tensor(value=weights)})
            graph.add_edge(
                beta, mul, **{'src_out_port': 0, 'dst_in_port': 2, 'tensor': Tensor(value=biases)})
            graph.add_edge(
                mean, mul, **{'src_out_port': 0, 'dst_in_port': 3, 'tensor': Tensor(value=mean_value)})
            graph.add_edge(
                var, mul, **{'src_out_port': 0, 'dst_in_port': 4, 'tensor': Tensor(value=var_value)})

    if matched:
        clear_redundant_nodes(graph)


def fuse_pad(graph):
    pad_op_list = ['Pad']
    op_has_padding_list = list(set(OpHasPaddingStrides.get_concrete_subclass_names(
    )).difference(['ConvTranspose']).intersection(OnnxOp.get_concrete_subclass_names()))
    pad_fusing_combinations = itertools.product(
        pad_op_list, op_has_padding_list)
    for pad_op, op_has_padding in pad_fusing_combinations:
        matches = two_nodes_matcher(graph, pad_op, op_has_padding)
        for m in matches:
            pad, op_has_padding = m['begin'], m['end']
            pad_out_edges = graph.sorted_out_edges(pad)
            pad_obj = NodeWrap(graph, pad)['object']
            if pad_obj is not None:
                if len(pad_out_edges) == 1 and pad_obj.is_fusable():
                    space_pads = pad_obj.space_pads()
                    op_has_padding_obj = NodeWrap(
                        graph, op_has_padding)['object']
                    init_pads = op_has_padding_obj.pads
                    fused_pads = np.reshape(np.array(init_pads, np.int64), newshape=(2, -1)) \
                        + np.reshape(np.array(space_pads, np.int64),
                                     newshape=(2, -1))
                    op_has_padding_obj.pads = fused_pads.flatten().tolist()
                    op_has_padding_obj.auto_pad = 'NOTSET'
                    pad_in_edges = graph.sorted_in_edges(pad, data=True)
                    src, _, attr = pad_in_edges[0]
                    graph.remove_edge(src, pad)
                    graph.add_edge(src, op_has_padding, **attr)
                    graph.remove_node(pad)
            else:
                WARN('[Parser]: Meets invalid Pad Node (%s) in fuse_pad!' % pad)


def convert_abnormal_reshape(graph):
    matches = single_node_matcher(graph, 'Reshape')
    for m in matches:
        reshape = m['target']
        reshape_obj = NodeWrap(graph, reshape)['object']
        input_shapes = reshape_obj.get_input_shapes()
        output_shapes = reshape_obj.get_output_shapes()
        if input_shapes and output_shapes:
            input_shape = input_shapes[0]
            output_shape = output_shapes[0]
            if input_shape is not None \
                    and output_shape is not None \
                    and len(input_shape) == 4 \
                    and len(output_shape) == 4 \
                    and input_shape != output_shape \
                    and sorted(input_shape) == sorted(output_shape) \
                    and 1 in input_shape:
                if Op.shape_nchw_to_nhwc(input_shape) == output_shape \
                        or Op.shape_nhwc_to_nchw(input_shape) == output_shape:
                    perm = [0, 2, 3, 1] if Op.shape_nchw_to_nhwc(
                        input_shape) == output_shape else [0, 3, 1, 2]
                    extra_in_edges = graph.sorted_in_edges(reshape)[1:]
                    graph.remove_edges_from(extra_in_edges)
                    transpose_attr = reshape_obj.copied_attr()
                    transpose_attr.update({'opset_version': 1, 'perm': perm})
                    NodeWrap(graph, reshape).replace_obj(
                        'Transpose', transpose_attr)


def convert_to_const(graph, op_type_name_list):
    if len(graph) and op_type_name_list:
        for node_name in graph.nodes:
            node = NodeWrap(graph, node_name)
            node_obj = node['object']
            if isinstance(node_obj, OpHasOneOutPort) and node_obj.type in op_type_name_list:
                out_tensors = node_obj.get_output_tensors()
                if len(out_tensors) >= 1 and out_tensors[0] is not None and node_obj.is_all_outputs_const():
                    new_attr = node_obj.copied_attr()
                    new_attr.update({'value': out_tensors[0].copy()})
                    node.replace_obj('Constant', new_attr)
                    const_in_edges = graph.sorted_in_edges(node_name)
                    graph.remove_edges_from(const_in_edges)
        clear_redundant_nodes(graph)
    else:
        WARN('[Parser]: Invalid params for convert_to_const!')


def convert_min_max_to_clip(graph):
    matched = False
    matches = [matched_patterns(graph,
                                nodes=[
                                    ('p1', {'op': pair[0]}),
                                    ('const_1', {'op': 'Constant'}),
                                    ('p2', {'op': pair[1]}),
                                    ('const_2', {'op': 'Constant'})
                                ],
                                edges=[
                                    ('const_1', 'p1'),
                                    ('p1', 'p2'),
                                    ('const_2', 'p2')
                                ]) for pair in [('Min', 'Max'), ('Max', 'Min')]]
    matches = extend_lists(matches)
    for m in matches:
        p1, p2 = m['p1'], m['p2']
        if not graph.has_node(p1) or not graph.has_node(p2):
            WARN('[Parser]: Node (%s or %s) cannot be found, graph maybe has been changed!' % (
                p1, p2))
            continue
        p1_out_edges = graph.sorted_out_edges(p1)
        c1, c2 = m['const_1'], m['const_2']
        const_value_1 = NodeWrap(graph, c1)['object'].value
        const_value_2 = NodeWrap(graph, c2)['object'].value
        if len(p1_out_edges) == 1 \
                and (const_value_1.size == 1 or np.all(const_value_1 == const_value_1.flatten()[0])) \
                and (const_value_2.size == 1 or np.all(const_value_2 == const_value_2.flatten()[0])):
            matched = True

            clip_value_1 = const_value_1.flatten()[0]
            clip_value_2 = const_value_2.flatten()[0]
            clip_min = clip_value_1 if (
                NodeWrap(graph, p1)['object'].type == 'Max') else clip_value_2
            clip_max = clip_value_1 if (
                NodeWrap(graph, p1)['object'].type == 'Min') else clip_value_2

            p1_in_edges = graph.sorted_in_edges(p1, data=True)
            p2_in_edges = graph.sorted_in_edges(p2)
            graph.remove_edges_from(p2_in_edges)
            for src, _, in_attr in p1_in_edges:
                if src != c1:
                    p2_in_attr = copy.deepcopy(in_attr)
                    p2_in_attr.update({'dst_in_port': 0})
                    graph.remove_edge(src, p1)
                    graph.add_edge(src, p2, **p2_in_attr)
                    break
            clip_attr = {'name': p2, 'opset_version': 6,
                         'min': clip_min, 'max': clip_max}
            NodeWrap(graph, p2).replace_obj('Clip', clip_attr)

            if p1 in graph._attr['output_names']:
                index = graph._attr['output_names'].index(p1)
                graph._attr['output_names'][index] = p2

    if matched:
        clear_redundant_nodes(graph)


def convert_reducemean_to_avgpool(graph):
    matches = single_node_matcher(graph, 'ReduceMean')
    for m in matches:
        mean = m['target']
        mean_obj = NodeWrap(graph, mean)['object']
        if mean_obj is None \
                or len(mean_obj.get_input_shapes()) < 1 \
                or len(mean_obj.get_output_shapes()) < 1:
            WARN(
                '[Parser]: Meets invalid ReduceMean Op(%s) in convert_reducemean_to_avgpool!' % mean)
            continue
        input_shape = mean_obj.get_input_shapes()[0]
        out_shape = mean_obj.get_output_shapes()[0]
        if input_shape is None \
                or len(input_shape) < 3 \
                or len(input_shape) > 5 \
                or any([s is None for s in input_shape]) \
                or out_shape is None \
                or any([s is None for s in out_shape]):
            continue
        axes = sorted(OpHasAxis.make_axes_non_negative(
            mean_obj.axes, len(input_shape)))
        if axes != list(range(len(input_shape)))[1:-1] \
                and axes != list(range(len(input_shape)))[2:]:
            continue
        keepdim_out_shape = []
        for index in range(len(input_shape)):
            keepdim_out_shape.append(
                1 if index in axes else input_shape[index])
        cur_data_format = mean_obj.data_format
        need_transpose = False
        if (cur_data_format == 'NCHW' and axes == list(range(len(input_shape)))[1:-1]) \
                or (cur_data_format == 'NHWC' and axes == list(range(len(input_shape)))[2:]):
            need_transpose = True
        kernel_shape = np.array(input_shape)[np.array(axes)].tolist()
        last_name = mean
        if need_transpose:
            if cur_data_format == 'NCHW':
                perm1 = [0, len(input_shape) - 1] + \
                    list(range(1, len(input_shape) - 1))
            else:
                perm1 = [0] + list(range(2, len(input_shape))) + [1]
            perm2 = Op.cal_inverse_perm(perm1)
            in_edges = graph.sorted_in_edges(mean, data=True)
            src, _, in_attr = in_edges[0]
            insert_transpose(graph, src, mean, in_attr, perm1)
            if not mean_obj.keepdims:
                for _, _, out_attr in graph.sorted_out_edges(mean, data=True):
                    if out_attr['tensor'].value is not None:
                        out_attr['tensor'].value = np.reshape(
                            out_attr['tensor'].value, keepdim_out_shape)
            last_name = insert_transpose_after(graph, mean, perm2)
            if mean in graph._attr['output_names']:
                index = graph._attr['output_names'].index(mean)
                graph._attr['output_names'][index] = last_name
        if not mean_obj.keepdims:
            reshape = insert_reshape_after(
                graph, last_name, out_shape, keepdim_out_shape)
            if last_name in graph._attr['output_names']:
                index = graph._attr['output_names'].index(last_name)
                graph._attr['output_names'][index] = reshape
        pool_rank = len(input_shape) - 2
        pool_attr_dict = mean_obj.copied_attr()
        pool_attr_dict.update({'opset_version': 11,
                               'data_format': cur_data_format,
                               'auto_pad': 'VALID',
                               'pads': [0] * pool_rank * 2,
                               'kernel_shape': kernel_shape,
                               'strides': [1] * pool_rank,
                               'dilations': [1] * pool_rank,
                               'count_include_pad': 0,
                               })
        NodeWrap(graph, mean).replace_obj('AveragePool', pool_attr_dict)


def convert_dequantizelinear(graph):
    matches = single_node_matcher(graph, 'DequantizeLinear')
    for m in matches:
        dequant = m['target']
        dequant_obj = NodeWrap(graph, dequant)['object']
        dequant_in_edges = graph.sorted_in_edges(dequant, data=True)
        if dequant_obj is None \
                or len(dequant_in_edges) != 3 \
                or len(dequant_obj.get_input_shapes()) != 3 \
                or any([s is None for s in dequant_obj.get_input_shapes()]) \
                or any([s is None for s in dequant_obj.get_input_shapes()[0]]):
            WARN(
                '[Parser]: Meets invalid DequantizeLinear Op(%s) in convert_dequantizelinear!' % dequant)
            continue
        input_shapes = dequant_obj.get_input_shapes()
        dequant_axis = dequant_obj.axis if len(input_shapes[1]) == 1 else -1
        dequant_axis = OpHasAxis.make_axes_non_negative(
            dequant_axis, len(input_shapes[0]))
        if len(input_shapes[0]) <= dequant_axis:
            WARN(
                '[Parser]: Meets invalid axis(%d) in DequantizeLinear Op(%s) in convert_dequantizelinear!' % (dequant_axis, dequant))
            continue
        axis_dim = input_shapes[0][dequant_axis]
        if input_shapes[1] != input_shapes[2] \
                or len(input_shapes[1]) not in (0, 1) \
                or (len(input_shapes[1]) == 1 and input_shapes[1][0] != axis_dim):
            WARN(
                '[Parser]: Meets different shapes of x_scale and x_zero_point in DequantizeLinear Op(%s) in convert_dequantizelinear!' % dequant)
            continue
        inp, _, inp_in_attr = dequant_in_edges[0]
        scale, _, scale_in_attr = dequant_in_edges[1]
        scale_is_const = scale_in_attr['tensor'].is_const
        zp, _, zp_in_attr = dequant_in_edges[2]
        zp_is_const = zp_in_attr['tensor'].is_const
        graph.remove_edges_from(dequant_in_edges)
        if scale_is_const and zp_is_const:
            scale_value = scale_in_attr['tensor'].value
            zp_value = zp_in_attr['tensor'].value if zp else np.array(
                0, np.int32)
            tiled_const_scale = np.tile(
                scale_value, axis_dim) if scale_value.size == 1 else scale_value
            tiled_const_zp = np.tile(zp_value,
                                     axis_dim) if zp_value.size == 1 else zp_value
            gamma_value = tiled_const_scale.astype(np.float32)
            beta_value = (-(tiled_const_zp * tiled_const_scale)
                          ).astype(np.float32)
            mean_value = np.zeros((axis_dim, ), np.float32)
            var_value = np.ones((axis_dim,), np.float32)
            gamma = get_valid_node_name(graph, dequant + '_gamma')
            beta = get_valid_node_name(graph, dequant + '_beta')
            mean = get_valid_node_name(graph, dequant + '_mean')
            var = get_valid_node_name(graph, dequant + '_var')

            graph.add_edge(inp, dequant, **inp_in_attr)
            graph.add_edge(
                gamma, dequant, **{'src_out_port': 0, 'dst_in_port': 1, 'tensor': Tensor(value=gamma_value)})
            graph.add_edge(
                beta, dequant, **{'src_out_port': 0, 'dst_in_port': 2, 'tensor': Tensor(value=beta_value)})
            graph.add_edge(
                mean, dequant, **{'src_out_port': 0, 'dst_in_port': 3, 'tensor': Tensor(value=mean_value)})
            graph.add_edge(
                var, dequant, **{'src_out_port': 0, 'dst_in_port': 4, 'tensor': Tensor(value=var_value)})

            batchnorm_attr = dequant_obj.copied_attr()
            batchnorm_attr.update(
                {'opset_version': 9, 'epsilon': 0, 'data_format': 'NHWC'})
            NodeWrap(graph, dequant).replace_obj(
                'BatchNormalization', batchnorm_attr)

            gamma_attr = {'name': gamma, 'value': gamma_value,
                          'data_format': 'NHWC', 'opset_version': 1}
            beta_attr = {'name': beta, 'value': beta_value,
                         'data_format': 'NHWC', 'opset_version': 1}
            mean_attr = {'name': mean, 'value': mean_value,
                         'data_format': 'NHWC', 'opset_version': 1}
            var_attr = {'name': var, 'value': var_value,
                        'data_format': 'NHWC', 'opset_version': 1}
            NodeWrap(graph, gamma).replace_obj('Constant', gamma_attr)
            NodeWrap(graph, beta).replace_obj('Constant', beta_attr)
            NodeWrap(graph, mean).replace_obj('Constant', mean_attr)
            NodeWrap(graph, var).replace_obj('Constant', var_attr)

            cast = insert_cast(graph, inp, dequant, 'float32')

            if dequant_axis != len(input_shapes[0]) - 1:
                pre_perm = [idx for idx in range(
                    len(input_shapes[0])) if idx != dequant_axis] + [dequant_axis]
                _, _, dequant_in_attr = graph.sorted_in_edges(dequant, data=True)[
                    0]
                insert_transpose(graph, cast, dequant,
                                 dequant_in_attr, pre_perm)
                post_trans = insert_transpose_after(
                    graph, dequant, Op.cal_inverse_perm(pre_perm))
                if dequant in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(dequant)
                    graph._attr['output_names'][index] = post_trans
            continue

        sub = get_valid_node_name(graph, dequant + '_sub')
        graph.add_node(sub)
        graph.add_edge(inp, sub, **inp_in_attr)
        new_zp_in_attr = copy.deepcopy(zp_in_attr)
        new_zp_in_attr.update({'dst_in_port': 1})
        graph.add_edge(zp, sub, **new_zp_in_attr)
        sub_out_attr = copy.deepcopy(inp_in_attr)
        sub_out_attr.update({'src_out_port': 0, 'dst_in_port': 0})
        graph.add_edge(sub, dequant, **sub_out_attr)
        NodeWrap(graph, sub).replace_obj(
            'Sub', {'name': sub, 'opset_version': 13})

        graph.add_edge(scale, dequant, **scale_in_attr)
        mul_attr = dequant_obj.copied_attr()
        mul_attr.update({'opset_version': 13})
        NodeWrap(graph, dequant).replace_obj('Mul', mul_attr)

        insert_cast(graph, sub, dequant, 'float32')

        if len(input_shapes[1]) == 1 and dequant_axis != len(input_shapes[0]) - 1:
            dim = [1 if idx != dequant_axis else axis_dim for idx in range(
                len(input_shapes[0]))]
            insert_reshape(graph, scale, dequant, scale_in_attr, dim)
            insert_reshape(graph, zp, sub, new_zp_in_attr, dim)


def convert_quantizelinear(graph):
    matches = single_node_matcher(graph, 'QuantizeLinear')
    for m in matches:
        quant = m['target']
        quant_obj = NodeWrap(graph, quant)['object']
        quant_in_edges = graph.sorted_in_edges(quant, data=True)
        if quant_obj is None \
                or len(quant_in_edges) != 3 \
                or len(quant_obj.get_input_shapes()) != 3 \
                or any(s is None for s in quant_obj.get_input_shapes()) \
                or any(s is None for s in quant_obj.get_input_shapes()[0]):
            WARN(
                '[Parser]: Meets invalid QuantizeLinear Op(%s) in convert_quantizelinear!' % quant)
            continue
        input_shapes = quant_obj.get_input_shapes()
        quant_axis = OpHasAxis.make_axes_non_negative(
            quant_obj.axis, len(input_shapes[0]))
        if input_shapes[1] != input_shapes[2] \
                or len(input_shapes[1]) not in (0, 1) \
                or (len(input_shapes[1]) == 1 and input_shapes[1][0] != input_shapes[0][quant_axis]):
            WARN(
                '[Parser]: Meets different shapes of y_scale and y_zero_point in QuantizeLinear Op(%s) in convert_quantizelinear!' % quant)
            continue
        inp, _, inp_in_attr = quant_in_edges[0]
        scale, _, scale_in_attr = quant_in_edges[1]
        zp, _, zp_in_attr = quant_in_edges[2]
        graph.remove_edges_from(quant_in_edges)

        div = get_valid_node_name(graph, quant + '_div')
        graph.add_edge(inp, div, **inp_in_attr)
        div_in_attr = copy.deepcopy(scale_in_attr)
        div_in_attr.update({'dst_in_port': 1})
        graph.add_edge(scale, div, **div_in_attr)
        NodeWrap(graph, div).replace_obj(
            'Div', {'name': div, 'opset_version': 13})
        # Insert cast before quant if input dtype is not float32
        if inp_in_attr['tensor'].value.dtype != 'float32':
            insert_cast(graph, inp, div, 'float32')

        # For (x / y_scale), it's rounding to nearest ties to even
        round_div = get_valid_node_name(graph, quant + '_round')
        common_attr = copy.deepcopy(inp_in_attr)
        common_attr.update({'src_out_port': 0,
                            'tensor': Tensor(value=np.random.ranf(input_shapes[0]).astype(np.float32))})
        graph.add_edge(div, round_div, **common_attr)
        NodeWrap(graph, round_div).replace_obj(
            'Round', {'name': round_div, 'opset_version': 11})

        add = get_valid_node_name(graph, quant + '_add')
        add_in_attr = copy.deepcopy(zp_in_attr)
        add_in_attr.update({'dst_in_port': 1})
        graph.add_edge(zp, add, **add_in_attr)
        graph.add_edge(round_div, add, **common_attr)
        NodeWrap(graph, add).replace_obj(
            'Add', {'name': add, 'opset_version': 13})
        add_operand = zp
        if len(input_shapes[1]) == 1 and quant_axis != len(input_shapes[0]) - 1:
            dim = [1 if idx != quant_axis else input_shapes[0][quant_axis]
                   for idx in range(len(input_shapes[0]))]
            insert_reshape(graph, scale, div, div_in_attr, dim)
            add_operand = insert_reshape(graph, zp, add, add_in_attr, dim)
        insert_cast(graph, add_operand, add, 'float32')

        # Insert clip and cast after quant
        clip = get_valid_node_name(graph, quant + '_clip')
        graph.add_edge(add, clip, **common_attr)
        zp_dtype = zp_in_attr['tensor'].value.dtype
        NodeWrap(graph, clip).replace_obj('Clip', {'name': clip,
                                                   'opset_version': 1,
                                                   'max': np.iinfo(zp_dtype).max,
                                                   'min': np.iinfo(zp_dtype).min})

        graph.add_edge(clip, quant, **common_attr)
        post_cast_attr = quant_obj.copied_attr()
        post_cast_attr.update({'opset_version': 1, 'to': str(zp_dtype)})
        NodeWrap(graph, quant).replace_obj('Cast', post_cast_attr)


def merge_batchnorm(graph):
    bn_matches = matched_patterns(graph,
                                  nodes=[
                                      ('input', {}),
                                      ('y', {'op': 'Constant'}),
                                      ('variance', {'op': 'Constant'}),
                                      ('mean', {'op': 'Constant'}),
                                      ('beta', {'op': 'Constant'}),
                                      ('gamma', {'op': 'Constant'}),
                                      ('add', {'op': 'Add'}),
                                      ('sqrt', {'op': 'Sqrt'}),
                                      ('reciprocal', {'op': 'Reciprocal'}),
                                      ('mul', {'op': 'Mul'}),
                                      ('mul_1', {'op': 'Mul'}),
                                      ('mul_2', {'op': 'Mul'}),
                                      ('sub', {'op': 'Sub'}),
                                      ('add_1', {'op': 'Add'}),
                                  ],
                                  edges=[
                                      ('input', 'mul_1'),
                                      ('variance', 'add'),
                                      ('y', 'add', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                      ('add', 'sqrt'),
                                      ('sqrt', 'reciprocal'),
                                      ('reciprocal', 'mul'),
                                      ('gamma', 'mul'),
                                      ('mul', 'mul_1'),
                                      ('mul', 'mul_2'),
                                      ('mul_2', 'sub'),
                                      ('beta', 'sub'),
                                      ('mean', 'mul_2'),
                                      ('sub', 'add_1'),
                                      ('mul_1', 'add_1')
                                  ]
                                  )
    matched = False
    for bn in bn_matches:
        matched = True
        inp, begin, end = bn['input'], bn['mul_1'], bn['add_1']
        begin_node = NodeWrap(graph, begin)
        begin_node_obj = begin_node['object']
        epsilon = NodeWrap(graph, bn['y'])['object'].value
        mean_value = NodeWrap(graph, bn['mean'])['object'].value
        variance_value = NodeWrap(graph, bn['variance'])['object'].value
        gamma_value = NodeWrap(graph, bn['gamma'])['object'].value
        beta_value = NodeWrap(graph, bn['beta'])['object'].value

        graph.remove_edge(bn['mul'], begin)
        end_out_edges = graph.sorted_out_edges(end, data=True)
        graph.remove_edge(begin, end)
        for _, dst, attr in end_out_edges:
            graph.remove_edge(end, dst)
            graph.add_edge(begin, dst, **attr)

        gamma = get_valid_node_name(graph, begin + '_gamma')
        beta = get_valid_node_name(graph, begin + '_beta')
        mean = get_valid_node_name(graph, begin + '_mean')
        var = get_valid_node_name(graph, begin + '_var')
        graph.add_edge(gamma, begin, **{'src_out_port': 0,
                                        'dst_in_port': 1, 'tensor': Tensor(value=gamma_value)})
        graph.add_edge(beta, begin, **{'src_out_port': 0,
                                       'dst_in_port': 2, 'tensor': Tensor(value=beta_value)})
        graph.add_edge(mean, begin, **{'src_out_port': 0,
                                       'dst_in_port': 3, 'tensor': Tensor(value=mean_value)})
        graph.add_edge(
            var, begin, **{'src_out_port': 0, 'dst_in_port': 4, 'tensor': Tensor(value=variance_value)})

        batchnorm_attr = begin_node_obj.copied_attr()
        batchnorm_attr.update({'opset_version': 9, 'epsilon': epsilon})
        begin_node.replace_obj('BatchNormalization', batchnorm_attr)

        gamma_attr = {'name': gamma, 'value': gamma_value,
                      'data_format': 'NHWC', 'opset_version': 1}
        beta_attr = {'name': beta, 'value': beta_value,
                     'data_format': 'NHWC', 'opset_version': 1}
        mean_attr = {'name': mean, 'value': mean_value,
                     'data_format': 'NHWC', 'opset_version': 1}
        var_attr = {'name': var, 'value': variance_value,
                    'data_format': 'NHWC', 'opset_version': 1}
        NodeWrap(graph, gamma).replace_obj('Constant', gamma_attr)
        NodeWrap(graph, beta).replace_obj('Constant', beta_attr)
        NodeWrap(graph, mean).replace_obj('Constant', mean_attr)
        NodeWrap(graph, var).replace_obj('Constant', var_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_channel_shuffle(graph):
    matched = False
    cs_matches = matched_patterns(graph,
                                  nodes=[
                                      ('reshape1', {'op': 'Reshape'}),
                                      ('transpose', {'op': 'Transpose'}),
                                      ('reshape2', {'op': 'Reshape'}),
                                  ],
                                  edges=[
                                      ('reshape1', 'transpose'),
                                      ('transpose', 'reshape2'),
                                  ])
    for m in cs_matches:
        reshape1, transpose, reshape2 = m['reshape1'], m['transpose'], m['reshape2']
        reshape1_obj, transpose_obj, reshape2_obj = [
            NodeWrap(graph, name)['object'] for name in [reshape1, transpose, reshape2]]
        if reshape1_obj is not None and transpose_obj is not None and reshape2_obj is not None:
            reshape1_out_edges = graph.sorted_out_edges(reshape1, data=True)
            transpose_out_edges = graph.sorted_out_edges(transpose, data=True)
            if len(reshape1_out_edges) == 1 and len(transpose_out_edges) == 1:
                reshape1_in_shape = reshape1_obj.get_input_shapes()[0]
                reshape1_out_shape = reshape1_obj.get_output_shapes()[0]
                reshape2_out_shape = reshape2_obj.get_output_shapes()[0]
                if reshape1_in_shape is not None \
                        and reshape1_out_shape is not None \
                        and len(reshape1_out_shape) > 2:
                    need_insert_reshape = False
                    if (len(reshape1_in_shape) == len(reshape1_out_shape) or len(reshape1_in_shape) + 1 == len(reshape1_out_shape)) \
                        and (int(np.prod(reshape1_out_shape[-2:])) == reshape1_in_shape[-1]
                             if transpose_obj.data_format == 'NHWC'
                             else int(np.prod(reshape1_out_shape[1:3])) == reshape1_in_shape[1]):
                        pass
                    elif len(reshape1_in_shape) == len(reshape1_out_shape) + 1 \
                        and (reshape1_out_shape[-2:] == reshape1_in_shape[-2:]
                             if transpose_obj.data_format == 'NHWC'
                             else reshape1_out_shape[1:3] == reshape1_in_shape[1:3]):
                        need_insert_reshape = True
                    else:
                        continue

                    perm_dim = len(transpose_obj.perm)
                    ref_perm = list(range(perm_dim - 2)) + [perm_dim - 1, perm_dim - 2] \
                        if transpose_obj.data_format == 'NHWC' \
                        else [0, 2, 1] + list(range(3, perm_dim))
                    if transpose_obj.perm == ref_perm:
                        matched = True
                        group = reshape1_out_shape[-2] if transpose_obj.data_format == 'NHWC' else reshape1_out_shape[1]
                        splits = 1
                        in_edges = graph.sorted_in_edges(reshape1, data=True)
                        src, _, in_attr = in_edges[0]
                        if need_insert_reshape:
                            insert_reshape(graph, src, reshape1,
                                           in_attr, reshape2_out_shape)
                            in_edges = graph.sorted_in_edges(
                                reshape1, data=True)
                            src, _, in_attr = in_edges[0]
                        reshape2_in_edges = graph.sorted_in_edges(
                            reshape2, data=True)

                        graph.remove_edges_from(in_edges + reshape2_in_edges)
                        graph.add_edge(src, reshape2, **in_attr)
                        cs_attr = reshape2_obj.copied_attr()
                        cs_attr.update({'group': group, 'splits': splits})
                        NodeWrap(graph, reshape2).replace_obj(
                            'ChannelShuffle', cs_attr)
        else:
            WARN('[Parser]]: Meets invalid Op in merge_channel_shuffle!')
    if matched:
        clear_redundant_nodes(graph)


def merge_channel_shuffle_with_split(graph):
    matched = False
    cs_matches = matched_patterns(graph,
                                  nodes=[
                                      ('reshape1', {'op': 'Reshape'}),
                                      ('transpose', {'op': 'Transpose'}),
                                      ('reshape2', {'op': 'Reshape'}),
                                      ('split', {'op': 'Split'})
                                  ],
                                  edges=[
                                      ('reshape1', 'transpose'),
                                      ('transpose', 'reshape2'),
                                      ('reshape2', 'split'),
                                  ])
    for m in cs_matches:
        reshape1, transpose, reshape2, split = m['reshape1'], m['transpose'], m['reshape2'], m['split']
        reshape1_obj, transpose_obj, reshape2_obj, split_obj = [NodeWrap(
            graph, name)['object'] for name in [reshape1, transpose, reshape2, split]]
        if reshape1_obj is not None and transpose_obj is not None and reshape2_obj is not None and split_obj is not None:
            reshape1_out_edges = graph.sorted_out_edges(reshape1, data=True)
            transpose_out_edges = graph.sorted_out_edges(transpose, data=True)
            reshape2_out_edges = graph.sorted_out_edges(reshape2, data=True)
            if len(reshape1_out_edges) == 1 and len(transpose_out_edges) == 1 and len(reshape2_out_edges) == 1:
                reshape1_in_shape = reshape1_obj.get_input_shapes()[0]
                reshape1_out_shape = reshape1_obj.get_output_shapes()[0]
                reshape2_out_shape = reshape2_obj.get_output_shapes()[0]
                if reshape1_in_shape is not None \
                        and reshape1_out_shape is not None \
                        and reshape2_out_shape is not None \
                        and len(reshape1_in_shape) > 2 \
                        and len(reshape2_out_shape) > 2:
                    nhw_dim = len(reshape2_out_shape) - 1
                    if (int(np.prod(reshape1_in_shape[nhw_dim:])) == reshape2_out_shape[-1]
                            if transpose_obj.data_format == 'NHWC'
                            else int(np.prod([reshape1_in_shape[0]] + reshape1_in_shape[-(nhw_dim - 1):])) == reshape2_out_shape[1]):
                        perm_dim = len(transpose_obj.perm)
                        ref_perm = list(range(perm_dim - 2)) + [perm_dim - 1, perm_dim - 2] \
                            if transpose_obj.data_format == 'NHWC' \
                            else [0, 2, 1] + list(range(3, perm_dim))
                        if transpose_obj.perm == ref_perm:
                            group = reshape1_out_shape[-2] if transpose_obj.data_format == 'NHWC' else reshape1_out_shape[1]
                            if len(split_obj.split) == group \
                                    and all([split_obj.split[0] == s for s in split_obj.split[1:]]) \
                                    and (split_obj.axis in (-1, perm_dim - 1) if split_obj.data_format == 'NHWC' else split_obj.axis == 1):
                                matched = True
                                graph.remove_edges_from(reshape2_out_edges)
                                reshape1_in_edges = graph.sorted_in_edges(
                                    reshape1, data=True)
                                if reshape1_in_shape == reshape2_out_shape:
                                    graph.remove_edge(reshape1, transpose)
                                    src, _, in_attr = reshape1_in_edges[0]
                                    graph.remove_edges_from(reshape1_in_edges)
                                    graph.add_edge(src, split, **in_attr)
                                else:
                                    _, _, out_attr = reshape1_out_edges[0]
                                    graph.remove_edge(reshape1, transpose)
                                    graph.add_edge(reshape1, split, **out_attr)
                                    graph.remove_edges_from(
                                        reshape1_in_edges[1:])
                                    reshape1_attr = reshape1_obj.copied_attr()
                                    reshape1_attr.update({'opset_version': 5})
                                    NodeWrap(graph, reshape1).replace_obj(
                                        'Reshape', reshape1_attr)
                                    insert_constant(graph,
                                                    reshape1 + '_shape',
                                                    np.array(
                                                        reshape2_out_shape, np.int64),
                                                    reshape1,
                                                    in_port=1,
                                                    data_format=reshape1_obj.data_format)
                                cs_attr = split_obj.copied_attr()
                                cs_attr.update(
                                    {'group': group, 'splits': group})
                                NodeWrap(graph, split).replace_obj(
                                    'ChannelShuffle', cs_attr)
        else:
            WARN('[Parser]]: Meets invalid Op in merge_channel_shuffle_with_split!')
    if matched:
        clear_redundant_nodes(graph)


def merge_channel_shuffle_with_pack(graph):
    matched = False
    cs_matches = matched_patterns(graph,
                                  nodes=[
                                      ('pack', {'op': 'ConcatFromSequence'}),
                                      ('transpose', {'op': 'Transpose'}),
                                      ('reshape', {'op': 'Reshape'}),
                                  ],
                                  edges=[
                                      ('pack', 'transpose'),
                                      ('transpose', 'reshape'),
                                  ])
    for m in cs_matches:
        pack, transpose, reshape = m['pack'], m['transpose'], m['reshape']
        pack_obj, transpose_obj, reshape_obj = [
            NodeWrap(graph, name)['object'] for name in [pack, transpose, reshape]]
        if pack_obj is None or transpose_obj is None or reshape_obj is None:
            WARN('[Parser]]: Meets invalid Op in merge_channel_shuffle!')
            continue

        if not pack_obj.new_axis:
            continue

        pack_out_edges = graph.sorted_out_edges(pack, data=True)
        _, _, pack_out_attr = pack_out_edges[0]
        pack_inputs = pack_obj.get_input_tensors()
        concat_output_value = np.concatenate(
            [*pack_inputs], axis=pack_obj.axis)
        reshape_out_shape = reshape_obj.get_output_shapes()[0]
        transpose_out_edges = graph.sorted_out_edges(transpose, data=True)
        # Check whether pack and transpose both have only 1 out edge;
        # Check if concat pack's inputs, output shape is same as reshape's shape.
        if len(pack_out_edges) != 1 or len(transpose_out_edges) != 1 \
                or list(concat_output_value.shape) != reshape_out_shape:
            continue

        pack_in_shape = pack_obj.get_input_shapes()[0]
        pack_out_shape = pack_obj.get_output_shapes()[0]
        # Check pack's input/output shape and the length of pack's output shape
        if pack_in_shape is None or pack_out_shape is None \
                or len(pack_out_shape) <= 2:
            continue

        perm_dim = len(transpose_obj.perm)
        ref_perm = list(range(perm_dim - 2)) + [perm_dim - 1, perm_dim - 2] \
            if transpose_obj.data_format == 'NHWC' \
            else [0, 2, 1] + list(range(3, perm_dim))
        is_channel_axis = pack_obj.axis in (-1, perm_dim - 2) \
            if pack_obj.data_format == 'NHWC' else pack_obj.axis == 1
        # Check transpose perm and whether the axis of pack is channel axis
        if transpose_obj.perm != ref_perm or not is_channel_axis:
            continue

        matched = True

        reshape_in_edges = graph.sorted_in_edges(reshape, data=True)
        graph.remove_edges_from(pack_out_edges + reshape_in_edges)
        graph.add_edge(pack, reshape, **pack_out_attr)
        pack_out_attr['tensor'].value = concat_output_value
        concat_attr = pack_obj.copied_attr()
        concat_attr.update({'opset_version': 4})
        NodeWrap(graph, pack).replace_obj('Concat', concat_attr)

        group = pack_out_shape[-2] if transpose_obj.data_format == 'NHWC' else pack_out_shape[1]
        splits = 1
        cs_attr = reshape_obj.copied_attr()
        cs_attr.update({'group': group, 'splits': splits})
        NodeWrap(graph, reshape).replace_obj(
            'ChannelShuffle', cs_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_gelu_1(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('inp', {}),
                                   ('div', {'op': 'Div'}),
                                   ('divc', {'op': 'Constant'}),
                                   ('erf', {'op': 'Erf'}),
                                   ('add_1', {'op': 'Add'}),
                                   ('addc', {'op': 'Constant'}),
                                   ('mul_1', {'op': 'Mul'}),
                                   ('mul_2', {'op': 'Mul'}),
                               ],
                               edges=[
                                   ('inp', 'div'),
                                   ('divc', 'div', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('inp', 'mul_1'),
                                   ('div', 'erf'),
                                   ('addc', 'add_1', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('erf', 'add_1'),
                                   ('add_1', 'mul_1', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('mul_1', 'mul_2'),
                               ]
                               )
    for m in matches:
        key_names = ['inp', 'div', 'erf', 'add_1', 'mul_1',
                     'mul_2']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if all([obj is not None for obj in node_objs.values()]):

            div_in_edges = graph.sorted_in_edges(m['div'], data=True)
            mul_1_in_edges = graph.sorted_in_edges(m['mul_1'], data=True)
            mul_2_in_edges = graph.sorted_in_edges(m['mul_2'], data=True)

            div_out_edges = graph.sorted_out_edges(m['div'])
            mul_1_out_edges = graph.sorted_out_edges(m['mul_1'])
            add_1_out_edges = graph.sorted_out_edges(m['add_1'])
            erf_out_edges = graph.sorted_out_edges(m['erf'])

            if len(div_out_edges) != 1 \
                    or len(mul_1_out_edges) != 1 \
                    or len(add_1_out_edges) != 1 \
                    or len(erf_out_edges) != 1\
                    or div_in_edges[0][2]['src_out_port'] != mul_1_in_edges[0][2]['src_out_port'] \
                    or len(node_objs['div'].sorted_in_consts()) != 1\
                    or len(node_objs['add_1'].sorted_in_consts()) != 1\
                    or len(node_objs['mul_2'].sorted_in_consts()) != 1\
                    or FLOAT_EQUAL(node_objs['div'].sorted_in_consts()[0][2], 1.4142135) is False \
                    or FLOAT_EQUAL(node_objs['add_1'].sorted_in_consts()[0][2], 1.0) is False \
                    or FLOAT_EQUAL(node_objs['mul_2'].sorted_in_consts()[0][2], 0.5) is False:
                continue

            matched = True
            _, _, in_attr = mul_1_in_edges[0]

            graph.remove_edge(m['inp'], m['div'])
            graph.remove_edge(m['inp'], m['mul_1'])
            graph.remove_edges_from(mul_2_in_edges)
            graph.add_edge(m['inp'], m['mul_2'], **in_attr)
            gelu_attr = node_objs['mul_2'].copied_attr()
            gelu_attr.update({'approximate': 'tanh'})
            NodeWrap(graph, m['mul_2']).replace_obj('Gelu', gelu_attr)
        else:
            WARN('[Parser]: Meets invalid nodes in merge_gelu!')
    if matched:
        clear_redundant_nodes(graph)


def merge_gelu_2(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('pow', {'op': 'Pow'}),
                                   ('mul_1', {'op': 'Mul'}),
                                   ('mul_1c', {'op': 'Constant'}),
                                   ('add_1', {'op': 'Add'}),
                                   ('mul_2', {'op': 'Mul'}),
                                   ('mul_2c', {'op': 'Constant'}),
                                   ('tanh', {'op': 'Tanh'}),
                                   ('add_2', {'op': 'Add'}),
                                   ('add_2c', {'op': 'Constant'}),
                                   ('mul_3', {'op': 'Mul'}),
                                   ('mul_4c', {'op': 'Constant'}),
                                   ('mul_4', {'op': 'Mul'}),
                               ],
                               edges=[
                                   ('pow', 'mul_1'),
                                   ('mul_1c', 'mul_1'),
                                   ('mul_1', 'add_1', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                   ('add_1', 'mul_2'),
                                   ('mul_2', 'tanh'),
                                   ('mul_2c', 'mul_2'),
                                   ('add_2c', 'add_2'),
                                   ('tanh', 'add_2'),
                                   ('add_2', 'mul_3', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                   ('mul_4c', 'mul_4'),
                                   ('mul_4', 'mul_3'),
                               ]
                               )
    for m in matches:
        key_names = ['pow', 'mul_1', 'add_1', 'mul_2',
                     'tanh', 'add_2', 'mul_3', 'mul_4']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if all([obj is not None for obj in node_objs.values()]):

            pow_in_edges = graph.sorted_in_edges(m['pow'], data=True)
            mul_4_in_edges = graph.sorted_in_edges(m['mul_4'], data=True)
            add_1_in_edges = graph.sorted_in_edges(m['add_1'], data=True)

            if len(pow_in_edges) != 2 \
                    or len(mul_4_in_edges) != 2 \
                    or len(add_1_in_edges) != 2:
                continue
            pow_src, _, in_attr1 = pow_in_edges[0]
            mul_4_src, _, in_attr2 = mul_4_in_edges[1]
            add_1_src, _, in_attr3 = add_1_in_edges[0]

            if pow_src != mul_4_src \
                    or pow_src != add_1_src \
                    or in_attr1['src_out_port'] != in_attr2['src_out_port'] \
                    or in_attr1['src_out_port'] != in_attr3['src_out_port']:
                continue

            mul_3_in_edges = graph.sorted_in_edges(m['mul_3'], data=True)

            pow_out_edges = graph.sorted_out_edges(m['pow'])
            mul_1_out_edges = graph.sorted_out_edges(m['mul_1'])
            add_1_out_edges = graph.sorted_out_edges(m['add_1'])
            mul_2_out_edges = graph.sorted_out_edges(m['mul_2'])
            tanh_out_edges = graph.sorted_out_edges(m['tanh'])
            add_2_out_edges = graph.sorted_out_edges(m['add_2'])
            mul_4_out_edges = graph.sorted_out_edges(m['mul_4'])

            if len(pow_out_edges) != 1 \
                    or len(mul_1_out_edges) != 1 \
                    or len(add_1_out_edges) != 1 \
                    or len(mul_2_out_edges) != 1 \
                    or len(tanh_out_edges) != 1 \
                    or len(add_2_out_edges) != 1 \
                    or len(mul_4_out_edges) != 1\
                    or len(node_objs['pow'].sorted_in_consts()) != 1\
                    or len(node_objs['add_2'].sorted_in_consts()) != 1\
                    or len(node_objs['mul_2'].sorted_in_consts()) != 1\
                    or len(node_objs['mul_1'].sorted_in_consts()) != 1\
                    or len(node_objs['mul_4'].sorted_in_consts()) != 1\
                    or FLOAT_EQUAL(node_objs['pow'].sorted_in_consts()[0][2], 3.0) is False\
                    or FLOAT_EQUAL(node_objs['mul_1'].sorted_in_consts()[0][2], 0.044714998453855515) is False \
                    or FLOAT_EQUAL(node_objs['mul_2'].sorted_in_consts()[0][2], 0.7978845834732056) is False\
                    or FLOAT_EQUAL(node_objs['mul_4'].sorted_in_consts()[0][2], 0.5) is False\
                    or FLOAT_EQUAL(node_objs['add_2'].sorted_in_consts()[0][2], 1.0) is False:
                continue

            matched = True
            graph.remove_edge(pow_src, m['pow'])
            graph.remove_edge(pow_src, m['mul_4'])
            graph.remove_edges_from(mul_3_in_edges)
            graph.add_edge(pow_src, m['mul_3'], **in_attr1)
            gelu_attr = node_objs['mul_3'].copied_attr()
            gelu_attr.update({'approximate': 'tanh'})
            NodeWrap(graph, m['mul_3']).replace_obj('Gelu', gelu_attr)
        else:
            WARN('[Parser]: Meets invalid nodes in merge_gelu!')
    if matched:
        clear_redundant_nodes(graph)


def merge_gelu_3(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('div', {'op': 'Div'}),
                                   ('divc', {'op': 'Constant'}),
                                   ('erf', {'op': 'Erf'}),
                                   ('add', {'op': 'Add'}),
                                   ('addc2', {'op': 'Constant'}),
                                   ('mul_1', {'op': 'Mul'}),
                                   ('mulc', {'op': 'Constant'}),
                                   ('mul_2', {'op': 'Mul'}),
                               ],
                               edges=[
                                   ('divc', 'div', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                   ('mulc', 'mul_1'),
                                   ('div', 'erf'),
                                   ('addc2', 'add'),
                                   ('erf', 'add'),
                                   ('mul_1', 'mul_2'),
                                   ('add', 'mul_2')
                               ]
                               )
    for m in matches:
        key_names = ['div', 'divc', 'erf', 'add', 'addc2',
                     'mul_1', 'mulc', 'mul_2']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if all([obj is not None for obj in node_objs.values()]):

            mul_1_in_edges = graph.sorted_in_edges(m['mul_1'], data=True)
            div_in_edges = graph.sorted_in_edges(m['div'], data=True)

            if len(mul_1_in_edges) != 2 \
                    or len(div_in_edges) != 2:
                continue
            mul_1_src, _, in_attr1 = mul_1_in_edges[1]
            div_src, _, in_attr2 = div_in_edges[0]

            if mul_1_src != div_src \
                    or in_attr1['src_out_port'] != in_attr2['src_out_port']:
                continue

            mul_2_in_edges = graph.sorted_in_edges(m['mul_2'], data=True)
            div_out_edges = graph.sorted_out_edges(m['div'])
            mul_1_out_edges = graph.sorted_out_edges(m['mul_1'])
            add_out_edges = graph.sorted_out_edges(m['add'])
            erf_out_edges = graph.sorted_out_edges(m['erf'])

            if len(div_out_edges) != 1 \
                    or len(mul_1_out_edges) != 1 \
                    or len(add_out_edges) != 1 \
                    or len(erf_out_edges) != 1\
                    or len(node_objs['div'].sorted_in_consts()) != 1\
                    or len(node_objs['add'].sorted_in_consts()) != 1\
                    or len(node_objs['mul_1'].sorted_in_consts()) != 1\
                    or FLOAT_EQUAL(node_objs['div'].sorted_in_consts()[0][2], 1.4142135381698608) is False \
                    or FLOAT_EQUAL(node_objs['add'].sorted_in_consts()[0][2], 1.0) is False \
                    or FLOAT_EQUAL(node_objs['mul_1'].sorted_in_consts()[0][2], 0.5) is False:
                continue

            matched = True
            graph.remove_edge(div_src, m['mul_1'])
            graph.remove_edge(div_src, m['div'])
            graph.remove_edges_from(mul_2_in_edges)
            graph.add_edge(div_src, m['mul_2'], **in_attr1)
            gelu_attr = node_objs['mul_2'].copied_attr()
            gelu_attr.update({'approximate': 'tanh'})
            NodeWrap(graph, m['mul_2']).replace_obj('Gelu', gelu_attr)
        else:
            WARN('[Parser]: Meets invalid nodes in merge_gelu!')
    if matched:
        clear_redundant_nodes(graph)


def merge_dilated_conv(graph):
    _REFERENCE_PERM = {'NHWC': [3, 1, 2, 0], 'NCHW': [1, 0, 2, 3]}

    def _check_pads(pad_obj, trans1_obj, data_format):
        perm = trans1_obj.perm[:]
        exclude_pad, spatial_pads, true_in_shape = False, [0] * ((len(perm) - 2) * 2), None
        ref_perm = _REFERENCE_PERM[data_format]
        new_inner_perm = Op.cal_inserting_before_perm(perm, ref_perm)
        new_perm = Op.cal_inverse_perm(new_inner_perm)
        if pad_obj.type != 'Pad':
            exclude_pad = True
        else:
            pads = np.reshape(np.array(pad_obj.pads), [2, -1])
            pads = np.transpose(pads)
            if new_perm is not None:
                pads = pads[np.array(perm)]
            if (data_format == 'NHWC' and np.all(pads[0, :] == 0) and np.all(pads[-1, :] == 0)) \
                    or (data_format == 'NCHW' and np.all(pads[:2, :] == 0)):
                if data_format == 'NHWC':
                    spatial_pads = pads[1:-1, :]
                else:
                    spatial_pads = pads[2:, :]
                spatial_pads = np.transpose(spatial_pads).flatten().tolist()
            else:
                exclude_pad = True
        if (exclude_pad and len(trans1_obj.get_input_shapes()) > 0 and all(s is not None for s in trans1_obj.get_input_shapes()[0])) \
                or ((not exclude_pad) and len(pad_obj.get_input_shapes()) > 0 and all(s is not None for s in pad_obj.get_input_shapes()[0])):
            true_in_shape = trans1_obj.get_input_shapes()[0] if exclude_pad else pad_obj.get_input_shapes()[0]
            if new_perm is not None:
                inverse_perm = Op.cal_inverse_perm(ref_perm)
                true_in_shape = np.array(true_in_shape)[np.array(perm)]
                true_in_shape = true_in_shape[np.array(inverse_perm)].tolist()
        return exclude_pad, spatial_pads, new_perm, true_in_shape

    def _check_crops(slice_obj, trans4_obj, data_format):
        perm = trans4_obj.perm[:]
        exclude_slice, spatial_crops, true_out_shape = False, [0] * ((len(perm) - 2) * 2), None
        ref_perm = _REFERENCE_PERM[data_format]
        new_inner_perm = Op.cal_inserting_after_perm(perm, ref_perm)
        new_perm = Op.cal_inverse_perm(new_inner_perm)
        if slice_obj.type != 'Slice' or any(s != 1 for s in slice_obj.steps):
            exclude_slice = True
        else:
            slice_in_shapes = slice_obj.get_input_shapes()
            if len(slice_in_shapes) < 1 \
                    or slice_in_shapes[0] is None \
                    or any(d is None for d in slice_in_shapes[0]):
                exclude_slice = True
            else:
                in_shape = slice_in_shapes[0]
                axes = slice_obj.axes[:]
                starts = slice_obj.starts[:]
                ends = slice_obj.ends[:]
                if len(axes) < len(perm):
                    full_starts = np.array([0] * len(in_shape))
                    full_ends = np.array(in_shape)
                    axes = np.array(axes)
                    full_starts[axes] = np.array(starts)
                    full_ends[axes] = np.array(ends)
                    starts = full_starts.tolist()
                    ends = full_ends.tolist()
                sliced = type(slice_obj).cal_sliced(starts, ends, in_shape)
                crops = np.reshape(np.array(sliced), [2, -1])
                crops = np.transpose(crops)
                if new_perm is not None:
                    inverse_perm = Op.cal_inverse_perm(perm)
                    crops = crops[np.array(inverse_perm)]
                if (data_format == 'NHWC' and np.all(crops[0, :] == 0) and np.all(crops[-1, :] == 0)) \
                        or (data_format == 'NCHW' and np.all(crops[:2, :] == 0)):
                    if data_format == 'NHWC':
                        spatial_crops = crops[1:-1, :]
                    else:
                        spatial_crops = crops[2:, :]
                    spatial_crops = np.transpose(spatial_crops).flatten().tolist()
                else:
                    exclude_slice = True
            if (exclude_slice and len(trans4_obj.get_output_shapes()) > 0 and all(s is not None for s in trans4_obj.get_output_shapes()[0])) \
                    or ((not exclude_slice) and len(slice_obj.get_output_shapes()) > 0 and all(s is not None for s in slice_obj.get_output_shapes()[0])):
                true_out_shape = trans4_obj.get_output_shapes(
                )[0] if exclude_slice else slice_obj.get_output_shapes()[0]
                if new_perm is not None:
                    inverse_perm = Op.cal_inverse_perm(perm)
                    true_out_shape = np.array(true_out_shape)[np.array(inverse_perm)]
                    true_out_shape = true_out_shape[np.array(ref_perm)].tolist()
        return exclude_slice, spatial_crops, new_perm, true_out_shape

    matched = False
    conv_ops_list = list(set(BaseConvOp.get_concrete_subclass_names()).intersection(
        OnnxOp.get_concrete_subclass_names()))
    matches = matched_patterns(graph,
                               nodes=[
                                   ('pad', {}),
                                   ('transpose_1', {'op': 'Transpose'}),
                                   ('space_to_depth', {'op': 'SpaceToDepth'}),
                                   ('transpose_2', {'op': 'Transpose'}),
                                   ('conv', {'op': conv_ops_list}),
                                   ('transpose_3', {'op': 'Transpose'}),
                                   ('depth_to_space', {'op': 'DepthToSpace'}),
                                   ('transpose_4', {'op': 'Transpose'}),
                                   ('slice', {}),
                               ],
                               edges=[
                                   ('pad', 'transpose_1'),
                                   ('transpose_1', 'space_to_depth'),
                                   ('space_to_depth', 'transpose_2'),
                                   ('transpose_2', 'conv'),
                                   ('conv', 'transpose_3'),
                                   ('transpose_3', 'depth_to_space'),
                                   ('depth_to_space', 'transpose_4'),
                                   ('transpose_4', 'slice'),
                               ]
                               )
    for m in matches:
        names = ['pad', 'transpose_1', 'space_to_depth', 'transpose_2',
                 'conv', 'transpose_3', 'depth_to_space', 'transpose_4', 'slice']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in obj_dict.values()]):
            WARN('[Parser]: Meets invalid Op in merge_dilated_conv!')
            continue
        if obj_dict['depth_to_space'].mode != 'DCR':
            continue
        conv_out_edges = graph.sorted_out_edges(m['conv'])
        if len(conv_out_edges) != 1:
            continue
        conv_in_shapes = obj_dict['conv'].get_input_shapes()
        if len(conv_in_shapes) < 1 or len(conv_in_shapes[0]) != 4:
            continue
        if isinstance(obj_dict['conv'], BaseDeconvOp) \
                and obj_dict['conv'].output_padding \
                and any(p != 0 for p in obj_dict['conv'].output_padding):
            continue
        data_format = obj_dict['conv'].data_format
        if obj_dict['space_to_depth'].data_format != data_format \
                or obj_dict['depth_to_space'].data_format != data_format:
            continue
        if obj_dict['space_to_depth'].blocksize != obj_dict['depth_to_space'].blocksize:
            continue
        if (obj_dict['transpose_2'].perm != _REFERENCE_PERM[data_format]) \
                or (obj_dict['transpose_3'].perm != _REFERENCE_PERM[data_format]):
            continue

        matched = True
        exclude_pad, spatial_pads, new_in_perm, true_in_shape \
            = _check_pads(obj_dict['pad'], obj_dict['transpose_1'], data_format)
        exclude_slice, spatial_crops, new_out_perm, true_out_shape \
            = _check_crops(obj_dict['slice'], obj_dict['transpose_4'], data_format)
        first = m['transpose_1'] if exclude_pad else m['pad']
        last = m['transpose_4'] if exclude_slice else m['slice']
        block_size = obj_dict['space_to_depth'].blocksize
        obj_dict['conv'].dilations = (np.array(obj_dict['conv'].dilations) * block_size).tolist()
        pads = np.reshape(np.array(obj_dict['conv'].pads), (2, -1)) \
            + np.reshape(np.array(spatial_pads), (2, -1)) \
            - np.reshape(spatial_crops, (2, -1))
        obj_dict['conv'].pads = pads.flatten().tolist()

        if isinstance(obj_dict['conv'], BaseDeconvOp):
            obj_dict['conv'].pads_updated = False
            if hasattr(obj_dict['conv'], 'output_shape') \
                    and true_out_shape is not None:
                obj_dict['conv'].output_shape = true_out_shape[1:-1] \
                    if data_format == 'NHWC' \
                    else true_out_shape[2:]
            if true_in_shape is not None:
                conv_sptial_in_shape = true_in_shape[1:-1] \
                    if data_format == 'NHWC' \
                    else true_in_shape[2:]
                obj_dict['conv'].update_pads(conv_sptial_in_shape)
        else:
            obj_dict['conv'].auto_pad = 'NOTSET'

        in_edges = graph.sorted_in_edges(first, data=True)
        out_edges = graph.sorted_out_edges(last, data=True)
        conv_in_edges = graph.sorted_in_edges(m['conv'], data=True)
        src, _, in_attr = in_edges[0]
        graph.remove_edges_from(conv_in_edges + conv_out_edges)
        graph.add_edge(src, m['conv'], **in_attr)
        for _, dst, out_attr in out_edges:
            graph.remove_edge(last, dst)
            graph.add_edge(m['conv'], dst, **out_attr)
        if new_in_perm:
            conv_in_edges = graph.sorted_in_edges(m['conv'], data=True)
            conv_src, _, conv_in_attr = conv_in_edges[0]
            insert_transpose(graph, conv_src, m['conv'], conv_in_attr, new_in_perm)
        if new_out_perm:
            new_out_transpose = insert_transpose_after(graph, m['conv'], new_out_perm)
        else:
            new_out_transpose = None

        if last in graph._attr['output_names']:
            index = graph._attr['output_names'].index(last)
            if new_out_transpose:
                graph._attr['output_names'][index] = new_out_transpose
            else:
                graph._attr['output_names'][index] = m['conv']
    if matched:
        clear_redundant_nodes(graph)


def merge_dilated_conv_group(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('pad', {'op': 'Pad'}),
                                   ('transpose_1', {'op': 'Transpose'}),
                                   ('space_to_depth', {'op': 'SpaceToDepth'}),
                                   ('transpose_2', {'op': 'Transpose'}),
                                   ('conv_1', {'op': 'Conv'}),
                                   ('transpose_3', {'op': 'Transpose'}),
                                   ('depth_to_space_1', {
                                    'op': 'DepthToSpace'}),
                                   ('transpose_4', {'op': 'Transpose'}),
                                   ('slice_1', {'op': 'Slice'}),
                                   ('bias_add_1', {'op': 'Add'}),
                                   ('conv_2', {'op': 'Conv'}),
                                   ('transpose_5', {'op': 'Transpose'}),
                                   ('depth_to_space_2', {
                                    'op': 'DepthToSpace'}),
                                   ('transpose_6', {'op': 'Transpose'}),
                                   ('slice_2', {'op': 'Slice'}),
                                   ('bias_add_2', {'op': 'Add'}),
                               ],
                               edges=[
                                   ('pad', 'transpose_1'),
                                   ('transpose_1', 'space_to_depth'),
                                   ('space_to_depth', 'transpose_2'),
                                   ('transpose_2', 'conv_1'),
                                   ('conv_1', 'transpose_3'),
                                   ('transpose_3', 'depth_to_space_1'),
                                   ('depth_to_space_1', 'transpose_4'),
                                   ('transpose_4', 'slice_1'),
                                   ('slice_1', 'bias_add_1'),
                                   ('transpose_2', 'conv_2'),
                                   ('conv_2', 'transpose_5'),
                                   ('transpose_5', 'depth_to_space_2'),
                                   ('depth_to_space_2', 'transpose_6'),
                                   ('transpose_6', 'slice_2'),
                                   ('slice_2', 'bias_add_2'),
                               ]
                               )
    for m in matches:
        pad, s2d, conv_1, d2s_1, bias_add_1, conv_2, d2s_2, bias_add_2 \
            = m['pad'], m['space_to_depth'], m['conv_1'], m['depth_to_space_1'], m['bias_add_1'], \
            m['conv_2'], m['depth_to_space_2'], m['bias_add_2']
        slice_1, slice_2 = m['slice_1'], m['slice_2']
        pad_obj = NodeWrap(graph, pad)['object']
        s2d_obj = NodeWrap(graph, s2d)['object']
        conv_1_obj = NodeWrap(graph, conv_1)['object']
        d2s_1_obj = NodeWrap(graph, d2s_1)['object']
        bias_add_1_obj = NodeWrap(graph, bias_add_1)['object']
        conv_2_obj = NodeWrap(graph, conv_2)['object']
        d2s_2_obj = NodeWrap(graph, d2s_2)['object']
        bias_add_2_obj = NodeWrap(graph, bias_add_2)['object']
        slice_1_obj = NodeWrap(graph, slice_1)['object']
        slice_2_obj = NodeWrap(graph, slice_2)['object']

        if pad_obj is not None \
                and s2d_obj is not None \
                and conv_1_obj is not None \
                and d2s_1_obj is not None \
                and bias_add_1_obj is not None \
                and conv_2_obj is not None \
                and d2s_2_obj is not None \
                and bias_add_2_obj is not None \
                and slice_1_obj is not None \
                and slice_2_obj is not None:
            if d2s_1_obj.mode != 'DCR' or d2s_2_obj.mode != 'DCR':
                continue
            pad_in_edges = graph.sorted_in_edges(pad, data=True)
            if len(pad_in_edges) < 1:
                WARN('[Parser]: The length of in_edges of Pad(%s) is invalid in merge_dilated_conv_group!' % pad)
                continue

            matched = True
            block_size = s2d_obj.blocksize
            pad_pads = np.reshape(
                np.array(pad_obj.space_pads(), np.int64), newshape=(2, -1))

            sliced_pads_1 = type(slice_1_obj).cal_sliced(slice_1_obj.starts if len(slice_1_obj.starts) == 2 else slice_1_obj.starts[1:3],
                                                         slice_1_obj.ends if len(
                                                             slice_1_obj.ends) == 2 else slice_1_obj.ends[1:3],
                                                         slice_1_obj.get_input_shapes()[0][1:3])
            fused_pads1 = np.reshape(np.array(conv_1_obj.pads, np.int64), newshape=(2, -1)) \
                + pad_pads \
                - np.reshape(np.array(sliced_pads_1, np.int64),
                             newshape=(2, -1))
            if isinstance(conv_1_obj, BaseDeconvOp):
                conv_1_obj.pads_updated = False
                pad_in_shape = pad_obj.get_input_shapes()[0]
                if pad_in_shape is not None and all(s is not None for s in pad_in_shape):
                    conv_sptial_in_shape = pad_in_shape[1:-1] if conv_1_obj.data_format == 'NHWC' else pad_in_shape[-2:]
                    conv_1_obj.update_pads(conv_sptial_in_shape)
            else:
                conv_1_obj.auto_pad = 'NOTSET'
            conv_1_obj.pads = fused_pads1.flatten().tolist()
            conv_1_obj.dilations = [block_size, block_size]
            conv_1_obj.biases += bias_add_1_obj.sorted_in_consts()[0][2]

            sliced_pads_2 = type(slice_2_obj).cal_sliced(slice_2_obj.starts if len(slice_2_obj.starts) == 2 else slice_2_obj.starts[1:3],
                                                         slice_2_obj.ends if len(
                                                             slice_2_obj.ends) == 2 else slice_2_obj.ends[1:3],
                                                         slice_2_obj.get_input_shapes()[0][1:3])
            fused_pads2 = np.reshape(np.array(conv_2_obj.pads, np.int64), newshape=(2, -1)) \
                + pad_pads \
                - np.reshape(np.array(sliced_pads_2, np.int64),
                             newshape=(2, -1))
            if isinstance(conv_2_obj, BaseDeconvOp):
                conv_2_obj.pads_updated = False
                pad_in_shape = pad_obj.get_input_shapes()[0]
                if pad_in_shape is not None and all(s is not None for s in pad_in_shape):
                    conv_sptial_in_shape = pad_in_shape[1:-1] if conv_2_obj.data_format == 'NHWC' else pad_in_shape[-2:]
                    conv_2_obj.update_pads(conv_sptial_in_shape)
            else:
                conv_2_obj.auto_pad = 'NOTSET'
            conv_2_obj.pads = fused_pads2.flatten().tolist()
            conv_2_obj.dilations = [block_size, block_size]
            conv_2_obj.biases += bias_add_2_obj.sorted_in_consts()[0][2]

            bias_add_1_out_edges = graph.sorted_out_edges(
                bias_add_1, data=True)
            bias_add_2_out_edges = graph.sorted_out_edges(
                bias_add_2, data=True)

            conv_1_in_edges = graph.sorted_in_edges(conv_1)
            conv_1_out_edges = graph.sorted_out_edges(conv_1)
            conv_2_in_edges = graph.sorted_in_edges(conv_2)
            conv_2_out_edges = graph.sorted_out_edges(conv_2)
            graph.remove_edges_from(
                conv_1_in_edges + conv_1_out_edges + conv_2_in_edges + conv_2_out_edges)

            pad_src, _, in_attr = pad_in_edges[0]
            graph.remove_edge(pad_src, pad)
            graph.add_edge(pad_src, conv_1, **in_attr)
            graph.add_edge(pad_src, conv_2, **in_attr)

            conv1_post_trans = get_valid_node_name(
                graph, conv_1 + '_post_transpose')
            conv2_post_trans = get_valid_node_name(
                graph, conv_2 + '_post_transpose')
            graph.add_edge(conv_1, conv1_post_trans)
            graph.add_edge(conv_2, conv2_post_trans)
            for _, out1, out_attr1 in bias_add_1_out_edges:
                graph.remove_edge(bias_add_1, out1)
                graph.add_edge(conv1_post_trans, out1, **out_attr1)
            for _, out2, out_attr2 in bias_add_2_out_edges:
                graph.remove_edge(bias_add_2, out2)
                graph.add_edge(conv2_post_trans, out2, **out_attr2)
            if bias_add_1 in graph._attr['output_names']:
                index = graph._attr['output_names'].index(bias_add_1)
                graph._attr['output_names'][index] = conv1_post_trans
            if bias_add_2 in graph._attr['output_names']:
                index = graph._attr['output_names'].index(bias_add_2)
                graph._attr['output_names'][index] = conv2_post_trans

            NodeWrap(graph, conv1_post_trans).replace_obj('Transpose', {
                'name': conv1_post_trans, 'opset_version': 1, 'perm': [0, 2, 3, 1]})
            NodeWrap(graph, conv2_post_trans).replace_obj('Transpose', {
                'name': conv2_post_trans, 'opset_version': 1, 'perm': [0, 2, 3, 1]})
        else:
            WARN('[Parser]: Meets invalid node in merge_dilated_conv_group!')
    if matched:
        clear_redundant_nodes(graph)


def merge_hardswish(graph):
    matched = False
    matches = [matched_patterns(graph,
                                nodes=[
                                    ('input', {}),
                                    ('add', {'op': 'Add'}),
                                    ('const_1', {'op': 'Constant'}),
                                    ('relu6', {'op': 'Clip'}),
                                    ('mul', {'op': 'Mul'}),
                                    ('const_2', {'op': 'Constant'}),
                                    ('mul_or_div', {'op': mul_or_div_type}),
                                ],
                                edges=[
                                    ('input', 'add'),
                                    ('input', 'mul'),
                                    ('add', 'relu6'),
                                    ('relu6', 'mul'),
                                    ('mul', 'mul_or_div'),
                                    ('const_1', 'add'),
                                    ('const_2', 'mul_or_div'),
                                ]) for mul_or_div_type in ['Mul', 'Div']]
    matches = extend_lists(matches)
    for m in matches:
        const_1, const_2, clip, add, mul, mul_or_div = m['const_1'], m[
            'const_2'], m['relu6'], m['add'], m['mul'], m['mul_or_div']
        node_objs = {name: NodeWrap(graph, name)['object'] for name in [
            const_1, const_2, clip, add, mul, mul_or_div]}
        if all([obj is not None for obj in node_objs.values()]):
            add_out_edges = graph.sorted_out_edges(add)
            relu6_out_edges = graph.sorted_out_edges(clip)
            mul_out_edges = graph.sorted_out_edges(mul)
            const_1_out_edges = graph.sorted_out_edges(const_1)
            const_2_out_edges = graph.sorted_out_edges(const_2)
            if len(add_out_edges) == 1 \
                    and len(relu6_out_edges) == 1 \
                    and len(mul_out_edges) == 1 \
                    and len(const_1_out_edges) == 1 \
                    and len(const_2_out_edges) == 1 \
                    and FLOAT_EQUAL(node_objs[const_1].value, 3) \
                    and ((node_objs[mul_or_div].type == 'Mul' and FLOAT_EQUAL(node_objs[const_2].value, 1 / 6.0))
                         or (node_objs[mul_or_div].type == 'Div' and FLOAT_EQUAL(node_objs[const_2].value, 6.0))) \
                    and node_objs[clip].min == 0 and node_objs[clip].max == 6:
                matched = True
                add_in_edges = graph.sorted_in_edges(add, keys=True, data=True)
                mul_in_edges = graph.sorted_in_edges(mul)
                mul_or_div_in_edges = graph.sorted_in_edges(mul_or_div)
                graph.remove_edges_from(mul_in_edges + mul_or_div_in_edges)
                for src, _, k, in_attr in add_in_edges:
                    if src == m['input']:
                        new_in_attr = copy.deepcopy(in_attr)
                        new_in_attr.update({'dst_in_port': 0})
                        graph.remove_edge(src, add, key=k)
                        graph.add_edge(src, mul_or_div, **new_in_attr)
                hw_attr = NodeWrap(graph, mul_or_div)['object'].copied_attr()
                hw_attr.update({'opset_version': 1})
                NodeWrap(graph, mul_or_div).replace_obj('HardSwish', hw_attr)
        else:
            WARN('[Parser]: Meets invalid Op in merge_hardswish!')
    if matched:
        clear_redundant_nodes(graph)


def merge_hardswish2(graph):
    matched = False
    matches = [matched_patterns(graph,
                                nodes=[
                                    ('input', {}),
                                    ('add', {'op': 'Add'}),
                                    ('const_1', {'op': 'Constant'}),
                                    ('relu6', {'op': 'Clip'}),
                                    ('mul_or_div', {'op': mul_or_div_type}),
                                    ('mul', {'op': 'Mul'}),
                                    ('const_2', {'op': 'Constant'}),
                                ],
                                edges=[
                                    ('input', 'add'),
                                    ('input', 'mul'),
                                    ('const_1', 'add'),
                                    ('add', 'relu6'),
                                    ('relu6', 'mul_or_div'),
                                    ('const_2', 'mul_or_div'),
                                    ('mul_or_div', 'mul'),
                                ]) for mul_or_div_type in ['Mul', 'Div']]
    matches = extend_lists(matches)
    for m in matches:
        const_1, const_2, clip, add, mul, mul_or_div = m['const_1'], m[
            'const_2'], m['relu6'], m['add'], m['mul'], m['mul_or_div']
        node_objs = {name: NodeWrap(graph, name)['object'] for name in [
            const_1, const_2, clip, add, mul, mul_or_div]}
        if all([obj is not None for obj in node_objs.values()]):
            add_out_edges = graph.sorted_out_edges(add)
            relu6_out_edges = graph.sorted_out_edges(clip)
            const_1_out_edges = graph.sorted_out_edges(const_1)
            const_2_out_edges = graph.sorted_out_edges(const_2)
            mul_or_div_out_edges = graph.sorted_out_edges(mul_or_div)
            if len(add_out_edges) == 1 \
                    and len(relu6_out_edges) == 1 \
                    and len(const_1_out_edges) == 1 \
                    and len(const_2_out_edges) == 1 \
                    and len(mul_or_div_out_edges) == 1 \
                    and FLOAT_EQUAL(node_objs[const_1].value, 3) \
                    and ((node_objs[mul_or_div].type == 'Mul' and FLOAT_EQUAL(node_objs[const_2].value, 1 / 6.0))
                         or (node_objs[mul_or_div].type == 'Div' and FLOAT_EQUAL(node_objs[const_2].value, 6.0))) \
                    and node_objs[clip].min == 0 and node_objs[clip].max == 6:
                matched = True
                add_in_edges = graph.sorted_in_edges(add, keys=True, data=True)
                mul_in_edges = graph.sorted_in_edges(mul)
                mul_or_div_in_edges = graph.sorted_in_edges(mul_or_div)
                graph.remove_edges_from(mul_in_edges + mul_or_div_in_edges)
                for src, _, k, in_attr in add_in_edges:
                    if src == m['input']:
                        new_in_attr = copy.deepcopy(in_attr)
                        new_in_attr.update({'dst_in_port': 0})
                        graph.remove_edge(src, add, key=k)
                        graph.add_edge(src, mul, **new_in_attr)
                hw_attr = NodeWrap(graph, mul)['object'].copied_attr()
                hw_attr.update({'opset_version': 1})
                NodeWrap(graph, mul).replace_obj('HardSwish', hw_attr)
        else:
            WARN('[Parser]: Meets invalid Op in merge_hardswish!')
    if matched:
        clear_redundant_nodes(graph)


def merge_hardsigmoid(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('input', {}),
                                   ('add', {'op': 'Add'}),
                                   ('const_1', {'op': 'Constant'}),
                                   ('relu6', {'op': 'Clip'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('const_2', {'op': 'Constant'}),
                               ],
                               edges=[
                                   ('input', 'add'),
                                   ('const_1', 'add'),
                                   ('add', 'relu6'),
                                   ('relu6', 'mul'),
                                   ('const_2', 'mul'),
                               ])
    for m in matches:
        const_1, const_2, clip, add, mul = m['const_1'], m['const_2'], m['relu6'], m['add'], m['mul']
        const_1_obj, const_2_obj, clip_obj = [NodeWrap(graph, n)['object'] for n in [
            const_1, const_2, clip]]
        if FLOAT_EQUAL(const_1_obj.value, 3) \
                and (FLOAT_EQUAL(const_2_obj.value, 1 / 6.0) or FLOAT_EQUAL(const_2_obj.value, 0.16667)) \
                and clip_obj.min == 0 and clip_obj.max == 6:
            matched = True
            inp = m['input']
            add_in_edges = graph.sorted_in_edges(add, data=True)
            add_out_edges = graph.sorted_out_edges(add)
            mul_in_edges = graph.sorted_in_edges(mul)
            graph.remove_edges_from(add_out_edges)
            graph.remove_edges_from(mul_in_edges)
            for src, _, in_attr in add_in_edges:
                graph.remove_edge(src, add)
                if src == inp:
                    graph.add_edge(src, mul, **in_attr)
            hs_attr = NodeWrap(graph, mul)['object'].copied_attr()
            hs_attr.update({'opset_version': 6, 'alpha': 1 / 6, 'beta': 0.5})
            NodeWrap(graph, mul).replace_obj('HardSigmoid', hs_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_hargsigmoid2(graph):
    # max(clip_min, min(clip_max, alpha * x + beta)) / clip_max
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('add', {'op': 'Add'}),
                                   ('const_1', {'op': 'Constant'}),
                                   ('clip', {'op': 'Clip'}),
                                   ('mul_or_div', {'op': ['Mul', 'Div']}),
                                   ('const_2', {'op': 'Constant'}),
                               ],
                               edges=[
                                   ('const_1', 'add'),
                                   ('add', 'clip'),
                                   ('clip', 'mul_or_div'),
                                   ('const_2', 'mul_or_div'),
                               ])
    for m in matches:
        names = ['add', 'const_1', 'clip', 'mul_or_div', 'const_2']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in obj_dict.values()]):
            WARN('[Parser]: Meets invalid Op in merge_hargsigmoid2!')
            continue
        mul_div_in_edges = graph.sorted_in_edges(m['mul_or_div'], data=True)
        if obj_dict['mul_or_div'].type == 'Div' \
                and mul_div_in_edges[1][0] != m['const_2']:
            continue
        c1 = obj_dict['const_1'].value
        c2 = obj_dict['const_2'].value \
            if obj_dict['mul_or_div'].type == 'Div' \
            else (1 / obj_dict['const_2'].value)
        clip_min, clip_max = obj_dict['clip'].min, obj_dict['clip'].max
        if not FLOAT_EQUAL(c2, clip_max):
            continue
        add_in_edges = graph.sorted_in_edges(m['add'], data=True)
        if len(add_in_edges) != 2:
            WARN(
                '[Parser]: Invalid number of inputs of Add Op(%s) in merge_hargsigmoid2!' % m['add'])
            continue
        inp, in_attr = None, None
        for src, _, attr in add_in_edges:
            if src != m['const_1']:
                inp = src
                in_attr = copy.deepcopy(attr)
                break
        if inp is None or in_attr is None:
            WARN('[Parser]: Meets invalid pattern of hargsigmoid!')
            continue
        matched = True
        in_attr['dst_in_port'] = 0
        graph.remove_edges_from(add_in_edges + mul_div_in_edges)
        graph.add_edge(inp, m['mul_or_div'], **in_attr)
        hs_attr = obj_dict['mul_or_div'].copied_attr()
        hs_attr.update({'alpha': 1.,
                        'beta': float(c1),
                        'clip_min': float(clip_min),
                        'clip_max': float(clip_max),
                        })
        NodeWrap(graph, m['mul_or_div']).replace_obj('HardSigmoid', hs_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_l2pool(graph):
    # CB-2482: Have similarity issue if merging to L2Pool
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('pow', {'op': 'Pow'}),
                                   ('avgpool', {'op': 'AveragePool'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('sqrt', {'op': 'Sqrt'}),
                               ],
                               edges=[
                                   ('pow', 'avgpool'),
                                   ('avgpool', 'mul'),
                                   ('mul', 'sqrt'),
                               ])
    for m in matches:
        names = ['pow', 'avgpool', 'mul', 'sqrt']
        pow_in_edges = graph.sorted_in_edges(m['pow'], data=True)
        sqrt_in_edges = graph.sorted_in_edges(m['sqrt'])
        nodes_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in nodes_dict.values()]) or \
                len(pow_in_edges) < 1 or \
                len(sqrt_in_edges) != 1:
            WARN('[Parser]: Meets invalid Op in merge_l2pool!')
            continue
        if len(nodes_dict['mul'].sorted_in_consts()) != 1 or \
                nodes_dict['mul'].sorted_in_consts()[0][2].size != 1 or \
                len(nodes_dict['pow'].sorted_in_consts()) != 1 or \
                not FLOAT_EQUAL(nodes_dict['pow'].sorted_in_consts()[0][2], 2):
            continue
        mul_y = nodes_dict['mul'].sorted_in_consts()[0][2].item()
        ksize = nodes_dict['avgpool'].kernel_shape
        if len(ksize) != 2 or not FLOAT_EQUAL(mul_y, ksize[0] * ksize[1]):
            continue
        matched = True
        graph.remove_edges_from(sqrt_in_edges)
        src, _, in_attr = pow_in_edges[0]
        graph.add_edge(src, m['sqrt'], **in_attr)
        l2pool_attr = nodes_dict['avgpool'].copied_attr()
        l2pool_attr.update({'name': m['sqrt'], 'opset_version': 11, 'p': 2})
        NodeWrap(graph, m['sqrt']).replace_obj('LpPool', l2pool_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_leaky_relu(graph):
    matched = False
    matches1 = matched_patterns(graph,
                                nodes=[
                                    ('input', {}),
                                    ('relu', {'op': 'Relu'}),
                                    ('neg', {'op': 'Neg'}),
                                    ('relu_1', {'op': 'Relu'}),
                                    ('const', {'op': 'Constant'}),
                                    ('mul', {'op': 'Mul'}),
                                    ('end', {'op': 'Sub'}),
                                ],
                                edges=[
                                    ('input', 'relu'),
                                    ('input', 'neg'),
                                    ('relu', 'end'),
                                    ('neg', 'relu_1'),
                                    ('relu_1', 'mul', {'dst_in_port': 1}),
                                    ('const', 'mul'),
                                    ('mul', 'end', {'dst_in_port': 1}),
                                ])

    matches2 = [matched_patterns(graph,
                                 nodes=[
                                     ('input', {}),
                                     ('relu', {'op': 'Relu'}),
                                     ('neg', {'op': 'Neg'}),
                                     ('end', {'op': 'Add'}),
                                     ('relu_1', {'op': 'Relu'}),
                                     ('const', {'op': 'Constant'}),
                                     ('mul', {'op': 'Mul'}),
                                     ('neg_1', {'op': 'Neg'}),
                                 ],
                                 edges=[
                                     ('input', 'relu'),
                                     ('input', 'neg'),
                                     ('relu', 'end'),
                                     ('neg', 'relu_1'),
                                     ('relu_1', 'mul', {
                                         'dst_in_port': 1 - const_in_port}),
                                     ('const', 'mul', {
                                         'dst_in_port': const_in_port}),
                                     ('mul', 'neg_1'),
                                     ('neg_1', 'end', {'dst_in_port': 1}),
                                 ]) for const_in_port in [0, 1]]
    matches2 = extend_lists(matches2)
    matches = matches1 + matches2
    for m in matches:
        inp, relu, neg, const, end = m['input'], m['relu'], m['neg'], m['const'], m['end']
        if not graph.has_node(inp) \
                or not graph.has_node(relu) \
                or not graph.has_node(neg) \
                or not graph.has_node(const) \
                or not graph.has_node(end):
            WARN('[Parser]: Node (%s or %s or %s or %s or %s) cannot be found, graph maybe has been changed in merge_leaky_relu!' % (
                inp, relu, neg, const, end))
            continue

        relu_in_edges = graph.sorted_in_edges(relu, data=True)
        neg_in_edges = graph.sorted_in_edges(neg, data=True)
        if len(relu_in_edges) != 1 \
                or len(neg_in_edges) != 1:
            WARN('[Parser]: Meets invalid in_edges in merge_leaky_relu!')
            continue
        if relu_in_edges[0][2]['src_out_port'] != neg_in_edges[0][2]['src_out_port']:
            continue

        matched = True
        end_in_edges = graph.sorted_in_edges(end)
        end_out_edges = graph.sorted_out_edges(end, data=True)
        graph.remove_edge(inp, neg)
        graph.remove_edges_from(end_in_edges)
        for _, dst, out_attr in end_out_edges:
            graph.remove_edge(end, dst)
            graph.add_edge(relu, dst, **out_attr)
        alpha = NodeWrap(graph, const)['object'].value
        leaky_attr = NodeWrap(graph, relu)['object'].copied_attr()
        leaky_attr.update({'opset_version': 6, 'alpha': float(alpha)})
        NodeWrap(graph, relu).replace_obj('LeakyRelu', leaky_attr)
        if end in graph._attr['output_names']:
            index = graph._attr['output_names'].index(end)
            graph._attr['output_names'][index] = relu
    if matched:
        clear_redundant_nodes(graph)


def merge_logical_xor(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('or', {'op': 'Or'}),
                                   ('and_1', {'op': 'And'}),
                                   ('not', {'op': 'Not'}),
                                   ('and_2', {'op': 'And'}),
                               ],
                               edges=[
                                   ('or', 'and_2'),
                                   ('and_1', 'not'),
                                   ('not', 'and_2'),
                               ])
    for m in matches:
        or_in_edges = graph.sorted_in_edges(m['or'], data=True)
        and1_in_edges = graph.sorted_in_edges(m['and_1'], data=True)
        and2_in_edges = graph.sorted_in_edges(m['and_2'])
        if len(or_in_edges) != 2 or len(and1_in_edges) != 2 or len(and2_in_edges) != 2:
            WARN('[Parser]: Meets invalid Op in merge_logical_xor!')
            continue
        # Check whether Or op and first And op have same inputs(src and src_out_port)
        same_inputs = [False, False]
        for idx, or_in_edge in enumerate(or_in_edges):
            or_src, _, or_in_attr = or_in_edge
            for and_src, _, and_in_attr in and1_in_edges:
                if or_src == and_src and or_in_attr['src_out_port'] == and_in_attr['src_out_port']:
                    same_inputs[idx] = True
                    break
        if not all(same_inputs):
            continue
        matched = True
        graph.remove_edges_from(or_in_edges + and1_in_edges + and2_in_edges)
        for src, _, in_attr in or_in_edges:
            graph.add_edge(src, m['and_2'], **in_attr)
        xor_attr = NodeWrap(graph, m['and_2'])['object'].copied_attr()
        xor_attr.update({'opset_version': 7})
        NodeWrap(graph, m['and_2']).replace_obj('Xor', xor_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_meshgrid(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('reshape1', {'op': 'Reshape'}),
                                   ('mul1', {'op': 'Mul'}),
                                   ('reshape2', {'op': 'Reshape'}),
                                   ('mul2', {'op': 'Mul'}),
                                   ('ones', {'op': 'Constant'}),
                               ],
                               edges=[
                                   ('reshape1', 'mul1', {'dst_in_port': 0}),
                                   ('reshape2', 'mul2', {'dst_in_port': 0}),
                                   ('ones', 'mul1', {'dst_in_port': 1}),
                                   ('ones', 'mul2', {'dst_in_port': 1})
                               ])
    for m in matches:
        names = ['reshape1', 'mul1', 'reshape2', 'mul2', 'ones']
        nodes_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in nodes_dict.values()]):
            WARN('[Parser]: Meets invalid Op in merge_meshgrid!')
            continue
        reshape1_in_edges = graph.sorted_in_edges(
            m['reshape1'], keys=True, data=True)
        reshape2_in_edges = graph.sorted_in_edges(
            m['reshape2'], keys=True, data=True)
        if len(reshape1_in_edges) < 1 or len(reshape2_in_edges) < 1:
            WARN('[Parser]: Meets invalid Reshape Op(%s or %s) in merge_meshgrid!' % (
                m['reshape1'], m['reshape2']))
            continue
        ones = nodes_dict['ones'].value
        if not FLOAT_EQUAL(ones, 1):
            continue
        reshape1_in_shape = nodes_dict['reshape1'].get_input_shapes()[0]
        reshape2_in_shape = nodes_dict['reshape2'].get_input_shapes()[0]
        if reshape1_in_shape is None or reshape2_in_shape is None:
            continue
        if len(reshape1_in_shape) != 1 or len(reshape2_in_shape) != 1:
            continue
        if (reshape1_in_shape[0] == ones.shape[0] and reshape2_in_shape[0] == ones.shape[1]) \
                or (reshape1_in_shape[0] == ones.shape[1] and reshape2_in_shape[0] == ones.shape[0]):
            matched = True
            mul1_in_edges = graph.sorted_in_edges(m['mul1'])
            src1, _, k1, in_attr1 = reshape1_in_edges[0]
            src2, _, k2, in_attr2 = reshape2_in_edges[0]
            graph.remove_edge(src1, m['reshape1'], key=k1)
            graph.remove_edge(src2, m['reshape2'], key=k2)
            graph.remove_edges_from(mul1_in_edges)
            graph.add_edge(src1, m['mul1'], **in_attr1)
            new_in_attr2 = copy.deepcopy(in_attr2)
            new_in_attr2['dst_in_port'] = 1
            graph.add_edge(src2, m['mul1'], **new_in_attr2)
            for _, dst, out_attr in graph.sorted_out_edges(m['mul2'], data=True):
                graph.remove_edge(m['mul2'], dst)
                new_out_attr = copy.deepcopy(out_attr)
                new_out_attr['src_out_port'] = 1
                graph.add_edge(m['mul1'], dst, **new_out_attr)
            if m['mul2'] in graph._attr['output_names']:
                index = graph._attr['output_names'].index(m['mul2'])
                graph._attr['output_names'].remove(m['mul2'])
                if m['mul1'] not in graph._attr['output_names']:
                    graph._attr['output_names'][index] = m['mul1']
            if reshape1_in_shape[0] == ones.shape[0] and reshape2_in_shape[0] == ones.shape[1]:
                indexing = 'ij'
            else:
                indexing = 'xy'
            meshgrid_attr = nodes_dict['mul1'].copied_attr()
            meshgrid_attr.update({'indexing': indexing})
            NodeWrap(graph, m['mul1']).replace_obj('Meshgrid', meshgrid_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_prelu(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('input', {}),
                                   ('relu', {'op': 'Relu'}),
                                   ('abs', {'op': 'Abs'}),
                                   ('sub', {'op': 'Sub'}),
                                   ('mul1', {'op': 'Mul'}),
                                   ('mul2', {'op': 'Mul'}),
                                   ('add', {'op': 'Add'}),
                               ],
                               edges=[
                                   ('input', 'relu'),
                                   ('input', 'sub'),
                                   ('input', 'abs'),
                                   ('abs', 'sub'),
                                   ('sub', 'mul1'),
                                   ('mul1', 'mul2'),
                                   ('mul2', 'add'),
                                   ('relu', 'add'),
                               ])
    matched = False
    for m in matches:
        inp, relu, abs, sub, mul1, mul2, add = m['input'], m[
            'relu'], m['abs'], m['sub'], m['mul1'], m['mul2'], m['add']
        input_shapes = NodeWrap(graph, relu)['object'].get_input_shapes()
        mul1_in_consts = NodeWrap(graph, mul1)['object'].sorted_in_consts()
        mul2_in_consts = NodeWrap(graph, mul2)['object'].sorted_in_consts()
        if len(input_shapes) >= 1 \
                and input_shapes[0] is not None \
                and len(mul1_in_consts) == 1 \
                and mul1_in_consts[0][2] is not None \
                and mul1_in_consts[0][2].size == input_shapes[0][-1] \
                and len(mul2_in_consts) == 1 \
                and mul2_in_consts[0][2] is not None \
                and FLOAT_EQUAL(mul2_in_consts[0][2], 0.5):
            matched = True
            slope = mul1_in_consts[0][2]
            relu_out_edges = graph.sorted_out_edges(relu)
            add_out_edges = graph.sorted_out_edges(add, data=True)
            graph.remove_edge(inp, abs)
            graph.remove_edge(inp, sub)
            graph.remove_edges_from(relu_out_edges)
            for _, dst, out_attr in add_out_edges:
                graph.remove_edge(add, dst)
                graph.add_edge(relu, dst, **out_attr)
            relu_attr = NodeWrap(graph, relu)['object'].copied_attr()
            relu_attr.update({'opset_version': 9})
            NodeWrap(graph, relu).replace_obj('PRelu', relu_attr)
            insert_constant(graph, relu + '_slope', slope,
                            relu, in_port=1, data_format='NHWC')
            if add in graph._attr['output_names']:
                index = graph._attr['output_names'].index(add)
                graph._attr['output_names'][index] = relu
    if matched:
        clear_redundant_nodes(graph)

    matches = matched_patterns(graph,
                               nodes=[
                                   ('input', {}),
                                   ('relu', {'op': 'Relu'}),
                                   ('abs', {'op': 'Abs'}),
                                   ('sub', {'op': 'Sub'}),
                                   ('const', {'op': 'Constant'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('add', {'op': 'Add'}),
                               ],
                               edges=[
                                   ('input', 'relu'),
                                   ('input', 'sub'),
                                   ('input', 'abs'),
                                   ('abs', 'sub'),
                                   ('sub', 'mul'),
                                   ('const', 'mul'),
                                   ('mul', 'add'),
                                   ('relu', 'add'),
                               ])
    matched = False
    for m in matches:
        inp, relu, abs, sub, const, mul, add = m['input'], m[
            'relu'], m['abs'], m['sub'], m['const'], m['mul'], m['add']
        node_objs = {name: NodeWrap(graph, name)['object'] for name in [
            inp, relu, abs, sub, const, mul, add]}
        if all([obj is not None for obj in node_objs]) and node_objs[const].value is not None:
            matched = True
            slope = node_objs[const].value * 2
            relu_out_edges = graph.sorted_out_edges(relu)
            add_out_edges = graph.sorted_out_edges(add, data=True)
            graph.remove_edge(inp, abs)
            graph.remove_edge(inp, sub)
            graph.remove_edges_from(relu_out_edges)
            for _, dst, out_attr in add_out_edges:
                graph.remove_edge(add, dst)
                graph.add_edge(relu, dst, **out_attr)
            relu_attr = NodeWrap(graph, relu)['object'].copied_attr()
            relu_attr.update({'opset_version': 9})
            NodeWrap(graph, relu).replace_obj('PRelu', relu_attr)
            insert_constant(graph, relu + '_slope', slope,
                            relu, in_port=1, data_format='NHWC')
            if add in graph._attr['output_names']:
                index = graph._attr['output_names'].index(add)
                graph._attr['output_names'][index] = relu
    if matched:
        clear_redundant_nodes(graph)

    matches = matched_patterns(graph,
                               nodes=[
                                   ('inp', {}),
                                   ('relu', {'op': 'Relu'}),
                                   ('neg_1', {'op': 'Neg'}),
                                   ('relu_1', {'op': 'Relu'}),
                                   ('const', {'op': 'Constant'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('add', {'op': 'Add'}),
                               ],
                               edges=[
                                   ('inp', 'relu'),
                                   ('inp', 'neg_1'),
                                   ('relu', 'add'),
                                   ('neg_1', 'relu_1'),
                                   ('relu_1', 'mul'),
                                   ('const', 'mul'),
                                   ('mul', 'add')
                               ])
    matched = False
    for m in matches:
        inp, relu, neg_1, relu_1, const, mul, add = m['inp'], m['relu'], m[
            'neg_1'], m['relu_1'], m['const'], m['mul'], m['add']
        node_objs = {name: NodeWrap(graph, name)['object'] for name in [
            inp, relu, neg_1, relu_1, const, mul, add]}
        if all([obj is not None for obj in node_objs]) and node_objs[const].value is not None:
            matched = True
            slope = node_objs[const].value * -1.
            relu_out_edges = graph.sorted_out_edges(relu)
            add_out_edges = graph.sorted_out_edges(add, data=True)
            graph.remove_edge(inp, neg_1)
            graph.remove_edges_from(relu_out_edges)
            for _, dst, out_attr in add_out_edges:
                graph.remove_edge(add, dst)
                graph.add_edge(relu, dst, **out_attr)
            relu_attr = NodeWrap(graph, relu)['object'].copied_attr()
            relu_attr.update({'opset_version': 9})
            NodeWrap(graph, relu).replace_obj('PRelu', relu_attr)
            insert_constant(graph, relu + '_slope', slope,
                            relu, in_port=1, data_format='NHWC')
            if add in graph._attr['output_names']:
                index = graph._attr['output_names'].index(add)
                graph._attr['output_names'][index] = relu
    if matched:
        clear_redundant_nodes(graph)


def merge_softplus(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('exp', {'op': 'Exp'}),
                                   ('add', {'op': 'Add'}),
                                   ('log', {'op': 'Log'}),
                               ],
                               edges=[
                                   ('exp', 'add'),
                                   ('add', 'log'),
                               ])
    for m in matches:
        names = ['exp', 'add', 'log']
        obj_dict = {k: NodeWrap(graph, m[k])['object'] for k in names}
        in_edges = graph.sorted_in_edges(m['exp'], data=True)
        exp_out_edges = graph.sorted_out_edges(m['exp'])
        add_in_edges = graph.sorted_in_edges(m['add'])
        add_out_edges = graph.sorted_out_edges(m['add'])
        if all([obj is not None for obj in obj_dict.values()]) \
                and len(in_edges) == 1 \
                and len(exp_out_edges) == 1 \
                and len(add_in_edges) == 2 \
                and len(add_out_edges) == 1:
            add_2nd_input_index = 1 if add_in_edges[0][0] == m['exp'] else 0
            add_2nd_input = add_in_edges[add_2nd_input_index][0]
            if graph.has_node(add_2nd_input) \
                    and NodeWrap(graph, add_2nd_input)['object'] is not None \
                    and NodeWrap(graph, add_2nd_input)['object'].type == 'Constant' \
                    and FLOAT_EQUAL(NodeWrap(graph, add_2nd_input)['object'].value, 1.):
                matched = True
                src, _, in_attr = in_edges[0]
                graph.remove_edges_from(in_edges + add_out_edges)
                graph.add_edge(src, m['log'], **in_attr)
                softplus_attr = obj_dict['log'].copied_attr()
                softplus_attr.update({'opset_version': 1})
                NodeWrap(graph, m['log']).replace_obj(
                    'Softplus', softplus_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_softmax(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('exp', {'op': 'Exp'}),
                                   ('sum', {'op': 'ReduceSum'}),
                                   ('div', {'op': 'Div'}),
                               ],
                               edges=[
                                   ('exp', 'sum'),
                                   ('sum', 'div', {'dst_in_port': 1}),
                                   ('exp', 'div'),
                               ])
    for m in matches:
        names = ['exp', 'sum', 'div']
        obj_dict = {k: NodeWrap(graph, m[k])['object'] for k in names}
        in_edges = graph.sorted_in_edges(m['exp'], data=True)
        exp_out_edges = graph.sorted_out_edges(m['exp'])
        sum_in_edges = graph.sorted_in_edges(m['sum'])
        sum_out_edges = graph.sorted_out_edges(m['sum'])
        div_in_edges = graph.sorted_in_edges(m['div'])
        if all([obj is not None for obj in obj_dict.values()]) \
                and len(in_edges) == 1 \
                and len(exp_out_edges) == 2 \
                and len(sum_in_edges) == 1 \
                and len(sum_out_edges) == 1 \
                and len(obj_dict['sum'].axes) == 1:
            matched = True
            src, _, in_attr = in_edges[0]
            graph.remove_edges_from(in_edges + div_in_edges)
            graph.add_edge(src, m['div'], **in_attr)
            softmax_attr = obj_dict['div'].copied_attr()
            softmax_attr.update(
                {'opset_version': 1, 'axis': obj_dict['sum'].axes[0]})
            NodeWrap(graph, m['div']).replace_obj(
                'Softmax', softmax_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_mish(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('inp', {}),
                                   ('softplus', {'op': 'Softplus'}),
                                   ('tanh', {'op': 'Tanh'}),
                                   ('mul', {'op': 'Mul'}),
                               ],
                               edges=[
                                   ('inp', 'softplus'),
                                   ('softplus', 'tanh'),
                                   ('tanh', 'mul'),
                                   ('inp', 'mul'),
                               ])
    for m in matches:
        names = ['inp', 'softplus', 'tanh', 'mul']
        obj_dict = {k: NodeWrap(graph, m[k])['object'] for k in names}
        softplus_in_edges = graph.sorted_in_edges(m['softplus'], data=True)
        mul_in_edges = graph.sorted_in_edges(m['mul'], data=True)
        softplus_out_edges = graph.sorted_out_edges(m['softplus'])
        tanh_out_edges = graph.sorted_out_edges(m['tanh'])
        out_edges = graph.sorted_out_edges(m['mul'])
        if all([obj is not None for obj in obj_dict.values()]) \
                and len(softplus_in_edges) == 1 \
                and len(mul_in_edges) == 2:
            if len(softplus_out_edges) == 1 \
                    and len(tanh_out_edges) == 1:
                matched = True
                src_out_port = softplus_in_edges[0][2]['src_out_port']
                for mul_src, _, in_attr in mul_in_edges:
                    if mul_src == m['inp'] and in_attr['src_out_port'] != src_out_port:
                        matched = False
                        break
                if not matched:
                    continue
                graph.remove_edges_from(softplus_in_edges + tanh_out_edges)
                mish_attr = obj_dict['mul'].copied_attr()
                mish_attr.update({'opset_version': 1})
                NodeWrap(graph, m['mul']).replace_obj(
                    'Mish', mish_attr)
        else:
            WARN('[Parser]: Meets invalid Node in merge_mish!')
    if matched:
        clear_redundant_nodes(graph)


def multidirectional_broadcasting(graph):
    op_type_list = OpNeedBroadcast.get_concrete_subclass_names()
    for op_type in op_type_list:
        matches = single_node_matcher(graph, op_type)
        for m in matches:
            broadcast = m['target']
            broadcast_obj = NodeWrap(graph, broadcast)['object']
            in_edges = graph.sorted_in_edges(broadcast, keys=True, data=True)
            if broadcast_obj is not None and len(in_edges) >= 2:
                in_tensors = broadcast_obj.get_input_tensors()
                if len(broadcast_obj.sorted_in_consts()) == len(in_edges):
                    WARN(
                        '[Parser]: Broadcast op (%s) with Constant inputs should not be fused in multidirectional_broadcasting!' % broadcast)
                if any([t is None or t.shape == [] for t in in_tensors]):
                    WARN(
                        '[Parser]: Meets Broadcast op (%s) with empty inputs in multidirectional_broadcasting!' % broadcast)
                    continue
                if broadcast_obj.type == 'BitShift' and list(in_tensors[1].shape) == [1]:
                    DEBUG(
                        '[Parser]: Meets Broadcast op (%s) with shift shape is [1], no need to broadcast in multidirectional_broadcasting!' % broadcast)
                    continue
                dims_and_reps = OpNeedBroadcast.cal_reshape_and_tile(
                    [t.shape for t in in_tensors])
                if len(dims_and_reps) == len(in_edges):
                    for i, dr in enumerate(dims_and_reps):
                        if dr['reshape'] is not None:
                            src, _, k, in_attr = in_edges[i]
                            insert_reshape(graph, src, broadcast,
                                           in_attr, dr['reshape'], key=k)
                            in_edges = graph.sorted_in_edges(
                                broadcast, keys=True, data=True)
                        if dr['tile'] is not None:
                            src, _, k, in_attr = in_edges[i]
                            insert_tile(graph, src, broadcast,
                                        in_attr, dr['tile'], key=k)
                            in_edges = graph.sorted_in_edges(
                                broadcast, keys=True, data=True)
                else:
                    WARN(
                        '[Parser]: Failed to calculate Broadcast op (%s) broadcast in multidirectional_broadcasting!' % broadcast)
            else:
                WARN(
                    '[Parser]: Meets Invalid broadcast Op (%s) that cannot be converted in multidirectional_broadcasting!' % broadcast)


def reshape_prelu_slope(graph):
    matches = single_node_matcher(graph, 'PRelu')
    for m in matches:
        prelu = m['target']
        prelu_obj = NodeWrap(graph, prelu)['object']
        if prelu_obj is not None and len(prelu_obj.get_input_shapes()) == 2:
            in_edges = graph.sorted_in_edges(prelu, keys=True, data=True)
            in_shape_0, in_shape_1 = prelu_obj.get_input_shapes()
            if in_shape_0[1:] != in_shape_1:
                reshape_tile = OpNeedUniBroadcast.cal_reshape_and_tile(
                    [in_shape_0, in_shape_1])
                if len(reshape_tile) == 1:
                    if reshape_tile[0]['reshape'] is not None:
                        slope_src, _, k, in_attr = in_edges[1]
                        insert_reshape(graph,
                                       slope_src,
                                       prelu,
                                       in_attr,
                                       reshape_tile[0]['reshape'],
                                       key=k)
                        in_edges = graph.sorted_in_edges(
                            prelu, keys=True, data=True)
                    if reshape_tile[0]['tile'] is not None:
                        slope_src, _, k, in_attr = in_edges[1]
                        insert_tile(graph,
                                    slope_src,
                                    prelu,
                                    in_attr,
                                    reshape_tile[0]['tile'],
                                    key=k)
                        in_edges = graph.sorted_in_edges(
                            prelu, keys=True, data=True)
                else:
                    WARN(
                        '[Parser]: Number of broadcast params is wrong in reshape_prelu_slope!')
        else:
            WARN('[Parser]: Meets invalid PRelu Op(%s) in reshape_prelu_slope!' % prelu)


def duplicate_moments_mean(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('mean1', {'op': 'ReduceMean'}),
                                   ('sub', {'op': 'Sub'}),
                                   ('pow', {'op': 'Pow'}),
                                   ('mean2', {'op': 'ReduceMean'}),
                               ],
                               edges=[
                                   ('mean1', 'sub'),
                                   ('sub', 'pow', {'dst_in_port': 0}),
                                   ('pow', 'mean2')
                               ])
    for m in matches:
        names = ['mean1', 'sub', 'pow', 'mean2']
        node_objs = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in node_objs.values()]):
            WARN('[Parser]: Meets invalid node in duplicate_moments_mean!')
            continue
        if len(graph.sorted_out_edges(m['mean1'])) <= 1:
            continue
        if any([len(graph.sorted_out_edges(m[n])) > 1 for n in ['sub', 'pow']]):
            continue
        mean1_in_edges = graph.sorted_in_edges(m['mean1'], data=True)
        sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
        if len(mean1_in_edges) != 1 or len(sub_in_edges) != 2:
            continue
        mean1_copy = get_valid_node_name(graph, m['mean1'] + '_copy')
        inp, _, mean1_in_attr = mean1_in_edges[0]
        graph.add_edge(inp, mean1_copy, **copy.deepcopy(mean1_in_attr))
        for _, dst, k, out_attr in graph.sorted_out_edges(m['mean1'], keys=True, data=True):
            if dst == m['sub']:
                graph.remove_edge(m['mean1'], dst, key=k)
                graph.add_edge(mean1_copy, dst, **out_attr)
                break
        mean1_attr = node_objs['mean1'].copied_attr()
        mean1_attr.update({'name': mean1_copy})
        NodeWrap(graph, mean1_copy).replace_obj('ReduceMean', mean1_attr)


def merge_reduce_variance(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('mean1', {'op': 'ReduceMean'}),
                                   ('sub', {'op': 'Sub'}),
                                   ('pow', {'op': 'Pow'}),
                                   ('mean2', {'op': 'ReduceMean'}),
                               ],
                               edges=[
                                   ('mean1', 'sub'),
                                   ('sub', 'pow', {'dst_in_port': 0}),
                                   ('pow', 'mean2')
                               ])
    for m in matches:
        names = ['mean1', 'sub', 'pow', 'mean2']
        node_objs = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in node_objs.values()]):
            WARN('[Parser]: Meets invalid node in merge_reduce_variance!')
            continue
        if any([len(graph.sorted_out_edges(m[n])) > 1 for n in ['mean1', 'sub', 'pow']]):
            continue
        mean1_in_edges = graph.sorted_in_edges(m['mean1'], data=True)
        sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
        if len(mean1_in_edges) != 1 or len(sub_in_edges) != 2:
            continue
        inp, _, mean1_in_attr = mean1_in_edges[0]
        inp_out_port = mean1_in_attr['src_out_port']
        found = False
        for sub_src, _, in_attr in sub_in_edges:
            if sub_src == inp and in_attr['src_out_port'] == inp_out_port:
                found = True
                break
        if not found:
            continue
        if not np.array_equal(np.sort(node_objs['mean1'].axes), np.sort(node_objs['mean2'].axes)):
            continue
        if len(node_objs['pow'].get_input_tensors()) != 2 \
                or not FLOAT_EQUAL(node_objs['pow'].get_input_tensors()[1], 2):
            continue
        matched = True
        mean2_in_edges = graph.sorted_in_edges(m['mean2'])
        graph.remove_edges_from(mean2_in_edges)
        graph.add_edge(inp, m['mean2'], **mean1_in_attr)
        mean2_attr = node_objs['mean2'].copied_attr()
        NodeWrap(graph, m['mean2']).replace_obj(
            'ReduceVariance', mean2_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_reduce_unbiased_variance(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('var', {'op': 'ReduceVariance'}),
                                   ('const_m', {'op': 'Constant'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('const_d', {'op': 'Constant'}),
                                   ('div', {'op': 'Div'}),
                               ],
                               edges=[
                                   ('var', 'mul'),
                                   ('const_m', 'mul'),
                                   ('mul', 'div', {
                                    'src_out_port': 0, 'dst_in_port': 0}),
                                   ('const_d', 'div', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                               ])
    for m in matches:
        names = ['var', 'const_m', 'mul', 'const_d', 'div']
        node_objs = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in node_objs.values()]):
            WARN('[Parser]: Meets invalid node in merge_reduce_unbiased_variance!')
            continue

        if node_objs['var'].unbiased:
            continue

        var_in_edges = graph.sorted_in_edges(m['var'], data=True)
        var_input_shapes = node_objs['var'].get_input_shapes()
        if len(var_in_edges) != 1 or len(var_input_shapes) != 1:
            WARN(
                '[Parser]: Meets invalid node (%s) in merge_reduce_unbiased_variance!' % m['var'])
            continue

        var_out_edges = graph.sorted_out_edges(m['var'])
        mul_out_edges = graph.sorted_out_edges(m['mul'])
        if var_input_shapes[0] is None or \
                len(var_out_edges) != 1 or len(mul_out_edges) != 1:
            continue

        ref_mul_operand = np.prod(
            np.take(var_input_shapes[0], node_objs['var'].axes))
        ref_div_operand = ref_mul_operand - 1
        if not FLOAT_EQUAL(ref_mul_operand, node_objs['const_m'].value) or \
                not FLOAT_EQUAL(ref_div_operand, node_objs['const_d'].value):
            continue

        matched = True
        div_in_edges = graph.sorted_in_edges(m['div'])
        graph.remove_edges_from(var_in_edges + div_in_edges)
        src, _, var_in_attr = var_in_edges[0]
        graph.add_edge(src, m['div'], **var_in_attr)

        unbiases_var_attr = node_objs['div'].copied_attr()
        unbiases_var_attr.update(
            {'opset_version': 1,
             'unbiased': 1,
             'keepdims': node_objs['var'].keepdims,
             'axes': node_objs['var'].axes})
        NodeWrap(graph, m['div']).replace_obj(
            'ReduceVariance', unbiases_var_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_normalized_moments(graph):
    normalized_moments_matches1 = matched_patterns(graph,
                                                   nodes=[
                                                       ('mul1', {'op': 'Mul'}),
                                                       ('mul2', {'op': 'Mul'}),
                                                       ('pow', {'op': 'Pow'}),
                                                       ('sub', {'op': 'Sub'}),
                                                       ('mul1_out'),
                                                   ],
                                                   edges=[
                                                       ('mul1', 'pow'),
                                                       ('mul1', 'mul1_out'),
                                                       ('pow', 'sub', {
                                                           'src_out_port': 0, 'dst_in_port': 1}),
                                                       ('mul2', 'sub'),
                                                   ])

    normalized_moments_matches2 = matched_patterns(graph,
                                                   nodes=[
                                                       ('mul1', {'op': 'Mul'}),
                                                       ('mul2', {'op': 'Mul'}),
                                                       ('pow', {'op': 'Pow'}),
                                                       ('sub', {'op': 'Sub'}),
                                                   ],
                                                   edges=[
                                                       ('mul1', 'pow'),
                                                       ('pow', 'sub', {
                                                           'src_out_port': 0, 'dst_in_port': 1}),
                                                       ('mul2', 'sub'),
                                                   ])

    normalized_moments_matches2 = list(filter(None, list(map(lambda y: y if sum(list(map(lambda x: len(
        x.items() & y.items()) > 0, normalized_moments_matches1))) == 0 else None, normalized_moments_matches2))))
    normalized_moments_matches = normalized_moments_matches1 + normalized_moments_matches2

    matched = False
    for m in normalized_moments_matches:
        key_names = ['mul1', 'mul2', 'pow', 'sub', 'mul1_out'] if 'mul1_out' in m else [
            'mul1', 'mul2', 'pow', 'sub']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if any([obj is None for obj in node_objs.values()]):
            WARN('[Parser]: Meets invalid nodes in merge_normalized_moments!')
            continue

        mul_1_in_edges = graph.sorted_in_edges(m['mul1'], data=True)
        mul_2_in_edges = graph.sorted_in_edges(m['mul2'], data=True)
        sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
        sub_out_edges = graph.sorted_out_edges(m['sub'], data=True)
        pow_out_edges = graph.sorted_out_edges(m['pow'], data=True)
        mul_1_out_edges = graph.sorted_out_edges(m['mul1'], data=True)
        mul_2_out_edges = graph.sorted_out_edges(m['mul2'], data=True)

        if len(pow_out_edges) != 1\
                or len(mul_2_out_edges) != 1\
                or len(node_objs['pow'].sorted_in_consts()) != 1\
                or len(node_objs['mul1'].sorted_in_consts()) != 1\
                or len(node_objs['mul2'].sorted_in_consts()) != 1\
                or FLOAT_EQUAL(node_objs['mul1'].sorted_in_consts()[0][2], node_objs['mul2'].sorted_in_consts()[0][2]) is False:
            continue

        count_value = round(1/node_objs['mul1'].sorted_in_consts()[0][2], 5)

        have_add = False
        have_add = True if len(
            mul_1_out_edges) == 2 and node_objs['mul1_out'].type == 'Add' else False

        # has_shift = False(2 inputs) or has_shift = True(3 inputs)
        if have_add is True:
            add_name = m['mul1_out']
            add_in_edges = graph.sorted_in_edges(add_name, data=True)
            add_out_edges = graph.sorted_out_edges(add_name, data=True)
            if len(add_in_edges) != 2:
                continue

            matched = True
            inp3_src, _, inp_attr3 = add_in_edges[1] if add_in_edges[0][0] == m['mul1'] else add_in_edges[0]
            graph.remove_edge(inp3_src, add_name)
            nor_moments_inp3_attr = copy.deepcopy(inp_attr3)
            nor_moments_inp3_attr.update({'dst_in_port': 2})
            graph.remove_edges_from(sub_in_edges)
            graph.add_edge(inp3_src, m['sub'], **nor_moments_inp3_attr)
            for _, dst, out_attr in add_out_edges:
                graph.remove_edge(add_name, dst)
                out_attr['src_out_port'] = 0
                graph.add_edge(m['sub'], dst, **out_attr)

            if add_name in graph._attr['output_names']:
                index = graph._attr['output_names'].index(add_name)
                graph._attr['output_names'].pop(index)

        else:
            matched = True
            inp_shape = node_objs['mul1'].get_input_shapes()[0]
            shift_name = get_valid_node_name(graph, 'NormalizedMoments_shift')
            shift_value = np.zeros(inp_shape).astype(np.float32)
            graph._attr['input_tensors'][shift_name] = Tensor(
                value=shift_value)
            graph.remove_edges_from(sub_in_edges)
            insert_constant(graph, shift_name, shift_value,
                            m['sub'], in_port=2)
            for _, dst, out_attr in mul_1_out_edges:
                graph.remove_edge(m['mul1'], dst)
                out_attr['src_out_port'] = 0
                graph.add_edge(m['sub'], dst, **out_attr)

        inp1_src, _, inp_attr1 = mul_1_in_edges[0]
        inp2_src, _, inp_attr2 = mul_2_in_edges[0]
        inp_attr2['dst_in_port'] = 1
        graph.remove_edge(inp1_src, m['mul1'])
        graph.remove_edge(inp2_src, m['mul2'])
        graph.add_edge(inp1_src, m['sub'], **inp_attr1)
        graph.add_edge(inp2_src, m['sub'], **inp_attr2)
        for _, dst, out_attr in sub_out_edges:
            out_attr['src_out_port'] = 1

        moment_attr = node_objs['sub'].copied_attr()
        moment_attr.update({'counts': count_value})
        NodeWrap(graph, m['sub']).replace_obj('NormalizedMoments', moment_attr)

        if m['mul1'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['mul1'])
            graph._attr['output_names'].pop(index)

    if matched:
        clear_redundant_nodes(graph)


def merge_moments(graph):
    moments_matches = matched_patterns(graph,
                                       nodes=[
                                           ('input', {}),
                                           ('mean', {'op': 'ReduceMean'}),
                                           ('variance', {
                                            'op': 'ReduceVariance'}),
                                       ],
                                       edges=[
                                           ('input', 'mean'),
                                           ('input', 'variance'),
                                       ])
    matched = False
    for m in moments_matches:
        key_names = ['mean', 'variance']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if any([obj is None for obj in node_objs.values()]):
            WARN('[Parser]: Meets invalid nodes in merge_moments!')
            continue
        if node_objs['variance'].unbiased:
            continue
        if node_objs['variance'].keepdims != node_objs['mean'].keepdims \
                or (not sorted(node_objs['variance'].axes) == sorted(node_objs['mean'].axes)):
            continue
        mean_in_edges = graph.sorted_in_edges(m['mean'], data=True)
        variance_in_edges = graph.sorted_in_edges(m['variance'], data=True)
        if variance_in_edges[0][2]['src_out_port'] != mean_in_edges[0][2]['src_out_port']:
            continue
        matched = True
        graph.remove_edges_from(variance_in_edges)
        for _, dst, out_attr in graph.sorted_out_edges(m['variance'], data=True):
            graph.remove_edge(m['variance'], dst)
            out_attr['src_out_port'] = 1
            graph.add_edge(m['mean'], dst, **out_attr)
        moments_attr = node_objs['mean'].copied_attr()
        NodeWrap(graph, m['mean']).replace_obj('Moments', moments_attr)
        if m['variance'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['variance'])
            graph._attr['output_names'].pop(index)
            if m['mean'] not in graph._attr['output_names']:
                graph._attr['output_names'].insert(index, m['mean'])
    if matched:
        clear_redundant_nodes(graph)


def merge_erosion(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('neg1', {'op': 'Neg'}),
                                      ('dilation', {'op': 'Dilation'}),
                                      ('neg2', {'op': 'Neg'}),
                                      ],
                               edges=[('neg1', 'dilation'),
                                      ('dilation', 'neg2'),
                                      ])
    for m in matches:
        names = ['neg1', 'dilation', 'neg2']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if all([obj is not None for obj in obj_dict.values()]):
            neg1_in_edges = graph.sorted_in_edges(m['neg1'], data=True)
            neg2_out_edges = graph.sorted_out_edges(m['neg2'], data=True)
            dilation_out_edges = graph.sorted_out_edges(m['dilation'], data=True)
            if len(neg1_in_edges) == 1 \
                    and len(dilation_out_edges) == 1:
                matched = True
                src1, dst1, in_attr = neg1_in_edges[0]
                act_attr = obj_dict['dilation'].copied_attr()
                graph.remove_edge(src1, m['neg1'])
                graph.remove_edge(m['neg1'], m['dilation'])
                graph.remove_edge(m['dilation'], m['neg2'])
                for src2, dst2, out_attr in neg2_out_edges:
                    graph.remove_edge(m['neg2'], dst2)
                    graph.add_edge(m['dilation'], dst2, **out_attr)
                graph.add_edge(src1, m['dilation'], **in_attr)
                if len(obj_dict['dilation'].weights.shape) == 3:
                    new_weights = np.flip(obj_dict['dilation'].weights, [1, 2])
                    act_attr.update({'weights': new_weights})

                NodeWrap(graph, m['dilation']).replace_obj('Erosion', act_attr)

                if m['neg2'] in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(m['neg2'])
                    graph._attr['output_names'][index] = m['dilation']
        else:
            WARN('[Parser]: Invalid node in merge_erosion!')
    if matched:
        clear_redundant_nodes(graph)


def merge_l2norm(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('input', {}),
                                   ('square', {'op': 'Mul'}),
                                   ('l2norm', {'op': 'Mul'}),
                                   ('sum', {'op': 'ReduceSum'}),
                                   ('max', {'op': 'Max'}),
                                   ('sqrt', {'op': 'Sqrt'}),
                                   ('recip', {'op': 'Reciprocal'}),
                               ],
                               edges=[
                                   ('input', 'square', {
                                    'src_out_port': 0, 'dst_in_port': 0}),
                                   ('input', 'square', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('input', 'l2norm'),
                                   ('square', 'sum'),
                                   ('sum', 'max'),
                                   ('max', 'sqrt'),
                                   ('sqrt', 'recip'),
                                   ('recip', 'l2norm', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                               ])
    matches2 = matched_patterns(graph,
                                nodes=[
                                    ('input', {}),
                                    ('pow_y', {'op': 'Constant'}),
                                    ('pow', {'op': 'Pow'}),
                                    ('l2norm', {'op': 'Mul'}),
                                    ('sum', {'op': 'ReduceSum'}),
                                    ('max', {'op': 'Max'}),
                                    ('sqrt', {'op': 'Sqrt'}),
                                    ('recip', {'op': 'Reciprocal'}),
                                ],
                                edges=[
                                    ('input', 'pow'),
                                    ('pow_y', 'pow'),
                                    ('input', 'l2norm'),
                                    ('pow', 'sum'),
                                    ('sum', 'max'),
                                    ('max', 'sqrt'),
                                    ('sqrt', 'recip'),
                                    ('recip', 'l2norm', {
                                        'src_out_port': 0, 'dst_in_port': 1}),
                                ])
    for m in matches + matches2:
        inp, sum, l2norm = m['input'], m['sum'], m['l2norm']
        if 'square' in m:
            square = m['square']
        else:
            square = m['pow']
            pow_y_obj = NodeWrap(graph, m['pow_y'])['object']
            if pow_y_obj is None or not FLOAT_EQUAL(pow_y_obj.value, 2.):
                continue
        sum_obj = NodeWrap(graph, sum)['object']
        if len(sum_obj.axes) == 1:
            matched = True
            axis = sum_obj.axes[0]
            square_in_edges = graph.sorted_in_edges(square)
            l2norm_in_edges = graph.sorted_in_edges(l2norm, data=True)
            graph.remove_edges_from(square_in_edges)
            for src, _, in_attr in l2norm_in_edges:
                if src == inp:
                    new_in_attr = copy.deepcopy(in_attr)
                    new_in_attr.update({'dst_in_port': 0})
                    graph.remove_edge(src, l2norm)
                    graph.add_edge(src, l2norm, **new_in_attr)
                else:
                    graph.remove_edge(src, l2norm)
            l2norm_attr = NodeWrap(graph, l2norm)['object'].copied_attr()
            l2norm_attr.update({'opset_version': 1, 'p': 2, 'axis': axis})
            NodeWrap(graph, l2norm).replace_obj('LpNormalization', l2norm_attr)
        else:
            WARN('[Parser]: Invalid ReduceSum axes for node(%s) in merge_l2norm!' % sum)
    if matched:
        clear_redundant_nodes(graph)


def merge_ln(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('inp', {}),
                                   ('mean_1', {'op': 'ReduceMean'}),
                                   ('sub', {'op': 'Sub'}),
                                   ('mul_1', {'op': 'Mul'}),
                                   ('div', {'op': 'Div'}),
                                   ('mean_2', {'op': 'ReduceMean'}),
                                   ('add_1', {'op': 'Add'}),
                                   ('epsilon', {'op': 'Constant'}),
                                   ('pow', {'op': 'Pow'}),
                                   ('pow_y', {'op': 'Constant'}),
                                   ('mul_2', {'op': 'Mul'}),
                                   ('gamma', {'op': 'Constant'}),
                                   ('add_2', {'op': 'Add'}),
                                   ('beta', {'op': 'Constant'}),
                               ],
                               edges=[
                                   ('inp', 'mean_1'),
                                   ('inp', 'sub'),
                                   ('mean_1', 'sub', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('sub', 'mul_1', {
                                    'src_out_port': 0, 'dst_in_port': 0}),
                                   ('sub', 'mul_1', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('sub', 'div'),
                                   ('mul_1', 'mean_2'),
                                   ('mean_2', 'add_1'),
                                   ('epsilon', 'add_1'),
                                   ('add_1', 'pow'),
                                   ('pow_y', 'pow'),
                                   ('pow', 'div'),
                                   ('div', 'mul_2'),
                                   ('gamma', 'mul_2'),
                                   ('mul_2', 'add_2'),
                                   ('beta', 'add_2'),
                               ]
                               )
    for m in matches:
        key_names = ['inp', 'sub', 'mean_1', 'add_2',
                     'epsilon', 'pow_y', 'gamma', 'beta']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if all([obj is not None for obj in node_objs.values()]) and FLOAT_EQUAL(node_objs['pow_y'].value, 0.5):
            matched = True
            epsilon = float(node_objs['epsilon'].value)
            gamma = node_objs['gamma'].value
            beta = node_objs['beta'].value
            mean_1_in_edges = graph.sorted_in_edges(m['mean_1'], data=True)
            add_2_in_edges = graph.sorted_in_edges(m['add_2'])
            _, _, in_attr = mean_1_in_edges[0]
            graph.remove_edge(m['inp'], m['mean_1'])
            graph.remove_edge(m['inp'], m['sub'])
            graph.remove_edges_from(add_2_in_edges)
            graph.add_edge(m['inp'], m['add_2'], **in_attr)
            ln_attr = node_objs['add_2'].copied_attr()
            ln_attr.update({'epsilon': epsilon, 'weights': gamma,
                            'biases': beta, 'axes': [-1]})
            NodeWrap(graph, m['add_2']).replace_obj('LayerNorm', ln_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_ln2(graph):
    matched = False
    ln_matches = matched_patterns(graph,
                                  nodes=[
                                      ('inp', {}),
                                      ('mean_1', {'op': 'ReduceMean'}),
                                      ('sub_1', {'op': 'Sub'}),
                                      ('pow', {'op': 'Pow'}),
                                      ('pow_y', {'op': 'Constant'}),
                                      ('mean_2', {'op': 'ReduceMean'}),
                                      ('add_1', {'op': 'Add'}),
                                      ('epsilon', {'op': 'Constant'}),
                                      ('sqrt', {'op': 'Sqrt'}),
                                      ('recip', {'op': 'Reciprocal'}),
                                      ('mul_1', {'op': 'Mul'}),
                                      ('mul_2', {'op': 'Mul'}),
                                      ('sub_2', {'op': 'Sub'}),
                                      ('beta', {'op': 'Constant'}),
                                      ('add_2', {'op': 'Add'}),
                                  ],
                                  edges=[
                                      ('inp', 'mean_1'),
                                      ('inp', 'mul_1'),
                                      ('inp', 'sub_1'),
                                      ('mean_1', 'sub_1', {'dst_in_port': 1}),
                                      ('sub_1', 'pow'),
                                      ('pow_y', 'pow', {'dst_in_port': 1}),
                                      ('pow', 'mean_2'),
                                      ('mean_2', 'add_1'),
                                      ('epsilon', 'add_1', {'dst_in_port': 1}),
                                      ('add_1', 'sqrt'),
                                      ('sqrt', 'recip'),
                                      ('recip', 'mul_1', {'dst_in_port': 1}),
                                      ('mean_1', 'mul_2'),
                                      ('recip', 'mul_2', {'dst_in_port': 1}),
                                      ('beta', 'sub_2'),
                                      ('mul_2', 'sub_2', {'dst_in_port': 1}),
                                      ('mul_1', 'add_2'),
                                      ('sub_2', 'add_2', {'dst_in_port': 1})
                                  ])
    ln_matches2 = matched_patterns(graph,
                                   nodes=[
                                       ('inp', {}),
                                       ('mean_1', {'op': 'ReduceMean'}),
                                       ('sub_1', {'op': 'Sub'}),
                                       ('pow', {'op': 'Pow'}),
                                       ('pow_y', {'op': 'Constant'}),
                                       ('mean_2', {'op': 'ReduceMean'}),
                                       ('add_1', {'op': 'Add'}),
                                       ('epsilon', {'op': 'Constant'}),
                                       ('sqrt', {'op': 'Sqrt'}),
                                       ('recip', {'op': 'Reciprocal'}),
                                       ('gamma', {'op': 'Constant'}),
                                       ('mul_gamma', {'op': 'Mul'}),
                                       ('mul_1', {'op': 'Mul'}),
                                       ('mul_2', {'op': 'Mul'}),
                                       ('sub_2', {'op': 'Sub'}),
                                       ('beta', {'op': 'Constant'}),
                                       ('add_2', {'op': 'Add'}),
                                   ],
                                   edges=[
                                       ('inp', 'mean_1'),
                                       ('inp', 'mul_1'),
                                       ('inp', 'sub_1'),
                                       ('mean_1', 'sub_1', {'dst_in_port': 1}),
                                       ('sub_1', 'pow'),
                                       ('pow_y', 'pow', {'dst_in_port': 1}),
                                       ('pow', 'mean_2'),
                                       ('mean_2', 'add_1'),
                                       ('epsilon', 'add_1',
                                        {'dst_in_port': 1}),
                                       ('add_1', 'sqrt'),
                                       ('sqrt', 'recip'),
                                       ('recip', 'mul_gamma'),
                                       ('gamma', 'mul_gamma',
                                           {'dst_in_port': 1}),
                                       ('mul_gamma', 'mul_1',
                                           {'dst_in_port': 1}),
                                       ('mean_1', 'mul_2'),
                                       ('mul_gamma', 'mul_2',
                                           {'dst_in_port': 1}),
                                       ('beta', 'sub_2'),
                                       ('mul_2', 'sub_2', {'dst_in_port': 1}),
                                       ('mul_1', 'add_2'),
                                       ('sub_2', 'add_2', {'dst_in_port': 1})
                                   ])
    ln_matches = ln_matches + ln_matches2
    for m in ln_matches:
        key_names = ['inp', 'mean_1', 'mean_2', 'mul_1',
                     'sub_1', 'pow_y', 'epsilon', 'beta', 'add_2']
        inp, mean_1, mean_2, mul_1, sub_1, pow_y, epsilon, beta, add_2 = [
            m[name] for name in key_names]
        objs_dict = {m[name]: NodeWrap(graph, m[name])[
            'object'] for name in key_names}
        if 'gamma' in m:
            gamma = m['gamma']
            objs_dict.update({gamma: NodeWrap(graph, gamma)['object']})
        else:
            gamma = ''
        if all(obj is not None for obj in objs_dict.values()):
            if objs_dict[epsilon].value.size > 1 \
                    and not FLOAT_EQUAL(objs_dict[epsilon].value.flatten()[1:], objs_dict[epsilon].value.item(0)):
                continue
            input_shapes = objs_dict[inp].get_output_shapes()
            mean_1_in_edges = graph.sorted_in_edges(mean_1, data=True)
            mul_1_in_edges = graph.sorted_in_edges(mul_1, data=True)
            sub_1_in_edges = graph.sorted_in_edges(sub_1, data=True)
            if len(input_shapes) > 0 \
                    and input_shapes[0] \
                    and len(input_shapes[0]) > 1 \
                    and len(mean_1_in_edges) >= 1 \
                    and len(mul_1_in_edges) == 2 \
                    and len(sub_1_in_edges) == 2 \
                    and mean_1_in_edges[0][2]['src_out_port'] == mul_1_in_edges[0][2]['src_out_port'] \
                    and mean_1_in_edges[0][2]['src_out_port'] == sub_1_in_edges[0][2]['src_out_port'] \
                    and FLOAT_EQUAL(objs_dict[pow_y].value, 2.) \
                    and objs_dict[mean_1].axes == objs_dict[mean_2].axes:
                in_shape = input_shapes[0]
                axes = OpHasAxis.make_axes_non_negative(
                    objs_dict[mean_1].axes, len(in_shape))
                axes.sort()
                biases = objs_dict[beta].value
                weights = objs_dict[gamma].value if gamma else None
                non_axes = [num for num in range(
                    len(in_shape)) if num not in axes]
                pre_perm = None
                is_in = False
                if len(non_axes) == 2 and non_axes[0] == 0:
                    in_biases = OpHasAxis.align_axes(
                        biases, non_axes[1], [in_shape[non_axes[1]]])
                    in_weights = None
                    if gamma:
                        in_weights = OpHasAxis.align_axes(
                            weights, non_axes[1], [in_shape[non_axes[1]]])
                    if in_biases is not None and (not gamma or in_weights is not None):
                        is_in = True
                        if non_axes[1] not in (1, len(in_shape) - 1):
                            pre_perm = [num for num in range(
                                len(in_shape)) if num != non_axes[1]] + [non_axes[1]]
                            channel_axis = len(in_shape) - 1
                            axes = list(range(1, channel_axis))
                        else:
                            channel_axis = non_axes[1]
                        data_format = 'NCHW' if channel_axis == 1 else 'NHWC'
                        biases = in_biases
                        weights = in_weights
                if not is_in:
                    exp_shape = [in_shape[axis] for axis in axes]
                    biases = OpHasAxis.align_axes(biases, axes, exp_shape)
                    weights = OpHasAxis.align_axes(
                        weights, axes, exp_shape) if gamma else None
                    if biases is None or (gamma and weights is None):
                        continue
                matched = True
                if weights is None:
                    weights = np.ones_like(biases)
                eps = float(objs_dict[epsilon].value.item(0))
                inp_out_attr = copy.deepcopy(mean_1_in_edges[0][2])
                inp_out_attr.update({'dst_in_port': 0})
                add_2_in_edges = graph.sorted_in_edges(add_2)
                graph.remove_edges_from(
                    mean_1_in_edges + mul_1_in_edges + sub_1_in_edges + add_2_in_edges)
                graph.add_edge(inp, add_2, **inp_out_attr)
                ln_attr = objs_dict[add_2].copied_attr()
                ln_attr.update(
                    {'epsilon': eps, 'weights': weights, 'biases': biases})
                if is_in:
                    ln_attr.update(
                        {'opset_version': 6, 'non_channel_axes': axes, 'data_format': data_format})
                    NodeWrap(graph, add_2).replace_obj(
                        'InstanceNormalization', ln_attr)
                else:
                    ln_attr.update({'axes': axes})
                    NodeWrap(graph, add_2).replace_obj('LayerNorm', ln_attr)
                if pre_perm is not None:
                    insert_transpose(graph, inp, add_2, inp_out_attr, pre_perm)
                    post_trans = insert_transpose_after(
                        graph, add_2, Op.cal_inverse_perm(pre_perm))
                    if add_2 in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(add_2)
                        graph._attr['output_names'][index] = post_trans
        else:
            WARN('[Parser]: Meets invalid nodes in merge_ln2!')
    if matched:
        clear_redundant_nodes(graph)


def merge_ln3(graph):
    matched = False
    ln_matches = matched_patterns(graph,
                                  nodes=[
                                      ('inp', {}),
                                      ('sub', {'op': 'Sub'}),
                                      ('mean', {'op': 'ReduceMean'}),
                                      ('sub_1', {'op': 'Sub'}),
                                      ('square', {'op': 'Pow'}),
                                      ('power', {'op': 'Constant'}),
                                      ('mean_1', {'op': 'ReduceMean'}),
                                      ('add', {'op': 'Add'}),
                                      ('epsilon', {'op': 'Constant'}),
                                      ('sqrt', {'op': 'Sqrt'}),
                                      ('recip', {'op': 'Reciprocal'}),
                                      ('mul', {'op': 'Mul'}),
                                      ('add_1', {'op': 'Add'}),
                                      ('beta', {'op': 'Constant'})
                                  ],
                                  edges=[
                                      ('inp', 'sub'),
                                      ('inp', 'mean'),
                                      ('inp', 'sub_1'),
                                      ('mean', 'sub', {'dst_in_port': 1}),
                                      ('mean', 'sub_1', {'dst_in_port': 1}),
                                      ('sub', 'square'),
                                      ('power', 'square', {'dst_in_port': 1}),
                                      ('square', 'mean_1'),
                                      ('mean_1', 'add'),
                                      ('epsilon', 'add', {'dst_in_port': 1}),
                                      ('add', 'sqrt'),
                                      ('sqrt', 'recip'),
                                      ('recip', 'mul', {'dst_in_port': 1}),
                                      ('sub_1', 'mul'),
                                      ('mul', 'add_1'),
                                      ('beta', 'add_1', {'dst_in_port': 1}),
                                  ])
    for m in ln_matches:
        key_names = ['inp', 'mean', 'sub', 'sub_1', 'square', 'power',
                     'mean_1', 'add', 'epsilon', 'sqrt', 'recip', 'mul', 'beta', 'add_1']
        inp, mean, sub, sub_1, square, power, mean_1, add, epsilon, sqrt, recip, mul_1, beta, add_1 = [
            m[name] for name in key_names]
        objs_dict = {m[name]: NodeWrap(graph, m[name])[
            'object'] for name in key_names}
        if all([obj is not None for obj in objs_dict.values()]):
            out_edges_len = [len(graph.sorted_out_edges(m[inner])) for inner in set(
                key_names).difference(['inp', 'mean', 'add_1'])]
            if any([l != 1 for l in out_edges_len]):
                continue
            input_shapes = objs_dict[inp].get_output_shapes()
            mean_in_edges = graph.sorted_in_edges(mean, data=True)
            sub_in_edges = graph.sorted_in_edges(sub, data=True)
            sub_1_in_edges = graph.sorted_in_edges(sub_1, data=True)
            if len(input_shapes) > 0 \
                    and input_shapes[0] \
                    and len(input_shapes[0]) > 1 \
                    and len(mean_in_edges) >= 1 \
                    and len(sub_in_edges) == 2 \
                    and len(sub_1_in_edges) == 2 \
                    and mean_in_edges[0][2]['src_out_port'] == sub_in_edges[0][2]['src_out_port'] \
                    and mean_in_edges[0][2]['src_out_port'] == sub_1_in_edges[0][2]['src_out_port'] \
                    and FLOAT_EQUAL(objs_dict[power].value, 2.) \
                    and objs_dict[mean].axes == objs_dict[mean_1].axes:
                beta = OpHasAxis.align_axes(objs_dict[beta].value,
                                            objs_dict[mean].axes,
                                            [input_shapes[0][axis] for axis in objs_dict[mean].axes])
                if beta is None:
                    continue
                matched = True
                axes = OpHasAxis.make_axes_non_negative(
                    objs_dict[mean].axes, len(input_shapes[0]))
                axes = sorted(axes)
                eps = float(objs_dict[epsilon].value)
                gamma = np.ones_like(beta)
                inp_out_attr = copy.deepcopy(mean_in_edges[0][2])
                inp_out_attr.update({'dst_in_port': 0})
                add_1_in_edges = graph.sorted_in_edges(add_1)
                graph.remove_edges_from(
                    mean_in_edges + sub_in_edges + sub_1_in_edges + add_1_in_edges)
                graph.add_edge(inp, add_1, **inp_out_attr)
                ln_attr = objs_dict[add_1].copied_attr()
                ln_attr.update({'axes': axes, 'epsilon': eps,
                                'weights': gamma, 'biases': beta})
                NodeWrap(graph, add_1).replace_obj('LayerNorm', ln_attr)
        else:
            WARN('[Parser]: Meets invalid nodes in merge_ln3!')
    if matched:
        clear_redundant_nodes(graph)


def merge_ln4(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('mean_1', {'op': 'ReduceMean'}),
                                   ('sub', {'op': 'Sub'}),
                                   ('pow', {'op': 'Pow'}),
                                   ('pow_y', {'op': 'Constant'}),
                                   ('mean_2', {'op': 'ReduceMean'}),
                                   ('add_1', {'op': 'Add'}),
                                   ('sqrt', {'op': 'Sqrt'}),
                                   ('div', {'op': 'Div'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('add_2', {'op': 'Add'}),
                                   ('eps', {'op': 'Constant'}),
                                   ('weight', {'op': 'Constant'}),
                                   ('bias', {'op': 'Constant'}),
                               ],
                               edges=[
                                   ('mean_1', 'sub', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('sub', 'pow'),
                                   ('pow', 'mean_2'),
                                   ('mean_2', 'add_1'),
                                   ('add_1', 'sqrt'),
                                   ('sqrt', 'div', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('sub', 'div'),
                                   ('div', 'mul'),
                                   ('pow_y', 'pow', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('mul', 'add_2'),
                                   ('eps', 'add_1', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('weight', 'mul', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('bias', 'add_2', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                               ]
                               )
    for m in matches:
        key_names = ['mean_1', 'sub', 'pow', 'pow_y',
                     'mean_2', 'add_1', 'sqrt', 'div', 'mul', 'add_2', 'eps', 'weight', 'bias']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if all([obj is not None for obj in node_objs.values()]):
            mean_1_in_edges = graph.sorted_in_edges(m['mean_1'], data=True)
            sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
            if len(mean_1_in_edges) != 1 \
                    or len(sub_in_edges) != 2 \
                    or mean_1_in_edges[0][0] != sub_in_edges[0][0] \
                    or mean_1_in_edges[0][2]['src_out_port'] != sub_in_edges[0][2]['src_out_port']:
                continue

            add_2_in_edges = graph.sorted_in_edges(m['add_2'])

            mean_1_out_edges = graph.sorted_out_edges(m['mean_1'], data=True)
            pow_out_edges = graph.sorted_out_edges(m['pow'])
            mean2_out_edges = graph.sorted_out_edges(m['mean_2'])
            add_1_out_edges = graph.sorted_out_edges(m['add_1'])
            sqrt_out_edges = graph.sorted_out_edges(m['sqrt'])
            div_out_edges = graph.sorted_out_edges(m['div'])
            mul_out_edges = graph.sorted_out_edges(m['mul'])

            input_shape = mean_1_in_edges[0][2]['tensor'].value.shape
            weight = node_objs['weight'].value
            bias = node_objs['bias'].value

            if weight.shape != bias.shape \
                    or np.ndim(weight) >= len(input_shape) \
                    or node_objs['mean_1'].axes != node_objs['mean_2'].axes \
                    or FLOAT_EQUAL(node_objs['pow_y'].value, 2.0) is False \
                    or len(mean_1_out_edges) != 1 \
                    or len(pow_out_edges) != 1 \
                    or len(mean2_out_edges) != 1 \
                    or len(add_1_out_edges) != 1 \
                    or len(sqrt_out_edges) != 1 \
                    or len(div_out_edges) != 1 \
                    or len(mul_out_edges) != 1:
                continue
            axes = OpHasAxis.make_axes_non_negative(
                node_objs['mean_1'].axes, len(input_shape))
            axes = sorted(axes)
            weight = OpHasAxis.align_axes(
                weight, axes, [input_shape[axis] for axis in axes])
            bias = OpHasAxis.align_axes(
                bias, axes, [input_shape[axis] for axis in axes])
            if weight is None or bias is None:
                continue
            matched = True
            eps = float(node_objs['eps'].value)
            inp, _, in_attr = mean_1_in_edges[0]
            for src, _, _ in mean_1_in_edges:
                graph.remove_edge(src, m['mean_1'])
            for src, _, _ in sub_in_edges:
                graph.remove_edge(src, m['sub'])
            graph.remove_edges_from(add_2_in_edges)
            graph.add_edge(inp, m['add_2'], **in_attr)
            ln_attr = node_objs['add_2'].copied_attr()
            ln_attr.update({'epsilon': eps, 'weights': weight,
                            'biases': bias, 'axes': node_objs['mean_2'].axes})
            NodeWrap(graph, m['add_2']).replace_obj('LayerNorm', ln_attr)
        else:
            WARN('[Parser]: Meets invalid nodes in merge_ln4!')
    if matched:
        clear_redundant_nodes(graph)


def merge_ln5(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('inp', {}),
                                   ('square_diff', {'op': 'Sub'}),
                                   ('mean', {'op': 'ReduceMean'}),
                                   ('sub_1', {'op': 'Sub'}),
                                   ('exponent_1', {'op': 'Constant'}),
                                   ('square', {'op': 'Pow'}),
                                   ('mean_1', {'op': 'ReduceMean'}),
                                   ('add', {'op': 'Add'}),
                                   ('eps', {'op': 'Constant'}),
                                   ('exponent_2', {'op': 'Constant'}),
                                   ('pow', {'op': 'Pow'}),
                                   ('div', {'op': 'Div'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('gamma', {
                                    'op': 'Constant', 'unique': False}),
                                   ('add_1', {'op': 'Add'}),
                                   ('beta', {
                                    'op': 'Constant', 'unique': False}),
                               ],
                               edges=[
                                   ('inp', 'square_diff'),
                                   ('inp', 'mean'),
                                   ('inp', 'sub_1'),
                                   ('mean', 'square_diff', {'dst_in_port': 1}),
                                   ('mean', 'sub_1', {'dst_in_port': 1}),
                                   ('square_diff', 'square'),
                                   ('exponent_1', 'square',
                                    {'dst_in_port': 1}),
                                   ('square', 'mean_1'),
                                   ('mean_1', 'add'),
                                   ('eps', 'add'),
                                   ('add', 'pow'),
                                   ('exponent_2', 'pow', {'dst_in_port': 1}),
                                   ('sub_1', 'div'),
                                   ('pow', 'div', {'dst_in_port': 1}),
                                   ('div', 'mul'),
                                   ('gamma', 'mul'),
                                   ('mul', 'add_1'),
                                   ('beta', 'add_1'),
                               ])
    for m in matches:
        names = ['inp', 'square_diff', 'mean', 'sub_1', 'exponent_1', 'square', 'mean_1',
                 'add', 'eps', 'exponent_2', 'pow', 'div', 'mul', 'gamma', 'add_1', 'beta']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in obj_dict.values()]):
            WARN('[Parser]: Meets invalid nodes in merge_ln5!')
            continue
        square_diff_in_edges = graph.sorted_in_edges(
            m['square_diff'], data=True)
        mean_in_edges = graph.sorted_in_edges(m['mean'], data=True)
        sub_1_in_edges = graph.sorted_in_edges(m['sub_1'], data=True)
        if len(square_diff_in_edges) != 2 \
                or len(mean_in_edges) != 1 \
                or len(sub_1_in_edges) != 2:
            WARN('[Parser]: Meets invalid nodes in merge_ln5!')
            continue
        inp = m['inp']
        inp_out_attr = mean_in_edges[0][2]
        inp_out_port = inp_out_attr['src_out_port']
        found_invalid_port = False
        for src, _, in_attr in square_diff_in_edges:
            if src == inp and in_attr['src_out_port'] != inp_out_port:
                found_invalid_port = True
                break
        if not found_invalid_port:
            for src, _, in_attr in sub_1_in_edges:
                if src == inp and in_attr['src_out_port'] != inp_out_port:
                    found_invalid_port = True
                    break
        if found_invalid_port:
            continue
        if not FLOAT_EQUAL(obj_dict['exponent_1'].value, 2) \
                or not FLOAT_EQUAL(obj_dict['exponent_2'].value, 0.5):
            continue
        if inp_out_attr['tensor'].value is None:
            continue
        input_shape = inp_out_attr['tensor'].value.shape
        axes1 = sorted(OpHasAxis.make_axes_non_negative(obj_dict['mean'].axes, len(input_shape)))
        axes2 = sorted(OpHasAxis.make_axes_non_negative(obj_dict['mean_1'].axes, len(input_shape)))
        if axes1 != axes2:
            continue
        matched = True
        gamma = obj_dict['gamma'].value
        beta = obj_dict['beta'].value
        weights = OpHasAxis.align_axes(
            gamma, axes1, [input_shape[axis] for axis in axes1])
        biases = OpHasAxis.align_axes(
            beta, axes1, [input_shape[axis] for axis in axes1])
        if weights is None or biases is None:
            last_node = m['div']
            new_op_type = 'MeanVarianceNormalization'
            node_attr = obj_dict['div'].copied_attr()
            node_attr.update({'opset_version': 13})
        else:
            last_node = m['add_1']
            new_op_type = 'LayerNorm'
            node_attr = obj_dict['add_1'].copied_attr()
            node_attr.update({'weights': weights, 'biases': biases})
        last_node_in_edges = graph.sorted_in_edges(last_node)
        graph.remove_edges_from(square_diff_in_edges +
                                mean_in_edges + sub_1_in_edges + last_node_in_edges)
        new_edge_attr = copy.deepcopy(inp_out_attr)
        new_edge_attr.update({'src_out_port': inp_out_port, 'dst_in_port': 0})
        graph.add_edge(inp, last_node, **new_edge_attr)
        eps = float(obj_dict['eps'].value)
        node_attr.update({'epsilon': eps,
                          'axes': axes1})
        NodeWrap(graph, last_node).replace_obj(new_op_type, node_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_ln6(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('reshape1', {'op': 'Reshape'}),
                                   ('con1', {'op': 'Constant'}),
                                   ('con2', {'op': 'Constant'}),
                                   ('bn', {'op': 'BatchNormalization'}),
                                   ('reshape2', {'op': 'Reshape'}),
                               ],
                               edges=[
                                   ('reshape1', 'bn'),
                                   ('con1', 'bn', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                   ('con2', 'bn', {
                                       'src_out_port': 0, 'dst_in_port': 2}),
                                   ('bn', 'reshape2'),

                               ])
    for m in matches:
        names = ['bn', 'con1', 'con2', 'reshape1', 'reshape2']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in obj_dict.values()]):
            WARN('[Parser]: Meets invalid nodes in merge_ln6!')
            continue
        if obj_dict['bn'].training_mode is False:
            continue
        inputs = obj_dict['bn'].get_input_tensors()
        scale = inputs[1]
        offset = inputs[2]
        if np.all(scale == 1) and np.all(offset == 0):
            reshape1_in_edges = graph.sorted_in_edges(m['reshape1'], data=True)
            reshape2_in_edges = graph.sorted_in_edges(m['reshape2'], data=True)

            if len(reshape2_in_edges) >= 1\
                    and len(reshape1_in_edges) >= 1:
                reshape_shape = obj_dict['reshape1'].shape
                reshape_back_shape = obj_dict['reshape2'].shape
                if reshape_shape == [1, np.prod(reshape_back_shape[:-1]), reshape_back_shape[-1], 1]:
                    axis = [-1 + len(reshape_back_shape)]
                elif reshape_shape == [1, reshape_back_shape[0], np.prod(reshape_back_shape[1:]), 1]:
                    axis = list(range(1, len(reshape_back_shape)))
                elif reshape_shape == [1, np.prod(reshape_back_shape[0:2]), np.prod(reshape_back_shape[2:]), 1]:
                    axis = list(range(2, len(reshape_back_shape)))
                elif reshape_shape == [1, 1, np.prod(reshape_back_shape[:]), 1]:
                    axis = list(range(0, len(reshape_back_shape)))
                else:
                    continue

                axis_shape = []
                for x in axis:
                    axis_shape.append(reshape_back_shape[x])

                matched = True
                src, _, in_attr = reshape1_in_edges[0]
                graph.remove_edges_from(reshape1_in_edges)
                graph.remove_edges_from(reshape2_in_edges)
                graph.add_edge(src, m['reshape2'], **in_attr)

                gamma = np.full(
                    (axis_shape), obj_dict['con1'].value[0], np.float32)
                beta = np.full(
                    (axis_shape), obj_dict['con2'].value[0], np.float32)
                eps = float(obj_dict['bn'].epsilon)
                ln_attr = obj_dict['reshape2'].copied_attr()
                ln_attr.update({'epsilon': eps,
                                'weights': gamma,
                                'biases': beta,
                                'axes': np.array(axis)})
                NodeWrap(graph, m['reshape2']).replace_obj(
                    'LayerNorm', ln_attr)
            else:
                WARN('[Parser]: Meets invalid nodes in merge_ln6!')

    if matched:
        clear_redundant_nodes(graph)


def merge_ln_reshape(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('reshape_1', {'op': 'Reshape'}),
                                   ('ln', {'op': 'LayerNorm'}),
                                   ('reshape_2', {'op': 'Reshape'}),
                               ],
                               edges=[
                                   ('reshape_1', 'ln'),
                                   ('ln', 'reshape_2'),
                               ]
                               )
    for m in matches:
        obj_dict = {name: NodeWrap(graph, m[name])['object']
                    for name in ['reshape_1', 'ln', 'reshape_2']}
        if any(obj is None for obj in obj_dict.values()) or \
                len(graph.sorted_in_edges(m['reshape_1'])) < 1:
            WARN('[Parser]: Meets invalid Node in merge_ln_reshape!')
            continue
        reshape1_in_edges = graph.sorted_in_edges(m['reshape_1'], data=True)
        reshape1_in_shapes = obj_dict['reshape_1'].get_input_shapes()
        ln_in_shape = obj_dict['reshape_1'].shape
        reshape2_shape = obj_dict['reshape_2'].shape
        if len(reshape1_in_edges) < 1 \
                or len(reshape1_in_shapes) < 1 \
                or reshape1_in_shapes[0] is None \
                or reshape2_shape != reshape1_in_shapes[0] \
                or len(reshape1_in_shapes[0]) < 3:
            continue
        ln_axes = np.sort(OpHasAxis.make_axes_non_negative(
            obj_dict['ln'].axes, ln_in_shape))
        if list(ln_axes) != list(range(ln_axes[0], ln_axes[0] + len(ln_axes))) \
                or any(ln_in_shape[axis] != 1 for axis in range(ln_axes[-1] + 1, len(ln_in_shape))):
            continue
        norm_shape_size = int(np.prod(ln_in_shape[ln_axes[0]:]))
        new_begin_axis = [axis for axis in range(0, len(reshape2_shape))
                          if int(np.prod(reshape2_shape[axis:])) == norm_shape_size]
        if len(new_begin_axis) == 0:
            continue
        matched = True
        new_begin_axis = new_begin_axis[0]
        new_ln_axes = list(range(new_begin_axis, len(reshape1_in_shapes[0])))
        exp_wb_shape = [reshape1_in_shapes[0][axis] for axis in new_ln_axes]
        weights = np.reshape(obj_dict['ln'].weights, exp_wb_shape)
        biases = np.reshape(obj_dict['ln'].biases, exp_wb_shape)

        src, _, in_attr = reshape1_in_edges[0]
        reshape2_in_edges = graph.sorted_in_edges(m['reshape_2'])
        graph.remove_edges_from(reshape2_in_edges)
        graph.add_edge(src, m['reshape_2'], **in_attr)
        new_ln_attr = obj_dict['reshape_2'].copied_attr()
        new_ln_attr.update({'epsilon': obj_dict['ln'].epsilon,
                            'weights': weights,
                            'biases': biases,
                            'axes': np.array(new_ln_axes)})
        NodeWrap(graph, m['reshape_2']).replace_obj('LayerNorm', new_ln_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_ln_mul_add(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('ln', {'op': 'LayerNorm'}),
                                   ('gamma', {'op': 'Constant'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('beta', {'op': 'Constant'}),
                                   ('add', {'op': 'Add'}),
                               ],
                               edges=[
                                   ('ln', 'mul'),
                                   ('gamma', 'mul'),
                                   ('mul', 'add'),
                                   ('beta', 'add'),
                               ]
                               )
    for m in matches:
        obj_dict = {name: NodeWrap(graph, m[name])['object']
                    for name in ['ln', 'gamma', 'beta', 'add']}
        if any(obj is None for obj in obj_dict.values()) or \
                len(graph.sorted_in_edges(m['ln'])) < 1:
            WARN('[Parser]: Meets invalid Node in merge_ln_mul_add!')
            continue
        ln_in_edges = graph.sorted_in_edges(m['ln'], data=True)
        ln_in_shapes = obj_dict['ln'].get_input_shapes()
        if len(ln_in_edges) < 1 \
                or len(ln_in_shapes) < 1 \
                or ln_in_shapes[0] is None:
            continue
        add_out_shapes = obj_dict['add'].get_output_shapes()
        if len(add_out_shapes) < 1 \
                or add_out_shapes[0] is None \
                or ln_in_shapes[0] != add_out_shapes[0]:
            continue
        if not FLOAT_EQUAL(obj_dict['ln'].weights, 1.0) \
                or not FLOAT_EQUAL(obj_dict['ln'].biases, 0.0):
            continue
        ln_axes = np.sort(OpHasAxis.make_axes_non_negative(
            obj_dict['ln'].axes, ln_in_shapes[0]))
        exp_wb_shape = [ln_in_shapes[0][axis] for axis in ln_axes]
        new_weights = OpHasAxis.align_axes(
            obj_dict['gamma'].value, ln_axes, exp_wb_shape)
        new_biases = OpHasAxis.align_axes(
            obj_dict['beta'].value, ln_axes, exp_wb_shape)
        if new_weights is None or new_biases is None:
            continue
        matched = True
        src, _, in_attr = ln_in_edges[0]
        add_in_edges = graph.sorted_in_edges(m['add'])
        graph.remove_edges_from(add_in_edges)
        graph.add_edge(src, m['add'], **in_attr)
        new_ln_attr = obj_dict['add'].copied_attr()
        new_ln_attr.update({'epsilon': obj_dict['ln'].epsilon,
                            'weights': new_weights,
                            'biases': new_biases,
                            'axes': np.array(ln_axes)})
        NodeWrap(graph, m['add']).replace_obj('LayerNorm', new_ln_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_mvn(graph):
    matched = False
    ln_matches = matched_patterns(graph,
                                  nodes=[
                                      ('inp', {}),
                                      ('mean_1', {'op': 'ReduceMean'}),
                                      ('neg', {'op': 'Neg'}),
                                      ('sub', {'op': 'Sub'}),
                                      ('pow', {'op': 'Pow'}),
                                      ('pow_y', {'op': 'Constant'}),
                                      ('mean_2', {'op': 'ReduceMean'}),
                                      ('add_1', {'op': 'Add'}),
                                      ('epsilon', {'op': 'Constant'}),
                                      ('sqrt', {'op': 'Sqrt'}),
                                      ('recip', {'op': 'Reciprocal'}),
                                      ('mul_1', {'op': 'Mul'}),
                                      ('mul_2', {'op': 'Mul'}),
                                      ('add_2', {'op': 'Add'}),
                                  ],
                                  edges=[
                                      ('inp', 'mean_1'),
                                      ('mean_1', 'neg'),
                                      ('mean_1', 'sub', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                      ('inp', 'sub'),
                                      ('sub', 'pow'),
                                      ('pow_y', 'pow', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                      ('pow', 'mean_2'),
                                      ('mean_2', 'add_1'),
                                      ('epsilon', 'add_1', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                      ('add_1', 'sqrt'),
                                      ('sqrt', 'recip'),
                                      ('recip', 'mul_1', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                      ('recip', 'mul_2', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                      ('neg', 'mul_2'),
                                      ('inp', 'mul_1'),
                                      ('mul_1', 'add_2'),
                                      ('mul_2', 'add_2', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                  ])
    for m in ln_matches:
        objs_dict = {m[name]: NodeWrap(graph, m[name])['object'] for name in [
            'inp', 'mean_1', 'sub', 'pow_y', 'epsilon', 'mul_1', 'add_2']}
        if all([graph.has_node(n) for n in objs_dict.keys()]) and all([v is not None for v in objs_dict.values()]):
            pow_y_value = objs_dict[m['pow_y']].value
            eps_value = objs_dict[m['epsilon']].value
            axes = objs_dict[m['mean_1']].axes
            if pow_y_value is not None and eps_value is not None and FLOAT_EQUAL(pow_y_value, 2) and len(axes) == 1 and axes[0] == 2:
                matched = True
                eps_value = float(eps_value)
                inp, mean_1, sub, mul_1, add_2 = [m[name] for name in [
                    'inp', 'mean_1', 'sub', 'mul_1', 'add_2']]
                mean_1_in_edges = graph.sorted_in_edges(mean_1, data=True)
                add_2_in_edges = graph.sorted_in_edges(add_2)
                _, _, in_attr = mean_1_in_edges[0]
                graph.remove_edge(inp, mean_1)
                graph.remove_edge(inp, sub)
                graph.remove_edge(inp, mul_1)
                graph.remove_edges_from(add_2_in_edges)
                graph.add_edge(inp, add_2, **in_attr)

                inp_obj = objs_dict[inp]
                ln_attr = objs_dict[m['add_2']].copied_attr()
                ln_attr.update({'axes': axes,
                                'epsilon': eps_value,
                                'opset_version': 13
                                })
                NodeWrap(graph, m['add_2']).replace_obj(
                    'MeanVarianceNormalization', ln_attr)
        else:
            WARN('[Parser]: Meets invalid nodes in merge_mvn!')
    if matched:
        clear_redundant_nodes(graph)


def merge_mvn2(graph):
    matched = False
    ln_matches = matched_patterns(graph,
                                  nodes=[
                                      ('inp', {}),
                                      ('mean_1', {'op': 'ReduceMean'}),
                                      ('sub_1', {'op': 'Sub'}),
                                      ('pow', {'op': 'Pow'}),
                                      ('pow_y', {'op': 'Constant'}),
                                      ('mean_2', {'op': 'ReduceMean'}),
                                      ('add_1', {'op': 'Add'}),
                                      ('epsilon', {'op': 'Constant'}),
                                      ('sqrt', {'op': 'Sqrt'}),
                                      ('recip', {'op': 'Reciprocal'}),
                                      ('mul_1', {'op': 'Mul'}),
                                      ('neg', {'op': 'Neg'}),
                                      ('mul_2', {'op': 'Mul'}),
                                      ('add_2', {'op': 'Add'}),
                                  ],
                                  edges=[
                                      ('inp', 'mean_1'),
                                      ('inp', 'mul_1'),
                                      ('inp', 'sub_1'),
                                      ('mean_1', 'sub_1', {'dst_in_port': 1}),
                                      ('sub_1', 'pow'),
                                      ('pow_y', 'pow', {'dst_in_port': 1}),
                                      ('pow', 'mean_2'),
                                      ('mean_2', 'add_1'),
                                      ('epsilon', 'add_1', {'dst_in_port': 1}),
                                      ('add_1', 'sqrt'),
                                      ('sqrt', 'recip'),
                                      ('recip', 'mul_1', {'dst_in_port': 1}),
                                      ('mean_1', 'neg'),
                                      ('neg', 'mul_2'),
                                      ('recip', 'mul_2', {'dst_in_port': 1}),
                                      ('mul_1', 'add_2'),
                                      ('mul_2', 'add_2', {'dst_in_port': 1})
                                  ])
    for m in ln_matches:
        key_names = ['inp', 'mean_1', 'mean_2', 'mul_1',
                     'sub_1', 'pow_y', 'epsilon', 'add_2']
        inp, mean_1, mean_2, mul_1, sub_1, pow_y, epsilon, add_2 = [
            m[name] for name in key_names]
        objs_dict = {m[name]: NodeWrap(graph, m[name])[
            'object'] for name in key_names}
        if all([obj is not None for obj in objs_dict.values()]):
            if objs_dict[epsilon].value.size > 1 and \
                    not all([FLOAT_EQUAL(val, objs_dict[epsilon].value.item(0)) for val in objs_dict[epsilon].value.flatten()[1:]]):
                continue
            input_shapes = objs_dict[inp].get_output_shapes()
            mean_1_in_edges = graph.sorted_in_edges(mean_1, data=True)
            mul_1_in_edges = graph.sorted_in_edges(mul_1, data=True)
            sub_1_in_edges = graph.sorted_in_edges(sub_1, data=True)
            if len(input_shapes) > 0 \
                    and input_shapes[0] \
                    and len(input_shapes[0]) > 1 \
                    and len(mean_1_in_edges) >= 1 \
                    and len(mul_1_in_edges) == 2 \
                    and len(sub_1_in_edges) == 2 \
                    and mean_1_in_edges[0][2]['src_out_port'] == mul_1_in_edges[0][2]['src_out_port'] \
                    and mean_1_in_edges[0][2]['src_out_port'] == sub_1_in_edges[0][2]['src_out_port'] \
                    and FLOAT_EQUAL(objs_dict[pow_y].value, 2.) \
                    and objs_dict[mean_1].axes == objs_dict[mean_2].axes:
                matched = True
                axes = OpHasAxis.make_axes_non_negative(
                    objs_dict[mean_1].axes, len(input_shapes[0]))
                axes = sorted(axes)
                eps = float(objs_dict[epsilon].value.item(0))
                inp_out_attr = copy.deepcopy(mean_1_in_edges[0][2])
                inp_out_attr.update({'dst_in_port': 0})
                add_2_in_edges = graph.sorted_in_edges(add_2)
                graph.remove_edges_from(
                    mean_1_in_edges + mul_1_in_edges + sub_1_in_edges + add_2_in_edges)
                graph.add_edge(inp, add_2, **inp_out_attr)
                ln_attr = objs_dict[add_2].copied_attr()
                ln_attr.update(
                    {'opset_version': 13, 'axes': axes, 'epsilon': eps})
                NodeWrap(graph, add_2).replace_obj(
                    'MeanVarianceNormalization', ln_attr)
        else:
            WARN('[Parser]: Meets invalid nodes in merge_mvn2!')
    if matched:
        clear_redundant_nodes(graph)


def merge_mvn3(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('trans_1', {'op': 'Transpose'}),
                                   ('mean_1', {'op': 'ReduceMean'}),
                                   ('sub_1', {'op': 'Sub'}),
                                   ('pow', {'op': 'Pow'}),
                                   ('mean_2', {'op': 'ReduceMean'}),
                                   ('add_1', {'op': 'Add'}),
                                   ('sqrt', {'op': 'Sqrt'}),
                                   ('recip', {'op': 'Reciprocal'}),
                                   ('mul_1', {'op': 'Mul'}),
                                   ('mul_2', {'op': 'Mul'}),
                                   ('sub_2', {'op': 'Sub'}),
                                   ('add_2', {'op': 'Add'}),
                                   ('trans_2', {'op': 'Transpose'})
                               ],
                               edges=[
                                   ('trans_1', 'mean_1'),
                                   ('trans_1', 'sub_1'),
                                   ('trans_1', 'mul_1'),
                                   ('mean_1', 'sub_1', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('sub_1', 'pow'),
                                   ('pow', 'mean_2'),
                                   ('mean_1', 'mul_2'),
                                   ('mean_2', 'add_1'),
                                   ('add_1', 'sqrt'),
                                   ('sqrt', 'recip'),
                                   ('recip', 'mul_1', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('recip', 'mul_2', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('mul_2', 'sub_2', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('mul_1', 'add_2'),
                                   ('sub_2', 'add_2', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('add_2', 'trans_2'),
                               ]
                               )
    for m in matches:
        names_check_out_edge_1 = ['sub_1', 'mean_2', 'pow',
                                  'add_1', 'sqrt', 'mul_1', 'mul_2', 'sub_2', 'add_2']
        names_check_out_edge_2 = ['mean_1', 'recip']
        names_check_out_edge_3 = ['trans_1']
        names_check_valid = names_check_out_edge_1 + \
            names_check_out_edge_2 + names_check_out_edge_3 + ['trans_2']
        obj_dict = {k: NodeWrap(graph, m[k])['object']
                    for k in names_check_valid}
        if any([obj is None for obj in obj_dict.values()]):
            WARN('[Parser]: Meets invalid Node in merge_mvn3!')
            continue
        if any([len(graph.sorted_out_edges(m[name])) != 1 for name in names_check_out_edge_1]):
            continue
        if any([len(graph.sorted_out_edges(m[name])) != 2 for name in names_check_out_edge_2]):
            continue
        if any([len(graph.sorted_out_edges(m[name])) != 3 for name in names_check_out_edge_3]):
            continue
        if len(graph.sorted_in_edges(m['trans_1'])) < 1:
            WARN(
                '[Parser]: Got invalid input edges of Transpose(%s) in merge_mvn3!' % m['trans_1'])
            continue
        if obj_dict['trans_1'].perm != [0, 2, 3, 1] \
                or obj_dict['trans_2'].perm != [0, 3, 1, 2]:
            continue
        if obj_dict['mean_1'].axes != [0, 1, 2] or obj_dict['mean_2'].axes != [0, 1, 2]:
            continue
        if len(obj_dict['mean_1'].get_input_shapes()) != 1 \
                or obj_dict['mean_1'].get_input_shapes()[0] is None \
                or len(obj_dict['mean_1'].get_input_shapes()[0]) != 4:
            WARN(
                '[Parser]: Got invalid input shape of Mean(%s) in merge_mvn3!' % m['mean_1'])
            continue
        if len(obj_dict['add_1'].sorted_in_consts()) != 1 \
                or obj_dict['add_1'].sorted_in_consts()[0][2] is None \
                or obj_dict['add_1'].sorted_in_consts()[0][2].size != 1:
            continue
        if len(obj_dict['sub_2'].sorted_in_consts()) != 1 \
                or obj_dict['sub_2'].sorted_in_consts()[0][2] is None \
                or not FLOAT_EQUAL(obj_dict['sub_2'].sorted_in_consts()[0][2], 0)\
                or not FLOAT_EQUAL(obj_dict['pow'].get_input_tensors()[1], 2):
            continue
        matched = True
        trans1_in_edges = graph.sorted_in_edges(m['trans_1'], data=True)
        trans2_in_edges = graph.sorted_in_edges(m['trans_2'])
        trans2_out_edges = graph.sorted_out_edges(m['trans_2'], data=True)
        src, _, in_attr = trans1_in_edges[0]
        graph.remove_edges_from(trans1_in_edges + trans2_in_edges)
        graph.add_edge(src, m['trans_2'], **in_attr)
        mean1_in_shape = obj_dict['mean_1'].get_input_shapes()[0]
        axes = [x for (i, x) in enumerate(
            obj_dict['mean_1'].axes) if mean1_in_shape[i] != 1]
        if not axes:
            axes = ['mean_1'].axes[:]
        axes = sorted([obj_dict['trans_1'].perm[x] for x in axes])
        eps = float(obj_dict['add_1'].sorted_in_consts()[0][2])
        mvn_attr = obj_dict['trans_2'].copied_attr()
        mvn_attr.update({'opset_version': 13, 'epsilon': eps, 'axes': axes})
        NodeWrap(graph, m['trans_2']).replace_obj(
            'MeanVarianceNormalization', mvn_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_gn(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('reshape_1', {'op': 'Reshape'}),
                                   ('norm', {
                                    'op': ['LayerNorm', 'MeanVarianceNormalization']}),
                                   ('reshape_2', {'op': 'Reshape'}),
                               ],
                               edges=[
                                   ('reshape_1', 'norm'),
                                   ('norm', 'reshape_2'),
                               ]
                               )
    for m in matches:
        obj_dict = {name: NodeWrap(graph, m[name])['object']
                    for name in ['reshape_1', 'norm', 'reshape_2']}
        if any([obj is None for obj in obj_dict.values()]) or \
                len(graph.sorted_in_edges(m['reshape_1'])) < 1:
            WARN('[Parser]: Meets invalid Node in merge_gn!')
            continue
        expand_shape = obj_dict['reshape_1'].shape
        origin_shape = obj_dict['reshape_2'].shape
        axes = obj_dict['norm'].axes
        if len(graph.sorted_out_edges(m['norm'])) != 1 or \
                len(expand_shape) != len(origin_shape) + 1 or \
                len(expand_shape) != len(axes) + 2:
            continue
        channels_axis = [index for index, s in enumerate(
            origin_shape) if origin_shape[index] != expand_shape[index]]
        channels_axis = channels_axis[0] if len(
            channels_axis) >= 1 else (len(origin_shape) - 1)
        if channels_axis == 0 or (0, channels_axis) in axes or \
                not np.array_equal(origin_shape[:channels_axis] + origin_shape[channels_axis + 1:], expand_shape[:channels_axis] + expand_shape[channels_axis + 2:]):
            continue

        weights_shape = [origin_shape[channels_axis]]
        if obj_dict['norm'].type == 'MeanVarianceNormalization':
            weights = np.ones(weights_shape, dtype=np.float32)
            biases = np.zeros(weights_shape, dtype=np.float32)
        else:
            weights = obj_dict['norm'].weights
            biases = obj_dict['norm'].biases
            perm = [i for i in range(
                len(origin_shape)) if i != channels_axis] + [channels_axis]
            if len(weights.shape) == len(expand_shape):
                new_weights_shape = list(
                    weights.shape[:channels_axis]) + [-1] + list(weights.shape[channels_axis + 2:])
                weights_c_last = np.transpose(
                    np.reshape(weights, new_weights_shape), perm)
                weights_2d = np.reshape(
                    weights_c_last, [-1, weights_c_last.shape[-1]])
                if weights_2d.shape[1] not in (1, weights_shape[0]) or \
                        not all([FLOAT_EQUAL(weights_2d[0], weights_2d[i]) for i in range(1, weights_2d.shape[0])]):
                    continue
                weights = np.array(weights_2d[0])
            if weights.size == 1:
                weights = np.tile(weights, weights_shape)
            elif weights.size == weights_shape[0]:
                weights = np.reshape(weights, weights_shape)
            else:
                WARN(
                    '[Parser]: Meets invalid weights of Node(%s) in merge_gn!' % m['norm'])
                continue
            if len(biases.shape) == len(expand_shape):
                new_biases_shape = list(
                    biases.shape[:channels_axis]) + [-1] + list(biases.shape[channels_axis + 2:])
                biases_c_last = np.transpose(
                    np.reshape(biases, new_biases_shape), perm)
                biases_2d = np.reshape(
                    biases_c_last, [-1, biases_c_last.shape[-1]])
                if biases_2d.shape[1] not in (1, weights_shape[0]) or \
                        not all([FLOAT_EQUAL(biases_2d[0], biases_2d[i]) for i in range(1, biases_2d.shape[0])]):
                    continue
                biases = np.array(biases_2d[0])
            if biases.size == 1:
                biases = np.tile(biases, weights_shape)
            elif biases.size == weights_shape[0]:
                biases = np.reshape(biases, weights_shape)
            else:
                WARN(
                    '[Parser]: Meets invalid biases of Node(%s) in merge_gn!' % m['norm'])
                continue
        matched = True
        group = expand_shape[channels_axis]
        src, _, in_attr = graph.sorted_in_edges(m['reshape_1'], data=True)[0]
        graph.remove_edge(m['reshape_1'], m['norm'])
        graph.add_edge(src, m['norm'], **in_attr)
        for _, dst, out_attr in graph.sorted_out_edges(m['reshape_2'], data=True):
            graph.remove_edge(m['reshape_2'], dst)
            graph.add_edge(m['norm'], dst, **out_attr)
        graph.remove_edge(m['norm'], m['reshape_2'])

        gn_attr = obj_dict['norm'].copied_attr()
        gn_attr.update({'group': group,
                        'axes': None,
                        'axis': channels_axis,
                        'weights': weights,
                        'biases': biases})
        NodeWrap(graph, m['norm']).replace_obj('ArmGroupNorm', gn_attr)
        if m['reshape_2'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['reshape_2'])
            graph._attr['output_names'][index] = m['norm']
    if matched:
        clear_redundant_nodes(graph)


def merge_gn2(graph):
    matched = False
    gn_matches = matched_patterns(graph,
                                  nodes=[
                                      ('reshape_1', {'op': 'Reshape'}),
                                      ('mean_1', {'op': 'ReduceMean'}),
                                      ('sub_1', {'op': 'Sub'}),
                                      ('pow', {'op': 'Pow'}),
                                      ('pow_y', {'op': 'Constant'}),
                                      ('mean_2', {'op': 'ReduceMean'}),
                                      ('add_1', {'op': 'Add'}),
                                      ('epsilon', {'op': 'Constant'}),
                                      ('sqrt', {'op': 'Sqrt'}),
                                      ('recip', {'op': 'Reciprocal'}),
                                      ('gamma', {'op': 'Constant'}),
                                      ('mul_gamma', {'op': 'Mul'}),
                                      ('mul_1', {'op': 'Mul'}),
                                      ('mul_2', {'op': 'Mul'}),
                                      ('sub_2', {'op': 'Sub'}),
                                      ('beta', {'op': 'Constant'}),
                                      ('add_2', {'op': 'Add'}),
                                      ('reshape_2', {'op': 'Reshape'}),
                                  ],
                                  edges=[
                                      ('reshape_1', 'mean_1'),
                                      ('reshape_1', 'mul_1'),
                                      ('reshape_1', 'sub_1'),
                                      ('mean_1', 'sub_1', {'dst_in_port': 1}),
                                      ('sub_1', 'pow'),
                                      ('pow_y', 'pow', {'dst_in_port': 1}),
                                      ('pow', 'mean_2'),
                                      ('mean_2', 'add_1'),
                                      ('epsilon', 'add_1'),
                                      ('add_1', 'sqrt'),
                                      ('sqrt', 'recip'),
                                      ('recip', 'mul_gamma'),
                                      ('gamma', 'mul_gamma'),
                                      ('mul_gamma', 'mul_1'),
                                      ('mean_1', 'mul_2'),
                                      ('mul_gamma', 'mul_2'),
                                      ('beta', 'sub_2'),
                                      ('mul_2', 'sub_2', {'dst_in_port': 1}),
                                      ('mul_1', 'add_2'),
                                      ('sub_2', 'add_2'),
                                      ('add_2', 'reshape_2'),
                                  ])
    for m in gn_matches:
        key_names = ['reshape_1', 'mean_1', 'mean_2',
                     'pow_y', 'epsilon', 'gamma', 'beta', 'reshape_2']
        reshape_1, mean_1, mean_2, pow_y, epsilon, gamma, beta, reshape_2 = [
            m[name] for name in key_names]
        objs_dict = {m[name]: NodeWrap(graph, m[name])[
            'object'] for name in key_names}
        if any(obj is None for obj in objs_dict.values()):
            WARN('[Parser]: Meets invalid nodes in merge_gn2!')
            continue
        if objs_dict[epsilon].value.size > 1 and \
                not FLOAT_EQUAL(objs_dict[epsilon].value, objs_dict[epsilon].value.item(0)):
            continue
        input_shapes = objs_dict[reshape_1].get_input_shapes()
        expanded_shape = objs_dict[reshape_1].shape
        output_shapes = objs_dict[reshape_2].get_output_shapes()
        reshape_1_in_edges = graph.sorted_in_edges(reshape_1, data=True)
        if len(input_shapes) < 1 \
                or input_shapes[0] is None \
                or expanded_shape is None \
                or len(output_shapes) < 1 \
                or output_shapes[0] is None \
                or input_shapes[0] != output_shapes[0] \
                or len(reshape_1_in_edges) < 1:
            continue
        if not FLOAT_EQUAL(objs_dict[pow_y].value, 2.) \
                or objs_dict[mean_1].axes != objs_dict[mean_2].axes:
            continue
        axes = sorted(OpHasAxis.make_axes_non_negative(
            objs_dict[mean_1].axes, len(expanded_shape)))
        non_axes = [num for num in range(
            len(expanded_shape)) if num not in axes]
        if len(non_axes) != 2 \
                or non_axes[0] != 0 \
                or non_axes[1] >= len(expanded_shape) - 1:
            continue
        channels_axis = non_axes[1]
        axes_after_reshape = [channels_axis, channels_axis + 1]
        exp_shape = [expanded_shape[axis] for axis in axes_after_reshape]
        biases = OpHasAxis.align_axes(
            objs_dict[beta].value, axes_after_reshape, exp_shape)
        weights = OpHasAxis.align_axes(
            objs_dict[gamma].value, axes_after_reshape, exp_shape)
        if biases is None or weights is None:
            continue
        matched = True
        eps = float(objs_dict[epsilon].value.item(0))
        biases = np.reshape(biases, [-1])
        weights = np.reshape(weights, [-1])
        inp, _, inp_out_attr = reshape_1_in_edges[0]
        reshape_2_in_edges = graph.sorted_in_edges(reshape_2)
        graph.remove_edges_from(reshape_2_in_edges)
        graph.add_edge(inp, reshape_2, **inp_out_attr)
        gn_attr = objs_dict[reshape_2].copied_attr()
        gn_attr.update(
            {'epsilon': eps, 'weights': weights, 'biases': biases,
             'group': expanded_shape[channels_axis], 'axes': None,
             'axis': channels_axis})
        NodeWrap(graph, reshape_2).replace_obj('ArmGroupNorm', gn_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_in(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('inp', {}),
                                   ('mean_1', {'op': 'ReduceMean'}),
                                   ('sub', {'op': 'Sub'}),
                                   ('pow', {'op': 'Pow'}),
                                   ('mean_2', {'op': 'ReduceMean'}),
                                   ('add_1', {'op': 'Add'}),
                                   ('sqrt', {'op': 'Sqrt'}),
                                   ('div', {'op': 'Div'}),
                               ],
                               edges=[
                                   ('inp', 'mean_1'),
                                   ('inp', 'sub'),
                                   ('mean_1', 'sub', {'dst_in_port': 1}),
                                   ('sub', 'pow'),
                                   ('pow', 'mean_2'),
                                   ('mean_2', 'add_1'),
                                   ('add_1', 'sqrt'),
                                   ('sqrt', 'div'),
                               ]
                               )
    for m in matches:
        names_check_out_edge_1 = ['mean_1', 'pow', 'mean_2', 'add_1', 'sqrt', ]
        names_check_out_edge_2 = ['sub']
        names_check_valid = names_check_out_edge_1 + \
            names_check_out_edge_2 + ['inp', 'div']
        obj_dict = {name: NodeWrap(graph, m[name])['object']
                    for name in names_check_valid}
        if any([obj is None for obj in obj_dict.values()]):
            WARN('[Parser]: Meets invalid Node in merge_in!')
            continue
        if any([len(graph.sorted_out_edges(m[name])) != 1 for name in names_check_out_edge_1]):
            continue
        if any([len(graph.sorted_out_edges(m[name])) != 2 for name in names_check_out_edge_2]):
            continue
        if len(obj_dict['pow'].sorted_in_consts()) != 1 or len(obj_dict['add_1'].sorted_in_consts()) != 1:
            continue
        if not FLOAT_EQUAL(obj_dict['pow'].sorted_in_consts()[0][2], 2):
            continue
        if obj_dict['mean_1'].axes != obj_dict['mean_2'].axes:
            continue
        input_shapes = obj_dict['inp'].get_output_shapes()
        output_shapes = obj_dict['div'].get_output_shapes()
        if len(input_shapes) < 1 or len(output_shapes) < 1:
            WARN('[Parser]: Meets invalid input/output shape in merge_in!')
            continue
        if input_shapes[0] is None \
                or output_shapes[0] is None \
                or len(input_shapes[0]) < 3 \
                or input_shapes[0] != output_shapes[0]:
            continue
        in_shape = input_shapes[0]
        data_format = obj_dict['mean_1'].data_format
        if (data_format == 'NCHW' and list(range(len(in_shape)))[2:] != obj_dict['mean_1'].axes) \
                or (data_format == 'NHWC' and list(range(len(in_shape)))[1:-1] != obj_dict['mean_1'].axes):
            continue
        matched = True
        eps = float(obj_dict['add_1'].sorted_in_consts()[0][2])
        channel_size = in_shape[1 if data_format == 'NCHW' else -1]
        gamma = np.ones((channel_size,), np.float32)
        beta = np.zeros((channel_size,), np.float32)
        mean_1_in_edges = graph.sorted_in_edges(m['mean_1'], data=True)
        sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
        div_in_edges = graph.sorted_in_edges(m['div'], data=True)
        _, _, in_attr = mean_1_in_edges[0]
        graph.remove_edges_from(mean_1_in_edges + sub_in_edges + div_in_edges)
        graph.add_edge(m['inp'], m['div'], **in_attr)
        instace_norm_attr = obj_dict['div'].copied_attr()
        instace_norm_attr.update({'opset_version': 6,
                                  'data_format': data_format,
                                  'non_channel_axes': obj_dict['mean_1'].axes,
                                  'epsilon': eps,
                                  'weights': gamma,
                                  'biases': beta})
        NodeWrap(graph, m['div']).replace_obj(
            'InstanceNormalization', instace_norm_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_norm(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('norm', {
                                    'op': ['InstanceNormalization', 'MeanVarianceNormalization', 'LayerNorm']}),
                                   ('reshape', {'op': 'Reshape'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('add', {'op': 'Add'}),
                                   ('reshape_dim', {'op': 'Constant'}),
                                   ('weight', {'op': 'Constant'}),
                                   ('bias', {'op': 'Constant'}),
                               ],
                               edges=[
                                   ('norm', 'reshape'),
                                   ('reshape_dim', 'reshape', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('reshape', 'mul'),
                                   ('weight', 'mul', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('mul', 'add'),
                                   ('bias', 'add', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                               ]
                               )
    for m in matches:
        key_names = ['norm', 'reshape', 'mul', 'add',
                     'reshape_dim', 'weight', 'bias']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if all([obj is not None for obj in node_objs.values()]):
            norm_in_edges = graph.sorted_in_edges(m['norm'])
            reshape_out_edges = graph.sorted_out_edges(m['reshape'])
            mul_in_edges = graph.sorted_in_edges(m['mul'])
            add_out_edges = graph.sorted_out_edges(m['add'], data=True)
            mul_out_edges = graph.sorted_out_edges(m['mul'])
            norm_out_edges = graph.sorted_out_edges(m['norm'])
            if len(norm_in_edges) >= 1:
                input_shape = node_objs['norm'].get_input_shapes()
                weight = node_objs['weight'].value
                bias = node_objs['bias'].value

                if weight is None or bias is None \
                        or weight.shape != bias.shape \
                        or len(reshape_out_edges) != 1 \
                        or len(norm_out_edges) != 1 \
                        or len(mul_out_edges) != 1 \
                        or len(input_shape) != 1 \
                        or input_shape[0] is None:
                    continue

                ins_attr = node_objs['norm'].copied_attr()
                if node_objs['norm'].type == 'InstanceNormalization':
                    if input_shape[0][-1] != weight.shape[0] \
                            or np.ndim(weight) != 1 \
                            or np.ndim(bias) != 1 \
                            or np.all(node_objs['norm'].weights == 1) is False \
                            or np.all(node_objs['norm'].biases == 0) is False:
                        continue
                    new_axis = node_objs['norm'].non_channel_axes
                    eps = node_objs['norm'].epsilon
                    ins_attr.update({'epsilon': eps, 'axes': new_axis})
                else:
                    axes = OpHasAxis.make_axes_non_negative(
                        node_objs['norm'].axes, len(input_shape[0]))
                    exp_shape = [input_shape[0][axis] for axis in axes]
                    if np.prod(weight.shape) != np.prod(exp_shape) \
                            or np.prod(bias.shape) != np.prod(exp_shape):
                        continue
                    weight = np.reshape(weight, exp_shape)
                    bias = np.reshape(bias, exp_shape)
                    if node_objs['norm'].type == 'LayerNorm':
                        if node_objs['norm'].weights.shape != weight.shape \
                                or node_objs['norm'].biases.shape != bias.shape:
                            continue
                        bias = bias + node_objs['norm'].biases * weight
                        weight = weight * node_objs['norm'].weights
                matched = True

                graph.remove_edges_from(mul_in_edges)
                graph.remove_edges_from(add_out_edges)
                for _, dst, out_attr in add_out_edges:
                    graph.add_edge(m['reshape'], dst, **out_attr)

                ins_attr.update({'weights': weight, 'biases': bias})

                NodeWrap(graph, m['norm']).replace_obj(
                    'LayerNorm', ins_attr)

                if m['add'] in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(m['add'])
                    graph._attr['output_names'][index] = m['reshape']
        else:
            WARN('[Parser]: Meets invalid nodes in merge_norm!')
    if matched:
        clear_redundant_nodes(graph)


def broadcast_ln_weights_biases(graph):
    matches = single_node_matcher(graph, 'LayerNorm')
    for m in matches:
        ln = m['target']
        ln_obj = NodeWrap(graph, ln)['object']
        if ln_obj is not None and ln_obj.weights is not None and ln_obj.biases is not None:
            input_shapes = ln_obj.get_output_shapes()
            if len(input_shapes) >= 1 \
                    and input_shapes[0] \
                    and len(input_shapes[0]) >= 2:
                input_rank = len(input_shapes[0])
                if list(ln_obj.weights.shape) != list(ln_obj.biases.shape):
                    axes = OpHasAxis.make_axes_non_negative(
                        ln_obj.axes, len(input_shapes[0]))
                    axes = sorted(axes)
                    if list(ln_obj.weights.shape) != [input_shapes[0][d] for d in axes]:
                        w_rank = len(ln_obj.weights.shape)
                        reshape_dim = [
                            1] * (len(axes) - w_rank) + list(ln_obj.weights.shape)
                        ln_obj.weights = np.reshape(
                            ln_obj.weights, reshape_dim)
                    if list(ln_obj.biases.shape) != [input_shapes[0][d] for d in axes]:
                        b_rank = len(ln_obj.biases.shape)
                        reshape_dim = [
                            1] * (len(axes) - b_rank) + list(ln_obj.biases.shape)
                        ln_obj.biases = np.reshape(ln_obj.biases, reshape_dim)
                    if list(ln_obj.weights.shape) != list(ln_obj.biases.shape):
                        max_reps = np.maximum(
                            np.array(ln_obj.weights.shape), np.array(ln_obj.biases.shape))
                        weights_reps = max_reps // np.array(
                            ln_obj.weights.shape)
                        if np.any(weights_reps > 1):
                            ln_obj.weights = np.tile(
                                ln_obj.weights, weights_reps.tolist())
                        biases_reps = max_reps // np.array(ln_obj.biases.shape)
                        if np.any(biases_reps > 1):
                            ln_obj.biases = np.tile(
                                ln_obj.biases, biases_reps.tolist())


def rearrange_fc_reshape_bn(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('fc', {'op': 'FullyConnected'}),
                                   ('reshape', {'op': 'Reshape'}),
                                   ('bn', {'op': 'BatchNormalization'})
                               ],
                               edges=[
                                   ('fc', 'reshape', {'dst_in_port': 0}),
                                   ('reshape', 'bn', {'dst_in_port': 0})
                               ])
    for m in matches:
        fc, reshape, bn = m['fc'], m['reshape'], m['bn']
        fc_obj = NodeWrap(graph, fc)['object']
        reshape_obj = NodeWrap(graph, reshape)['object']
        bn_obj = NodeWrap(graph, bn)['object']
        if fc_obj is None:
            WARN(
                '[Parser]: Meets invalid FullyConnected Op (%s) in rearrange_fc_reshape_bn!' % fc)
            continue
        if reshape_obj is None:
            WARN(
                '[Parser]: Meets invalid Reshape Op (%s) in rearrange_fc_reshape_bn!' % reshape)
            continue
        if bn_obj is None:
            WARN(
                '[Parser]: Meets invalid BatchNormalization Op (%s) in rearrange_fc_reshape_bn!' % bn)
            continue
        if bn_obj.training_mode:
            continue
        fc_out_edges = graph.sorted_out_edges(fc, data=True)
        reshape_out_edges = graph.sorted_out_edges(reshape)
        if len(fc_out_edges) != 1 or len(reshape_out_edges) != 1:
            continue
        if reshape_obj.shape is None \
                or len(reshape_obj.shape) != 3 \
                or len(bn_obj.sorted_in_consts()) < 1 \
                or bn_obj.sorted_in_consts()[0][2] is None \
                or bn_obj.sorted_in_consts()[0][2].size != fc_obj.num_output:
            continue
        if bn_obj.data_format == 'NHWC' and reshape_obj.shape[-1] != fc_obj.num_output:
            continue
        if bn_obj.data_format == 'NCHW' and reshape_obj.shape[1] != fc_obj.num_output:
            continue
        _, _, fc_out_attr = fc_out_edges[0]
        graph.remove_edges_from(fc_out_edges + reshape_out_edges)
        graph.add_edge(fc, bn, **fc_out_attr)
        for _, dst, out_attr in graph.sorted_out_edges(bn, data=True):
            graph.remove_edge(bn, dst)
            graph.add_edge(reshape, dst, **out_attr)
        graph.add_edge(bn, reshape)
        if bn in graph._attr['output_names']:
            index = graph._attr['output_names'].index(bn)
            graph._attr['output_names'][index] = reshape


def rearrange_matmul_reshape_bias(graph):
    matmul_reshape_bias_matches = matched_patterns(graph,
                                                   nodes=[
                                                       # 'MatMul'
                                                       ('matmul', {
                                                           'op': 'FullyConnected'}),
                                                       ('reshape', {
                                                           'op': 'Reshape'}),
                                                       ('bias_add', {
                                                           'op': ['Add', 'BatchNormalization']})
                                                   ],
                                                   edges=[
                                                       ('matmul', 'reshape'),
                                                       ('reshape', 'bias_add')
                                                   ])
    for m in matmul_reshape_bias_matches:
        matmul, reshape, bias = m['matmul'], m['reshape'], m['bias_add']
        matmul_obj, reshape_obj, bias_obj = [
            NodeWrap(graph, name)['object'] for name in [matmul, reshape, bias]]
        if matmul_obj is not None and reshape_obj is not None and bias_obj is not None:
            bias_obj_in_consts = bias_obj.sorted_in_consts()
            if len(bias_obj_in_consts) >= 1 \
                    and len(graph.sorted_out_edges(reshape)) == 1 \
                    and matmul_obj.num_output == bias_obj_in_consts[0][2].size:
                bias_out_edges = graph.sorted_out_edges(bias, data=True)
                matmul_out_edges = graph.sorted_out_edges(matmul, data=True)
                reshape_out_edges = graph.sorted_out_edges(reshape, data=True)
                graph.remove_edge(matmul, reshape)
                graph.remove_edge(reshape, bias)
                for _, dst, attr in bias_out_edges:
                    graph.remove_edge(bias, dst)
                    graph.add_edge(reshape, dst, **attr)
                graph.add_edge(matmul, bias, **matmul_out_edges[0][2])
                graph.add_edge(bias, reshape, **reshape_out_edges[0][2])


def rearrange_linear_concat_relu(graph):
    linear_op_list = list(set(BaseLinearOp.get_concrete_subclass_names()).intersection(
        OnnxOp.get_concrete_subclass_names())) + ['FullyConnected']
    concat_relu_matches = [matched_patterns(graph,
                                            nodes=[
                                                ('concat', {
                                                    'op': 'Concat'}),
                                                ('relu', {
                                                    'op': relu_op})
                                            ],
                                            edges=[
                                                ('concat', 'relu')
                                            ]) for relu_op in ['Relu', 'LeakyRelu']]
    concat_relu_matches = extend_lists(concat_relu_matches)
    for m in concat_relu_matches:
        concat, relu = m['concat'], m['relu']
        concat_in_edges = graph.sorted_in_edges(concat, data=True)
        concat_out_edges = graph.sorted_out_edges(concat)
        relu_out_edges = graph.sorted_out_edges(relu, data=True)
        if len(concat_in_edges) > 1 \
                and len(concat_out_edges) == 1:
            in_objects = [NodeWrap(graph, edge[0])['object']
                          for edge in concat_in_edges]
            if all([obj.type in linear_op_list for obj in in_objects]):
                relu_obj = NodeWrap(graph, relu)['object']
                for i, (src, _, in_attr) in enumerate(concat_in_edges):
                    meta_relu = get_valid_node_name(
                        graph, relu + '_before_concat_' + str(i + 1))
                    graph.remove_edge(src, concat)
                    meta_relu_in_attr = copy.deepcopy(in_attr)
                    meta_relu_in_attr.update({'dst_in_port': 0})
                    graph.add_edge(src, meta_relu, **meta_relu_in_attr)
                    meta_relu_out_attr = copy.deepcopy(in_attr)
                    meta_relu_out_attr.update({'src_out_port': 0})
                    graph.add_edge(meta_relu, concat, **meta_relu_out_attr)
                    meta_relu_attr = relu_obj.copied_attr()
                    meta_relu_attr.update({'name': meta_relu})
                    NodeWrap(graph, meta_relu).replace_obj(
                        relu_obj.type, meta_relu_attr)

                for _, dst, out_attr in relu_out_edges:
                    graph.remove_edge(relu, dst)
                    graph.add_edge(concat, dst, **out_attr)
                graph.remove_edge(concat, relu)

                if relu in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(relu)
                    graph._attr['output_names'][index] = concat
                graph.remove_node(relu)


def rearrange_linear_reshape_relu(graph):
    linear_ops = ['FullyConnected', 'Conv', 'ConvTranspose']
    relu_ops = ['Relu', 'LeakyRelu']
    linear_relu_combinations = itertools.product(linear_ops, relu_ops)
    for linear_type, relu_type in linear_relu_combinations:
        matches = matched_patterns(graph,
                                   nodes=[('linear', {'op': linear_type}),
                                          ('reshape', {'op': 'Reshape'}),
                                          ('relu', {'op': relu_type})
                                          ],
                                   edges=[('linear', 'reshape'),
                                          ('reshape', 'relu')
                                          ]
                                   )
        for m in matches:
            linear, reshape, relu = m['linear'], m['reshape'], m['relu']
            reshape_out_edges = graph.sorted_out_edges(reshape)
            relu_in_edges = graph.sorted_in_edges(relu, data=True)
            relu_out_edges = graph.sorted_out_edges(relu, data=True)
            if len(reshape_out_edges) == 1:
                for _, dst, attr in relu_out_edges:
                    graph.remove_edge(relu, dst)
                    graph.add_edge(reshape, dst, **attr)
                graph.add_edge(linear, relu, **relu_in_edges[0][2])
                graph.add_edge(relu, reshape)
                graph.remove_edge(linear, reshape)
                graph.remove_edge(reshape, relu)
                if relu in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(relu)
                    graph._attr['output_names'][index] = reshape


def rearrange_pack_concat(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('input_1', {}),
                                   ('pack_1', {'op': 'Unsqueeze'}),
                                   ('input_2', {}),
                                   ('pack_2', {'op': 'Unsqueeze'}),
                                   ('concat', {'op': 'Concat'}),
                                   ('transpose', {'op': 'Transpose'})
                               ],
                               edges=[
                                   ('input_1', 'pack_1'),
                                   ('input_2', 'pack_2'),
                                   ('pack_1', 'concat'),
                                   ('pack_2', 'concat', {
                                    'src_out_port': 0, 'dst_in_port': 1}),
                                   ('concat', 'transpose')
                               ]
                               )
    for m in matches:
        input_1, input_2, pack_1, pack_2, concat, transpose \
            = m['input_1'], m['input_2'], m['pack_1'], m['pack_2'], m['concat'], m['transpose']
        pack_1_obj = NodeWrap(graph, pack_1)['object']
        pack_2_obj = NodeWrap(graph, pack_2)['object']
        concat_obj = NodeWrap(graph, concat)['object']
        transpose_obj = NodeWrap(graph, transpose)['object']
        pack_1_out_shapes = pack_1_obj.get_output_shapes()
        pack_2_out_shapes = pack_2_obj.get_output_shapes()
        concat_out_shape = concat_obj.get_output_shapes()[0]
        transpose_out_shape = transpose_obj.get_output_shapes()[0]
        if len(pack_1_out_shapes) == 1 \
                and len(pack_2_out_shapes) == 1 \
                and pack_1_out_shapes[0] \
                and len(pack_1_out_shapes[0]) == 5 \
                and pack_1_out_shapes[0] == pack_2_out_shapes[0] \
                and pack_1_obj.axes == pack_2_obj.axes \
                and pack_1_obj.axes[0] == concat_obj.axis \
                and len(concat_out_shape) == 5 \
                and transpose_obj.perm == [0, 1, 2, 4, 3]:
            matched = True
            opset_version = graph._attr['opset_version']
            reshape_ver = ReshapeOp.cal_ver(opset_version)
            reshape_1 = get_valid_node_name(graph, concat + '_reshape')
            reshape_2 = get_valid_node_name(graph, transpose + '_reshape')
            graph.add_nodes_from([reshape_1, reshape_2])

            input_1_out_edges = graph.sorted_out_edges(input_1, data=True)
            for _, out1, out_edge1_attr in input_1_out_edges:
                if out1 == pack_1:
                    new_out_edge1_attr = copy.deepcopy(out_edge1_attr)
                    new_out_edge1_attr.update({'dst_in_port': 0})
                    graph.remove_edge(input_1, pack_1)
                    graph.remove_edge(pack_1, concat)
                    graph.add_edge(input_1, concat, **new_out_edge1_attr)

            input_2_out_edges = graph.sorted_out_edges(input_2, data=True)
            for _, out2, out_edge2_attr in input_2_out_edges:
                if out2 == pack_2:
                    new_out_edge2_attr = copy.deepcopy(out_edge2_attr)
                    new_out_edge2_attr.update({'dst_in_port': 1})
                    graph.remove_edge(input_2, pack_2)
                    graph.remove_edge(pack_2, concat)
                    graph.add_edge(input_2, concat, **new_out_edge2_attr)

            graph.remove_edge(concat, transpose)
            graph.add_edge(concat, reshape_1)
            graph.add_edge(reshape_1, transpose)
            reshape_1_attr = {'name': reshape_1,
                              'opset_version': opset_version}
            NodeWrap(graph, reshape_1).replace_obj('Reshape', reshape_1_attr)
            if reshape_ver >= 5:
                const_1 = get_valid_node_name(graph, reshape_1 + '_shape')
                graph.add_node(const_1)
                dim = np.array([concat_out_shape[0], concat_out_shape[1]
                                * concat_out_shape[2], *concat_out_shape[3:]], np.int64)
                const_1_node = NodeWrap(graph, const_1)
                const_1_attr = {'name': const_1, 'value': dim,
                                'data_format': 'NHWC', 'opset_version': opset_version}
                const_1_node.replace_obj('Constant', const_1_attr)
                edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                             'tensor': Tensor(value=dim)}
                graph.add_edge(const_1, reshape_1, **edge_attr)
            else:
                ERROR('[Parser]: Reshape version (%d) not implemented in rearrange_pack_concat!' %
                      reshape_ver)

            transpose_obj.perm = [0, 1, 3, 2]
            transpose_out_edges = graph.sorted_out_edges(transpose, data=True)
            for _, out, out_edge_attr in transpose_out_edges:
                graph.remove_edge(transpose, out)
                graph.add_edge(reshape_2, out, **out_edge_attr)
            graph.add_edge(transpose, reshape_2)

            reshape_2_attr = {'name': reshape_2,
                              'opset_version': opset_version}
            NodeWrap(graph, reshape_2).replace_obj('Reshape', reshape_2_attr)
            if reshape_ver >= 5:
                const_2 = get_valid_node_name(graph, reshape_2 + '_shape')
                graph.add_node(const_2)
                dim = np.array(transpose_out_shape, np.int64)
                const_2_node = NodeWrap(graph, const_2)
                const_2_attr = {'name': const_2, 'value': dim,
                                'data_format': 'NHWC', 'opset_version': opset_version}
                const_2_node.replace_obj('Constant', const_2_attr)
                edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                             'tensor': Tensor(value=dim)}
                graph.add_edge(const_2, reshape_2, **edge_attr)
            else:
                ERROR('[Parser]: Reshape version (%d) not implemented in rearrange_pack_concat!' %
                      reshape_ver)
    if matched:
        clear_redundant_nodes(graph)


def remove_preprocess(graph):
    matched = False
    linear_op_types = list(set(BaseLinearOp.get_concrete_subclass_names(
    )).intersection(OnnxOp.get_concrete_subclass_names()))
    preprocess_matches = [matched_patterns(graph,
                                           nodes=[('bn', {'op': 'BatchNormalization'}),
                                                  ('linear', {'op': linear_op})
                                                  ],
                                           edges=[('bn', 'linear')]
                                           ) for linear_op in linear_op_types]
    preprocess_matches2 = [matched_patterns(graph,
                                            nodes=[('bn', {'op': 'BatchNormalization'}),
                                                   ('transpose', {
                                                    'op': 'Transpose'}),
                                                   ('linear', {
                                                    'op': linear_op})
                                                   ],
                                            edges=[('bn', 'transpose'),
                                                   ('transpose', 'linear')]
                                            ) for linear_op in linear_op_types]
    preprocess_matches = extend_lists(preprocess_matches)
    preprocess_matches2 = extend_lists(preprocess_matches2)
    preprocess_matches += preprocess_matches2
    for m in preprocess_matches:
        bn, transpose = m['bn'], m.get('transpose', None)
        bn_obj = NodeWrap(graph, bn)['object']
        transpose_obj = NodeWrap(graph, transpose)[
            'object'] if transpose is not None else None
        if bn_obj is not None and (transpose is None or (transpose_obj is not None and len(graph.sorted_out_edges(transpose)) == 1)):
            if bn_obj.training_mode:
                continue
            bn_inputs = bn_obj.get_input_tensors()
            if len(bn_inputs) != 5:
                continue
            if len(bn_inputs[1].shape) > 1 \
                    or len(bn_inputs[2].shape) > 1 \
                    or len(bn_inputs[3].shape) > 1 \
                    or len(bn_inputs[4].shape) > 1:
                continue
            matched = True
            for inp_name in graph._attr['input_tensors'].keys():
                inp_out_tensors = NodeWrap(graph, inp_name)[
                    'object'].get_output_tensors()
                if len(inp_out_tensors) >= 1 \
                        and inp_out_tensors[0] is not None \
                        and str(inp_out_tensors[0].dtype).find('int') >= 0:
                    continue
                paths = list(all_simple_paths(graph, inp_name, bn))
                if not paths:
                    continue
                passes_nodes = set(extend_lists(paths))
                if all([NodeWrap(graph, pn)['object'].type not in linear_op_types for pn in passes_nodes]):
                    remove_node_safely(graph, m['bn'])
        else:
            WARN('[Parser]: Meets invalid Node in remove_preprocess!')
    if matched:
        clear_redundant_nodes(graph)


def rename_reshape_like(graph):
    matches = [single_node_matcher(graph, op_type)
               for op_type in ('Flatten', 'Squeeze', 'Unsqueeze')]
    matches = extend_lists(matches)
    for m in matches:
        name = m['target']
        node_obj = NodeWrap(graph, name)['object']
        in_edges = graph.sorted_in_edges(name)
        out_shapes = node_obj.get_output_shapes()
        if len(in_edges) >= 1 and len(out_shapes) >= 1 and out_shapes[0] is not None:
            graph.remove_edges_from(in_edges[1:])
            const = get_valid_node_name(graph, name + '_shape')
            graph.add_node(const)
            reshape_attr = node_obj.copied_attr()
            reshape_attr.update({'opset_version': 5})
            NodeWrap(graph, name).replace_obj('Reshape', reshape_attr)
            reshape_dim = np.array(out_shapes[0], np.int64)
            const_attr = {'name': const,
                          'value': reshape_dim,
                          'data_format': 'NHWC',
                          'opset_version': 9}
            NodeWrap(graph, const).replace_obj('Constant', const_attr)
            edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                         'tensor': Tensor(value=reshape_dim)}
            graph.add_edge(const, name, **edge_attr)


def rename_single_mul_or_add_or_sub(graph):
    mas = ['Mul', 'Div', 'Add', 'Sub']
    mas_matches = [single_node_matcher(graph, op_type) for op_type in mas]
    mas_matches = extend_lists(mas_matches)
    for m in mas_matches:
        n = m['target']
        n_obj = NodeWrap(graph, n)['object']
        if n_obj is None:
            WARN('[Parser]: Meets invalid Op(%s) in rename_single_mul_or_add_or_sub!' % n)
            continue
        if n_obj.quantize:
            continue
        in_shapes = n_obj.get_input_shapes()
        in_consts = n_obj.sorted_in_consts()
        in_edges = graph.sorted_in_edges(n, keys=True, data=True)
        out_edges = graph.sorted_out_edges(n, data=True)
        if len(in_edges) == 2 \
                and len(in_shapes) == 2 \
                and len(in_consts) == 1 \
                and ((in_shapes[0] is not None and len(in_shapes[0]) in (0, 1, 2, 3, 4, 5) and in_consts[0][1] == 1)
                     or (in_shapes[1] is not None and len(in_shapes[1]) in (0, 1, 2, 3, 4, 5) and in_consts[0][1] == 0)) \
                and in_consts[0][2] is not None \
                and np.ndim(in_consts[0][2]) in (0, 1):
            const_in_port = in_consts[0][1]
            main_input_port = 1 - const_in_port

            if n_obj.type == 'Div' and const_in_port == 0:
                continue

            if len(out_edges) == 1 \
                    and (NodeWrap(graph, out_edges[0][1])['object'].type in ('Relu', 'LeakyRelu')
                         or (NodeWrap(graph, out_edges[0][1])['object'].type == 'Clip'
                             and FLOAT_EQUAL(NodeWrap(graph, out_edges[0][1])['object'].min, 0)
                             and FLOAT_EQUAL(NodeWrap(graph, out_edges[0][1])['object'].max, 6)
                             )):
                continue

            if (n_obj.type in ('Mul', 'Div') and FLOAT_EQUAL(in_consts[0][2], 1.)) \
                    or (n_obj.type == 'Add' and FLOAT_EQUAL(in_consts[0][2], 0.)) \
                    or (n_obj.type == 'Sub' and FLOAT_EQUAL(in_consts[0][2], 0.) and const_in_port == 1):
                src, _, k, const_in_attr = in_edges[const_in_port]
                graph.remove_edge(src, n, key=k)
                remove_node_safely(graph, n)
            else:

                output_shapes = n_obj.get_output_shapes()
                if len(output_shapes) == 0 or output_shapes[0] is None:
                    WARN(
                        '[Parser]: Meets invalid output shape of Node(%s) in rename_single_mul_or_add_or_sub!' % n)
                    continue

                src, _, k, in_attr = in_edges[main_input_port]
                input_dtype = str(in_attr['tensor'].value.dtype)
                cast_inserted = False
                if 'float' not in input_dtype:
                    insert_cast(graph, src, n, 'float32', in_attr, key=k)
                    cast_inserted = True
                    in_edges = graph.sorted_in_edges(n, keys=True, data=True)

                original_shape = output_shapes[0]
                reshape_inserted = False
                if len(in_shapes[main_input_port]) in (0, 1):
                    pre_reshape_dim = [1, 1] \
                        if in_shapes[main_input_port] == [] \
                        else [1, int(np.prod(in_shapes[main_input_port]))]
                    src, _, _, in_attr = in_edges[main_input_port]
                    insert_reshape(graph, src, n, in_attr, pre_reshape_dim)
                    reshape_inserted = True
                    in_edges = graph.sorted_in_edges(n, keys=True, data=True)
                    in_shapes = n_obj.get_input_shapes()

                num_output = in_shapes[main_input_port][-1]

                tiled_const_value = np.tile(
                    in_consts[0][2], num_output) if in_consts[0][2].size == 1 else in_consts[0][2]
                tiled_const_value = tiled_const_value.astype(np.float32)
                if tiled_const_value.shape[-1] > num_output:
                    num_output = tiled_const_value.shape[-1]
                    src, _, _, in_attr = in_edges[main_input_port]
                    insert_tile(graph, src, n, in_attr, [
                                1] * (len(in_shapes[main_input_port]) - 1) + [num_output])
                    in_edges = graph.sorted_in_edges(n, keys=True, data=True)

                if n_obj.type == 'Sub' and in_consts[0][1] == 0:
                    gamma_value = - np.ones((num_output, ), np.float32)
                    beta_value = tiled_const_value
                else:
                    gamma_value = tiled_const_value \
                        if n_obj.type == 'Mul' \
                        else (1 / tiled_const_value if n_obj.type == 'Div' else np.ones((num_output, ), np.float32))
                    beta_value = np.zeros((num_output, ), np.float32) \
                        if n_obj.type in ('Mul', 'Div') \
                        else (tiled_const_value if n_obj.type == 'Add' else -tiled_const_value)
                mean_value = np.zeros((num_output, ), np.float32)
                var_value = np.ones((num_output,), np.float32)
                gamma = get_valid_node_name(graph, n + '_gamma')
                beta = get_valid_node_name(graph, n + '_beta')
                mean = get_valid_node_name(graph, n + '_mean')
                var = get_valid_node_name(graph, n + '_var')

                if main_input_port == 0:
                    graph.remove_edges_from(in_edges[1:])
                else:
                    src, _, _, in_attr = in_edges[main_input_port]
                    in_attr.update({'dst_in_port': 0})
                    graph.remove_edges_from(in_edges)
                    graph.add_edge(src, n, **in_attr)

                graph.add_edge(
                    gamma, n, **{'src_out_port': 0, 'dst_in_port': 1, 'tensor': Tensor(value=gamma_value)})
                graph.add_edge(
                    beta, n, **{'src_out_port': 0, 'dst_in_port': 2, 'tensor': Tensor(value=beta_value)})
                graph.add_edge(
                    mean, n, **{'src_out_port': 0, 'dst_in_port': 3, 'tensor': Tensor(value=mean_value)})
                graph.add_edge(
                    var, n, **{'src_out_port': 0, 'dst_in_port': 4, 'tensor': Tensor(value=var_value)})

                batchnorm_attr = n_obj.copied_attr()
                batchnorm_attr.update({'opset_version': 9, 'epsilon': 0})
                NodeWrap(graph, n).replace_obj(
                    'BatchNormalization', batchnorm_attr)
                NodeWrap(graph, n)['object'].data_format = 'NHWC'

                gamma_attr = {'name': gamma, 'value': gamma_value,
                              'data_format': 'NHWC', 'opset_version': 1}
                beta_attr = {'name': beta, 'value': beta_value,
                             'data_format': 'NHWC', 'opset_version': 1}
                mean_attr = {'name': mean, 'value': mean_value,
                             'data_format': 'NHWC', 'opset_version': 1}
                var_attr = {'name': var, 'value': var_value,
                            'data_format': 'NHWC', 'opset_version': 1}
                NodeWrap(graph, gamma).replace_obj('Constant', gamma_attr)
                NodeWrap(graph, beta).replace_obj('Constant', beta_attr)
                NodeWrap(graph, mean).replace_obj('Constant', mean_attr)
                NodeWrap(graph, var).replace_obj('Constant', var_attr)

                post_cast = None
                if cast_inserted:
                    to_dtype = get_converted_dtype(input_dtype, True) if input_dtype in (
                        'uint64', 'int64') else input_dtype
                    post_cast = get_valid_node_name(graph, n + '_post_cast')
                    graph.add_node(post_cast)
                    post_cast_attr = {'name': post_cast,
                                      'opset_version': 1, 'to': to_dtype}
                    NodeWrap(graph, post_cast).replace_obj(
                        'Cast', post_cast_attr)
                    for _, dst, out_attr in out_edges:
                        if out_attr.get('src_out_port', 0) == 0:
                            graph.remove_edge(n, dst)
                            new_out_attr = copy.deepcopy(out_attr)
                            new_out_attr['src_out_port'] = 0
                            graph.add_edge(post_cast, dst, **new_out_attr)
                    graph.add_edge(n, post_cast)

                post_reshape = None
                if reshape_inserted:
                    post_reshape = insert_reshape_after(
                        graph, n, original_shape)

                if n in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(n)
                    if cast_inserted and post_cast is not None:
                        graph._attr['output_names'][index] = post_cast
                    elif reshape_inserted and post_reshape is not None:
                        graph._attr['output_names'][index] = post_reshape


def remove_sub_add_pair(graph):
    matches_1 = matched_patterns(graph,
                                 nodes=[
                                     ('const_1', {'op': 'Constant'}),
                                     ('sub', {'op': 'Sub'}),
                                     ('const_2', {'op': 'Constant'}),
                                     ('add', {'op': 'Add'}),
                                 ],
                                 edges=[
                                     ('const_1', 'sub', {'dst_in_port': 1}),
                                     ('sub', 'add'),
                                     ('const_2', 'add', {'dst_in_port': 1}),
                                 ])
    matches_2 = matched_patterns(graph,
                                 nodes=[
                                     ('const_1', {'op': 'Constant'}),
                                     ('add', {'op': 'Add'}),
                                     ('const_2', {'op': 'Constant'}),
                                     ('sub', {'op': 'Sub'}),
                                 ],
                                 edges=[
                                     ('const_1', 'add', {'dst_in_port': 1}),
                                     ('add', 'sub'),
                                     ('const_2', 'sub', {'dst_in_port': 1}),
                                 ])
    all_matches = matches_1 + matches_2
    for m in all_matches:
        const_1, const_2, add, sub = m['const_1'], m['const_2'], m['add'], m['sub']
        if not graph.has_node(add) or not graph.has_node(sub):
            WARN('[Parser]: Node (%s or %s or %s or %s) cannot be found, graph maybe has been changed!' % (
                const_1, const_2, add, sub))
            continue
        inp = add if sub in graph.succ[add] else sub
        out = sub if inp == add else add
        inp_out_edges = graph.sorted_out_edges(inp)
        const_1_node = NodeWrap(graph, const_1)
        const_2_node = NodeWrap(graph, const_2)
        if len(inp_out_edges) == 1 \
                and const_1_node['object'].value is not None \
                and const_2_node['object'].value is not None \
                and const_1_node['object'].value.shape == const_2_node['object'].value.shape \
                and np.all(const_1_node['object'].value == const_2_node['object'].value):
            remove_node_safely(graph, out)
            remove_node_safely(graph, inp)


def split_special_bn(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('bn', {'op': 'BatchNormalization'}),
                                   ('gamma', {'op': 'Constant'}),
                                   ('beta', {'op': 'Constant'}),
                                   ('mean', {'op': 'Constant'}),
                                   ('var', {'op': 'Constant'}),
                                   ('x_input', {})
                               ],
                               edges=[
                                   ('x_input', 'bn', {'dst_in_port': 0}),
                                   ('gamma', 'bn', {'dst_in_port': 1}),
                                   ('beta', 'bn', {'dst_in_port': 2}),
                                   ('mean', 'bn', {'dst_in_port': 3}),
                                   ('var', 'bn', {'dst_in_port': 4}),
                               ])
    matched = False
    for m in matches:
        beta, bn, gamma, mean, var, x_input = \
            m['beta'], m['bn'], m['gamma'], m['mean'], m['var'], m['x_input']
        beta_obj, bn_obj, gamma_obj, mean_obj, var_obj, input_obj = \
            [NodeWrap(graph, name)['object'] for name in [
                beta, bn, gamma, mean, var, x_input]]
        if bn_obj is not None and \
                beta_obj is not None and \
                gamma_obj is not None and \
                mean_obj is not None and \
                var_obj is not None and \
                input_obj is not None:
            if bn_obj.training_mode:
                continue
            bn_in_edges = graph.sorted_in_edges(m['bn'], data=True)
            bn_out_edges = graph.sorted_out_edges(m['bn'], data=True)

            if len(bn_in_edges) > 0 \
                    and len(bn_out_edges) > 0 \
                    and len(beta_obj.value.shape) > 1 \
                    and len(gamma_obj.value.shape) > 1 \
                    and len(mean_obj.value.shape) > 1 \
                    and len(var_obj.value.shape) > 1:
                matched = True
                gamma = gamma_obj.value
                beta = beta_obj.value
                mean = mean_obj.value
                var = var_obj.value
                input_value = bn_in_edges[0][2]['tensor'].value
                weights = gamma / np.sqrt(var + bn_obj.epsilon)
                biases = beta - gamma * mean / np.sqrt(var + bn_obj.epsilon)
                if bn_obj.data_format.startswith('NC') \
                        and bn_obj.spatial:
                    reshape_dims = [-1] + (len(input_value) - 2) * [1]
                    weights = np.reshape(weights, reshape_dims)
                    biases = np.reshape(biases, reshape_dims)
                mul_tensor = input_value * weights

                bn_weight_mul = get_valid_node_name(graph, bn + '_weight_mul')
                bn_bias_add = get_valid_node_name(graph, bn + '_bias_add')

                src, _, in_attr = bn_in_edges[0]
                graph.remove_edges_from(bn_in_edges)
                graph.add_edge(src, bn_weight_mul, **in_attr)
                graph.add_edge(bn_weight_mul, bn_bias_add, **{
                               'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(value=mul_tensor)})
                for _, dst, out_attr in bn_out_edges:
                    graph.remove_edge(bn, dst)
                    graph.add_edge(bn_bias_add, dst, **out_attr)

                insert_constant(graph, bn_weight_mul + '_value',
                                weights, bn_weight_mul, in_port=1)
                insert_constant(graph, bn_bias_add + '_value',
                                biases, bn_bias_add, in_port=1)

                NodeWrap(graph, bn_weight_mul).replace_obj(
                    'Mul', {'name': bn_weight_mul, 'opset_version': 7})
                NodeWrap(graph, bn_bias_add).replace_obj(
                    'Add', {'name': bn_bias_add, 'opset_version': 7})

                if bn in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(bn)
                    graph._attr['output_names'].remove(bn)
                    graph._attr['output_names'].insert(index, bn_bias_add)
    if matched:
        clear_redundant_nodes(graph)


def split_group_conv(graph):
    matches = single_node_matcher(graph, 'Conv')
    for single_match in matches:
        conv = single_match['target']
        conv_obj = NodeWrap(graph, conv)['object']
        conv_in_shapes = conv_obj.get_input_shapes()
        if len(conv_in_shapes) >= 1 \
                and conv_in_shapes[0] is not None \
                and conv_obj.group > 1 \
                and conv_obj.weights.shape[0] != conv_obj.group:
            conv_out_tensor = conv_obj.get_output_tensors()[0]
            split_num = conv_obj.group
            weights_split = np.split(conv_obj.weights, split_num, axis=0)
            biases_split = np.split(conv_obj.biases, split_num)
            if conv_out_tensor is not None:
                conv_out_tensor_split = np.split(
                    conv_out_tensor, split_num, axis=3)
            else:
                conv_out_tensor_split = []
            conv_in_edges = graph.sorted_in_edges(conv, data=True)
            conv_out_edges = graph.sorted_out_edges(conv, data=True)

            split = get_valid_node_name(graph, conv + '_split')
            concat = get_valid_node_name(graph, conv + '_concat')
            graph.add_nodes_from([split, concat])
            for src, _, in_attr in conv_in_edges:
                graph.remove_edge(src, conv)
                graph.add_edge(src, split, **in_attr)
            for _, dst, out_attr in conv_out_edges:
                graph.remove_edge(conv, dst)
                graph.add_edge(concat, dst, **out_attr)

            split_attr = conv_obj.copied_attr()
            splits = [conv_in_shapes[0][3] // split_num] * split_num
            split_attr.update(
                {'name': split, 'opset_version': 2, 'axis': 3, 'split': splits})
            NodeWrap(graph, split).replace_obj('Split', split_attr)
            concat_attr = conv_obj.copied_attr()
            concat_attr.update({'name': concat, 'opset_version': 4, 'axis': 3})
            NodeWrap(graph, concat).replace_obj('Concat', concat_attr)

            for i in range(split_num):
                meta_conv = get_valid_node_name(
                    graph, conv + '_split_conv_' + str(i + 1))
                graph.add_edge(split, meta_conv, **
                               {'src_out_port': i, 'dst_in_port': 0})
                if conv_out_tensor_split:
                    graph.add_edge(meta_conv, concat, **{'src_out_port': 0,
                                                         'dst_in_port': i,
                                                         'tensor': Tensor(value=conv_out_tensor_split[i],
                                                                          is_const=conv_obj.is_all_inputs_const())
                                                         })
                else:
                    graph.add_edge(meta_conv, concat, **
                                   {'src_out_port': 0, 'dst_in_port': i})
                meta_conv_attr = conv_obj.copied_attr()
                meta_conv_attr.update({'name': meta_conv,
                                       'opset_version': 1,
                                       'weights': weights_split[i],
                                       'biases': biases_split[i],
                                       'num_output': conv_obj.num_output // split_num,
                                       'group': 1})
                NodeWrap(graph, meta_conv).replace_obj('Conv', meta_conv_attr)
            graph.remove_node(conv)


def split_conv_transpose(graph):
    matches = single_node_matcher(graph, 'ConvTranspose')
    for m in matches:
        conv_trans = m['target']
        conv_trans_obj = NodeWrap(graph, conv_trans)['object']
        in_edges = graph.sorted_in_edges(conv_trans, data=True)
        out_edges = graph.sorted_out_edges(conv_trans, data=True)
        if conv_trans_obj is None \
                or len(in_edges) < 1 \
                or len(out_edges) < 1 \
                or len(conv_trans_obj.get_output_shapes()) < 1 \
                or conv_trans_obj.get_output_shapes()[0] is None:
            WARN(
                '[Parser]: Meets invalid ConvTranspose (%s) in split_conv_transpose!' % conv_trans)
            continue
        if all(p == 0 for p in conv_trans_obj.output_padding):
            continue
        strides = np.array(conv_trans_obj.strides)
        dilations = np.array(conv_trans_obj.dilations)
        output_padding = np.array(conv_trans_obj.output_padding)
        if np.any(np.logical_and(output_padding >= strides, output_padding >= dilations)):
            WARN('[Parser]: Onnx %s (%s) output_padding should be less than stride or dilation!'
                 % (type(conv_trans_obj).__name__, conv_trans_obj.name))
            continue
        spatial_rank = len(conv_trans_obj.output_padding)
        ori_output_shape = conv_trans_obj.get_output_shapes()[0]
        data_format = conv_trans_obj.data_format
        if data_format == 'NCHW':
            ori_spatial_output_shape = ori_output_shape[2:]
            # xi_begins(i=0,1,2,3,...) + x0_end + x1_end + xi_ends(i=2,3,...)
            full_pads = [0] * (2 + spatial_rank + 2) + \
                conv_trans_obj.output_padding
        else:
            ori_spatial_output_shape = ori_output_shape[1:-1]
            full_pads = [0] * (2 + spatial_rank + 1) + \
                conv_trans_obj.output_padding + [0]
        new_spatial_output_shape = (np.array(ori_spatial_output_shape) + np.multiply(
            np.array(conv_trans_obj.strides) - 1, conv_trans_obj.output_padding)).tolist()
        # new_output_shape = ori_output_shape[:2] + new_spatial_output_shape

        src, _, in_attr = in_edges[0]
        graph.remove_edges_from(in_edges)
        pre_pad = get_valid_node_name(graph, conv_trans + '_pre_pad')
        graph.add_edge(src, pre_pad, **in_attr)
        graph.add_edge(pre_pad, conv_trans)
        pad_attr = {'name': pre_pad,
                    'opset_version': 2,
                    'pads': full_pads,
                    'data_format': data_format}
        NodeWrap(graph, pre_pad).replace_obj('Pad', pad_attr)

        if ori_spatial_output_shape != new_spatial_output_shape:
            begin = [0] * (2 + spatial_rank)
            size = ori_output_shape
            post_slice = insert_slice_after(
                graph, conv_trans, begin, size, data_format=data_format)
            if conv_trans in graph._attr['output_names']:
                index = graph._attr['output_names'].index(conv_trans)
                graph._attr['output_names'][index] = post_slice

        if conv_trans_obj.output_shape:
            assert(len(conv_trans_obj.output_shape) == spatial_rank), \
                '[Parser]: Meets invalid output_shape of ConvTranspose (%s) in split_conv_transpose!' % conv_trans
            conv_trans_obj.output_shape = new_spatial_output_shape
        conv_trans_obj.output_padding = [0] * spatial_rank


def split_negative_pads(graph):
    possible_op_types = [op_type.get_concrete_subclass_names()
                         for op_type in [OnnxOp, CommonOp]]
    possible_op_types = extend_lists(possible_op_types)
    op_with_pads_list = list(set(
        OpHasPaddingStrides.get_concrete_subclass_names()).difference(['ConvTranspose']).intersection(possible_op_types))
    matches = [single_node_matcher(graph, op) for op in op_with_pads_list]
    matches = extend_lists(matches)
    for m in matches:
        node = m['target']
        node_obj = NodeWrap(graph, node)['object']
        if node_obj is not None:
            if isinstance(node_obj, BaseOnnxPoolOp) \
                    and len(node_obj.get_out_ports()) > 1:
                continue
            if len(getattr(node_obj, 'pads', [])) == 4 \
                    and any([p < 0 for p in node_obj.pads]):
                input_shapes = node_obj.get_input_shapes()
                in_edges = graph.sorted_in_edges(node, data=True)
                if len(input_shapes) >= 1 and len(input_shapes[0]) == 4 and len(in_edges) == 1:
                    src, _, in_attr = in_edges[0]
                    in_shape = input_shapes[0]
                    begin = np.array([0] * len(in_shape), np.int64)
                    end = np.array(in_shape, np.int64)
                    if node_obj.data_format == 'NHWC':
                        negative_mask = node_obj.tf_pads < 0
                        begin[negative_mask[:, 0]] = node_obj.tf_pads[:,
                                                                      0][negative_mask[:, 0]] * (-1)
                        end[negative_mask[:, 1]] = (
                            node_obj.tf_pads[:, 1] + end)[negative_mask[:, 1]]
                    else:
                        onnx_pads = np.reshape(
                            np.array(node_obj.pads), (-1, 2))
                        negative_mask = onnx_pads < 0
                        begin[2:][negative_mask[0, :]] = onnx_pads[0,
                                                                   :][negative_mask[0, :]] * (-1)
                        end[2:][negative_mask[1, :]] = (
                            onnx_pads[1, :] + end[2:])[negative_mask[1, :]]
                    size = end - begin
                    insert_slice(graph, src, node, in_attr,
                                 begin.tolist(), size.tolist())
                    node_obj.pads = [0 if p < 0 else p for p in node_obj.pads]
                    node_obj.auto_pad = 'NOTSET'
        else:
            WARN(
                '[Parser]: Meets invalid Node (%s) in split_negative_pads!' % node)


def split_reduce_logsumexp(graph):
    matches = single_node_matcher(graph, 'ReduceLogSumExp')
    for m in matches:
        rlse = m['target']
        rlse_obj = NodeWrap(graph, rlse)['object']
        in_edges = graph.sorted_in_edges(rlse, data=True)
        if rlse_obj is None or len(in_edges) != 1:
            WARN(
                '[Parser]: Meets invalid ReduceLogSumExp (%s) in split_reduce_logsumexp!' % rlse)
            continue
        exp = get_valid_node_name(graph, rlse + '_exp')
        src, _, in_attr = in_edges[0]
        graph.remove_edge(src, rlse)
        graph.add_edge(src, exp, **in_attr)
        graph.add_edge(exp, rlse)
        exp_attr = rlse_obj.copied_attr()
        exp_attr.update({'name': exp, 'opset_version': 13})
        NodeWrap(graph, exp).replace_obj('Exp', exp_attr)
        log_attr = rlse_obj.copied_attr()
        log_attr.update({'opset_version': 13})
        NodeWrap(graph, rlse).replace_obj('ReduceLogSum', log_attr)


def split_reduce_logsum(graph):
    matches = single_node_matcher(graph, 'ReduceLogSum')
    for m in matches:
        rls = m['target']
        rls_obj = NodeWrap(graph, rls)['object']
        in_edges = graph.sorted_in_edges(rls, data=True)
        if rls_obj is None or len(in_edges) != 1:
            WARN(
                '[Parser]: Meets invalid ReduceLogSum (%s) in split_reduce_logsum!' % rls)
            continue
        reduce_sum = get_valid_node_name(graph, rls + '_reduce_sum')
        src, _, in_attr = in_edges[0]
        graph.remove_edge(src, rls)
        graph.add_edge(src, reduce_sum, **in_attr)
        graph.add_edge(reduce_sum, rls)
        reduce_sum_attr = rls_obj.copied_attr()
        reduce_sum_attr.update({'name': reduce_sum, 'opset_version': 11})
        NodeWrap(graph, reduce_sum).replace_obj('ReduceSum', reduce_sum_attr)
        log_attr = rls_obj.copied_attr()
        log_attr.update({'opset_version': 13})
        NodeWrap(graph, rls).replace_obj('Log', log_attr)


def split_reduce_sumsq(graph):
    matches = single_node_matcher(graph, 'ReduceSumSquare')
    for m in matches:
        rss = m['target']
        rss_obj = NodeWrap(graph, rss)['object']
        in_edges = graph.sorted_in_edges(rss, data=True)
        if rss_obj is not None and len(in_edges) == 1:
            pow = get_valid_node_name(graph, rss + '_pre_pow')
            src, _, in_attr = in_edges[0]
            graph.remove_edge(src, rss)
            graph.add_edge(src, pow, **in_attr)
            graph.add_edge(pow, rss)
            insert_constant(graph, pow + '_exponent',
                            np.array([2], np.int32), pow, in_port=1)
            pow_attr = rss_obj.copied_attr()
            pow_attr.update({'name': pow, 'opset_version': 7})
            NodeWrap(graph, pow).replace_obj('Pow', pow_attr)
            sum_attr = rss_obj.copied_attr()
            sum_attr.update({'opset_version': 11})
            NodeWrap(graph, rss).replace_obj('ReduceSum', sum_attr)


def split_roll(graph):
    # only support split roll with len(axis)==1 and len(shift)==1 currently
    matches = single_node_matcher(graph, 'Roll')
    matched = False
    from ....ops.common_ops import RollOp
    for single_match in matches:
        roll = single_match['target']
        roll_obj = NodeWrap(graph, roll)['object']
        roll_in_edges = graph.sorted_in_edges(roll, data=True)
        if roll_obj is not None and len(roll_in_edges) == 1:
            axis_value = roll_obj.axes[0]
            roll_shift = roll_obj.shift
            roll_shape = roll_obj.get_input_shapes()[0]
            if len(roll_shift) != 1:
                continue
            roll_shift = roll_shift[0]
            roll_shift, start1, end1, steps1, axes1, start2, end2, steps2, axes2 =\
                RollOp.cal_roll_parm(axis_value, roll_shift, roll_shape)

            if roll_shift == 0 \
                    or np.any(np.abs(start1) >= np.abs(end1)) \
                    or np.any(np.abs(start2) >= np.abs(end2)):
                continue

            matched = True
            slice1 = get_valid_node_name(graph, roll + '_slice1')
            slice2 = get_valid_node_name(graph, roll + '_slice2')

            src, dst, roll_in_attr = roll_in_edges[0]

            slice_in_attr = copy.deepcopy(roll_in_attr)

            graph.remove_edge(src, roll)
            graph.add_edge(src, slice1, **roll_in_attr)
            graph.add_edge(src, slice2, **slice_in_attr)
            graph.add_edge(slice1, roll, **
                           {'src_out_port': 0, 'dst_in_port': 0})
            graph.add_edge(slice2, roll, **
                           {'src_out_port': 0, 'dst_in_port': 1})

            slice1_attr = {'name': slice1, 'opset_version': 1,
                           'axes': axes1, 'starts': start1, 'ends': end1}
            slice2_attr = {'name': slice2, 'opset_version': 1,
                           'axes': axes2, 'starts': start2, 'ends': end2}

            concat_attr = roll_obj.copied_attr()
            concat_attr.update(
                {'opset_version': 11, 'axis': axis_value, 'axes': None})
            NodeWrap(graph, roll).replace_obj('Concat', concat_attr)

            NodeWrap(graph, slice1).replace_obj('Slice', slice1_attr)
            NodeWrap(graph, slice2).replace_obj('Slice', slice2_attr)

            if roll in graph._attr['output_names']:
                index = graph._attr['output_names'].index(roll)
                graph._attr['output_names'][index] = concat


def split_mean(graph):
    matches = single_node_matcher(graph, 'Mean')
    for m in matches:
        mean = m['target']
        mean_obj = NodeWrap(graph, mean)['object']
        if mean_obj is not None \
                and len(mean_obj.get_input_shapes()) >= 2 \
                and all([s is not None and len(s) >= 1 for s in mean_obj.get_input_shapes()]):
            input_shapes = mean_obj.get_input_shapes()
            if any([input_shapes[0] != s for s in input_shapes[1:]]):
                WARN(
                    '[Parser]: All input shapes of Mean Op(%s) should be broadcasted and equal in split_mean!' % mean)
                continue
            dim = [1] + input_shapes[0]
            concat = get_valid_node_name(graph, mean + '_concat')
            in_edges = graph.sorted_in_edges(mean, keys=True, data=True)
            for i, (src, _, k, in_attr) in enumerate(in_edges):
                reshape = get_valid_node_name(
                    graph, mean + '_pre_reshape_' + str(i))
                reshape_in_attr = copy.deepcopy(in_attr)
                reshape_in_attr['dst_in_port'] = 0
                graph.remove_edge(src, mean, key=k)
                graph.add_edge(src, reshape, **reshape_in_attr)
                graph.add_edge(reshape, concat, **
                               {'src_out_port': 0, 'dst_in_port': i})
                NodeWrap(graph, reshape).replace_obj(
                    'Reshape', {'name': reshape, 'opset_version': 5})
                insert_constant(graph,
                                reshape + '_dim',
                                np.array(dim, np.int32),
                                reshape,
                                in_port=1,
                                data_format=mean_obj.data_format)
            concat_out_attr = copy.deepcopy(in_edges[0][3])
            tensor_dtype = concat_out_attr['tensor'].value.dtype if \
                concat_out_attr['tensor'].value is not None else np.float32
            tensor_value = np.random.ranf(
                [len(in_edges)] + input_shapes[0]).astype(tensor_dtype)
            concat_out_attr.update({'src_out_port': 0,
                                    'tensor': Tensor(value=tensor_value)})
            graph.add_edge(concat, mean, **concat_out_attr)
            NodeWrap(graph, concat).replace_obj(
                'Concat', {'name': concat, 'opset_version': 11, 'axis': 0})
            reduce_mean_attr = mean_obj.copied_attr()
            reduce_mean_attr.update(
                {'opset_version': 11, 'axes': [0], 'keepdims': False})
            NodeWrap(graph, mean).replace_obj('ReduceMean', reduce_mean_attr)
        else:
            WARN('[Parser]: Meets invalid Mean Op(%s) in split_mean!' % mean)


def split_sum_or_max_or_min(graph, op_type_list=['Sum', 'Max', 'Min']):
    # a dict of new op_type and its opset version
    op_type_name_and_ver_dict = {'Sum': ['Add', 7],
                                 'Max': ['Max', 8],
                                 'Min': ['Min', 8],
                                 'TfKerasMultiply': ['Mul', 7]}
    if not isinstance(op_type_list, list) \
            or any((op_type not in op_type_name_and_ver_dict
                    or len(op_type_name_and_ver_dict[op_type]) != 2) for op_type in op_type_list):
        WARN('[Parser]: Meet invalid op_type %s in split_sum_or_max_or_min!' % str(op_type_list))
        return

    matches = single_node_matcher(graph, op_type_list)
    for single_match in matches:
        node = single_match['target']
        node_obj = NodeWrap(graph, node)['object']
        if node_obj is not None and len(node_obj.get_input_shapes()) >= 2:
            new_op_type, new_op_version = op_type_name_and_ver_dict[node_obj.type]
            split_num = len(node_obj.get_input_shapes()) - 2
            if split_num > 0:
                in_edges = graph.sorted_in_edges(node, keys=True, data=True)
                out_edges = graph.sorted_out_edges(node, data=True)
                graph.remove_edges_from(in_edges[2:])

                nodes_list = [node]
                for i in range(split_num):
                    cur_src, _, _, cur_in_attr = in_edges[2 + i]
                    new_node = get_valid_node_name(graph, node + '_expand_' + str(i + 1))
                    last_node_obj = NodeWrap(graph, nodes_list[-1])['object']
                    if last_node_obj is not None and all([inp is not None for inp in last_node_obj.get_input_tensors()]):
                        node_result = reduce(
                            lambda x, y: x + y, last_node_obj.get_input_tensors())
                    else:
                        node_result = None
                    graph.add_edge(
                        nodes_list[-1], new_node, **{'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(value=node_result)})
                    new_in_attr = copy.deepcopy(cur_in_attr)
                    new_in_attr.update({'dst_in_port': 1})
                    graph.add_edge(cur_src, new_node, **new_in_attr)

                    node_attr = node_obj.copied_attr()
                    node_attr.update(
                        {'name': new_node, 'opset_version': new_op_version})
                    NodeWrap(graph, new_node).replace_obj(new_op_type, node_attr)
                    nodes_list.append(new_node)

                for _, dst, out_attr in out_edges:
                    graph.remove_edge(node, dst)
                    graph.add_edge(nodes_list[-1], dst, **out_attr)

                if node in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(node)
                    graph._attr['output_names'][index] = nodes_list[-1]

            new_node_attr = node_obj.copied_attr()
            new_node_attr.update({'opset_version': new_op_version})
            NodeWrap(graph, node).replace_obj(new_op_type, new_node_attr)
        else:
            WARN('[Parser]: Invalid Op(%s) for splitting in split_sum_or_max_or_min!' % node)


def split_hardmax(graph):
    matches = single_node_matcher(graph, 'Hardmax')
    for m in matches:
        hardmax = m['target']
        hardmax_obj = NodeWrap(graph, hardmax)['object']

        in_edges = graph.sorted_in_edges(hardmax, data=True)
        if hardmax_obj is None or len(in_edges) != 1:
            WARN('[Parser]: Meet invalid Hardmax (%s) in split_hardmax!' % hardmax)
            continue

        input_shape = hardmax_obj.get_input_shapes()[0]
        hardmax_axis = hardmax_obj.axis + \
            len(input_shape) if hardmax_obj.axis < 0 else hardmax_obj.axis
        if hardmax_axis < 0 or hardmax_axis >= len(input_shape):
            WARN('[Parser]: Meet invalid axis (%s) of Hardmax (%s) in split_hardmax!' % (
                str(hardmax_axis), hardmax))
            continue

        input_tensors = hardmax_obj.get_input_tensors()
        if len(input_tensors) != 1 or input_tensors[0] is None:
            WARN(
                '[Parser]: Meet invalid input tensor of Hardmax (%s) in split_hardmax!' % hardmax)
            continue

        src, _, in_attr = in_edges[0]

        argmax = get_valid_node_name(graph, hardmax + '_argmax')
        graph.remove_edge(src, hardmax)
        graph.add_edge(src, argmax)
        graph.add_edge(argmax, hardmax)
        argmax_attr = hardmax_obj.copied_attr()
        argmax_attr.update(
            {'name': argmax, 'axis': hardmax_axis, 'keepdims': 0, 'opset_version': 13})
        NodeWrap(graph, argmax).replace_obj('ArgMax', argmax_attr)

        onehot = hardmax
        insert_constant(graph, onehot + '_depth',
                        np.array([input_shape[hardmax_axis]], np.int32), onehot, in_port=1)
        insert_constant(graph, onehot + '_values',
                        np.array([0, 1], input_tensors[0].dtype), onehot, in_port=2)
        onehot_attr = hardmax_obj.copied_attr()
        onehot_attr.update({'axis': hardmax_axis, 'opset_version': 11})
        NodeWrap(graph, hardmax).replace_obj('OneHot', onehot_attr)


def align_matmul_input(graph):
    matches = single_node_matcher(graph, ['MatMul', 'MatMulInteger'])
    for m in matches:
        matmul = m['target']
        obj = NodeWrap(graph, matmul)['object']
        in_edges = graph.sorted_in_edges(matmul, keys=True, data=True)
        if obj is not None \
                and len(in_edges) >= 2 \
                and len(obj.get_input_shapes()) >= 2 \
                and all([shape is not None for shape in obj.get_input_shapes()]) \
                and len(obj.get_input_shapes()[0]) != len(obj.get_input_shapes()[1]) \
                and len(obj.get_output_shapes()) >= 1 \
                and obj.get_output_shapes()[0]:
            input_shapes = obj.get_input_shapes()
            max_dim = max(*[len(s) for s in input_shapes])
            edge_index = 0 if max_dim == len(input_shapes[1]) else 1
            min_dim = len(input_shapes[edge_index])
            dim_diff = max_dim - len(input_shapes[edge_index])
            reshape_dim = [1] * dim_diff + list(input_shapes[edge_index]) \
                if edge_index == 0 \
                else [1] * (dim_diff - 2 + min_dim) + list(input_shapes[edge_index]) + [1] * (2 - min_dim)
            src, _, k, in_attr = in_edges[edge_index]
            insert_reshape(graph, src, matmul, in_attr, reshape_dim, key=k)
            out_shape = obj.get_output_shapes()[0]
            post_reshape = insert_reshape_after(
                graph, matmul, out_shape, old_dim=list(out_shape))
            if matmul in graph._attr['output_names']:
                index = graph._attr['output_names'].index(matmul)
                graph._attr['output_names'][index] = post_reshape


def adjust_scalar_to_1d(graph):
    op_type_list = OpNeedBroadcast.get_concrete_subclass_names()
    for op_type in op_type_list:
        matches = single_node_matcher(graph, op_type)
        for m in matches:
            broadcast = m['target']
            broadcast_obj = NodeWrap(graph, broadcast)['object']
            if broadcast_obj is not None:
                in_edges = graph.sorted_in_edges(
                    broadcast, keys=True, data=True)
                input_shapes = broadcast_obj.get_input_shapes()
                if len(in_edges) == len(input_shapes) \
                        and len(input_shapes) >= 2 \
                        and all([s is not None and len(s) == 0 for s in input_shapes]):
                    for i, (src, _, k, in_attr) in enumerate(in_edges):
                        reshape = get_valid_node_name(
                            graph, broadcast + '_pre_reshape_' + str(i))
                        reshape_in_attr = copy.deepcopy(in_attr)
                        reshape_in_attr['dst_in_port'] = 0

                        graph.remove_edge(src, broadcast, key=k)
                        graph.add_edge(src, reshape, **reshape_in_attr)
                        graph.add_edge(
                            reshape, broadcast, **{'src_out_port': 0, 'dst_in_port': in_attr['dst_in_port']})

                        NodeWrap(graph, reshape).replace_obj(
                            'Reshape', {'name': reshape, 'opset_version': 5})
                        insert_constant(graph,
                                        reshape + '_dim',
                                        np.array([1], np.int32),
                                        reshape,
                                        in_port=1,
                                        data_format=broadcast_obj.data_format)

                    post_reshapes = []
                    out_edges = graph.sorted_out_edges(broadcast, data=True)
                    for p in broadcast_obj.get_out_ports():
                        post_reshape = get_valid_node_name(
                            graph, broadcast + '_post_reshape_' + str(p))
                        for _, dst, out_attr in out_edges:
                            if out_attr['src_out_port'] == p:
                                reshape_out_attr = copy.deepcopy(out_attr)
                                reshape_out_attr['src_out_port'] = 0
                                graph.remove_edge(broadcast, dst)
                                graph.add_edge(post_reshape, dst,
                                               **reshape_out_attr)
                        graph.add_edge(broadcast, post_reshape,
                                       **{'src_out_port': p, 'dst_in_port': 0})
                        NodeWrap(graph, post_reshape).replace_obj(
                            'Reshape', {'name': post_reshape, 'opset_version': 5})
                        insert_constant(graph,
                                        post_reshape + '_dim',
                                        np.array([], np.int32),
                                        post_reshape,
                                        in_port=1,
                                        data_format=broadcast_obj.data_format)
                        post_reshapes.append(post_reshape)

                    if broadcast in graph._attr['output_names'] and post_reshapes:
                        index = graph._attr['output_names'].index(broadcast)
                        graph._attr['output_names'].remove(broadcast)
                        graph._attr['output_names'][index:index] = post_reshapes

            else:
                WARN(
                    '[Parser]: Meets invalid Node (%) in adjust_scalar_to_1d!' % broadcast)


def adjust_2d_to_4d(graph):
    pure_inputs_types = ['MatMul']
    mixed_inputs_types = []
    for op_type in pure_inputs_types:
        matches = single_node_matcher(graph, op_type)
        for m in matches:
            node_name = m['target']
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is not None:
                in_edges = graph.sorted_in_edges(
                    node_name, keys=True, data=True)
                out_edges = graph.sorted_out_edges(node_name, data=True)
                in_shapes = node_obj.get_input_shapes()
                out_shapes = node_obj.get_output_shapes()
                if len(in_edges) in (1, 2) \
                        and len(in_shapes) in (1, 2) \
                        and len(in_edges) == len(in_shapes) \
                        and all([s is not None and len(s) == 2 for s in in_shapes]) \
                        and len(out_edges) >= 1 \
                        and len(out_shapes) >= 1:
                    for in_port, (src, _, k, in_attr) in enumerate(in_edges):
                        pre_reshape_dim = [1, 1] + in_shapes[in_port]
                        insert_reshape(graph, src, node_name,
                                       in_attr, pre_reshape_dim, key=k)
                    post_reshape_dim = out_shapes[0]
                    post_reshape = get_valid_node_name(
                        graph, node_name + '_post_reshape')
                    for _, dst, out_attr in out_edges:
                        graph.remove_edge(node_name, dst)
                        graph.add_edge(post_reshape, dst, **out_attr)
                    graph.add_edge(node_name, post_reshape)

                    post_reshape_attr = node_obj.copied_attr()
                    post_reshape_attr.update(
                        {'name': post_reshape, 'opset_version': 5})
                    NodeWrap(graph, post_reshape).replace_obj(
                        'Reshape', post_reshape_attr)
                    const = get_valid_node_name(graph, post_reshape + '_shape')
                    insert_constant(graph, const, np.array(
                        post_reshape_dim, np.int64), post_reshape, in_port=1, data_format='NHWC')

                    if node_name in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(node_name)
                        graph._attr['output_names'][index] = post_reshape


def adjust_3d_to_4d(graph):
    pure_inputs_types = ['InstanceNormalization', 'LRN', 'MatMul', 'Moments']
    mixed_inputs_types = ['Resize']
    for op_type in pure_inputs_types + mixed_inputs_types:
        matches = single_node_matcher(graph, op_type)
        for m in matches:
            node_name = m['target']
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is not None:
                in_edges = graph.sorted_in_edges(
                    node_name, keys=True, data=True)
                out_edges = graph.sorted_out_edges(node_name, data=True)
                in_shapes = node_obj.get_input_shapes()
                out_shapes = node_obj.get_output_shapes()
                if (len(in_edges) in (1, 2) or node_obj.type in mixed_inputs_types) \
                        and (len(in_shapes) in (1, 2) or node_obj.type in mixed_inputs_types) \
                        and len(in_edges) == len(in_shapes) \
                        and (all([s is not None and len(s) == 3 for s in in_shapes])
                             or (node_obj.type in mixed_inputs_types and len(in_shapes[0]) == 3)) \
                        and len(out_edges) >= 1 \
                        and len(out_shapes) >= 1 \
                        and all([d is not None for shape in out_shapes for d in shape]):
                    for in_port, (src, _, k, in_attr) in enumerate(in_edges):
                        if node_obj.type in mixed_inputs_types and in_port > 0:
                            continue
                        if op_type in ('InstanceNormalization', 'LRN', 'Resize'):
                            reshape1_dim = in_shapes[in_port][0:-
                                                              1] + [1] + in_shapes[in_port][-1:]
                        else:
                            reshape1_dim = [1] + in_shapes[in_port]
                        insert_reshape(graph, src, node_name,
                                       in_attr, reshape1_dim, key=k)

                    ports_shape = OrderedDict()
                    for _, _, out_attr in out_edges:
                        if out_attr['src_out_port'] not in ports_shape:
                            ports_shape.update(
                                {out_attr['src_out_port']: list(out_attr['tensor'].value.shape)})

                    reshape2_nodes = []
                    for out_port in node_obj.get_out_ports():
                        reshape = insert_reshape_after(graph,
                                                       node_name,
                                                       ports_shape[out_port],
                                                       out_port=out_port)
                        reshape2_nodes.append(reshape)

                    if op_type == 'InstanceNormalization':
                        node_obj.non_channel_axes = [2, 3]
                    elif op_type == 'Moments':
                        moments_axes = OpHasAxis.make_axes_non_negative(node_obj.axes, len(in_shapes[0]))
                        node_obj.axes = [(axis + 1) for axis in moments_axes]
                    elif op_type == 'Resize':
                        in_edges = graph.sorted_in_edges(
                            node_name, keys=True, data=True)
                        if node_obj.scales is not None and node_obj.scales.size == 3:
                            scales = node_obj.scales.tolist()
                            scales.insert(2, 1.0)
                            scales = np.array(scales, np.float32)
                            node_obj.scales = scales
                            scales_inp, _, _, scales_in_attr = in_edges[2]
                            if scales_in_attr.get('tensor', None) is not None:
                                scales_in_attr['tensor'].value = scales
                            if NodeWrap(graph, scales_inp)['object'].type == 'Constant':
                                NodeWrap(graph, scales_inp)[
                                    'object'].value = scales
                        if node_obj.sizes is not None and node_obj.sizes.size == 3:
                            sizes = node_obj.sizes.tolist()
                            sizes.insert(2, 1)
                            sizes = np.array(sizes, np.int64)
                            node_obj.sizes = sizes
                            sizes_inp, _, _, sizes_in_attr = in_edges[3]
                            if sizes_in_attr.get('tensor', None) is not None:
                                sizes_in_attr['tensor'].value = sizes
                            if NodeWrap(graph, sizes_inp)['object'].type == 'Constant':
                                NodeWrap(graph, sizes_inp)[
                                    'object'].value = sizes

                    if node_name in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(node_name)
                        for idx, out_node in enumerate(reshape2_nodes):
                            if idx == 0:
                                graph._attr['output_names'][index] = out_node
                            else:
                                graph._attr['output_names'].insert(
                                    index, out_node)


def adjust_5d_to_4d(graph):
    convert_types = ['InstanceNormalization', ]
    for op_type in convert_types:
        matches = single_node_matcher(graph, op_type)
        for m in matches:
            node_name = m['target']
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is not None:
                in_edges = graph.sorted_in_edges(
                    node_name, keys=True, data=True)
                out_edges = graph.sorted_out_edges(node_name, data=True)
                in_shapes = node_obj.get_input_shapes()
                out_shapes = node_obj.get_output_shapes()
                if len(in_edges) == len(in_shapes) \
                        and len(out_edges) >= 1 \
                        and len(out_shapes) >= 1:
                    if op_type == 'InstanceNormalization' \
                            and in_shapes[0] is not None \
                            and len(in_shapes[0]) > 4 \
                            and NodeWrap(graph, in_edges[0][0])['object'].type == 'Transpose' \
                            and NodeWrap(graph, graph.pred[graph.pred[node_name][0]][0])['object'].type == 'Input':
                        new_w = 1
                        for i in range(2, len(in_shapes[0]) - 1):
                            new_w *= in_shapes[0][i]
                        src, _, k, in_attr = in_edges[0]
                        pre_reshape_dim = [
                            in_shapes[0][0], in_shapes[0][1], new_w, in_shapes[0][-1]]
                        insert_reshape(graph, src, node_name,
                                       in_attr, pre_reshape_dim, key=k)

                        src, dst, in_attr = out_edges[0]
                        post_reshape_dim = in_shapes[0]
                        new_out = insert_reshape(
                            graph, node_name, dst, in_attr, post_reshape_dim)
                        if node_name in graph._attr['output_names']:
                            index = graph._attr['output_names'].index(
                                node_name)
                            graph._attr['output_names'][index] = new_out


def broadcast_prelu(graph):
    op_type = 'PRelu'
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        broadcastop = m['target']
        broadcastop_obj = NodeWrap(graph, broadcastop)['object']
        in_edges = graph.sorted_in_edges(broadcastop, data=True)
        if broadcastop_obj is not None and len(in_edges) == 2:
            meta_ret = True
            in_types = [NodeWrap(graph, e[0])['object'].type for e in in_edges]
            in_tensors = [e[2]['tensor'].value for e in in_edges]
            if in_types.count('Constant') == 2:
                meta_ret = False
                WARN(
                    '[Parser]: broadcast op (%s) with Constant inputs should be fused in broadcast_prelu!' % broadcastop)
            elif len(in_tensors) == 2:
                if in_tensors[0] is not None and in_tensors[1] is not None:
                    if in_tensors[0].shape and list(in_tensors[0].shape) == list(in_tensors[1].shape):
                        pass
                    else:
                        dim_1, dim_2 = len(in_tensors[0].shape), len(
                            in_tensors[1].shape)
                        if dim_1 == dim_2:
                            input_shapes = broadcastop_obj.get_input_shapes()
                            if input_shapes[1][0] == 1:
                                reshape_shape = input_shapes[1][1:]
                                insert_reshape(
                                    graph, in_edges[1][0], broadcastop, in_edges[1][2], reshape_shape)
                            else:
                                WARN(
                                    '[Parser]: Invalid inputs of Node(%s) for broadcasting in broadcast_prelu!' % broadcastop)
                else:
                    meta_ret = False
                    WARN(
                        '[Parser]: Invalid inputs of Node(%s) for broadcasting in broadcast_prelu!' % broadcastop)
        else:
            WARN(
                '[Parser]: Meets Invalid broadcast Op (%s) that cannot be converted in broadcast_prelu!' % broadcastop)


def middle_passes(graph, params):
    '''
    Pass is an optimization based on IR to remove redundant operators and perform hardware-friendly operator transformation.
    Among them, middle_pass focuses on operator splitting and merging,
    while back_pass focuses on converting onnx operators into Arm operators defined in IR def.
    '''

    convert_to_const(graph, ['Shape', 'ConstantOfShape',
                             'Range', 'NonZero', 'EyeLike'])

    fuse_const(graph)

    convert_abnormal_reshape(graph)
    convert_fill(graph)
    convert_dequantizelinear(graph)
    convert_quantizelinear(graph)
    convert_bn_train(graph)
    clear_useless_concat_input(graph)
    remove_redundant_transpose(graph)
    merge_dilated_conv_group(graph)
    merge_dilated_conv(graph)
    remove_useless_op(graph, ['Concat',
                              'Dropout', 'Expand', 'Reshape', 'Slice', 'Transpose', 'Roll'])
    remove_redundant_transpose(graph)

    decompose_const_if(graph, params)
    rename_reshape_like(graph)

    split_negative_pads(graph)
    split_conv_transpose(graph)
    convert_to_const(graph, ['Concat', 'Mul', 'Shape',
                             'Slice', 'Tile', 'Unsqueeze'])
    merge_gelu_1(graph)
    merge_gelu_2(graph)
    merge_gelu_3(graph)

    split_special_bn(graph)
    split_hardmax(graph)
    split_reduce_logsumexp(graph)
    split_reduce_logsum(graph)
    split_reduce_sumsq(graph)
    split_roll(graph)
    convert_upsample_to_resize(graph)
    convert_special_resize(graph)
    convert_global_pool(graph)
    # merge_l2pool(graph)
    convert_special_clip_to_relu(graph)
    convert_sigmoid_mul_to_silu(graph)
    convert_sigmoid_mul_to_swish(graph)
    remove_useless_op(graph, ['Cast', 'Concat', 'Identity', 'Pad', 'Slice',
                              'Transpose', 'Reshape', 'AveragePool', 'MaxPool', 'Resize'])
    remove_sub_add_pair(graph)
    merge_leaky_relu(graph)
    merge_prelu(graph)
    reshape_prelu_slope(graph)

    merge_logical_xor(graph)
    merge_erosion(graph)
    merge_l2norm(graph)
    merge_hardswish(graph)
    merge_hardswish2(graph)
    merge_hardsigmoid(graph)
    merge_hargsigmoid2(graph)
    merge_softplus(graph)
    merge_softmax(graph)
    merge_mish(graph)
    merge_batchnorm(graph)
    merge_channel_shuffle(graph)
    merge_channel_shuffle_with_pack(graph)
    merge_ln(graph)
    merge_ln2(graph)
    merge_ln3(graph)
    merge_ln4(graph)
    merge_ln5(graph)
    merge_ln6(graph)
    merge_mvn(graph)
    merge_mvn2(graph)
    merge_mvn3(graph)
    merge_gn(graph)
    merge_gn2(graph)
    merge_in(graph)
    broadcast_ln_weights_biases(graph)
    merge_norm(graph)
    merge_ln_reshape(graph)
    merge_ln_mul_add(graph)
    duplicate_moments_mean(graph)
    merge_reduce_variance(graph)
    merge_reduce_unbiased_variance(graph)
    merge_moments(graph)
    merge_normalized_moments(graph)

    convert_gemm_to_fc(graph)
    convert_special_matmul_to_fc(graph)
    fuse_mul_add_or_sub(graph)
    fuse_gather_const_mul(graph)
    convert_gather_to_slice(graph)
    rearrange_matmul_reshape_bias(graph)
    rename_single_mul_or_add_or_sub(graph)
    rearrange_fc_reshape_bn(graph)
    fuse_pad(graph)
    fuse_bias(graph)
    fuse_linear_bn(graph)
    convert_1d_conv(graph)
    convert_reducemean_to_avgpool(graph)
    convert_1d_pooling(graph)

    decompose_pack(graph)
    remove_useless_op(graph, ['Concat', 'Split'])
    rearrange_pack_concat(graph)
    convert_min_max_to_clip(graph)
    remove_redundant_reshape(graph)
    rearrange_linear_reshape_relu(graph)
    rearrange_linear_concat_relu(graph)
    convert_special_transpose(graph)
    merge_meshgrid(graph)

    convert_einsum(graph)
    convert_nms(graph)
    align_matmul_input(graph)
    adjust_scalar_to_1d(graph)
    adjust_2d_to_4d(graph)
    adjust_3d_to_4d(graph)

    pass
