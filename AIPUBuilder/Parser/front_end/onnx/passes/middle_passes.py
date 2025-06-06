# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import itertools
import copy
from functools import reduce
from collections import OrderedDict
from ....common.defs import Tensor, FLOAT_EQUAL, FLOAT64_EQUAL, TYPE_MAX
from ....graph.graph import SubGraph
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL
from ....common.utils import extend_lists, get_converted_dtype
from ....graph.node_wrap import NodeWrap
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
# from ....graph.pattern_generator import match_patterns_from_expression
from ....graph.graph_algo import get_valid_node_name, clear_redundant_nodes, determined_sort, all_simple_paths, has_path
from ....ops.op import Op, BaseLinearOp, BaseConvOp, BaseDeconvOp, BaseOnnxPoolOp, OpHasOneOutPort, OpHasPaddingStrides, \
    OpHasAxis, \
    OnnxOp, CommonOp, OpNeedBroadcast, OpNeedUniBroadcast, OnnxReduceOp
from ....ops.onnx_ops.array_ops import ReshapeOp, CenterCropPadOp
from ....ops.onnx_ops.nn_ops import DeformConvOp
from .common_passes import fuse_const, remove_useless_op, remove_node_safely, insert_reshape, insert_reshape_after, \
    insert_cast, insert_constant, insert_slice, insert_slice_after, insert_tile, insert_transpose, \
    insert_transpose_after, insert_gather, \
    remove_redundant_reshape, remove_redundant_transpose, insert_cast_sub_mul_for_quant, \
    insert_mul_add_cast_after_for_dequant, \
    insert_repeat, convert_to_const, insert_cast_after, merge_same_op_at_out_port, convert_multi_outputs_to_const


def clear_useless_concat_input(graph):
    matched = False
    matches = single_node_matcher(graph, 'Concat')
    for m in matches:
        concat = m['target']
        concat_obj = NodeWrap(graph, concat)['object']
        if concat_obj is None:
            ERROR(
                '[Parser]: Meets invalid Concat Op (%s) in clear_useless_concat_input!' % concat)
            continue
        in_edges = graph.sorted_in_edges(concat, keys=True)
        if len(in_edges) < 2:
            continue
        input_shapes = concat_obj.get_input_shapes()
        if len(input_shapes) != len(in_edges):
            ERROR(
                '[Parser]: Meets invalid Concat Op (%s) in clear_useless_concat_input!' % concat)
            continue
        if any([s is None for s in input_shapes]):
            ERROR('[Parser]: Meets invalid input shape for Concat Op (%s) in clear_useless_concat_input!' % concat)
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
            node_obj.convert_version()
        else:
            ERROR('[Parser]: Meets invalid node(%s) in convert_onnx_version!' % node)


def convert_1d_conv(graph):
    for conv_type in ['Conv', 'ConvTranspose']:
        matches = single_node_matcher(graph, conv_type)
        for m in matches:
            conv = m['target']
            conv_obj = NodeWrap(graph, conv)['object']
            if conv_obj is None:
                ERROR('[Parser]: Meets invalid Conv node(%s) in convert_1d_conv!' % conv)
                continue
            if conv_obj.weights is not None and len(conv_obj.weights.shape) == 3:
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
                    insert_reshape(graph, src, conv, in_attr, reshape1_dim, quantize=conv_obj.quantize)

                    reshape2 = get_valid_node_name(
                        graph, conv + '_post_reshape')
                    for _, dst, out_attr in out_edges:
                        graph.remove_edge(conv, dst)
                        graph.add_edge(reshape2, dst, **out_attr)
                    conv_out_tensor = None
                    conv_out_attr = out_edges[0][2]
                    if conv_out_attr['tensor'] is not None:
                        conv_out_tensor = copy.deepcopy(conv_out_attr['tensor'])
                        if conv_out_tensor.value is not None:
                            conv_out_tensor.value = np.expand_dims(conv_out_tensor.value, 1 if is_channels_last else 2)
                    graph.add_edge(conv, reshape2, **{'tensor': conv_out_tensor})

                    reshape2_shape_const = get_valid_node_name(
                        graph, reshape2 + '_shape')
                    insert_constant(graph, reshape2_shape_const, np.array(
                        reshape2_dim, np.int32), reshape2, in_port=1, data_format='NHWC')
                    reshape2_attr = conv_obj.copied_attr()
                    reshape2_attr.update(
                        {'name': reshape2, 'opset_version': 5})
                    NodeWrap(graph, reshape2).replace_obj(
                        'Reshape', reshape2_attr)

                    conv_attr = conv_obj.copied_attr()
                    conv_attr['weights'] = np.expand_dims(
                        conv_attr['weights'], axis=2)
                    conv_attr['kernel_shape'] = [1] + conv_obj.kernel_shape
                    conv_attr['strides'] = [1] + conv_obj.strides
                    conv_attr['dilations'] = [1] + conv_obj.dilations
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
                    if pool_obj.data_format == 'NHWC':
                        pre_reshape_dim = [in_shape[0], 1] + in_shape[1:]
                        post_old_dim = [out_shape[0], 1] + out_shape[1:]
                    else:
                        pre_reshape_dim = in_shape[0:2] + [1, in_shape[-1]]
                        post_old_dim = out_shape[0:2] + [1, out_shape[-1]]
                    post_reshape_dim = out_shape
                    quantize = pool_obj.quantize
                    # Get attributes firstly, then update their value because default value of attributes
                    # can be affected by others(default value of dilations changes if kernel_shape updates).
                    kernel_shape = pool_obj.kernel_shape
                    strides = pool_obj.strides
                    dilations = pool_obj.dilations
                    pool_obj.pads = np.concatenate([np.array([[0], [0]]), np.transpose(
                        np.array([pool_obj.pads]))], axis=1).flatten().tolist()
                    if kernel_shape:
                        pool_obj.kernel_shape = [1] + kernel_shape
                    if strides:
                        pool_obj.strides = [1] + strides
                    if dilations:
                        pool_obj.dilations = [1] + dilations
                    src, _, in_attr = in_edges[0]
                    insert_reshape(
                        graph, src, pool, in_attr, pre_reshape_dim, data_format=pool_obj.data_format,
                        quantize=quantize)
                    post_reshape = insert_reshape_after(graph, pool, post_reshape_dim, old_dim=post_old_dim,
                                                        quantize=quantize)
                    if pool in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(pool)
                        graph._attr['output_names'][index] = post_reshape
            else:
                ERROR(
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
                and len(fusebnv3_obj.get_input_shapes()) == 5:
            if fusebnv3_obj.training_mode is False or fusebnv3_obj.quantize:
                continue
            input_shapes = fusebnv3_obj.get_input_shapes()
            x_shape = input_shapes[0]
            if x_shape is None or None in x_shape:
                ERROR(
                    '[Parser]: Meets invalid input shapes of BatchNormalization node(%s) in convert_bn_train!' % fusebnv3)
                continue
            input_dtypes = fusebnv3_obj.get_input_dtypes()
            if len(input_dtypes) < 5 or input_dtypes[0] is None or input_dtypes[3] is None:
                ERROR(
                    '[Parser]: Meets invalid input dtypes of BatchNormalization node(%s) in convert_bn_train!' % fusebnv3)
                continue
            matched = True
            x_src, _, x_out_attr = fusebnv3_in_edges[0]
            scale, _, scale_out_attr = fusebnv3_in_edges[1]
            offset, _, offset_out_attr = fusebnv3_in_edges[2]
            input_mean, _, inmean_out_attr = fusebnv3_in_edges[3]
            input_var, _, invar_out_attr = fusebnv3_in_edges[4]
            x_shape = tuple(x_shape)
            inp_rank = len(x_shape)
            eps = fusebnv3_obj.epsilon
            momentum = fusebnv3_obj.momentum
            input_dtype = input_dtypes[0]
            mean_var_dtype = input_dtypes[3]
            if fusebnv3_obj.data_format == 'NCHW':
                dims = [0] + list(range(2, inp_rank))
                reshape_dim = [x_shape[1]] + [1] * (inp_rank - 2)
                weights_shape = tuple([x_shape[1]])
            else:
                dims = list(range(0, inp_rank - 1))
                reshape_dim = [1] * (inp_rank - 2) + [x_shape[-1]]
                weights_shape = tuple([x_shape[-1]])
            # Step 1: Consider output Y
            # current_mean_value -> onnx input_mean; current_var_value -> onnx input_var
            inp_shape = np.array(x_shape, np.dtype(np.int32))
            reduce_dims = np.take(inp_shape, np.array(dims, np.int32), axis=0)
            cnt_float = np.prod(reduce_dims, axis=tuple(
                [0]), keepdims=False).astype(mean_var_dtype)
            # current_mean_value = np.mean(x, axis=tuple(dims), keepdims=True)
            current_mean_shape = tuple([1 if shape in dims else shape for shape in x_shape])
            # sub_value = np.subtract(x, current_mean_value)
            # var_squeezed_value = np.sum(
            #     np.square(sub_value), axis=tuple(dims), keepdims=False)
            # current_var_value = np.true_divide(var_squeezed_value, cnt_float)
            # # weights_value -> onnx 1/sqrt(input_var+eps)=(input_var+eps)^(-0.5)
            # current_var_eps_value = np.add(current_var_value, eps)
            # sqrt_value = np.sqrt(current_var_eps_value)
            # weights_value = np.true_divide(1, sqrt_value)
            # reshaped_weights_value = np.reshape(weights_value, reshape_dim)
            # # y = reshaped_weights_value * (x - input_mean) * scale + B
            # #   = reshaped_weights_value * sub_value * scale + offset
            # #   = reshaped_weights_value * sub_value * inputs[1] + inputs[2]
            # mul_sub_value = reshaped_weights_value * sub_value

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
            current_mean_out_attr = {'src_out_port': 0, 'dst_in_port': 1, 'tensor': Tensor(shape=current_mean_shape)}
            graph.add_edge(current_mean, sub, **current_mean_out_attr)
            graph.add_edge(sub, var_squeezed)
            graph.add_edge(var_squeezed, current_var, **
                           {'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(shape=weights_shape)})
            graph.add_edge(current_var, current_var_eps, **{'tensor': Tensor(shape=weights_shape)})
            graph.add_edge(current_var_eps, weights, **{'tensor': Tensor(shape=weights_shape)})

            # reshaped_weights_value * sub
            graph.add_edge(sub, mul_sub, **{'tensor': Tensor(shape=x_shape)})
            graph.add_edge(weights, mul_sub, **
                           {'src_out_port': 0, 'dst_in_port': 1, 'tensor': Tensor(shape=weights_shape)})
            # reshaped_weights_value * sub * scale
            graph.add_edge(mul_sub, mul_scale, **{'tensor': Tensor(shape=x_shape)})
            graph.add_edge(scale, mul_scale, **scale_out_attr)
            # reshaped_weights_value * sub * scale + offset
            offset_out_attr = copy.deepcopy(offset_out_attr)
            offset_out_attr.update({'dst_in_port': 1})
            graph.add_edge(mul_scale, y, **{'tensor': Tensor(shape=x_shape)})
            graph.add_edge(offset, y, **offset_out_attr)

            insert_constant(graph, current_var_eps + '_add', np.array(eps).astype(mean_var_dtype),
                            current_var_eps, in_port=1, data_format='NHWC')
            insert_constant(graph, weights + '_pow', np.array(-0.5).astype(mean_var_dtype),
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
                mul_sub_in_attr = {'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(shape=weights_shape)}
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
            insert_constant(graph, mul_in_mean + '_momentum', np.array(momentum).astype(mean_var_dtype),
                            mul_in_mean, in_port=1, data_format='NHWC')
            # reshaped_current_mean * (1 - momentum)
            graph.add_node(mul_cur_mean)
            mul_cur_mean_in_attr = {'dst_in_port': 0, 'tensor': Tensor(shape=current_mean_shape)}
            reshaped_current_mean = insert_reshape(graph, current_mean, mul_cur_mean, mul_cur_mean_in_attr,
                                                   weights_shape) \
                if fusebnv3_obj.data_format == 'NCHW' else current_mean
            insert_constant(graph, mul_cur_mean + '_momentum', np.array(1 - momentum).astype(mean_var_dtype),
                            mul_cur_mean, in_port=1, data_format='NHWC')
            # input_mean * momentum + reshaped_current_mean * (1 - momentum)
            mul_in_mean_value = None if mul_in_mean_in_attr['tensor'].value is None \
                else (mul_in_mean_in_attr['tensor'].value * momentum)
            run_mean_in_attr = copy.deepcopy(inmean_out_attr)
            run_mean_in_attr.update({'src_out_port': 0, 'dst_in_port': 0})
            if mul_in_mean_value is not None:
                run_mean_in_attr['tensor'].value = mul_in_mean_value
            else:
                run_mean_in_attr['tensor'].shape = weights_shape
            graph.add_edge(mul_in_mean, running_mean, **run_mean_in_attr)
            run_mean_in_attr1 = {'dst_in_port': 1, 'tensor': Tensor(shape=weights_shape)}
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
            insert_constant(graph, mul_in_var + '_momentum', np.array(momentum).astype(mean_var_dtype),
                            mul_in_var, in_port=1, data_format='NHWC')
            # current_var * (1 - momentum)
            graph.add_edge(current_var, mul_cur_var, **{'tensor': Tensor(shape=weights_shape)})
            insert_constant(graph, mul_cur_var + '_momentum', np.array(1 - momentum).astype(mean_var_dtype),
                            mul_cur_var, in_port=1, data_format='NHWC')
            # input_var * momentum + current_var * (1 - momentum)
            mul_in_var_value = None if mul_in_var_in_attr['tensor'].value is None \
                else mul_in_var_in_attr['tensor'].value * momentum
            run_var_in_attr = copy.deepcopy(invar_out_attr)
            run_var_in_attr.update({'src_out_port': 0, 'dst_in_port': 0})
            if mul_in_var_value is not None:
                run_var_in_attr['tensor'].value = mul_in_var_value
            else:
                run_var_in_attr['tensor'].shape = weights_shape
            graph.add_edge(mul_in_var, running_var, **run_var_in_attr)
            graph.add_edge(mul_cur_var, running_var, **{'dst_in_port': 1, 'tensor': Tensor(shape=weights_shape)})

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
            ERROR(
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
                    ERROR(
                        '[Parser]: box_shape or score_shape of node (%s) is None.' % nms)
                    continue

                onnx_batch = box_shape[0]
                box_num = box_shape[1]
                class_num = score_shape[1]
                max_output_boxes_per_class = nms_obj.max_output_boxes_per_class
                if class_num > 1 or onnx_batch > 1:
                    WARN(
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
                    need_insert_num = min(need_insert_num1,
                                          need_insert_num2) if need_insert_num1 > 0 and need_insert_num2 > 0 else max(
                        need_insert_num1, need_insert_num2)
                    complete_res = np.zeros(need_insert_num)
                    complete_res = np.reshape(complete_res, (-1, 1))
                    batch_list = np.concatenate(
                        [batch_list, complete_res], axis=0)
                    class_num_list = np.concatenate(
                        [class_num_list, complete_res], axis=0)

                # Manipulate edges and nodes
                if center_box is True:
                    origin_box_num = box_shape[1]
                    add_num = np.array(
                        onnx_batch * [[-0.5] * origin_box_num + [-0.5] * origin_box_num +
                                      [0.5] * origin_box_num + [0.5] * origin_box_num],
                        dtype=np.float32)
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
                    'Slice',
                    {'name': post_slice, 'opset_version': 1, 'starts': post_slice_start, 'ends': post_slice_end,
                     'steps': post_slice_step})
                NodeWrap(graph, post_reshape).replace_obj(
                    'Reshape', {'name': post_reshape, 'opset_version': 1, 'shape': [-1, 1]})
                NodeWrap(graph, post_concatenate).replace_obj(
                    'Concat', {'name': post_concatenate, 'opset_version': 11, 'axis': 1})

                if nms in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(nms)
                    graph._attr['output_names'].remove(nms)
                    graph._attr['output_names'].insert(index, post_concatenate)

            else:
                ERROR('[Parser]: The in_edge length of the node (%s) is illegal.' % nms)
        else:
            ERROR('[Parser]: Meets invalid node (%s) in NonMaxSuppression!' % nms)


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
            ERROR('[Parser]: Meets invalid Node in convert_sigmoid_mul_to_swish!')
            continue

        mul1_in_edges = graph.sorted_in_edges(m['mul1'], data=True)
        mul1_out_edges = graph.sorted_out_edges(m['mul1'], data=True)
        mul2_in_edges = graph.sorted_in_edges(m['mul2'], data=True)
        sigmoid_out_edges = graph.sorted_out_edges(m['sigmoid'], data=True)
        if len(mul1_in_edges) == 2 \
                and len(mul2_in_edges) == 2 \
                and len(mul1_out_edges) == 1 \
                and len(sigmoid_out_edges) == 1 \
                and mul1_in_edges[1][2]['tensor'].is_const \
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
                    and np.ndim(in_consts[0][2]) in (0, 1) \
                    and in_consts[0][2].size == 1:
                graph.remove_edges_from(in_edges[1:])
                in_shape = input_shapes[0]
                indices_rank = np.ndim(in_consts[0][2])
                indices = in_consts[0][2].item()
                indices = in_shape[gather_obj.axis] + indices if indices < 0 else indices

                starts = [0] * len(in_shape)
                ends = copy.deepcopy(in_shape)
                starts[gather_obj.axis] = int(indices)
                ends[gather_obj.axis] = starts[gather_obj.axis] + 1
                axes = list(range(len(in_shape)))
                slice_attr = gather_obj.copied_attr()
                slice_attr.update(
                    {'name': gather, 'opset_version': 1, 'axes': axes, 'starts': starts, 'ends': ends})
                NodeWrap(graph, gather).replace_obj('Slice', slice_attr)
                if indices_rank == 0:  # No need to reshape if indices_rank is 1(q=1) because output rank=q+(r-1)=r
                    old_dim = np.array(ends, np.int64) - np.array(starts, np.int64)
                    reshape_dim = np.delete(old_dim, gather_obj.axis)
                    reshape = insert_reshape_after(graph, gather, reshape_dim.tolist(), old_dim.tolist(),
                                                   quantize=gather_obj.quantize)
                    if gather in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(gather)
                        graph._attr['output_names'].remove(gather)
                        graph._attr['output_names'].insert(index, reshape)


def convert_gemm_to_fc(graph):
    matches = single_node_matcher(graph, 'Gemm')
    for m in matches:
        gemm = m['target']
        gemm_obj = NodeWrap(graph, gemm)['object']
        if gemm_obj is None:
            ERROR('[Parser]: Meets invalid Gemm (%s) in convert_gemm_to_fc!' % gemm)
            continue
        is_quantized = gemm_obj.quantize
        if is_quantized and (not FLOAT_EQUAL(gemm_obj.alpha, 1) or not FLOAT_EQUAL(gemm_obj.beta, 1)):
            continue
        gemm_in_edges = graph.sorted_in_edges(gemm, data=True)
        if len(gemm_in_edges) in (2, 3) \
                and gemm_in_edges[0][2]['tensor'] is not None \
                and gemm_in_edges[0][2]['tensor'].shape is not None \
                and len(gemm_in_edges[0][2]['tensor'].shape) == 2:
            input2 = gemm_in_edges[1][0]
            input3 = gemm_in_edges[2][0] if len(gemm_in_edges) == 3 else ''
            if NodeWrap(graph, input2)['object'].type == 'Constant' \
                    and (not input3 or NodeWrap(graph, input3)['object'].type == 'Constant'):
                W = NodeWrap(graph, input2)['object'].value
                if bool(gemm_obj.transB):
                    W = np.transpose(W)
                num_output = W.shape[-1]
                b = NodeWrap(graph, input3)[
                    'object'].value if input3 else np.zeros((num_output,), np.float32)
                fc_attr = gemm_obj.copied_attr()
                if is_quantized:
                    b = np.array(b, np.int32)
                    if len(gemm_in_edges) == 3:
                        biases_scale_zp = list(gemm_in_edges[2][2]['tensor'].scale_zp)
                    else:
                        biases_scale_zp = [np.array(1.0, dtype=np.float32), np.array(0, dtype=np.int32)]
                    fc_attr.update({'weights_scale_zp': list(gemm_in_edges[1][2]['tensor'].scale_zp),
                                    'biases_scale_zp': biases_scale_zp})
                else:
                    W = W * gemm_obj.alpha
                    b = np.array(b * gemm_obj.beta, dtype=np.float32)
                fc_attr.update({'weights': np.transpose(W), 'biases': b})
                NodeWrap(graph, gemm).replace_obj('FullyConnected', fc_attr)
                graph.remove_edge(input2, gemm)
                if input3:
                    graph.remove_edge(input3, gemm)
                if bool(gemm_obj.transA):
                    src, _, in_attr = gemm_in_edges[0]
                    perm = [1, 0]
                    insert_transpose(graph, src, gemm, in_attr, perm)


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
    def _warn_unsupported(node_name, equation):
        WARN('[Parser]: This equation(%s) of Einsum(%s) is currently not supported in convert_einsum!' %
             (equation, node_name))
        return

    def _is_consecutive(nums):
        return set(nums) == set(range(min(nums), max(nums) + 1))

    matches = single_node_matcher(graph, 'Einsum')
    for m in matches:
        einsum = m['target']
        einsum_obj = NodeWrap(graph, einsum)['object']
        if einsum_obj is not None and einsum_obj.equation is not None:
            in_edges = graph.sorted_in_edges(einsum, data=True)
            if len(in_edges) == 2:
                equation = einsum_obj.equation.replace(' ', '')
                equ_list = equation.split('->')
                add_list = equ_list[0].split(',')
                if len(add_list) != 2:
                    ERROR('[Parser]: Meets invalid equation in convert_einsum!')
                    continue
                if len(equ_list) == 1:
                    if len(add_list[0]) == len(add_list[1]):
                        if len(add_list[0]) == 1:
                            in_shapes = einsum_obj.get_input_shapes()
                            if add_list[0] == add_list[1]:
                                # 'i,i'
                                reshape0_shape = [1, in_shapes[0][0]]
                                reshape1_shape = [in_shapes[1][0], 1]
                                insert_reshape(graph, in_edges[0][0], einsum, in_edges[0][-1],
                                               reshape0_shape, quantize=einsum_obj.quantize)
                                insert_reshape(graph, in_edges[1][0], einsum, in_edges[1][-1],
                                               reshape1_shape, quantize=einsum_obj.quantize)
                                post_reshape = insert_reshape_after(
                                    graph, einsum, [], old_dim=[1, 1], quantize=einsum_obj.quantize)
                                matmul_attr = einsum_obj.copied_attr()
                                matmul_attr.update({'opset_version': 13})
                                NodeWrap(graph, einsum).replace_obj('MatMul', matmul_attr)
                                if einsum in graph._attr['output_names']:
                                    index = graph._attr['output_names'].index(einsum)
                                    graph._attr['output_names'][index] = post_reshape
                            else:
                                # 'i,j'
                                reshape0_shape = [in_shapes[0][0], 1]
                                reshape1_shape = [1, in_shapes[1][0]]
                                insert_reshape(graph, in_edges[0][0], einsum, in_edges[0][-1],
                                               reshape0_shape, quantize=einsum_obj.quantize)
                                insert_reshape(graph, in_edges[1][0], einsum, in_edges[1][-1],
                                               reshape1_shape, quantize=einsum_obj.quantize)
                                matmul_attr = einsum_obj.copied_attr()
                                matmul_attr.update({'opset_version': 13})
                                NodeWrap(graph, einsum).replace_obj('MatMul', matmul_attr)
                        elif len(add_list[0]) == 2 and add_list[0][-1] == add_list[1][-2]:
                            # 'ij,jk'
                            # TODO: support batch matmul
                            matmul_attr = einsum_obj.copied_attr()
                            matmul_attr.update({'opset_version': 13})
                            NodeWrap(graph, einsum).replace_obj('MatMul', matmul_attr)
                        else:
                            _warn_unsupported(einsum, equation)
                    else:
                        _warn_unsupported(einsum, equation)
                    continue
                out_term = equ_list[1]
                if len(add_list[1]) >= 2 and len(add_list[0]) >= 2 \
                        and set(add_list[0]).symmetric_difference(add_list[1]) == set(out_term) \
                        and ''.join([in0 for in0, out in zip(add_list[0], out_term) if in0 == out]) + \
                        ''.join(reversed([in1 for in1, out in zip(reversed(add_list[1]), reversed(out_term)) if
                                          in1 == out])) == out_term:
                    # abcd,cdef->abef
                    in_edges = graph.sorted_in_edges(einsum, data=True)
                    in_shapes = einsum_obj.get_input_shapes()
                    if len(in_edges) < 2 or len(in_shapes) < 2 \
                            or any(shape is None or None in shape for shape in in_shapes):
                        ERROR('[Parser]: Meets invalid inputs of Einsum(%s) in convert_einsum!' % einsum)
                        continue
                    in0_keep_dim = [s for s in add_list[0] if s in out_term]  # [a,b]
                    in0_keep_shape = in_shapes[0][:len(in0_keep_dim)]  # [a,b]
                    in0_shape = in0_keep_shape + [-1]  # [a,b,-1]
                    in1_reduce_dim = [s for s in add_list[1] if s not in out_term]  # [c,d]
                    in1_keep_shape = in_shapes[1][len(in1_reduce_dim):]  # [e,f]
                    in1_keep_prod = int(np.prod(in1_keep_shape))  # e*f
                    in1_shape = [-1, in1_keep_prod]  # [-1,e*f]
                    matmul_out_shape = in0_keep_shape + [in1_keep_prod]  # [a,b,e*f]
                    einsum_out_shape = in0_keep_shape + in1_keep_shape  # [a,b,e,f]
                    in0, _, in0_attr = in_edges[0]
                    in1, _, in1_attr = in_edges[1]
                    quantize = einsum_obj.quantize
                    insert_reshape(graph, in0, einsum, in0_attr, in0_shape, quantize=quantize)
                    insert_reshape(graph, in1, einsum, in1_attr, in1_shape, quantize=quantize)
                    post_reshape = insert_reshape_after(
                        graph, einsum, einsum_out_shape, old_dim=matmul_out_shape, quantize=quantize)
                    matmul_attr = einsum_obj.copied_attr()
                    matmul_attr.update({'opset_version': 13})
                    NodeWrap(graph, einsum).replace_obj('MatMul', matmul_attr)
                    if einsum in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(einsum)
                        graph._attr['output_names'][index] = post_reshape
                elif len(add_list[1]) >= 2 and len(add_list[0]) >= 2 \
                        and out_term[-2] in add_list[0][-2:] and out_term[-1] in add_list[1][-2:]:
                    a_equ = list(add_list[0])
                    b_equ = list(add_list[1])
                    # find disappeared shape in out_term
                    dis_shapes = set(add_list[0]) & set(add_list[1]) - set(out_term)
                    dis_shape = None
                    if len(dis_shapes) > 1:
                        for s in list(dis_shapes):
                            if s in a_equ[-2:] and s in b_equ[-2:]:
                                dis_shape = s
                    elif len(dis_shapes) == 1:
                        dis_shape = list(dis_shapes)[0]
                    else:
                        _warn_unsupported(einsum, equation)
                        continue

                    if not dis_shape:
                        _warn_unsupported(einsum, equation)
                        continue
                    idx_0 = a_equ.index(dis_shape)
                    idx_1 = b_equ.index(dis_shape)
                    in_edges = graph.sorted_in_edges(einsum, data=True)
                    if idx_0 != len(a_equ) - 1:
                        perm = list(range(len(a_equ)))
                        perm[idx_0] = len(a_equ) - 1
                        perm[-1] = idx_0
                        tmp_equ = a_equ.copy()
                        tmp_equ[idx_0] = a_equ[-1]
                        tmp_equ[-1] = a_equ[idx_0]
                        a_equ = tmp_equ
                        ein_src0, _, ein_in_attr0 = in_edges[0]
                        insert_transpose(graph, ein_src0, einsum, ein_in_attr0, perm,
                                         quantize=einsum_obj.quantize)
                    if idx_1 != len(b_equ) - 2:
                        perm = list(range(len(b_equ)))
                        perm[idx_1] = len(b_equ) - 2
                        perm[-2] = idx_1
                        tmp_equ = b_equ.copy()
                        tmp_equ[idx_1] = b_equ[-2]
                        tmp_equ[-2] = b_equ[idx_1]
                        b_equ = tmp_equ
                        ein_src1, _, ein_in_attr1 = in_edges[1]
                        insert_transpose(graph, ein_src1, einsum, ein_in_attr1, perm,
                                         quantize=einsum_obj.quantize)

                    # bsht, bthd-> bshd ---> bsht, bhtd
                    while a_equ[:-2] != b_equ[:-2]:
                        if len(a_equ) != len(b_equ):
                            in_edges = graph.sorted_in_edges(einsum, data=True)
                            a_input_shape = list(in_edges[0][-1]['tensor'].shape)
                            b_input_shape = list(in_edges[1][-1]['tensor'].shape)
                            if len(a_equ) < len(b_equ):
                                diff_idx = -1
                                for i, (_a, _b) in enumerate(list(zip(a_equ, b_equ))):
                                    if _a != _b:
                                        diff_idx = i
                                        break
                                if diff_idx == -1:
                                    a_input_shape.insert(-1, 1)
                                    insert_reshape(graph, in_edges[0][0], einsum, in_edges[0][-1],
                                                   a_input_shape,
                                                   quantize=einsum_obj.quantize)
                                    a_equ.insert(-1, 1)
                                else:
                                    a_input_shape.insert(diff_idx, 1)
                                    reshape0 = insert_reshape(graph, in_edges[0][0], einsum, in_edges[0][-1],
                                                              a_input_shape,
                                                              quantize=einsum_obj.quantize)
                                    if b_equ[diff_idx] == out_term[diff_idx]:
                                        tile_reps = [1] * len(a_input_shape)
                                        tile_reps[diff_idx] = b_input_shape[diff_idx]
                                        _, _, reshape0_out_attr = graph.sorted_out_edges(reshape0, data=True)[0]
                                        insert_tile(graph, reshape0, einsum, reshape0_out_attr, tile_reps,
                                                    quantize=einsum_obj.quantize)
                                        a_equ.insert(diff_idx, b_equ[diff_idx])
                                    else:
                                        a_equ.insert(diff_idx, 1)
                            else:
                                diff_idx = -1
                                for i, (_a, _b) in enumerate(list(zip(a_equ, b_equ))):
                                    if _a != _b:
                                        diff_idx = i
                                        break
                                if diff_idx == -1:
                                    b_input_shape.append(1)
                                    insert_reshape(graph, in_edges[1][0], einsum, in_edges[1][-1],
                                                   b_input_shape,
                                                   quantize=einsum_obj.quantize)
                                    b_equ.append(1)
                                else:
                                    b_input_shape.insert(diff_idx, 1)
                                    reshape0 = insert_reshape(graph, in_edges[1][0], einsum, in_edges[1][-1],
                                                              b_input_shape,
                                                              quantize=einsum_obj.quantize)
                                    if a_equ[diff_idx] == out_term[diff_idx]:
                                        tile_reps = [1] * len(b_input_shape)
                                        tile_reps[diff_idx] = a_input_shape[diff_idx]
                                        _, _, reshape0_out_attr = graph.sorted_out_edges(reshape0, data=True)[0]
                                        insert_tile(graph, reshape0, einsum, reshape0_out_attr, tile_reps,
                                                    quantize=einsum_obj.quantize)
                                        b_equ.insert(diff_idx, a_equ[diff_idx])
                                    else:
                                        b_equ.insert(diff_idx, 1)
                        a_prefix = a_equ[:-2]
                        b_prefix = b_equ[:-2]
                        for i, (_a, _b) in enumerate(list(zip(a_prefix, b_prefix))):
                            if _a != _b:
                                out = out_term[i]
                                in_edges = graph.sorted_in_edges(einsum, data=True)
                                a_input_shape = list(in_edges[0][-1]['tensor'].shape)
                                b_input_shape = list(in_edges[1][-1]['tensor'].shape)
                                if _a == out:
                                    b_input_shape.insert(i, 1)
                                    tile_reps = [1] * len(b_input_shape)
                                    tile_reps[i] = a_input_shape[i]

                                    reshape0 = insert_reshape(graph, in_edges[1][0], einsum, in_edges[1][-1],
                                                              b_input_shape,
                                                              quantize=einsum_obj.quantize)
                                    _, _, reshape0_out_attr = graph.sorted_out_edges(reshape0, data=True)[0]
                                    insert_tile(graph, reshape0, einsum, reshape0_out_attr, tile_reps,
                                                quantize=einsum_obj.quantize)
                                    b_equ.insert(i, _a)
                                elif _b == out:
                                    a_input_shape.insert(i, 1)
                                    tile_reps = [1] * len(a_input_shape)
                                    tile_reps[i] = b_input_shape[i]

                                    reshape0 = insert_reshape(graph, in_edges[0][0], einsum, in_edges[0][-1],
                                                              a_input_shape,
                                                              quantize=einsum_obj.quantize)
                                    _, _, reshape0_out_attr = graph.sorted_out_edges(reshape0, data=True)[0]
                                    insert_tile(graph, reshape0, einsum, reshape0_out_attr, tile_reps,
                                                quantize=einsum_obj.quantize)
                                    a_equ.insert(i, _b)
                                else:
                                    _warn_unsupported(einsum, equation)
                    new_out_term = a_equ[:-1] + b_equ[-1:]
                    if new_out_term != list(out_term) and list(set(new_out_term) - set(out_term)) == [1]:
                        need_post_reshape = True
                    else:
                        need_post_reshape = False
                    matmul_attr = einsum_obj.copied_attr()
                    matmul_attr.update({'opset_version': 13})
                    NodeWrap(graph, einsum).replace_obj('MatMul', matmul_attr)

                    if need_post_reshape:
                        in_edges = graph.sorted_in_edges(einsum, data=True)
                        a_input_shape = list(in_edges[0][-1]['tensor'].shape)
                        b_input_shape = list(in_edges[1][-1]['tensor'].shape)
                        matmul_out_shape = a_input_shape[:-1] + b_input_shape[-1:]
                        out_shape = einsum_obj.get_output_shapes()[0]
                        post_reshape = insert_reshape_after(
                            graph, einsum, out_shape, old_dim=matmul_out_shape, quantize=einsum_obj.quantize)
                        if einsum in graph._attr['output_names']:
                            index = graph._attr['output_names'].index(einsum)
                            graph._attr['output_names'][index] = post_reshape

                    if add_list[0][:-2] == add_list[1][:-2] and add_list[0][:-2] != '...' and \
                            add_list[0][:-2] != out_term[:-2]:
                        # need reduce sum
                        axes = []
                        origin_list = add_list[0][:-2]
                        dst_list = out_term[:-2]
                        for axis, v in enumerate(origin_list):
                            if v not in dst_list:
                                axes.append(axis)
                        out_edges = graph.sorted_out_edges(einsum, data=True)
                        reduce_sum = get_valid_node_name(graph, f'{einsum}_ReduceSum')
                        graph.add_node(reduce_sum)
                        reduce_sum_attr = einsum_obj.copied_attr()
                        reduce_sum_attr.update({'name': reduce_sum, 'opset_version': 11, 'keepdims': 0,
                                                'axes': axes})
                        NodeWrap(graph, reduce_sum).replace_obj('ReduceSum', reduce_sum_attr)

                        for src, dst, out_attr in out_edges:
                            graph.remove_edge(src, dst)
                            new_out_attr = copy.deepcopy(out_attr)
                            new_out_attr['src_out_port'] = 0
                            graph.add_edge(reduce_sum, dst, **new_out_attr)

                        in_shapes = einsum_obj.get_input_shapes()
                        graph.add_edge(einsum, reduce_sum, **{'src_out_port': 0, 'dst_in_port': 0,
                                                              'tensor': Tensor(
                                                                  shape=in_shapes[0][:-1] + [in_shapes[1][-1]])})

                        if einsum in graph._attr['output_names']:
                            index = graph._attr['output_names'].index(einsum)
                            graph._attr['output_names'][index] = reduce_sum
                elif len(add_list[1]) >= 2 and len(add_list[0]) >= 2 \
                        and len(add_list[0]) == len(add_list[1]) \
                        and len(set(add_list[0])) == len(set(add_list[1])) \
                        and len(add_list[0]) == len(out_term):  # abcd, aecd -> acbe(sum out d), or aebd(sum out c.)...
                    src0_sum_out_dim = list(set(add_list[0]).difference(out_term))
                    src1_sum_out_dim = list(set(add_list[1]).difference(out_term))
                    # make sure only 1 summing out dim
                    if len(src0_sum_out_dim) != 1 or len(src1_sum_out_dim) != 1 \
                            or src0_sum_out_dim[0] != src1_sum_out_dim[0]:
                        _warn_unsupported(einsum, equation)
                        continue
                    sum_out_dim = src0_sum_out_dim[0]
                    src0_dim_not_in_src1 = list(set(add_list[0]).difference(add_list[1]))
                    src1_dim_not_in_src0 = list(set(add_list[1]).difference(add_list[0]))
                    if len(src0_dim_not_in_src1) != 1 or len(src1_dim_not_in_src0) != 1:
                        _warn_unsupported(einsum, equation)
                        continue
                    src0_dim_not_in_src1 = src0_dim_not_in_src1[0]
                    src1_dim_not_in_src0 = src1_dim_not_in_src0[0]
                    ein_src0, _, ein0_in_attr = in_edges[0]
                    ein_src1, _, ein1_in_attr = in_edges[1]
                    # if out_perm is acbe, then src0(abcd) should convert to acbd
                    src0_target_shape = add_list[0].replace(sum_out_dim, '').replace(
                        src0_dim_not_in_src1, '') + src0_dim_not_in_src1 + sum_out_dim
                    if src0_target_shape != add_list[0]:
                        src0_perm = [add_list[0].index(shape) for shape in src0_target_shape]
                        insert_transpose(graph, ein_src0, einsum, ein0_in_attr,
                                         src0_perm, quantize=einsum_obj.quantize)
                    # if out_perm is acbe, then src1(aecd) should convert to acde
                    src1_target_shape = src0_target_shape[:-2] + sum_out_dim + src1_dim_not_in_src0
                    if src1_target_shape != add_list[1]:
                        src1_perm = [add_list[1].index(shape) for shape in src1_target_shape]
                        insert_transpose(graph, ein_src1, einsum, ein1_in_attr,
                                         src1_perm, quantize=einsum_obj.quantize)
                    # for inputs abcd and aecd, after transpose, they become acbd and acde, then matmul out shape is acbe
                    matmul_out_shape = src0_target_shape[:-1] + src1_dim_not_in_src0
                    if matmul_out_shape != out_term:
                        out_perm = [matmul_out_shape.index(shape) for shape in out_term]
                        post_trans = insert_transpose_after(graph, einsum, out_perm, quantize=einsum_obj.quantize)
                        if einsum in graph._attr['output_names']:
                            index = graph._attr['output_names'].index(einsum)
                            graph._attr['output_names'][index] = post_trans
                    matmul_attr = einsum_obj.copied_attr()
                    matmul_attr.update({'opset_version': 13})
                    NodeWrap(graph, einsum).replace_obj('MatMul', matmul_attr)
                elif len(add_list[0]) == 4 and \
                        add_list[0][:3] == '...' and \
                        len(add_list[1]) == 2 and \
                        len(out_term) == 5 and \
                        add_list[0][-1] == add_list[1][-1] and \
                        add_list[1][-2:] == out_term[-2:]:
                    # ...d, hd -> ...hd
                    in_shapes = einsum_obj.get_input_shapes()
                    inp_0_shape = in_shapes[0]
                    inp_1_shape = in_shapes[1]
                    h, d = inp_1_shape
                    x = int(np.prod(inp_0_shape[:-1]))
                    reshape0_shape = [x, 1, d]
                    reshape1_shape = [1, h, d]
                    insert_reshape(graph, in_edges[0][0], einsum, in_edges[0][-1], reshape0_shape,
                                   quantize=einsum_obj.quantize)
                    insert_reshape(graph, in_edges[1][0], einsum, in_edges[1][-1], reshape1_shape,
                                   quantize=einsum_obj.quantize)

                    mul_attr = einsum_obj.copied_attr()
                    mul_attr.update({'opset_version': 13})
                    NodeWrap(graph, einsum).replace_obj(
                        'Mul', mul_attr)

                    reshape2_shape = inp_0_shape[:-1] + [h, d]
                    reshape2 = insert_reshape_after(graph, einsum, reshape2_shape, quantize=einsum_obj.quantize)

                    if einsum in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(einsum)
                        graph._attr['output_names'][index] = reshape2
                elif len(add_list[1]) >= 2 and len(add_list[0]) >= 2 and \
                        len(set(add_list[0]).difference(set(out_term))) > 0 and \
                        set(add_list[0]).difference(set(out_term)) == set(add_list[1]).difference(set(out_term)) and \
                        len(set(add_list[0]).intersection(add_list[1]).intersection(out_term)) > 0:
                    # bmchw,bnmc -> bmhwn
                    batch = set(add_list[0]).intersection(add_list[1]).intersection(out_term)
                    c = set(add_list[0]).difference(set(out_term))
                    # bmchw --> bhc, bnmc --> bcw
                    in_edges = graph.sorted_in_edges(einsum, data=True)
                    in_shapes = einsum_obj.get_input_shapes()
                    out_shapes = einsum_obj.get_output_shapes()
                    if len(in_edges) < 2 or len(in_shapes) < 2 \
                            or any(shape is None or None in shape for shape in in_shapes):
                        ERROR('[Parser]: Meets invalid inputs of Einsum(%s) in convert_einsum!' % einsum)
                        continue
                    inp_0_shape = in_shapes[0]
                    inp_1_shape = in_shapes[1]
                    need_reshape_0 = True if len(inp_0_shape) > 3 else False
                    need_reshape_1 = True if len(inp_1_shape) > 3 else False
                    need_reshape_out = True if len(out_term) > 3 else False
                    batch_idx_input0 = [add_list[0].index(b) for b in batch]
                    batch_idx_input0.sort()
                    batch_idx_input1 = [add_list[1].index(b) for b in batch]
                    batch_idx_input1.sort()
                    batch_idx_output = [out_term.index(b) for b in batch]
                    batch_idx_output.sort()
                    c_idx_input0 = [add_list[0].index(b) for b in c]
                    c_idx_input0.sort()
                    c_idx_input1 = [add_list[1].index(b) for b in c]
                    c_idx_input1.sort()
                    need_transpose_0_c = False if max(c_idx_input0) == len(inp_0_shape) - 1 else True
                    need_transpose_1_c = True if max(c_idx_input1) == len(inp_1_shape) - 1 else False

                    batch_shape = [inp_0_shape[axis] for axis in batch_idx_input0]
                    c_shape = [inp_0_shape[axis] for axis in c_idx_input0]
                    new_batch_shape = int(np.prod(batch_shape))
                    new_c_shape = int(np.prod(c_shape))
                    matmul_output_shape = [new_batch_shape]
                    if _is_consecutive(batch_idx_input0):
                        if _is_consecutive(c_idx_input0):
                            if need_reshape_0:
                                if need_transpose_0_c:
                                    h_shape = inp_0_shape[max(c_idx_input0) + 1:]
                                    new_h_shape = int(np.prod(h_shape))
                                    reshape_0_shape = [new_batch_shape, new_c_shape, new_h_shape]
                                    reshape0 = insert_reshape(graph, in_edges[0][0], einsum, in_edges[0][-1],
                                                              reshape_0_shape,
                                                              quantize=einsum_obj.quantize)
                                    in_edges = graph.sorted_in_edges(einsum, data=True)
                                    insert_transpose(graph, reshape0, einsum, in_edges[0][-1], [0, 2, 1],
                                                     quantize=einsum_obj.quantize)
                                else:
                                    h_shape = inp_0_shape[max(batch_idx_input0) + 1:min(c_idx_input0)]
                                    new_h_shape = int(np.prod(h_shape))
                                    reshape_0_shape = [new_batch_shape, new_h_shape, new_c_shape]
                                    insert_reshape(graph, in_edges[0][0], einsum, in_edges[0][-1],
                                                   reshape_0_shape,
                                                   quantize=einsum_obj.quantize)
                            else:
                                if need_transpose_0_c:
                                    new_h_shape = inp_0_shape[2]
                                    insert_transpose(graph, in_edges[0][0], einsum, in_edges[0][-1], [0, 2, 1],
                                                     quantize=einsum_obj.quantize)
                                else:
                                    new_h_shape = inp_0_shape[1]
                            matmul_output_shape.append(new_h_shape)
                        else:
                            _warn_unsupported(einsum, einsum_obj.equation)
                    else:
                        _warn_unsupported(einsum, einsum_obj.equation)

                    if _is_consecutive(batch_idx_input1):
                        if _is_consecutive(c_idx_input1):
                            if need_reshape_1:
                                if need_transpose_1_c:
                                    w_shape = inp_1_shape[max(batch_idx_input1) + 1:min(c_idx_input1)]
                                    new_w_shape = int(np.prod(w_shape))
                                    reshape_1_shape = [new_batch_shape, new_w_shape, new_c_shape]
                                    reshape1 = insert_reshape(graph, in_edges[1][0], einsum, in_edges[1][-1],
                                                              reshape_1_shape,
                                                              quantize=einsum_obj.quantize)
                                    in_edges = graph.sorted_in_edges(einsum, data=True)
                                    insert_transpose(graph, reshape1, einsum, in_edges[1][-1], [0, 2, 1],
                                                     quantize=einsum_obj.quantize)
                                else:
                                    w_shape = inp_1_shape[max(c_idx_input1) + 1:]
                                    new_w_shape = int(np.prod(w_shape))
                                    reshape_1_shape = [new_batch_shape, new_c_shape, new_w_shape]
                                    insert_reshape(graph, in_edges[1][0], einsum, in_edges[1][-1],
                                                   reshape_1_shape,
                                                   quantize=einsum_obj.quantize)
                            else:
                                if need_transpose_1_c:
                                    new_w_shape = inp_1_shape[1]
                                    insert_transpose(graph, in_edges[1][0], einsum, in_edges[1][-1], [0, 2, 1],
                                                     quantize=einsum_obj.quantize)
                                else:
                                    new_w_shape = inp_1_shape[2]
                            matmul_output_shape.append(new_w_shape)
                        else:
                            _warn_unsupported(einsum, einsum_obj.equation)
                    else:
                        # insert transpose first
                        perm = batch_idx_input1 + c_idx_input1
                        left_perm = []
                        for i in range(len(inp_1_shape)):
                            if i not in perm:
                                left_perm.append(i)
                        perm = batch_idx_input1 + c_idx_input1 + left_perm
                        transpose1 = insert_transpose(graph, in_edges[1][0], einsum, in_edges[1][-1], perm,
                                                      quantize=einsum_obj.quantize)
                        if need_reshape_1:
                            new_w_shape = int(np.prod(inp_1_shape)) // (new_batch_shape * new_c_shape)
                            reshape_1_shape = [new_batch_shape, new_c_shape, new_w_shape]
                            in_edges = graph.sorted_in_edges(einsum, data=True)
                            insert_reshape(graph, transpose1, einsum, in_edges[1][-1],
                                           reshape_1_shape,
                                           quantize=einsum_obj.quantize)
                            matmul_output_shape.append(new_w_shape)

                    matmul_attr = einsum_obj.copied_attr()
                    matmul_attr.update({'opset_version': 13})
                    NodeWrap(graph, einsum).replace_obj('MatMul', matmul_attr)

                    _, dst, matmul_out_attr = graph.sorted_out_edges(einsum, data=True)[0]
                    if matmul_out_attr['tensor'].value is not None:
                        in_edges = graph.sorted_in_edges(einsum, data=True)
                        matmul_out_attr['tensor'].value = np.matmul(in_edges[0][-1]['tensor'].value,
                                                                    in_edges[1][-1]['tensor'].value)
                    matmul_out_attr['tensor'].shape = tuple(matmul_output_shape.copy())

                    matmul_output_term = [add_list[0][idx] for idx in batch_idx_input0]
                    if need_transpose_0_c:
                        matmul_output_term += [add_list[0][idx]
                                               for idx in range(max(c_idx_input0) + 1, len(add_list[0]))]
                    else:
                        matmul_output_term += [add_list[0][idx] for idx in
                                               range(max(batch_idx_input0) + 1, min(c_idx_input0))]
                    for i in range(len(add_list[1])):
                        if i not in batch_idx_input1 and i not in c_idx_input1:
                            matmul_output_term.append(add_list[1][i])

                    need_transpose_out = ''.join(matmul_output_term) != out_term
                    if need_transpose_out:
                        perm = [0, 2, 1]
                        trans_out_attr = copy.deepcopy(matmul_out_attr)
                        transpose_out = insert_transpose(graph, einsum, dst, trans_out_attr, perm,
                                                         quantize=einsum_obj.quantize)
                        matmul_output_shape_updated = [matmul_output_shape[axis] for axis in perm]
                        matmul_output_shape = matmul_output_shape_updated.copy()
                    if need_reshape_out:
                        reshape_out_shape = out_shapes[0]
                        reshape_src = transpose_out if need_transpose_out else einsum
                        reshape_out = insert_reshape_after(graph, reshape_src, reshape_out_shape,
                                                           old_dim=matmul_output_shape,
                                                           quantize=einsum_obj.quantize)
                    if need_transpose_out or need_reshape_out:
                        if einsum in graph._attr['output_names']:
                            index = graph._attr['output_names'].index(einsum)
                            last_out = reshape_out if need_reshape_out else transpose_out
                            graph._attr['output_names'][index] = last_out
                else:
                    _warn_unsupported(einsum, equation)
            elif len(in_edges) == 1:
                equation = einsum_obj.equation.replace(' ', '')
                equ_list = equation.split('->')
                if len(equ_list[0]) == len(equ_list[1]) and set(equ_list[0]) == set(equ_list[1]):
                    # convert to transpose
                    perm = []
                    for v in equ_list[0]:
                        perm.append(equ_list[1].index(v))
                    trans_attr = einsum_obj.copied_attr()
                    trans_attr.update({'opset_version': 13, 'perm': perm})
                    NodeWrap(graph, einsum).replace_obj('Transpose', trans_attr)
                elif len(equ_list[0]) == len(equ_list[1]) + 1:
                    if equ_list[0][:len(equ_list[1])] == equ_list[1] and \
                            equ_list[0][-1] == equ_list[0][-2]:
                        # batch_diagonal, convert to GatherND
                        input_shape = einsum_obj.get_input_shapes()[0]
                        output_shape = einsum_obj.get_output_shapes()[0]
                        need_reshape = True if len(input_shape) > 2 else False
                        reshape_shape = [int(np.prod(input_shape[:-1])), input_shape[-1]]
                        if need_reshape:
                            insert_reshape(graph, in_edges[0][0], einsum, in_edges[0][-1],
                                           reshape_shape,
                                           quantize=einsum_obj.quantize)
                        indices = []
                        for i in range(reshape_shape[0]):
                            indices.append([i, i % reshape_shape[1]])
                        insert_constant(graph, einsum + '_indices',
                                        np.array(indices, np.int64), einsum, in_port=1)
                        gather_attr = einsum_obj.copied_attr()
                        gather_attr.update({'opset_version': 13, 'batch_dims': 0})
                        NodeWrap(graph, einsum).replace_obj('GatherND', gather_attr)
                        if need_reshape:
                            reshape_out = insert_reshape_after(graph, einsum, output_shape,
                                                               quantize=einsum_obj.quantize)
                            if einsum in graph._attr['output_names']:
                                index = graph._attr['output_names'].index(einsum)
                                graph._attr['output_names'][index] = reshape_out
                    elif len(set(equ_list[0]).difference(set(equ_list[1]))) == 1:
                        # ReduceSum, 'ij->i'
                        sum_shape = list(set(equ_list[0]).difference(set(equ_list[1])))[0]
                        sum_axis = equ_list[0].index(sum_shape)
                        reduce_sum_attr = einsum_obj.copied_attr()
                        reduce_sum_attr.update({'opset_version': 11, 'keepdims': 0,
                                                'axes': [sum_axis]})
                        NodeWrap(graph, einsum).replace_obj('ReduceSum', reduce_sum_attr)
                    else:
                        _warn_unsupported(einsum, einsum_obj.equation)
                else:
                    _warn_unsupported(einsum, einsum_obj.equation)
            else:
                _warn_unsupported(einsum, einsum_obj.equation)
        else:
            ERROR('[Parser]: Meets invalid node in convert_einsum!')


def convert_special_clip_to_relu(graph):
    matches = single_node_matcher(graph, 'Clip')
    for m in matches:
        clip = m['target']
        clip_obj = NodeWrap(graph, clip)['object']
        if clip_obj is not None:
            inputs = clip_obj.get_input_tensors()
            in_edges = graph.sorted_in_edges(clip, data=True)
            if len(inputs) == 3 \
                    and len(in_edges) == 3 \
                    and inputs[1] is not None \
                    and inputs[2] is not None \
                    and FLOAT_EQUAL(inputs[1], 0) \
                    and in_edges[0][2]['tensor'].get_dtype() is not None \
                    and inputs[2] >= TYPE_MAX(in_edges[0][2]['tensor'].get_dtype()) \
                    and not clip_obj.quantize:
                graph.remove_edges_from(in_edges[1:])
                NodeWrap(graph, clip).replace_obj(
                    'Relu', {'name': clip, 'opset_version': 6})
        else:
            ERROR(
                '[Parser]: Meets invalid Clip Node (%s) in convert_special_clip_to_relu!' % clip)


def convert_special_matmul_to_fc(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('w', {'op': 'Constant', 'unique': False}),
                                   ('matmul', {'op': 'MatMul'})
                               ],
                               edges=[
                                   ('w', 'matmul', {'src_out_port': 0, 'dst_in_port': 1}),
                               ])
    for m in matches:
        matmul, w = m['matmul'], m['w']
        matmul_obj = NodeWrap(graph, matmul)['object']
        w_obj = NodeWrap(graph, w)['object']
        in_edges = graph.sorted_in_edges(matmul, data=True)
        if matmul_obj is None \
                or w_obj is None \
                or w_obj.value is None \
                or len(in_edges) != 2 \
                or in_edges[1][2]['tensor'] is None:
            ERROR('[Parser]: Meets invalid MatMul Node (%s) in convert_special_matmul_to_fc!' % matmul)
            continue

        if len(matmul_obj.sorted_in_consts()) != 1 \
                or len(w_obj.value.shape) != 2:
            continue

        input_shapes = matmul_obj.get_input_shapes()
        if len(input_shapes) != 2 or any(shape is None or None in shape for shape in input_shapes):
            continue
        output_shapes = matmul_obj.get_output_shapes()
        if len(output_shapes) < 1 or output_shapes[0] is None or None in output_shapes[0]:
            continue
        if len(input_shapes[0]) >= 2:
            matched = True
            weights = np.transpose(w_obj.value)
            graph.remove_edge(w, matmul)
            matmul_attr = matmul_obj.copied_attr()
            if matmul_obj.quantize:
                biases = np.zeros([weights.shape[0]], np.int32)
                biases_scale = np.ones([weights.shape[0]], np.float32)
                biases_zp = np.zeros([weights.shape[0]], np.int32)
                matmul_attr.update({'weights': weights,
                                    'weights_scale_zp': list(in_edges[1][2]['tensor'].scale_zp),
                                    'biases': biases,
                                    'biases_scale_zp': [biases_scale, biases_zp]})
            else:
                biases = np.zeros([weights.shape[0]], w_obj.value.dtype)
                matmul_attr.update({'weights': weights, 'biases': biases})
            NodeWrap(graph, matmul).replace_obj('FullyConnected', matmul_attr)
            if len(input_shapes[0]) > 2:
                src, _, in_attr = in_edges[0]
                insert_reshape(graph, src, matmul, in_attr, [-1, input_shapes[0][-1]], quantize=matmul_obj.quantize)
                post_reshape = insert_reshape_after(graph,
                                                    matmul,
                                                    output_shapes[0],
                                                    old_dim=[int(np.prod(output_shapes[0][:-1])), output_shapes[0][-1]],
                                                    quantize=matmul_obj.quantize)
                if matmul in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(matmul)
                    graph._attr['output_names'][index] = post_reshape
    if matched:
        clear_redundant_nodes(graph)


def convert_upsample_to_resize(graph):
    matches = single_node_matcher(graph, 'Upsample')
    for m in matches:
        upsample = m['target']
        upsample_obj = NodeWrap(graph, upsample)['object']
        in_edges = graph.sorted_in_edges(upsample, data=True)
        if upsample_obj is not None and len(in_edges) == 2:
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
            ERROR(
                '[Parser]: Meets invalid Upsample Op (%s) in convert_upsample_to_resize!' % upsample)


def convert_special_cast(graph):
    '''Convert Cast op whose dst type is 'bool' to Equal(B=0)+Not instead of uint8
    because convertting to uint8 may cause similarity issue.
    For example, float 256.0 becomes 0 if converted to uint8(truncate) but expects
    True if converted to bool.
    '''
    matches = single_node_matcher(graph, 'Cast')
    for m in matches:
        cast = m['target']
        cast_obj = NodeWrap(graph, cast)['object']
        if cast_obj is None:
            ERROR('[Parser]: Meets invalid Cast Op (%s) in convert_special_cast!' % cast)
            continue
        if cast_obj.to != 'bool':
            continue
        cast_in_edges = graph.sorted_in_edges(cast, data=True)
        if len(cast_in_edges) < 1:
            ERROR('[Parser]: Meets invalid input of Cast Op (%s) in convert_special_cast!' % cast)
            continue
        src, _, in_attr = cast_in_edges[0]
        if in_attr['tensor'] is None \
                or in_attr['tensor'].get_dtype() is None \
                or in_attr['tensor'].get_shape() is None \
                or any(s is None for s in in_attr['tensor'].get_shape()):
            ERROR('[Parser]: Meets invalid input of Cast Op (%s) in convert_special_cast!' % cast)
            continue
        graph.remove_edges_from(cast_in_edges)

        equal_node = get_valid_node_name(graph, cast + '_equal')
        graph.add_edge(src, equal_node, **in_attr)
        input_shape = in_attr['tensor'].get_shape()
        input_dtype = in_attr['tensor'].get_dtype()
        const_zeros_value = np.zeros(input_shape, dtype=np.dtype(input_dtype))
        insert_constant(graph, equal_node + '_zeros', const_zeros_value, equal_node, in_port=1)
        equal_out_attr = {'tensor': Tensor(shape=input_shape, dtype=input_dtype)}
        graph.add_edge(equal_node, cast, **equal_out_attr)

        NodeWrap(graph, equal_node).replace_obj('Equal', {'name': equal_node, 'opset_version': 13})
        cast_attr = cast_obj.copied_attr()
        cast_attr.update({'opset_version': 1})
        NodeWrap(graph, cast).replace_obj('Not', cast_attr)


def convert_special_conv_to_mul(graph):
    matches = single_node_matcher(graph, 'Conv')
    for m in matches:
        conv = m['target']
        conv_obj = NodeWrap(graph, conv)['object']
        if conv_obj is None:
            ERROR('[Parser]: Meets invalid Conv(%s) in convert_special_conv_to_mul!' % conv)
            continue
        if conv_obj.weights is not None or conv_obj.quantize:
            continue
        in_edges = graph.sorted_in_edges(conv, keys=True, data=True)
        input_shapes = conv_obj.get_input_shapes()
        if len(in_edges) < 2 \
                or in_edges[1][3]['tensor'].is_const \
                or len(input_shapes) < 2 \
                or any(shape is None for shape in input_shapes) \
                or len(input_shapes[1]) < 3:
            WARN('[Parser]: Meets invalid Conv(%s) in convert_special_conv_to_mul!' % conv)
            continue
        inp_shape, w_shape = input_shapes[:2]
        group = conv_obj.group
        if inp_shape[1] != w_shape[0] \
                or any(s != 1 for s in w_shape[2:]) \
                or group not in (1, inp_shape[1]):
            continue
        if any(d != 1 for d in conv_obj.dilations) \
                or any(s != 1 for s in conv_obj.strides) \
                or any(p != 0 for p in conv_obj.pads) \
                or conv_obj.auto_pad != 'NOTSET':
            continue
        w_name, _, w_k, w_in_attr = in_edges[1]
        dim = [w_shape[0]] + [1] * (len(w_shape) - 2)
        if group == inp_shape[1]:
            insert_reshape(graph, w_name, conv, w_in_attr, dim, key=w_k, data_format='NCHW')
            NodeWrap(graph, conv).replace_obj('Mul', {'name': conv, 'opset_version': 7})
        else:
            # Convert to torch.sum(torch.reshape(x, [N, 1, C, H, W]) * w, dim=2, keepdim=False)
            mul_node = get_valid_node_name(graph, conv + '_mul')
            graph.remove_edges_from(in_edges)
            inp_name, _, _, inp_in_attr = in_edges[0]
            graph.add_edge(inp_name, mul_node, **inp_in_attr)
            inp_reshape_dim = [inp_shape[0], 1] + list(inp_shape[1:])
            insert_reshape(graph, inp_name, mul_node, inp_in_attr, inp_reshape_dim)
            graph.add_edge(w_name, mul_node, **w_in_attr)
            graph.add_edge(mul_node, conv)
            NodeWrap(graph, mul_node).replace_obj('Mul', {'name': mul_node, 'opset_version': 7})
            reduce_sum_attr = conv_obj.copied_attr()
            reduce_sum_attr.update({'opset_version': 11, 'axes': [2], 'keepdims': False})
            NodeWrap(graph, conv).replace_obj('ReduceSum', reduce_sum_attr)
        if len(in_edges) == 3:
            # + torch.reshape(b, [-1, 1, 1])
            add = get_valid_node_name(graph, conv + '_add')
            b_name, _, b_k, b_in_attr = in_edges[2]
            graph.remove_edges_from(in_edges[2:])
            b_in_attr.update({'dst_in_port': 1})
            graph.add_edge(b_name, add, **b_in_attr)
            insert_reshape(graph, b_name, add, b_in_attr, dim, key=b_k, data_format='NCHW')
            for _, dst, out_attr in graph.sorted_out_edges(conv, data=True):
                graph.remove_edge(conv, dst)
                graph.add_edge(add, dst, **out_attr)
            graph.add_edge(conv, add)
            NodeWrap(graph, add).replace_obj('Add', {'name': add, 'opset_version': 7})
            if conv in graph._attr['output_names']:
                index = graph._attr['output_names'].index(conv)
                graph._attr['output_names'][index] = add


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
            ERROR(
                '[Parser]: Meets invalid Resize Op (%s) in convert_special_resize!' % resize)


def convert_special_pow(graph):
    matched = False
    matches = single_node_matcher(graph, 'Pow')
    for m in matches:
        power = m['target']
        pow_obj = NodeWrap(graph, power)['object']
        if pow_obj is not None:
            if pow_obj.quantize:
                continue
            in_edges = graph.sorted_in_edges(power, data=True)
            pow_y_obj = NodeWrap(graph, in_edges[1][0])['object']
            if pow_y_obj.type == 'Constant' and FLOAT_EQUAL(pow_y_obj.value, 0.5):
                matched = True
                graph.remove_edges_from(in_edges[1:])
                NodeWrap(graph, power).replace_obj(
                    'Sqrt', {'name': power, 'opset_version': 13})
        else:
            ERROR(
                '[Parser]: Meets invalid Pow Op (%s) in convert_special_pow!' % power)
    if matched:
        clear_redundant_nodes(graph)


def convert_special_mul(graph):
    matched = False
    matches = single_node_matcher(graph, 'Mul')
    for m in matches:
        mul = m['target']
        mul_obj = NodeWrap(graph, mul)['object']
        if mul_obj is not None:
            if mul_obj.quantize:
                continue
            in_edges = graph.sorted_in_edges(mul, data=True)
            if len(in_edges) == 2 and in_edges[0][0] == in_edges[1][0]:
                matched = True
                graph.remove_edges_from(in_edges[1:])
                insert_constant(graph, f'{mul}_pow_y', np.array(2., dtype=np.float32), mul, in_port=1)
                NodeWrap(graph, mul).replace_obj(
                    'Pow', {'name': mul, 'opset_version': 13})
        else:
            ERROR(
                '[Parser]: Meets invalid Mul Op (%s) in convert_special_mul!' % mul)
    if matched:
        clear_redundant_nodes(graph)


def convert_special_scatternd(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('indices', {'op': 'Constant', 'unique': False}),
                                      ('updates', {'unique': False}),
                                      ('scatternd', {'op': 'ScatterND'})],
                               edges=[('indices', 'scatternd', {'src_out_port': 0, 'dst_in_port': 1}),
                                      ('updates', 'scatternd', {'dst_in_port': 2})])
    for m in matches:
        indices, scatternd, updates = m['indices'], m['scatternd'], m['updates']
        indices_obj, scatternd_obj = [NodeWrap(graph, name)['object'] for name in [
            indices, scatternd]]
        scatternd_in_edges = graph.sorted_in_edges(scatternd, data=True)
        if indices_obj is None or scatternd_obj is None or len(scatternd_in_edges) < 3:
            ERROR('[Parser]: Meet invalid node in convert_special_scatternd!')
            continue
        if scatternd_obj.reduction != 'none':
            continue
        input_shapes = scatternd_obj.get_input_shapes()
        if len(input_shapes) < 3 \
                or np.any([None in input_shape for input_shape in input_shapes]) \
                or np.any([item is None for input_shape in input_shapes for item in input_shape]) \
                or len(input_shapes[0]) < 1 \
                or len(indices_obj.value.shape) < 2:
            continue
        input_shape = input_shapes[0]
        indices_shape = input_shapes[1]
        update_shape = input_shapes[2]

        if len(input_shape) != len(update_shape) \
                or len(indices_shape) != (len(update_shape) + 1):
            continue
        axis = [idx for idx, (inp_shape, u_shape) in enumerate(zip(input_shape, update_shape)) if inp_shape != u_shape]
        if len(axis) > 1:
            continue
        axis = (len(input_shape) - 1) if len(axis) == 0 else axis[0]
        input_dim_at_axis = input_shape[axis]
        indices_value = indices_obj.value
        indices_len_at_axis = indices_value.shape[axis]
        start_indice = indices_value.item(axis)
        start_indices = np.array([0] * indices_value.shape[-1])
        start_indices[axis] = start_indice
        if start_indice < 0:
            # -2, -1 is continuous(split to [:-2] and [-2:]) and
            # -2, -1, 0, 1, ...x is not(concat of split[-2:] and split[0:x])
            if start_indice + indices_len_at_axis > 0:
                continue
            start_indice = input_dim_at_axis + start_indice
        exp_shape = indices_value.shape[:-1]
        indices_exp_value = list(np.ndindex(*exp_shape))
        indices_exp_value = np.reshape(
            np.array(indices_exp_value) + start_indices, indices_value.shape)
        if not np.array_equal(indices_exp_value, indices_value) \
                or (indices_len_at_axis + start_indice) > input_dim_at_axis:
            continue
        matched = True
        graph.remove_edges_from(scatternd_in_edges)
        _, _, updates_out_attr = scatternd_in_edges[2]
        if indices_len_at_axis == input_dim_at_axis:
            scatternd_out_edges = graph.sorted_out_edges(scatternd, data=True)
            graph.remove_edges_from(scatternd_out_edges)
            updates_out_port = updates_out_attr['src_out_port']
            for _, dst, out_attr in scatternd_out_edges:
                out_attr['src_out_port'] = updates_out_port
                graph.add_edge(updates, dst, **out_attr)
            if scatternd in graph._attr['output_names']:
                index = graph._attr['output_names'].index(scatternd)
                graph._attr['output_names'][index] = updates
        else:
            src, _, src_out_attr = scatternd_in_edges[0]
            split_node = get_valid_node_name(graph, scatternd + '_split')
            graph.add_edge(src, split_node, **src_out_attr)
            split_in_tensor = src_out_attr['tensor'].value
            if start_indice == 0:
                split_out_attr = {'src_out_port': 1, 'dst_in_port': 1}
                updates_out_attr.update({'dst_in_port': 0})
                split = [indices_len_at_axis,
                         input_dim_at_axis - indices_len_at_axis]
                if split_in_tensor is not None:
                    split_out_tensor = np.split(split_in_tensor, split, axis=axis)[split_out_attr['src_out_port']]
                    split_out_attr['tensor'] = Tensor(value=split_out_tensor, shape=split_out_tensor.shape)
                graph.add_edge(split_node, scatternd, **split_out_attr)
            else:
                split_out_0_attr = {'src_out_port': 0, 'dst_in_port': 0}
                split_mid_len = indices_len_at_axis
                split_end_len = input_dim_at_axis - start_indice - split_mid_len
                if split_end_len != 0:
                    split_out_2_attr = {'src_out_port': 2, 'dst_in_port': 2}
                    split = [start_indice, split_mid_len, split_end_len]
                    if split_in_tensor is not None:
                        split_out_2_tensor = np.split(split_in_tensor, split, axis=axis)[
                            split_out_2_attr['src_out_port']]
                        split_out_2_attr['tensor'] = Tensor(value=split_out_2_tensor, shape=split_out_2_tensor.shape)
                    graph.add_edge(split_node, scatternd, **split_out_2_attr)
                else:
                    split = [start_indice, split_mid_len]
                if split_in_tensor is not None:
                    split_out_tensor = np.split(split_in_tensor, split, axis=axis)[split_out_0_attr['src_out_port']]
                    split_out_0_attr['tensor'] = Tensor(value=split_out_tensor, shape=split_out_tensor.shape)
                updates_out_attr.update({'dst_in_port': 1})
                graph.add_edge(split_node, scatternd, **split_out_0_attr)
            graph.add_edge(updates, scatternd, **updates_out_attr)

            NodeWrap(graph, split_node).replace_obj(
                'Split', {'name': split_node, 'opset_version': 11, 'axis': axis, 'split': split})
            concat_attr = scatternd_obj.copied_attr()
            concat_attr.update({'opset_version': 13, 'axis': axis})
            NodeWrap(graph, scatternd).replace_obj('Concat', concat_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_special_scatternd2(graph):
    '''Remove special scatternd that all the inputs are replaced by updates.
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('indices', {'op': 'Constant'}),
                                      ('updates', {}),
                                      ('scatternd', {'op': 'ScatterND'})],
                               edges=[('indices', 'scatternd', {'src_out_port': 0, 'dst_in_port': 1}),
                                      ('updates', 'scatternd', {'dst_in_port': 2})])
    for m in matches:
        indices, scatternd, updates = m['indices'], m['scatternd'], m['updates']
        indices_obj, scatternd_obj = [NodeWrap(graph, name)['object'] for name in [
            indices, scatternd]]
        scatternd_in_edges = graph.sorted_in_edges(scatternd, data=True)
        if indices_obj is None or scatternd_obj is None or len(scatternd_in_edges) < 3:
            ERROR('[Parser]: Meet invalid node in convert_special_scatternd!')
            continue
        if scatternd_obj.reduction != 'none':
            continue
        input_shapes = scatternd_obj.get_input_shapes()
        if len(input_shapes) < 3 \
                or any((input_shape is None or None in input_shape) for input_shape in input_shapes):
            continue
        input_shape = input_shapes[0]
        indices_shape = input_shapes[1]
        update_shape = input_shapes[2]
        if input_shape != update_shape \
                or indices_shape != update_shape[:-1] + [len(update_shape) - 1]:
            continue
        exp_indices = np.expand_dims(
            np.array(list(np.ndindex(*update_shape[:-1]))), list(range(len(indices_shape) - 2)))
        if not np.array_equal(indices_obj.value, exp_indices):
            continue
        matched = True
        scatternd_out_edges = graph.sorted_out_edges(scatternd, data=True)
        graph.remove_edges_from(scatternd_out_edges)
        updates_out_port = scatternd_in_edges[2][2]['src_out_port']
        for _, dst, out_attr in scatternd_out_edges:
            out_attr.update({'src_out_port': updates_out_port})
            graph.add_edge(updates, dst, **out_attr)
        if scatternd in graph._attr['output_names']:
            index = graph._attr['output_names'].index(scatternd)
            graph._attr['output_names'][index] = updates
    if matched:
        clear_redundant_nodes(graph)


def convert_multi_scatternd_to_concat(graph):
    '''Convert multiple connected ScatterND whose indice is sorted and regular to one Concat op.
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('input', {'unique': False}),
                                      ('indices', {'op': 'Constant'}),
                                      ('scatternd', {'op': 'ScatterND'}),
                                      ],
                               edges=[('input', 'scatternd', {'dst_in_port': 0}),
                                      ('indices', 'scatternd', {'src_out_port': 0, 'dst_in_port': 1}),
                                      ])
    scatters_dict = {}  # save {scatter_node_name: (input_node_name, start_indice, axis), ...}
    last_scatters = []  # save the last scatter node names
    for m in matches:
        names = ['indices', 'input', 'scatternd']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(obj is None for obj in obj_dict.values()):
            ERROR('[Parser]: Meet invalid node in convert_multi_scatternd_to_concat!')
            continue
        if obj_dict['scatternd'].reduction != 'none':
            continue
        input_shapes = obj_dict['scatternd'].get_input_shapes()
        if len(input_shapes) < 3 \
                or any((input_shape is None or None in input_shape) for input_shape in input_shapes):
            continue
        input_shape = input_shapes[0]
        indices_shape = input_shapes[1]
        update_shape = input_shapes[2]
        indices_depth = indices_shape[-1]
        if len(indices_shape) != indices_depth + 1 \
                or indices_shape[:-1] != update_shape[:indices_depth]:
            continue
        axes = [idx for idx, (s1, s2) in enumerate(zip(input_shape, indices_shape[:-1])) if s1 != s2]
        if len(axes) != 1:
            continue
        axis = axes[0]
        concat_nodes_num = input_shape[axis]
        base_indices = np.reshape(list(np.ndindex(*indices_shape[:-1])), indices_shape)
        start_indice = obj_dict['indices'].value.item(axis)
        exp_indices = base_indices + np.array([start_indice if idx == axis else 0 for idx in range(indices_depth)])
        if not np.array_equal(exp_indices, obj_dict['indices'].value):
            continue
        scatters_dict[m['scatternd']] = (m['input'], start_indice, axis)
        if start_indice == (concat_nodes_num - 1):
            last_scatters.append(m['scatternd'])
    for last_scatter in last_scatters:
        last_scatter_obj = NodeWrap(graph, last_scatter)['object']
        if last_scatter_obj is None:
            ERROR('[Parser]: Meet invalid ScatterND node (%s) in convert_multi_scatternd_to_concat!' % last_scatter)
            continue
        largest_indice, exp_axis = scatters_dict[last_scatter][1:]
        concat_nodes_num = largest_indice + 1
        scatter = last_scatter
        scatter_nodes = [None] * concat_nodes_num
        src_nodes_info = [None] * concat_nodes_num
        for exp_indice in range(largest_indice, -1, -1):
            inp, indice, axis = scatters_dict[scatter]
            if (indice != 0 and inp not in scatters_dict) \
                    or indice != exp_indice or axis != exp_axis:
                break
            scatter_nodes[indice] = scatter
            scatter_in_edges = graph.sorted_in_edges(scatter, data=True)
            if len(scatter_in_edges) < 3:
                break
            update, _, in_attr = scatter_in_edges[2]
            src_nodes_info[indice] = (update, in_attr)
            scatter = inp
        if None in scatter_nodes:
            continue
        matched = True
        last_scatter_in_edges = graph.sorted_in_edges(last_scatter, data=True)
        graph.remove_edges_from(last_scatter_in_edges)
        for idx, (src, in_attr) in enumerate(src_nodes_info):
            new_in_attr = copy.deepcopy(in_attr)
            new_in_attr.update({'dst_in_port': idx})
            graph.add_edge(src, last_scatter, **new_in_attr)
        concat_attr = last_scatter_obj.copied_attr()
        concat_attr.update({'axis': axis, 'opset_version': 13})
        NodeWrap(graph, last_scatter).replace_obj('Concat', concat_attr)
    if matched:
        clear_redundant_nodes(graph)


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
                quantize = transpose_obj.quantize
                insert_reshape(graph, src, transpose, in_attr, dim1, quantize=quantize)
                post_reshape = insert_reshape_after(graph, transpose, dim2, quantize=quantize)
                if transpose in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(transpose)
                    graph._attr['output_names'][index] = post_reshape

        else:
            ERROR(
                '[Parser]: Meets invalid Transpose Op(%s) in convert_special_transpose!' % transpose)


def _decompose_const_if(graph, params):
    matched = False
    matches = single_node_matcher(graph, 'If')
    for m in matches:
        if_name = m['target']
        if_obj = NodeWrap(graph, if_name)['object']
        if if_obj is not None \
                and len(if_obj.sorted_in_consts()) >= 1 \
                and if_obj.sorted_in_consts()[0][1] == 0:
            condition = if_obj.sorted_in_consts()[0][2].item()
            if_in_edges = graph.sorted_in_edges(if_name, data=True)
            if_out_edges = graph.sorted_out_edges(if_name, data=True)
            keep_branch = if_obj.then_branch if condition else if_obj.else_branch
            remove_branch_name = if_obj.else_branch.name if condition else if_obj.then_branch.name
            sub_parent_node_map = {}
            matched = True

            is_subgraph = isinstance(graph, SubGraph)

            if is_subgraph:
                if if_name in graph._root._attr['subgraphs'] and \
                        remove_branch_name in graph._root._attr['subgraphs'][if_name]:
                    graph._root._attr['subgraphs'][if_name].pop(remove_branch_name)
            else:
                if if_name in graph._attr['subgraphs'] and \
                        remove_branch_name in graph._attr['subgraphs'][if_name]:
                    graph._attr['subgraphs'][if_name].pop(remove_branch_name)

            for n in keep_branch.nodes:
                n_obj = keep_branch.nodes[n]['object']
                if n_obj is None:
                    ERROR(
                        f'[Parser]: Meet invalid Node({n}) of root node({if_name}) in decompose_if.')
                n_in_edges = keep_branch.sorted_in_edges(n, data=True)
                if n_obj.type in ['Input', 'DummyInput']:
                    continue
                elif n_obj.type == 'Constant':
                    sub_parent_node_map[n] = n
                    if not graph.has_node(n):
                        graph.add_node(n)
                        cur_obj_attr = n_obj.copied_attr()
                        cur_obj_attr.update({'in_subgraph': is_subgraph})
                        NodeWrap(graph, n).replace_obj('Constant', cur_obj_attr)
                elif n_obj.type == 'Out':
                    o_port = n_in_edges[0][2]['src_out_port']
                    for _, dst, out_attr in if_out_edges:
                        if out_attr['src_out_port'] == o_port:
                            in_attr = copy.deepcopy(out_attr)
                            graph.add_edge(n_in_edges[0][0], dst, **in_attr)
                else:
                    parent_g_node_name = get_valid_node_name(graph, n)
                    graph.add_node(parent_g_node_name)
                    sub_parent_node_map[n] = parent_g_node_name
                    cur_obj_attr = n_obj.copied_attr()
                    cur_obj_attr.update({'in_subgraph': is_subgraph, 'name': parent_g_node_name})
                    if n_obj.type.startswith('Plugin'):
                        NodeWrap(graph, parent_g_node_name).replace_obj(n_obj.type[6:], cur_obj_attr)
                    else:
                        NodeWrap(graph, parent_g_node_name).replace_obj(n_obj.type, cur_obj_attr)
                    for in_e in n_in_edges:
                        src, dst, n_in_attr = in_e
                        src_obj = keep_branch.nodes[src]['object']
                        if src_obj.type == 'DummyInput':
                            if is_subgraph:
                                assert graph._root.has_node(src), f'{src} is DummyInput but not in main graph.'
                            else:
                                assert graph.has_node(src), f'{src} is DummyInput but not in main graph.'
                            if not graph.has_node(src) and is_subgraph:
                                graph.add_node(src)
                                sub_parent_node_map[src] = src
                                cur_obj_attr = src_obj.copied_attr()
                                cur_obj_attr.update({'in_subgraph': is_subgraph})
                                NodeWrap(graph, src).replace_obj(src_obj.type, cur_obj_attr)
                            in_attr = copy.deepcopy(n_in_attr)
                            graph.add_edge(src, parent_g_node_name, **in_attr)
                        elif src_obj.type == 'Constant':
                            if not graph.has_node(src):
                                graph.add_node(src)
                                sub_parent_node_map[src] = src
                                cur_obj_attr = src_obj.copied_attr()
                                cur_obj_attr.update({'in_subgraph': is_subgraph})
                                NodeWrap(graph, src).replace_obj('Constant', cur_obj_attr)
                            in_attr = copy.deepcopy(n_in_attr)
                            graph.add_edge(src, parent_g_node_name, **in_attr)
                        else:
                            in_attr = copy.deepcopy(n_in_attr)
                            graph.add_edge(sub_parent_node_map[src], parent_g_node_name, **in_attr)

            graph.remove_edges_from(if_in_edges)
            graph.remove_edges_from(if_out_edges)

            if if_name in graph._attr['output_names']:
                index = graph._attr['output_names'].index(if_name)
                graph._attr['output_names'][index] = keep_branch._attr['output_names'][0]
                for out in keep_branch._attr['output_names'][1:]:
                    index += 1
                    graph._attr['output_names'].insert(index, out)
            if params.get('input_names', []) and not is_subgraph:
                for inp in params['input_names'][:]:
                    _has_path = False
                    for out in graph._attr['output_names']:
                        if has_path(graph, inp, out):
                            _has_path = True
                            break
                    if _has_path is False:
                        inp_obj = NodeWrap(graph, inp)['object']
                        if inp_obj is not None and inp_obj.depend_nodes:
                            for op in inp_obj.depend_nodes:
                                if graph.has_node(op) and op != if_name:
                                    for out in graph._attr['output_names']:
                                        if has_path(graph, op, out):
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
            if params.get('output_names', []) and not is_subgraph:
                params['output_names'] = graph._attr['output_names']
            # clear subgraph
            if is_subgraph:
                if if_name in graph._root._attr['subgraphs']:
                    graph._root._attr['subgraphs'].pop(if_name)
            else:
                if if_name in graph._attr['subgraphs']:
                    graph._attr['subgraphs'].pop(if_name)
    if matched:
        clear_redundant_nodes(graph)


def _decompose_const_loop(graph, params):
    matched = False
    matches = single_node_matcher(graph, 'Loop')
    for m in matches:
        loop = m['target']
        loop_obj = NodeWrap(graph, loop)['object']
        loop_in_edges = graph.sorted_in_edges(loop, data=True)
        loop_out_edges = graph.sorted_out_edges(loop, data=True)
        if loop_obj is not None \
                and len(loop_in_edges) >= 2 and len(loop_out_edges) >= 1 and \
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

            if not condition:
                # loop_out_ports = loop_obj.get_out_ports()
                # # if any(p >= 2 for p in loop_out_ports) \
                # #         or (1 in loop_out_ports and len(loop_in_edges) != 3):
                # #     WARN('[Parser]: Meets unsupported Loop Node(%s) in decompose_const_loop!' % loop)
                # #     continue
                const_list = []
                for i in range(K):
                    v_initial, _, v_initial_in_attr = loop_in_edges[2 + i]
                    shape = get_valid_node_name(graph, f'{loop}_shape_{i}')
                    shape_in_attr = copy.deepcopy(v_initial_in_attr)
                    shape_in_attr.update({'dst_in_port': 0})
                    graph.add_edge(v_initial, shape, **shape_in_attr)
                    concat = get_valid_node_name(graph, f'{shape}_concat_{i}')
                    graph.add_edge(shape, concat, **{'src_out_port': 0, 'dst_in_port': 1})
                    insert_constant(graph, f'{concat}_zero_{i}', np.array([0], dtype=np.int64), concat, in_port=0)
                    const = get_valid_node_name(graph, f'{shape}_const_{i}')
                    graph.add_edge(concat, const)
                    NodeWrap(graph, shape).replace_obj('Shape', {'name': shape, 'opset_version': 1})
                    NodeWrap(graph, concat).replace_obj('Concat', {'name': concat, 'opset_version': 13, 'axis': 0})
                    NodeWrap(graph, const).replace_obj('ConstantOfShape', {'name': const, 'opset_version': 9})
                    const_list.append(const)
                for _, dst, out_attr in loop_out_edges:
                    graph.remove_edge(loop, dst)
                    out_port = out_attr['src_out_port']
                    if out_port < N:
                        graph.add_edge(loop_in_edges[2 + out_port][0], dst, **out_attr)
                    else:
                        const_out_attr = copy.deepcopy(out_attr)
                        const_out_attr.update({'src_out_port': 0})
                        graph.add_edge(const_list[out_port - N], dst, **const_out_attr)
                if loop in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(loop)
                    graph._attr['output_names'].pop(index)
                    for i in range(N):
                        graph._attr['output_names'].insert(index + i, loop_in_edges[2 + i][0])
                    for c in const_list:
                        if c is not None:
                            WARN(
                                f'[Parser]: The output({c}) of Node({loop}) has zero shape, which will be removed from graph!')
                # clear subgraph
                if loop in graph._attr['subgraphs']:
                    graph._attr['subgraphs'].pop(loop)
                continue

            if not loop_in_edges[0][2]['tensor'].is_const or \
                    loop_in_edges[0][2]['tensor'].value is None:
                continue
            if loop_obj.real_loop_cnt is None:
                continue

            graph.remove_edges_from(loop_in_edges)

            loop_cnt = loop_obj.real_loop_cnt
            loop_res = OrderedDict()
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
                                assert src in loop_obj.body._attr[
                                    'input_tensors'], f'{src} is Input but not in subgraph input tensors.'
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
                                        graph.add_edge(loop_res[i - 1][inp_idx - 1], main_g_node_name, **in_attr)
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
                            if i in loop_res:
                                loop_res[i][out_idx] = sub_main_node_map[n]
                            else:
                                loop_res[i] = {out_idx: sub_main_node_map[n]}
                        else:
                            scan_outs_name = list(k_carried_dict.keys())[out_idx - 1 - N]
                            k_carried_dict[scan_outs_name].append(sub_main_node_map[n])

            graph.remove_edges_from(loop_out_edges)

            # Loop have N + K outputs
            for i in range(N):
                _, dst, out_edge = loop_out_edges[i]
                out_attr = copy.deepcopy(out_edge)
                out_attr['src_out_port'] = 0
                graph.add_edge(loop_res[loop_cnt - 1][i + 1], dst, **out_attr)
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
                    loop_outputs.append(loop_res[loop_cnt - 1][i + 1])
                for i in range(K):
                    loop_outputs.append(list(k_carried_dict.keys())[i])
                graph._attr['output_names'][index:index] = loop_outputs

            # clear subgraph
            if loop in graph._attr['subgraphs']:
                graph._attr['subgraphs'].pop(loop)

    if matched:
        clear_redundant_nodes(graph)


def decompose_const_if_loop(graph, params):
    def check_decompose_const_if(graph):
        matched = False
        matches = single_node_matcher(graph, 'If')
        for m in matches:
            if_name = m['target']
            if_obj = NodeWrap(graph, if_name)['object']
            if if_obj is not None \
                    and len(if_obj.sorted_in_consts()) >= 1 \
                    and if_obj.sorted_in_consts()[0][1] == 0:
                matched = True
                break
        return matched

    def check_decompose_const_loop(graph):
        matched = False
        matches = single_node_matcher(graph, 'Loop')
        for m in matches:
            loop = m['target']
            loop_obj = NodeWrap(graph, loop)['object']
            loop_in_edges = graph.sorted_in_edges(loop, data=True)
            loop_out_edges = graph.sorted_out_edges(loop, data=True)
            if loop_obj is not None \
                    and len(loop_in_edges) >= 2 and len(loop_out_edges) >= 1 and \
                    loop_in_edges[1][2]['tensor'].is_const and \
                    loop_in_edges[1][2]['tensor'].value is not None:

                condition = loop_in_edges[1][2]['tensor'].value
                if not condition:
                    matched = True
                    break
                else:
                    if not loop_in_edges[0][2]['tensor'].is_const or \
                            loop_in_edges[0][2]['tensor'].value is None:
                        continue
                    if loop_obj.real_loop_cnt is None:
                        continue
                    matched = True
                    break
        return matched

    matched_if = check_decompose_const_if(graph)
    matched_loop = check_decompose_const_loop(graph)

    cnt = 0

    while (matched_if or matched_loop) and cnt < 50:
        cnt += 1
        if matched_if:
            _decompose_const_if(graph, params)
        if matched_loop:
            _decompose_const_loop(graph, params)

        matched_if = check_decompose_const_if(graph)
        matched_loop = check_decompose_const_loop(graph)

    if cnt > 0:
        from ....graph.graph_algo import infer
        infer(graph)

    if cnt >= 50:
        ERROR('[Parser]More than 50 times in decompose_const_if_loop, maybe in dead loop, Please double check!!!')


def decompose_pack(graph):
    matches = single_node_matcher(graph, 'ConcatFromSequence')
    for m in matches:
        pack_or_concat = m['target']
        pack_or_concat_obj = NodeWrap(graph, pack_or_concat)['object']
        if pack_or_concat_obj is not None:
            if graph._attr.get('quantize', False) \
                    and pack_or_concat_obj.quantize:
                quantize = True
            else:
                quantize = False
            if pack_or_concat_obj.new_axis:
                in_edges = graph.sorted_in_edges(
                    pack_or_concat, keys=True, data=True)
                input_shapes = pack_or_concat_obj.get_input_shapes()
                if len(in_edges) == len(input_shapes) \
                        and all([in_shape is not None for in_shape in input_shapes]) \
                        and all(meta_shape is not None for in_shape in input_shapes for meta_shape in in_shape):
                    reshape_dim = list(input_shapes[0])
                    pos = pack_or_concat_obj.axis if pack_or_concat_obj.axis >= 0 else (
                        len(reshape_dim) + 1 + pack_or_concat_obj.axis)
                    reshape_dim.insert(pos, 1)
                    for src, _, k, in_attr in in_edges:
                        insert_reshape(
                            graph, src, pack_or_concat, in_attr, reshape_dim, key=k, data_format='NCHW',
                            quantize=quantize)
                else:
                    ERROR(
                        '[Parser]: Meets invalid input shapes for Node (%s) in decompose_pack!' % pack_or_concat)
            node_attr = pack_or_concat_obj.copied_attr()
            node_attr.update({'opset_version': 4})
            NodeWrap(graph, pack_or_concat).replace_obj('Concat', node_attr)
        else:
            ERROR(
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
        if linear_obj.quantize:
            continue
        bias_obj = NodeWrap(graph, bias)['object']
        transpose_obj = NodeWrap(graph, transpose)['object']
        if linear_obj is not None and bias_obj is not None and (transpose is None or transpose_obj is not None):
            if transpose_obj is not None and (
                    len(graph.sorted_out_edges(transpose)) != 1 or transpose_obj.perm != [0, 2, 3, 1]):
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
            ERROR('[Parser]: Meets invalid Op in fuse_bias!')


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
                edges = [('const', 'gather_%s' % str(i), {'src_out_port': 0, 'dst_in_port': 0}) for i in
                         range(gather_num)] \
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
            ERROR('[Parser]: Meets invalid Operator object in fuse_gather_const_mul!')


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
            if linear_obj.weights is None or linear_obj.biases is None:
                WARN('[Parser]: Meets invalid %s Op (%s) in fuse_linear_bn!' % (linear_obj.type, linear))
                continue
            if bn_obj.training_mode:
                continue
            if len(graph.sorted_out_edges(linear)) != 1:
                continue
            if transpose_obj is not None and (
                    len(graph.sorted_out_edges(transpose)) != 1 or transpose_obj.perm != [0, 2, 3, 1]):
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
            ERROR('[Parser]: Meets invalid node in fuse_linear_bn!')
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
                                       ('const_2', 'add_sub', {'dst_in_port': 1})
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
            mul_in_edges = graph.sorted_in_edges(mul, data=True)
            mul_in_attr = [in_attr for (src, _, in_attr) in mul_in_edges if src != const_1]
            if len(mul_in_attr) < 1 \
                    or mul_in_attr[0]['tensor'] is None \
                    or mul_in_attr[0]['tensor'].get_dtype() is None \
                    or 'float' not in mul_in_attr[0]['tensor'].get_dtype():
                continue
            main_in_port = mul_in_attr[0]['dst_in_port']
            input_out_shape = input_shapes[0]
            if input_out_shape is None or input_out_shape == []:
                continue

            weights = NodeWrap(graph, const_1)['object'].value
            biases = NodeWrap(graph, const_2)['object'].value
            if weights is None or biases is None:
                continue

            if len(input_out_shape) > 2 and input_out_shape[1] == weights.size \
                    and len(weights.shape) > 1 and weights.shape[-1] == 1:
                num_output = input_out_shape[1]
                data_format = 'NCHW'
            else:
                num_output = input_out_shape[-1]
                data_format = 'NHWC'

            if (weights.size != num_output and weights.size != 1) \
                    or (biases.size != num_output and biases.size != 1):
                continue

            if len(input_out_shape) > 1 \
                    and len(weights.shape) > 1 \
                    and num_output not in list(weights.shape):
                continue

            if len(input_out_shape) > 1 \
                    and len(biases.shape) > 1 \
                    and num_output not in list(biases.shape):
                continue

            matched = True

            if weights.size < num_output and weights.size == 1:
                weights = np.tile(np.reshape(weights, [-1]), num_output)
            if biases.size < num_output and biases.size == 1:
                biases = np.tile(np.reshape(biases, [-1]), num_output)
            if add_sub_obj.type == 'Sub':
                biases = (-1.0 * biases).astype(np.float32)
            if np.ndim(weights) != 1:
                weights = np.reshape(weights, [-1])
            if np.ndim(biases) != 1:
                biases = np.reshape(biases, [-1])
            mean_value = np.zeros((num_output,), np.float32)
            var_value = np.ones((num_output,), np.float32)

            graph.remove_edge(const_1, mul)
            if main_in_port == 1:
                mul_in_edges[1][2]['dst_in_port'] = 0

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
            bn_attr.update({'epsilon': 0, 'data_format': data_format})
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
    )).difference(['ConvTranspose', 'MaxPool', 'MaxUnpool']).intersection(OnnxOp.get_concrete_subclass_names()))
    pad_fusing_combinations = itertools.product(
        pad_op_list, op_has_padding_list)
    for pad_op, op_has_padding in pad_fusing_combinations:
        matches = two_nodes_matcher(graph, pad_op, op_has_padding)
        for m in matches:
            pad, op_has_padding = m['begin'], m['end']
            pad_out_edges = graph.sorted_out_edges(pad)
            pad_obj = NodeWrap(graph, pad)['object']
            op_has_padding_obj = NodeWrap(graph, op_has_padding)['object']
            if pad_obj is not None and op_has_padding_obj is not None:
                if len(pad_out_edges) == 1 and pad_obj.is_fusable():
                    space_pads = pad_obj.space_pads()
                    op_has_padding_obj = NodeWrap(
                        graph, op_has_padding)['object']
                    init_pads = op_has_padding_obj.pads
                    fused_pads = np.reshape(np.array(init_pads, np.int64), newshape=(2, -1)) \
                        + np.reshape(np.array(space_pads, np.int64),
                                     newshape=(2, -1))
                    new_pads = fused_pads.flatten().tolist()
                    if op_has_padding_obj.type == 'AveragePool':
                        if any(pad != 0 for pad in new_pads) and not op_has_padding_obj.count_include_pad:
                            # Cannot fuse pad with the op with padding if not all the pads are 0 and count_include_pad is False
                            continue
                        op_has_padding_obj.count_include_pad = True
                    op_has_padding_obj.pads = new_pads
                    op_has_padding_obj.auto_pad = 'NOTSET'
                    pad_in_edges = graph.sorted_in_edges(pad, data=True)
                    src, _, attr = pad_in_edges[0]
                    graph.remove_edge(src, pad)
                    graph.add_edge(src, op_has_padding, **attr)
                    graph.remove_node(pad)
            else:
                ERROR('[Parser]: Meets invalid Pad Node (%s) in fuse_pad!' % pad)


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


def convert_center_crop_pad(graph):
    '''Convert onnx CenterCropPad to Slice+Pad.
    '''
    matched = False
    matches = single_node_matcher(graph, 'CenterCropPad')
    for m in matches:
        crop_pad = m['target']
        crop_pad_obj = NodeWrap(graph, crop_pad)['object']
        input_shapes = crop_pad_obj.get_input_shapes()
        in_edges = graph.sorted_in_edges(crop_pad, data=True)
        if crop_pad_obj is None or len(input_shapes) < 2 or len(in_edges) < 2:
            ERROR(
                '[Parser]: Meets invalid CenterCropPad Op(%s) in convert_center_crop_pad!' % crop_pad)
            continue
        input_shape = input_shapes[0]
        if input_shape is None or None in input_shape:
            ERROR(
                '[Parser]: Meets invalid input of CenterCropPad Op(%s) in convert_center_crop_pad!' % crop_pad)
            continue
        new_shape_in_attr = in_edges[1][2]
        if new_shape_in_attr['tensor'] is None \
                or not new_shape_in_attr['tensor'].is_const:
            WARN(
                '[Parser]: Meets unsupported non-constant shape(second input) of CenterCropPad Op(%s) in convert_center_crop_pad!'
                % crop_pad)
            continue
        matched = True
        new_shape_at_axes = new_shape_in_attr['tensor'].value
        axes = OpHasAxis.make_axes_non_negative(crop_pad_obj.axes, len(input_shape))
        new_shape, crop_slices, pad_slices = CenterCropPadOp.get_shape_and_crop_pad_slices(
            input_shape, new_shape_at_axes, axes)
        slice_begin = [sl.start for sl in crop_slices]
        slice_size = [input_shape[idx] if sl.stop is None else (
            sl.stop - sl.start) for idx, sl in enumerate(crop_slices)]
        pad_begin = [sl.start for sl in pad_slices]
        pad_end = [0 if sl.stop is None else (new_shape[idx] - sl.stop) for idx, sl in enumerate(pad_slices)]
        graph.remove_edges_from(in_edges)
        src, _, in_attr = in_edges[0]
        insert_slice(graph, src, crop_pad, in_attr, slice_begin, slice_size)
        pad_attr = crop_pad_obj.copied_attr()
        pad_attr.update({'mode': 'constant', 'paddings': pad_begin + pad_end, 'opset_version': 1})
        NodeWrap(graph, crop_pad).replace_obj('Pad', pad_attr)
    if matched:
        clear_redundant_nodes(graph)


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
            ERROR('[Parser]: Node (%s or %s) cannot be found, graph maybe has been changed!' % (
                p1, p2))
            continue
        p1_out_edges = graph.sorted_out_edges(p1)
        c1, c2 = m['const_1'], m['const_2']
        const_value_1 = NodeWrap(graph, c1)['object'].value
        const_value_2 = NodeWrap(graph, c2)['object'].value

        if NodeWrap(graph, p1)['object'].quantize and NodeWrap(graph, p2)['object'].quantize:
            quantized = True
        else:
            quantized = False

        if quantized:
            p1_out_edges = graph.sorted_out_edges(p1, data=True)
            p2_out_edges = graph.sorted_out_edges(p2, data=True)
            p1_scale_zp = p1_out_edges[0][2]['tensor'].scale_zp
            p2_scale_zp = p2_out_edges[0][2]['tensor'].scale_zp
            if p1_scale_zp != p2_scale_zp:
                continue
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
                         'min': clip_min, 'max': clip_max, 'quantize': quantized}
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
            ERROR(
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
        kernel_shape = np.array(input_shape)[np.array(axes)].tolist()
        if len(input_shape) == 5 and int(np.prod(kernel_shape)) > 256:
            continue
        if graph._attr.get('quantize', False) \
                and mean_obj.quantize:
            quantize = True
        else:
            quantize = False
        keepdim_out_shape = []
        for index in range(len(input_shape)):
            keepdim_out_shape.append(
                1 if index in axes else input_shape[index])
        cur_data_format = mean_obj.data_format
        need_transpose = False
        if (cur_data_format == 'NCHW' and axes == list(range(len(input_shape)))[1:-1]) \
                or (cur_data_format == 'NHWC' and axes == list(range(len(input_shape)))[2:]):
            need_transpose = True
        last_name = mean
        in_edges = graph.sorted_in_edges(mean, data=True)
        if need_transpose:
            if cur_data_format == 'NCHW':
                perm1 = [0, len(input_shape) - 1] + \
                    list(range(1, len(input_shape) - 1))
            else:
                perm1 = [0] + list(range(2, len(input_shape))) + [1]
            perm2 = Op.cal_inverse_perm(perm1)
            src, _, in_attr = in_edges[0]
            insert_transpose(graph, src, mean, in_attr, perm1, quantize=quantize)
            if not mean_obj.keepdims:
                for _, _, out_attr in graph.sorted_out_edges(mean, data=True):
                    if out_attr['tensor'].value is not None:
                        out_attr['tensor'].value = np.reshape(
                            out_attr['tensor'].value, keepdim_out_shape)
                    else:
                        out_attr['tensor'].shape = tuple(keepdim_out_shape)
            last_name = insert_transpose_after(graph, mean, perm2, quantize=quantize)
            if mean in graph._attr['output_names']:
                index = graph._attr['output_names'].index(mean)
                graph._attr['output_names'][index] = last_name
        if not mean_obj.keepdims:
            reshape = insert_reshape_after(
                graph, last_name, out_shape, keepdim_out_shape, quantize=quantize)
            if last_name in graph._attr['output_names']:
                index = graph._attr['output_names'].index(last_name)
                graph._attr['output_names'][index] = reshape
        graph.remove_edges_from(in_edges[1:])
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
            ERROR(
                '[Parser]: Meets invalid DequantizeLinear Op(%s) in convert_dequantizelinear!' % dequant)
            continue
        inp, _, inp_in_attr = dequant_in_edges[0]
        scale, _, scale_in_attr = dequant_in_edges[1]
        zp, _, zp_in_attr = dequant_in_edges[2]
        if graph._attr.get('quantize', False) \
                and scale_in_attr['tensor'].is_const and zp_in_attr['tensor'].is_const:
            inp_obj = NodeWrap(graph, inp)['object']
            if inp_obj is not None and not inp_obj.quantize:
                inp_obj.quantize = True
                if inp_in_attr['tensor'] is None:
                    inp_in_attr['tensor'] = Tensor()
                inp_in_attr['tensor'].dtype = zp_in_attr['tensor'].get_dtype()
                inp_in_attr['tensor'].scale_zp = (scale_in_attr['tensor'].value, zp_in_attr['tensor'].value)
                inp_obj.activation_quantization_axis = dequant_obj.axis
        else:
            input_shapes = dequant_obj.get_input_shapes()
            if dequant_obj.axis is not None and len(input_shapes[1]) == 1:
                dequant_axis = dequant_obj.axis
                dequant_axis = OpHasAxis.make_axes_non_negative(
                    dequant_axis, len(input_shapes[0]))
                if len(input_shapes[0]) <= dequant_axis:
                    ERROR(
                        '[Parser]: Meets invalid axis(%d) in DequantizeLinear Op(%s) in convert_dequantizelinear!' % (
                            dequant_axis, dequant))
                    continue
            else:
                dequant_axis = None
            if input_shapes[1] != input_shapes[2] \
                    or len(input_shapes[1]) not in (0, 1) \
                    or (dequant_axis is not None
                        and len(input_shapes[1]) == 1
                        and input_shapes[1][0] != input_shapes[0][dequant_axis]):
                ERROR(
                    '[Parser]: Meets different shapes of x_scale and x_zero_point in DequantizeLinear Op(%s) in convert_dequantizelinear!' % dequant)
                continue
            graph.remove_edges_from(dequant_in_edges)

            sub = get_valid_node_name(graph, dequant + '_sub')
            graph.add_node(sub)
            graph.add_edge(inp, sub, **inp_in_attr)
            new_zp_in_attr = copy.deepcopy(zp_in_attr)
            new_zp_in_attr.update({'dst_in_port': 1})
            graph.add_edge(zp, sub, **new_zp_in_attr)
            sub_out_attr = copy.deepcopy(inp_in_attr)
            sub_out_attr.update({'src_out_port': 0, 'dst_in_port': 0})
            if sub_out_attr['tensor'] is not None:
                if sub_out_attr['tensor'].value is not None:
                    sub_out_attr['tensor'].value = sub_out_attr['tensor'].value.astype(np.float32)
                else:
                    sub_out_attr['tensor'].dtype = 'float32'
            graph.add_edge(sub, dequant, **sub_out_attr)
            NodeWrap(graph, sub).replace_obj(
                'Sub', {'name': sub, 'opset_version': 13})

            graph.add_edge(scale, dequant, **scale_in_attr)
            mul_attr = dequant_obj.copied_attr()
            mul_attr.update({'opset_version': 13, 'quantize': False})
            NodeWrap(graph, dequant).replace_obj('Mul', mul_attr)

            insert_cast(graph, inp, sub, 'float32', inp_in_attr)
            float_zp = insert_cast(graph, zp, sub, 'float32', new_zp_in_attr)

            if len(input_shapes[1]) == 1 and dequant_axis is not None \
                    and dequant_axis != len(input_shapes[0]) - 1:
                dim = [1 if idx != dequant_axis else input_shapes[0][dequant_axis] for idx in range(
                    len(input_shapes[0]))]
                insert_reshape(graph, scale, dequant, scale_in_attr, dim)
                insert_reshape(graph, zp, float_zp, new_zp_in_attr, dim)


def convert_quantizelinear(graph):
    '''Convert QuantizeLinear to other ops(Div, Round, Add, Clip) if the node is not quantized or
    scale/zp is not Constant, otherwise just set dtype/scale_zp for out edges(the node is quantized
    and scale/zp is Constant).
    '''
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
            ERROR(
                '[Parser]: Meets invalid QuantizeLinear Op(%s) in convert_quantizelinear!' % quant)
            continue
        input_shapes = quant_obj.get_input_shapes()
        if quant_obj.axis is not None and len(input_shapes[1]) == 1:
            quant_axis = quant_obj.axis
        else:
            quant_axis = -1
        quant_axis = OpHasAxis.make_axes_non_negative(
            quant_axis, len(input_shapes[0]))
        if input_shapes[1] != input_shapes[2] \
                or len(input_shapes[1]) not in (0, 1) \
                or (len(input_shapes[1]) == 1 and input_shapes[1][0] != input_shapes[0][quant_axis]):
            ERROR(
                '[Parser]: Meets different shapes of y_scale and y_zero_point in QuantizeLinear Op(%s) in convert_quantizelinear!' % quant)
            continue
        quant_out_edges = graph.sorted_out_edges(quant, data=True)
        scale, _, scale_in_attr = quant_in_edges[1]
        zp, _, zp_in_attr = quant_in_edges[2]
        zp_dtype = zp_in_attr['tensor'].get_dtype()
        if zp_dtype is None:
            ERROR(
                '[Parser]: Meets invalid zp dtype of QuantizeLinear Op(%s) in convert_quantizelinear!' % quant)
            continue
        if graph._attr.get('quantize', False) \
                and scale_in_attr['tensor'].is_const \
                and zp_in_attr['tensor'].is_const:
            for _, dst, out_attr in quant_out_edges:
                if out_attr['tensor'] is None:
                    out_attr['tensor'] = Tensor()
                out_attr['tensor'].dtype = str(zp_dtype)
                out_attr['tensor'].scale_zp = (scale_in_attr['tensor'].value, zp_in_attr['tensor'].value)
                out_attr['tensor'].activation_quantization_axis = quant_obj.axis
        else:
            inp, _, inp_in_attr = quant_in_edges[0]
            inp_dtype = inp_in_attr['tensor'].get_dtype()
            if inp_dtype is None:
                ERROR(
                    '[Parser]: Meets invalid input dtype of QuantizeLinear Op(%s) in convert_quantizelinear!' % quant)
                continue
            graph.remove_edges_from(quant_in_edges)

            div = get_valid_node_name(graph, quant + '_div')
            graph.add_edge(inp, div, **inp_in_attr)
            div_in_attr = copy.deepcopy(scale_in_attr)
            div_in_attr.update({'dst_in_port': 1})
            graph.add_edge(scale, div, **div_in_attr)
            NodeWrap(graph, div).replace_obj(
                'Div', {'name': div, 'opset_version': 13})
            # Insert cast before quant if input dtype is not float32
            if inp_dtype != 'float32':
                insert_cast(graph, inp, div, 'float32', div_in_attr)

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
            cast = insert_cast(graph, zp, add, 'float32', add_in_attr)
            if len(input_shapes[1]) == 1 and quant_axis != len(input_shapes[0]) - 1:
                dim = [1 if idx != quant_axis else input_shapes[0][quant_axis]
                       for idx in range(len(input_shapes[0]))]
                insert_reshape(graph, scale, div, div_in_attr, dim)
                insert_reshape(graph, zp, cast, add_in_attr, dim)

            # Insert clip and cast after quant
            clip = get_valid_node_name(graph, quant + '_clip')
            graph.add_edge(add, clip, **common_attr)
            NodeWrap(graph, clip).replace_obj('Clip', {'name': clip,
                                                       'opset_version': 1,
                                                       'max': np.iinfo(zp_dtype).max,
                                                       'min': np.iinfo(zp_dtype).min})

            graph.add_edge(clip, quant, **common_attr)
            if graph._attr.get('quantize', False):
                for _, dst, out_attr in quant_out_edges:
                    if out_attr['tensor'] is None:
                        out_attr['tensor'] = Tensor()
                    out_attr['tensor'].dtype = str(zp_dtype)
                    out_attr['tensor'].scale_zp = [np.array([1.0]).astype(np.float32), np.array([0]).astype(np.int32)]
            post_cast_attr = quant_obj.copied_attr()
            post_cast_attr.update({'opset_version': 1, 'to': str(zp_dtype), 'quantize': False})
            NodeWrap(graph, quant).replace_obj('Cast', post_cast_attr)


def convert_qadd(graph):
    matched = False
    matches = single_node_matcher(graph, 'QLinearAddMs')
    for m in matches:
        qadd = m['target']
        qadd_obj = NodeWrap(graph, qadd)['object']
        in_edges = graph.sorted_in_edges(qadd, data=True)
        if qadd_obj is None or len(in_edges) < 7:
            ERROR('[Parser]: Meets invalid QLinearAddMs node(%s) in convert_qadd!' % qadd)
            continue
        qadd_src_a, qadd_src_b = in_edges[0][0], in_edges[1][0]
        qadd_src_a_obj = NodeWrap(graph, qadd_src_a)['object']
        qadd_src_b_obj = NodeWrap(graph, qadd_src_b)['object']
        if qadd_src_a_obj is None or qadd_src_b_obj is None:
            ERROR('[Parser]: Meets invalid input nodes(%s, %s) of QLinearAddMs in convert_qadd!' % (
                qadd_src_a, qadd_src_b))
            continue
        if any(e[2]['tensor'] is None or not e[2]['tensor'].is_const for e in (in_edges[1:3] + in_edges[4:])):
            WARN('[Parser]: Only supports QLinearAddMs(%s) with constant scale/zp in convert_qadd!' % qadd)
            continue
        a_dtype = str(qadd_obj.A_zero_point.dtype)
        b_dtype = str(qadd_obj.B_zero_point.dtype)
        y_dtype = str(qadd_obj.C_zero_point.dtype)
        if any(dtype not in ('int8', 'uint8') for dtype in (a_dtype, b_dtype, y_dtype)):
            ERROR('[Parser]: Meets invalid zero_point dtype of QLinearAddMs node(%s) in convert_qadd!' % qadd)
            continue
        matched = True
        graph.remove_edges_from(in_edges[1:])
        a_scale, a_zp = qadd_obj.A_scale, qadd_obj.A_zero_point
        b_scale, b_zp = qadd_obj.B_scale, qadd_obj.B_zero_point
        y_scale, y_zp = qadd_obj.C_scale, qadd_obj.C_zero_point
        qadd_attr = qadd_obj.copied_attr()
        input_a, _, input_a_in_attr = in_edges[0]
        input_b, _, input_b_in_attr = in_edges[3]
        input_b_in_attr.update({'dst_in_port': 1})
        graph.add_edge(input_b, qadd, **input_b_in_attr)
        if graph._attr.get('quantize', False):
            qadd_src_a_obj.quantize = True
            input_a_in_attr['tensor'].dtype = a_dtype
            input_a_in_attr['tensor'].scale_zp = (a_scale, a_zp)
            qadd_src_b_obj.quantize = True
            input_b_in_attr['tensor'].dtype = b_dtype
            input_b_in_attr['tensor'].scale_zp = (b_scale, b_zp)

            out_edges = graph.sorted_out_edges(qadd, data=True)
            for _, _, out_attr in out_edges:
                out_attr['tensor'].dtype = y_dtype
                out_attr['tensor'].scale_zp = (y_scale, y_zp)

            qadd_attr.update({'opset_version': 13, 'quantize': True})
        else:
            insert_cast_sub_mul_for_quant(graph, input_a, qadd, a_scale, a_zp,
                                          input_a_in_attr)
            insert_cast_sub_mul_for_quant(graph, input_b, qadd, b_scale, b_zp,
                                          input_b_in_attr)
            out_cast = insert_mul_add_cast_after_for_dequant(graph, qadd, y_dtype, y_scale,
                                                             y_zp)
            qadd_attr.update({'opset_version': 13})
            if qadd in graph._attr['output_names']:
                index = graph._attr['output_names'].index(qadd)
                graph._attr['output_names'][index] = out_cast
        NodeWrap(graph, qadd).replace_obj('Add', qadd_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_qavgpool(graph):
    matched = False
    matches = single_node_matcher(graph, 'QLinearAveragePoolMs')
    for m in matches:
        qpool = m['target']
        qpool_obj = NodeWrap(graph, qpool)['object']
        in_edges = graph.sorted_in_edges(qpool, data=True)
        if qpool_obj is None or len(in_edges) != 5:
            ERROR('[Parser]: Meets invalid QLinearAveragePoolMs node(%s) in convert_qavgpool!' % qpool)
            continue
        qpool_src = in_edges[0][0]
        qpool_src_obj = NodeWrap(graph, qpool_src)['object']
        if qpool_src_obj is None:
            ERROR('[Parser]: Meets invalid input node(%s) of QLinearAveragePoolMs in convert_qavgpool!' % qpool_src)
            continue
        if any((e[2]['tensor'] is None or not e[2]['tensor'].is_const) for e in in_edges[1:]):
            WARN('[Parser]: Only supports QLinearAveragePoolMs(%s) with constant scale/zp in convert_qavgpool!' % qpool)
            continue
        x_dtype = str(qpool_obj.x_zero_point.dtype)
        y_dtype = str(qpool_obj.y_zero_point.dtype)
        if any(dtype not in ('int8', 'uint8') for dtype in (x_dtype, y_dtype)):
            ERROR(
                '[Parser]: Meets invalid zero_point dtype of QLinearAveragePoolMs node(%s) in convert_qavgpool!' % qpool)
            continue
        matched = True
        graph.remove_edges_from(in_edges[1:])
        x_scale, x_zp = qpool_obj.x_scale, qpool_obj.x_zero_point
        y_scale, y_zp = qpool_obj.y_scale, qpool_obj.y_zero_point
        qpool_attr = qpool_obj.copied_attr()
        qpool_attr.update({'data_format': ('NCHW' if qpool_obj.channels_last == 0 else 'NHWC'),
                           'opset_version': 1})
        src, _, src_in_attr = in_edges[0]
        if graph._attr.get('quantize', False):
            qpool_src_obj.quantize = True
            src_in_attr['tensor'].dtype = str(x_dtype)
            src_in_attr['tensor'].scale_zp = (x_scale, x_zp)

            out_edges = graph.sorted_out_edges(qpool, data=True)
            for _, _, out_attr in out_edges:
                out_attr['tensor'].dtype = str(y_dtype)
                out_attr['tensor'].scale_zp = (y_scale, y_zp)

            qpool_attr.update({'quantize': True})
        else:
            insert_cast_sub_mul_for_quant(graph, src, qpool, x_scale, x_zp,
                                          src_in_attr)
            out_cast = insert_mul_add_cast_after_for_dequant(graph, qpool, y_dtype, y_scale,
                                                             y_zp)
            if qpool in graph._attr['output_names']:
                index = graph._attr['output_names'].index(qpool)
                graph._attr['output_names'][index] = out_cast
        NodeWrap(graph, qpool).replace_obj('AveragePool', qpool_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_qconcat(graph):
    matched = False
    op_type_name = 'QLinearConcatMs'
    matches = single_node_matcher(graph, op_type_name)
    for m in matches:
        qconcat = m['target']
        qconcat_obj = NodeWrap(graph, qconcat)['object']
        in_edges = graph.sorted_in_edges(qconcat, data=True)
        if qconcat_obj is None or len(in_edges) < 5:
            ERROR(f'[Parser]: Meets invalid {op_type_name} node({qconcat}) in convert_qconcat!')
            continue
        x_dtypes = []
        inp_invalid = False
        for i in range(2, len(in_edges), 3):
            qconcat_src = in_edges[i][0]
            qconcat_src_obj = NodeWrap(graph, qconcat_src)['object']
            if qconcat_src_obj is None:
                inp_invalid = True
                ERROR(f'[Parser]: Meets invalid input node({qconcat_src}) of {op_type_name} in convert_qconcat!')
                break
            if any((e[2]['tensor'] is None or not e[2]['tensor'].is_const) for e in in_edges[i + 1:i + 3]):
                inp_invalid = True
                WARN(f'[Parser]: Only supports {op_type_name}({qconcat}) with constant scale/zp in convert_qconcat!')
                break
            x_dtypes.append(in_edges[i + 2][2]['tensor'].dtype)

        if inp_invalid:
            continue
        y_dtype = str(qconcat_obj.y_zero_point.dtype)
        if any(dtype not in ('int8', 'uint8') for dtype in x_dtypes + [y_dtype]):
            ERROR(
                f'[Parser]: Meets invalid zero_point dtype of {op_type_name} node({qconcat}) in convert_qconcat!')
            continue
        matched = True
        y_scale, y_zp = qconcat_obj.y_scale, qconcat_obj.y_zero_point
        qconcat_attr = qconcat_obj.copied_attr()
        qconcat_attr.update({'opset_version': 4})
        graph.remove_edges_from(in_edges[:2])
        for i in range(2, len(in_edges), 3):
            graph.remove_edges_from(in_edges[i + 1:i + 3])
            src, _, src_in_attr = in_edges[i]
            qconcat_src_obj = NodeWrap(graph, src)['object']
            x_scale = in_edges[i + 1][2]['tensor'].value
            x_zp = in_edges[i + 2][2]['tensor'].value
            x_dtype = in_edges[i][2]['tensor'].dtype
            if graph._attr.get('quantize', False):
                qconcat_src_obj.quantize = True
                src_in_attr['tensor'].dtype = str(x_dtype)
                src_in_attr['tensor'].scale_zp = (x_scale, x_zp)
                out_edges = graph.sorted_out_edges(qconcat, data=True)
                for _, _, out_attr in out_edges:
                    out_attr['tensor'].dtype = str(y_dtype)
                    out_attr['tensor'].scale_zp = (y_scale, y_zp)
            else:
                insert_cast_sub_mul_for_quant(graph, src, qconcat, x_scale, x_zp,
                                              src_in_attr)

        if graph._attr.get('quantize', False):
            qconcat_attr.update({'quantize': True})
        else:
            out_cast = insert_mul_add_cast_after_for_dequant(graph, qconcat, y_dtype, y_scale,
                                                             y_zp)
            if qconcat in graph._attr['output_names']:
                index = graph._attr['output_names'].index(qconcat)
                graph._attr['output_names'][index] = out_cast
        NodeWrap(graph, qconcat).replace_obj('Concat', qconcat_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_qconv(graph):
    matched = False
    matches = single_node_matcher(graph, 'QLinearConv')
    for m in matches:
        qconv = m['target']
        qconv_obj = NodeWrap(graph, qconv)['object']
        in_edges = graph.sorted_in_edges(qconv, data=True)
        if qconv_obj is None or qconv_obj.num_output is None or len(in_edges) != 9:
            ERROR('[Parser]: Meets invalid QLinearConv node(%s) in convert_qconv!' % qconv)
            continue
        qconv_src = in_edges[0][0]
        qconv_src_obj = NodeWrap(graph, qconv_src)['object']
        if qconv_src_obj is None:
            ERROR('[Parser]: Meets invalid input node(%s) of QLinearConv in convert_qconv!' % qconv_src)
            continue
        if any(e[2]['tensor'].value is None for e in in_edges[1:]):
            ERROR('[Parser]: Meets invalid QLinearConv node(%s) to in convert_qconv!' % qconv)
            continue
        if any(not e[2]['tensor'].is_const for e in in_edges[1:]):
            WARN(
                '[Parser]: Only supports QLinearConv(%s) with constant weights/biases/scale/zp in convert_qconv!' % qconv)
            continue
        x_dtype = qconv_obj.x_zero_point.dtype
        y_dtype = qconv_obj.y_zero_point.dtype
        if str(x_dtype) not in ('int8', 'uint8') or str(y_dtype) not in ('int8', 'uint8'):
            ERROR('[Parser]: Meets invalid QLinearConv node(%s) in convert_qconv!' % qconv)
            continue
        matched = True
        x_scale, x_zp = qconv_obj.x_scale, qconv_obj.x_zero_point
        w_scale, w_zp = qconv_obj.w_scale, qconv_obj.w_zero_point
        y_scale, y_zp = qconv_obj.y_scale, qconv_obj.y_zero_point
        weights = qconv_obj.w
        biases = qconv_obj.B
        graph.remove_edges_from(in_edges[1:])
        conv_attr = qconv_obj.copied_attr()
        if graph._attr.get('quantize', False):
            b_scale = x_scale * w_scale
            b_zp = np.zeros(b_scale.shape, np.int32)
            qconv_src_obj.quantize = True
            in_edges[0][2]['tensor'].dtype = str(x_dtype)
            in_edges[0][2]['tensor'].scale_zp = (x_scale, x_zp)

            out_edges = graph.sorted_out_edges(qconv, data=True)
            for _, _, out_attr in out_edges:
                out_attr['tensor'].dtype = str(y_dtype)
                out_attr['tensor'].scale_zp = (y_scale, y_zp)

            conv_attr.update({'opset_version': 1,
                              'quantize': True,
                              'weights': weights,
                              'weights_scale_zp': [w_scale, w_zp],
                              'biases': biases,
                              'biases_scale_zp': [b_scale, b_zp]})

        else:
            spatial_len = len(weights.shape) - 2
            w_scal_zp_reshape_dim = [-1] + [1] * (spatial_len + 1)
            weights = (weights.astype(np.int32) - np.reshape(w_zp, w_scal_zp_reshape_dim)) \
                * np.reshape(w_scale, w_scal_zp_reshape_dim)
            weights = weights.astype(np.float32)
            biases = biases.astype(np.int32) * x_scale * w_scale
            biases = biases.astype(np.float32)
            conv_attr.update({'opset_version': 1, 'weights': weights, 'biases': biases})

            src, _, in_attr = in_edges[0]
            insert_cast_sub_mul_for_quant(graph, src, qconv, x_scale, x_zp,
                                          in_attr, data_format=qconv_obj.data_format)
            out_cast = insert_mul_add_cast_after_for_dequant(graph, qconv, y_dtype, y_scale,
                                                             y_zp,
                                                             data_format=qconv_obj.data_format)

            if qconv in graph._attr['output_names']:
                index = graph._attr['output_names'].index(qconv)
                graph._attr['output_names'][index] = out_cast

        NodeWrap(graph, qconv).replace_obj('Conv', conv_attr)

    if matched:
        clear_redundant_nodes(graph)


def convert_qgemm(graph):
    matched = False
    matches = single_node_matcher(graph, 'QGemmMs')
    for m in matches:
        qgemm = m['target']
        qgemm_obj = NodeWrap(graph, qgemm)['object']
        in_edges = graph.sorted_in_edges(qgemm, data=True)
        if qgemm_obj is None or len(in_edges) < 6:
            ERROR('[Parser]: Meets invalid QLinearGemmMs node(%s) in convert_qgemm!' % qgemm)
            continue
        qgemm_src_a, qgemm_src_b = in_edges[0][0], in_edges[1][0]
        qgemm_src_a_obj = NodeWrap(graph, qgemm_src_a)['object']
        qgemm_src_b_obj = NodeWrap(graph, qgemm_src_b)['object']
        if qgemm_src_a_obj is None or qgemm_src_b_obj is None:
            ERROR('[Parser]: Meets invalid input nodes(%s, %s) of QLinearAddMs in convert_qgemm!' %
                  (qgemm_src_a, qgemm_src_b))
            continue
        if any((e[2]['tensor'] is None or not e[2]['tensor'].is_const) for e in
               (in_edges[1:3] + in_edges[4:6] + in_edges[7:])):
            WARN('[Parser]: Only supports QLinearGemmMs(%s) with constant scale/zp in convert_qgemm!' % qgemm)
            continue
        a_dtype = str(qgemm_obj.a_zero_point.dtype)
        b_dtype = str(qgemm_obj.b_zero_point.dtype)
        if any(dtype not in ('int8', 'uint8') for dtype in (a_dtype, b_dtype)):
            ERROR('[Parser]: Meets invalid zero_point dtype of QLinearGemmMs node(%s) in convert_qgemm!' % qgemm)
            continue
        matched = True
        a_scale, a_zp = qgemm_obj.a_scale, qgemm_obj.a_zero_point
        b_scale, b_zp = qgemm_obj.b_scale, qgemm_obj.b_zero_point
        y_scale, y_zp = qgemm_obj.y_scale, qgemm_obj.y_zero_point
        c_scale = qgemm_obj.alpha * a_scale * b_scale
        c_zp = np.array(0, dtype=np.int32)
        qgemm_attr = qgemm_obj.copied_attr()
        qgemm_attr.update({'opset_version': 13})
        a_src, _, a_src_in_attr = in_edges[0]
        b_src, _, b_src_in_attr = in_edges[3]
        c_src, _, c_src_in_attr = in_edges[6]
        c_is_not_none = (c_src_in_attr['tensor'] is None
                         or not c_src_in_attr['tensor'].is_const
                         or c_src_in_attr['tensor'].value is not None)

        graph.remove_edges_from(in_edges[1:])
        if graph._attr.get('quantize', False):
            if y_scale is None:  # y_dtype is float32
                graph.remove_edge(a_src, qgemm)
                a_dequant = get_valid_node_name(graph, qgemm + '_dequant_A')
                graph.add_edge(a_src, a_dequant, **a_src_in_attr)
                insert_constant(graph, a_dequant + '_scale', np.array(a_scale), a_dequant, in_port=1)
                insert_constant(graph, a_dequant + '_zp', np.array(a_zp), a_dequant, in_port=2)
                graph.add_edge(a_dequant, qgemm)
                NodeWrap(graph, a_dequant).replace_obj('DequantizeLinear',
                                                       {'name': a_dequant, 'opset_version': 13, 'axis': -1})
                b_dequant = get_valid_node_name(graph, qgemm + '_dequant_B')
                b_dequant_in_attr = copy.deepcopy(b_src_in_attr)
                b_dequant_in_attr.update({'dst_in_port': 0})
                graph.add_edge(b_src, b_dequant, **b_dequant_in_attr)
                insert_constant(graph, b_dequant + '_scale', np.array(b_scale), b_dequant, in_port=1)
                insert_constant(graph, b_dequant + '_zp', np.array(b_zp), b_dequant, in_port=2)
                graph.add_edge(b_dequant, qgemm, **{'dst_in_port': 1})
                NodeWrap(graph, b_dequant).replace_obj('DequantizeLinear',
                                                       {'name': b_dequant, 'opset_version': 13, 'axis': -1})
                if c_is_not_none:
                    c_dequant = get_valid_node_name(graph, qgemm + '_dequant_C')
                    c_dequant_in_attr = copy.deepcopy(c_src_in_attr)
                    c_dequant_in_attr.update({'dst_in_port': 0})
                    graph.add_edge(c_src, c_dequant, **c_dequant_in_attr)
                    insert_constant(graph, c_dequant + '_scale', np.array(c_scale), c_dequant, in_port=1)
                    insert_constant(graph, c_dequant + '_zp', np.array(c_zp), c_dequant, in_port=2)
                    graph.add_edge(c_dequant, qgemm, **{'dst_in_port': 2})
                    NodeWrap(graph, c_dequant).replace_obj('DequantizeLinear',
                                                           {'name': c_dequant, 'opset_version': 13, 'axis': -1})

                qgemm_attr.update({'quantize': False})
            else:
                qgemm_src_a_obj.quantize = True
                a_src_in_attr['tensor'].dtype = a_dtype
                a_src_in_attr['tensor'].scale_zp = (a_scale, a_zp)
                qgemm_src_b_obj.quantize = True
                b_src_in_attr['tensor'].dtype = b_dtype
                b_src_in_attr['tensor'].scale_zp = (b_scale, b_zp)
                b_src_in_attr['dst_in_port'] = 1
                graph.add_edge(b_src, qgemm, **b_src_in_attr)
                if c_is_not_none:
                    c_src_in_attr['tensor'].dtype = 'int32'
                    c_src_in_attr['tensor'].scale_zp = (c_scale, c_zp)
                    c_src_in_attr['dst_in_port'] = 2
                    graph.add_edge(c_src, qgemm, **c_src_in_attr)
                out_edges = graph.sorted_out_edges(qgemm, data=True)
                for _, _, out_attr in out_edges:
                    out_attr['tensor'].dtype = np.dtype(a_dtype)
                    out_attr['tensor'].scale_zp = (y_scale, y_zp)

                qgemm_attr.update({'quantize': True})
        else:
            insert_cast_sub_mul_for_quant(graph, a_src, qgemm, a_scale, a_zp,
                                          a_src_in_attr)
            b_src_in_attr['dst_in_port'] = 1
            graph.add_edge(b_src, qgemm, **b_src_in_attr)
            insert_cast_sub_mul_for_quant(graph, b_src, qgemm, b_scale, b_zp,
                                          b_src_in_attr)
            if c_is_not_none:
                c_src_in_attr['dst_in_port'] = 2
                graph.add_edge(c_src, qgemm, **c_src_in_attr)
                insert_cast_sub_mul_for_quant(graph, c_src, qgemm, c_scale, c_zp,
                                              c_src_in_attr)
            if y_scale is not None and y_zp is not None:
                out_cast = insert_mul_add_cast_after_for_dequant(graph, qgemm, a_dtype, y_scale,
                                                                 y_zp)
                if qgemm in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(qgemm)
                    graph._attr['output_names'][index] = out_cast
        NodeWrap(graph, qgemm).replace_obj('Gemm', qgemm_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_qglobal_avgpool(graph):
    matched = False
    matches = single_node_matcher(graph, 'QLinearGlobalAveragePoolMs')
    for m in matches:
        qpool = m['target']
        qpool_obj = NodeWrap(graph, qpool)['object']
        in_edges = graph.sorted_in_edges(qpool, data=True)
        if qpool_obj is None or len(in_edges) != 5:
            ERROR('[Parser]: Meets invalid QLinearGlobalAveragePoolMs node(%s) in convert_qglobal_avgpool!' % qpool)
            continue
        qpool_src = in_edges[0][0]
        qpool_src_obj = NodeWrap(graph, qpool_src)['object']
        if qpool_src_obj is None:
            ERROR('[Parser]: Meets invalid input node(%s) of QLinearConv in convert_qpool!' % qpool_src)
            continue
        if any((e[2]['tensor'] is None or not e[2]['tensor'].is_const) for e in in_edges[1:]):
            WARN(
                '[Parser]: Only supports QLinearGlobalAveragePoolMs(%s) with constant scale/zp in convert_qglobal_avgpool!' % qpool)
            continue
        x_dtype = str(qpool_obj.x_zero_point.dtype)
        y_dtype = str(qpool_obj.y_zero_point.dtype)
        if any(dtype not in ('int8', 'uint8') for dtype in (x_dtype, y_dtype)):
            ERROR(
                '[Parser]: Meets invalid zero_point dtype of QLinearGlobalAveragePoolMs node(%s) in convert_qglobal_avgpool!' % qpool)
            continue
        matched = True
        graph.remove_edges_from(in_edges[1:])
        x_scale, x_zp = qpool_obj.x_scale, qpool_obj.x_zero_point
        y_scale, y_zp = qpool_obj.y_scale, qpool_obj.y_zero_point
        qpool_attr = qpool_obj.copied_attr()
        qpool_attr.update({'data_format': ('NCHW' if qpool_obj.channels_last == 0 else 'NHWC'),
                           'opset_version': 1})
        src, _, src_in_attr = in_edges[0]
        if graph._attr.get('quantize', False):
            qpool_src_obj.quantize = True
            src_in_attr['tensor'].dtype = str(x_dtype)
            src_in_attr['tensor'].scale_zp = (x_scale, x_zp)

            out_edges = graph.sorted_out_edges(qpool, data=True)
            for _, _, out_attr in out_edges:
                out_attr['tensor'].dtype = str(y_dtype)
                out_attr['tensor'].scale_zp = (y_scale, y_zp)

            qpool_attr.update({'quantize': True})
        else:
            insert_cast_sub_mul_for_quant(graph, src, qpool, x_scale, x_zp,
                                          src_in_attr)
            out_cast = insert_mul_add_cast_after_for_dequant(graph, qpool, y_dtype, y_scale,
                                                             y_zp)
            if qpool in graph._attr['output_names']:
                index = graph._attr['output_names'].index(qpool)
                graph._attr['output_names'][index] = out_cast
        NodeWrap(graph, qpool).replace_obj('GlobalAveragePool', qpool_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_qleakyrelu(graph):
    matched = False
    matches = single_node_matcher(graph, 'QLinearLeakyReluMs')
    for m in matches:
        qleakyrelu = m['target']
        qleakyrelu_obj = NodeWrap(graph, qleakyrelu)['object']
        in_edges = graph.sorted_in_edges(qleakyrelu, data=True)
        if qleakyrelu_obj is None or len(in_edges) < 4:
            ERROR('[Parser]: Meets invalid QLinearLeakyReluMsOp node(%s) in convert_qleakyrelu!' % qleakyrelu)
            continue
        qleakyrelu_src = in_edges[0][0]
        qleakyrelu_src_obj = NodeWrap(graph, qleakyrelu_src)['object']
        if qleakyrelu_src_obj is None:
            ERROR(
                '[Parser]: Meets invalid input node(%s) of QLinearLeakyReluMsOp in convert_qleakyrelu!' % qleakyrelu_src)
            continue
        if any((e[2]['tensor'] is None or not e[2]['tensor'].is_const) for e in in_edges[1:]):
            WARN(
                '[Parser]: Only supports QLinearLeakyReluMsOp(%s) with constant scale/zp in convert_qleakyrelu!' % qleakyrelu)
            continue
        x_dtype = str(qleakyrelu_obj.X_zero_point.dtype)
        y_dtype = str(qleakyrelu_obj.Y_zero_point.dtype)
        if any(dtype not in ('int8', 'uint8') for dtype in (x_dtype, y_dtype)):
            ERROR(
                '[Parser]: Meets invalid zero_point dtype of QLinearLeakyReluMsOp node(%s) in convert_qleakyrelu!' % qleakyrelu)
            continue
        matched = True
        graph.remove_edges_from(in_edges[1:])
        x_scale, x_zp = qleakyrelu_obj.X_scale, qleakyrelu_obj.X_zero_point
        y_scale, y_zp = qleakyrelu_obj.Y_scale, qleakyrelu_obj.Y_zero_point
        qleakyrelu_attr = qleakyrelu_obj.copied_attr()
        qleakyrelu_attr.update({'opset_version': 6})
        src, _, src_in_attr = in_edges[0]
        if graph._attr.get('quantize', False):
            qleakyrelu_src_obj.quantize = True
            src_in_attr['tensor'].dtype = str(x_dtype)
            src_in_attr['tensor'].scale_zp = (x_scale, x_zp)

            out_edges = graph.sorted_out_edges(qleakyrelu, data=True)
            for _, _, out_attr in out_edges:
                out_attr['tensor'].dtype = str(y_dtype)
                out_attr['tensor'].scale_zp = (y_scale, y_zp)

            qleakyrelu_attr.update({'quantize': True})
        else:
            insert_cast_sub_mul_for_quant(graph, src, qleakyrelu, x_scale, x_zp,
                                          src_in_attr)
            out_cast = insert_mul_add_cast_after_for_dequant(graph, qleakyrelu, y_dtype, y_scale,
                                                             y_zp)
            if qleakyrelu in graph._attr['output_names']:
                index = graph._attr['output_names'].index(qleakyrelu)
                graph._attr['output_names'][index] = out_cast
        NodeWrap(graph, qleakyrelu).replace_obj('LeakyRelu', qleakyrelu_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_qmatmul(graph):
    matched = False
    matches = single_node_matcher(graph, 'QLinearMatMul')
    for m in matches:
        qmatmul = m['target']
        qmatmul_obj = NodeWrap(graph, qmatmul)['object']
        in_edges = graph.sorted_in_edges(qmatmul, data=True)
        if qmatmul_obj is None or len(in_edges) != 8:
            ERROR('[Parser]: Meets invalid QLinearMatMul node(%s) in convert_qmatmul!' % qmatmul)
            continue
        qmatmul_src_a, qmatmul_src_b = in_edges[0][0], in_edges[3][0]
        qmatmul_src_a_obj = NodeWrap(graph, qmatmul_src_a)['object']
        qmatmul_src_b_obj = NodeWrap(graph, qmatmul_src_b)['object']
        if qmatmul_src_a_obj is None or qmatmul_src_b_obj is None:
            ERROR('[Parser]: Meets invalid input nodes(%s, %s) of QLinearAddMs in convert_qmatmul!' %
                  (qmatmul_src_a, qmatmul_src_b))
            continue
        if any((e[2]['tensor'] is None or e[2]['tensor'].value is None) for e in (in_edges[1:3] + in_edges[4:])):
            ERROR('[Parser]: Meets invalid QLinearMatMul node(%s) to in convert_qmatmul!' % qmatmul)
            continue
        if any(not e[2]['tensor'].is_const for e in (in_edges[1:3] + in_edges[4:])):
            WARN('[Parser]: Only supports QLinearMatMul(%s) with constant scale/zp in convert_qmatmul!' % qmatmul)
            continue
        a_dtype = qmatmul_obj.a_zero_point.dtype
        b_dtype = qmatmul_obj.b_zero_point.dtype
        y_dtype = qmatmul_obj.y_zero_point.dtype
        if any(str(dtype) not in ('int8', 'uint8') for dtype in (a_dtype, b_dtype, y_dtype)):
            ERROR('[Parser]: Meets invalid zero_point dtype of QLinearMatMul node(%s) in convert_qmatmul!' % qmatmul)
            continue
        matched = True
        graph.remove_edges_from(in_edges[1:])
        a_scale, a_zp = qmatmul_obj.a_scale, qmatmul_obj.a_zero_point
        b_scale, b_zp = qmatmul_obj.b_scale, qmatmul_obj.b_zero_point
        y_scale, y_zp = qmatmul_obj.y_scale, qmatmul_obj.y_zero_point
        qmatmul_attr = qmatmul_obj.copied_attr()
        input_a, _, input_a_in_attr = in_edges[0]
        input_b, _, input_b_in_attr = in_edges[3]
        input_b_in_attr.update({'dst_in_port': 1})
        graph.add_edge(input_b, qmatmul, **input_b_in_attr)
        if graph._attr.get('quantize', False):
            qmatmul_src_a_obj.quantize = True
            input_a_in_attr['tensor'].dtype = str(a_dtype)
            input_a_in_attr['tensor'].scale_zp = (a_scale, a_zp)
            qmatmul_src_b_obj.quantize = True
            input_b_in_attr['tensor'].dtype = str(b_dtype)
            input_b_in_attr['tensor'].scale_zp = (b_scale, b_zp)

            out_edges = graph.sorted_out_edges(qmatmul, data=True)
            for _, _, out_attr in out_edges:
                out_attr['tensor'].dtype = str(y_dtype)
                out_attr['tensor'].scale_zp = (y_scale, y_zp)

            qmatmul_attr.update({'opset_version': 13, 'quantize': True})
        else:
            insert_cast_sub_mul_for_quant(graph, input_a, qmatmul, a_scale, a_zp,
                                          input_a_in_attr)
            insert_cast_sub_mul_for_quant(graph, input_b, qmatmul, b_scale, b_zp,
                                          input_b_in_attr)
            out_cast = insert_mul_add_cast_after_for_dequant(graph, qmatmul, y_dtype, y_scale,
                                                             y_zp)
            qmatmul_attr.update({'opset_version': 13})
            if qmatmul in graph._attr['output_names']:
                index = graph._attr['output_names'].index(qmatmul)
                graph._attr['output_names'][index] = out_cast
        NodeWrap(graph, qmatmul).replace_obj('MatMul', qmatmul_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_qsigmoid(graph):
    matched = False
    matches = single_node_matcher(graph, 'QLinearSigmoidMs')
    for m in matches:
        qsigmoid = m['target']
        qsigmoid_obj = NodeWrap(graph, qsigmoid)['object']
        in_edges = graph.sorted_in_edges(qsigmoid, data=True)
        if qsigmoid_obj is None or len(in_edges) < 4:
            ERROR('[Parser]: Meets invalid QLinearSigmoidMsOp node(%s) in convert_qsigmoid!' % qsigmoid)
            continue
        qsigmoid_src = in_edges[0][0]
        qsigmoid_src_obj = NodeWrap(graph, qsigmoid_src)['object']
        if qsigmoid_src_obj is None:
            ERROR('[Parser]: Meets invalid input node(%s) of QLinearSigmoidMsOp in convert_qsigmoid!' % qsigmoid_src)
            continue
        if any((e[2]['tensor'] is None or not e[2]['tensor'].is_const) for e in in_edges[1:]):
            WARN(
                '[Parser]: Only supports QLinearSigmoidMsOp(%s) with constant scale/zp in convert_qsigmoid!' % qsigmoid)
            continue
        x_dtype = str(qsigmoid_obj.X_zero_point.dtype)
        y_dtype = str(qsigmoid_obj.Y_zero_point.dtype)
        if any(dtype not in ('int8', 'uint8') for dtype in (x_dtype, y_dtype)):
            ERROR(
                '[Parser]: Meets invalid zero_point dtype of QLinearSigmoidMsOp node(%s) in convert_qsigmoid!' % qsigmoid)
            continue
        matched = True
        graph.remove_edges_from(in_edges[1:])
        x_scale, x_zp = qsigmoid_obj.X_scale, qsigmoid_obj.X_zero_point
        y_scale, y_zp = qsigmoid_obj.Y_scale, qsigmoid_obj.Y_zero_point
        qsigmoid_attr = qsigmoid_obj.copied_attr()
        qsigmoid_attr.update({'opset_version': 6})
        src, _, src_in_attr = in_edges[0]
        if graph._attr.get('quantize', False):
            qsigmoid_src_obj.quantize = True
            src_in_attr['tensor'].dtype = str(x_dtype)
            src_in_attr['tensor'].scale_zp = (x_scale, x_zp)

            out_edges = graph.sorted_out_edges(qsigmoid, data=True)
            for _, _, out_attr in out_edges:
                out_attr['tensor'].dtype = str(y_dtype)
                out_attr['tensor'].scale_zp = (y_scale, y_zp)

            qsigmoid_attr.update({'quantize': True})
        else:
            insert_cast_sub_mul_for_quant(graph, src, qsigmoid, x_scale, x_zp,
                                          src_in_attr)
            out_cast = insert_mul_add_cast_after_for_dequant(graph, qsigmoid, y_dtype, y_scale,
                                                             y_zp)
            if qsigmoid in graph._attr['output_names']:
                index = graph._attr['output_names'].index(qsigmoid)
                graph._attr['output_names'][index] = out_cast
        NodeWrap(graph, qsigmoid).replace_obj('Sigmoid', qsigmoid_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_qnorm_to_float(graph):
    if not graph._attr.get('quantize', False):
        return
    from .common_passes import insert_dequant_quant
    matches = matched_patterns(graph,
                               nodes=[
                                   ('in', {'op': 'InstanceNormalization'}),
                                   ('relu', {'op': 'Relu'}),
                               ],
                               edges=[
                                   ('in', 'relu'),
                               ])
    matches2 = single_node_matcher(graph, 'LayerNormalization')
    matches += matches2
    for m in matches:
        qnorm = m['in'] if 'in' in m else m['target']
        qrelu = m['relu'] if 'relu' in m else None
        qnorm_obj = NodeWrap(graph, qnorm)['object']
        if qrelu is not None:
            qrelu_obj = NodeWrap(graph, qrelu)['object']
        in_edges = graph.sorted_in_edges(qnorm, data=True)
        if qnorm_obj is None:
            ERROR(f'[Parser]: Meets invalid QNorm node({qnorm}) in convert_qnorm_to_float!')
            continue
        src, _, in_attr = in_edges[0]
        insert_dequant_quant(graph, src, qnorm, in_attr, 'DequantizeLinear')

        # convert qnorm to float
        if len(in_edges) == 1:
            q_weights = qnorm_obj.weights
            w_scale, w_zp = qnorm_obj.weights_scale_zp
            f_weights = (q_weights - w_zp) * w_scale
            f_weights = f_weights.astype(np.float32)
            qnorm_obj.weights = f_weights
            qnorm_obj.weights_scale_zp = ()
            q_biases = qnorm_obj.biases
            b_scale, b_zp = qnorm_obj.biases_scale_zp
            f_biases = (q_biases - b_zp) * b_scale
            f_biases = f_biases.astype(np.float32)
            qnorm_obj.biases = f_biases
            qnorm_obj.biases_scale_zp = ()
            qnorm_obj.quantize = False
        else:
            q_weights_obj = NodeWrap(graph, in_edges[1][0])['object']
            q_weights = q_weights_obj.value
            q_weights_obj.quantize = False
            w_scale, w_zp = in_edges[1][-1]['tensor'].scale_zp
            f_weights = (q_weights - w_zp) * w_scale
            f_weights = f_weights.astype(np.float32)
            q_weights_obj.value = f_weights
            in_edges[1][-1]['tensor'].scale_zp = ()
            in_edges[1][-1]['tensor'].dtype = f_weights.dtype
            if len(in_edges) > 2:
                q_biases_obj = NodeWrap(graph, in_edges[2][0])['object']
                q_biases_obj.quantize = False
                q_biases = q_biases_obj.value
                b_scale, b_zp = in_edges[2][-1]['tensor'].scale_zp
                f_biases = (q_biases - b_zp) * b_scale
                f_biases = f_biases.astype(np.float32)
                q_biases_obj.value = f_biases
                in_edges[2][-1]['tensor'].scale_zp = ()
                in_edges[2][-1]['tensor'].dtype = f_biases.dtype
            qnorm_obj.quantize = False

        if qrelu is not None:
            qrelu_obj.quantize = False
            out_edges = graph.sorted_out_edges(qrelu, data=True)
        else:
            out_edges = graph.sorted_out_edges(qnorm, data=True)

        for _, dst, out_attr in out_edges:
            if qrelu is not None:
                insert_dequant_quant(graph, qrelu, dst, out_attr, 'QuantizeLinear')
            else:
                insert_dequant_quant(graph, qnorm, dst, out_attr, 'QuantizeLinear')


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
            reshape2_out_edges = graph.sorted_out_edges(reshape2, data=True)
            if len(reshape1_out_edges) == 1 and len(transpose_out_edges) == 1 \
                    and len(reshape2_out_edges) >= 1:
                if reshape1_obj.quantize:
                    reshape1_out_scale_zp = reshape1_out_edges[0][2]['tensor'].scale_zp
                    transpose_out_scale_zp = transpose_out_edges[0][2]['tensor'].scale_zp
                    reshape2_out_scale_zp = reshape2_out_edges[0][2]['tensor'].scale_zp
                    if len(reshape1_out_scale_zp) != 2 \
                            or len(transpose_out_scale_zp) != 2 \
                            or len(reshape2_out_scale_zp) != 2 \
                            or not FLOAT_EQUAL(reshape1_out_scale_zp[0], transpose_out_scale_zp[0]) \
                            or not FLOAT_EQUAL(reshape2_out_scale_zp[0], transpose_out_scale_zp[0]) \
                            or not np.array_equal(reshape1_out_scale_zp[1], transpose_out_scale_zp[1]) \
                            or not np.array_equal(reshape2_out_scale_zp[1], transpose_out_scale_zp[1]):
                        continue
                reshape1_in_shape = reshape1_obj.get_input_shapes()[0]
                reshape1_out_shape = reshape1_obj.get_output_shapes()[0]
                reshape2_out_shape = reshape2_obj.get_output_shapes()[0]
                if reshape1_in_shape is not None \
                        and reshape1_out_shape is not None \
                        and len(reshape1_out_shape) > 2:
                    need_insert_front_reshape = False
                    need_insert_after_reshape = False
                    if reshape1_in_shape != reshape2_out_shape:
                        need_insert_after_reshape = True
                    if (len(reshape1_in_shape) == len(reshape1_out_shape) or len(reshape1_in_shape) + 1 == len(
                            reshape1_out_shape)) \
                            and (int(np.prod(reshape1_out_shape[-2:])) == reshape1_in_shape[-1]
                                 if transpose_obj.data_format == 'NHWC'
                                 else int(np.prod(reshape1_out_shape[1:3])) == reshape1_in_shape[1]):
                        pass
                    elif len(reshape1_in_shape) == len(reshape1_out_shape) + 1 \
                            and (reshape1_out_shape[-2:] == reshape1_in_shape[-2:]
                                 if transpose_obj.data_format == 'NHWC'
                                 else reshape1_out_shape[1:3] == reshape1_in_shape[1:3]):
                        need_insert_front_reshape = True
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
                        if need_insert_front_reshape:
                            insert_reshape(graph, src, reshape1,
                                           in_attr, reshape2_out_shape,
                                           quantize=reshape1_obj.quantize)
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

                        if need_insert_after_reshape:
                            out_edges = graph.sorted_out_edges(reshape2, data=True)
                            _, dst, out_attr = out_edges[0]
                            after_rs = insert_reshape(graph, reshape2, dst,
                                                      out_attr, reshape2_out_shape,
                                                      quantize=reshape2_obj.quantize)
                            if reshape2 in graph._attr['output_names']:
                                index = graph._attr['output_names'].index(reshape2)
                                graph._attr['output_names'][index] = after_rs
        else:
            ERROR('[Parser]]: Meets invalid Op in merge_channel_shuffle!')
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
                        else int(np.prod([reshape1_in_shape[0]] + reshape1_in_shape[-(nhw_dim - 1):])) ==
                            reshape2_out_shape[1]):
                        perm_dim = len(transpose_obj.perm)
                        ref_perm = list(range(perm_dim - 2)) + [perm_dim - 1, perm_dim - 2] \
                            if transpose_obj.data_format == 'NHWC' \
                            else [0, 2, 1] + list(range(3, perm_dim))
                        if transpose_obj.perm == ref_perm:
                            group = reshape1_out_shape[-2] if transpose_obj.data_format == 'NHWC' else \
                                reshape1_out_shape[1]
                            if len(split_obj.split) == group \
                                    and all([split_obj.split[0] == s for s in split_obj.split[1:]]) \
                                    and (split_obj.axis in (
                                    -1, perm_dim - 1) if split_obj.data_format == 'NHWC' else split_obj.axis == 1):
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
            ERROR('[Parser]]: Meets invalid Op in merge_channel_shuffle_with_split!')
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
            ERROR('[Parser]]: Meets invalid Op in merge_channel_shuffle!')
            continue

        if not pack_obj.new_axis:
            continue

        pack_out_edges = graph.sorted_out_edges(pack, data=True)
        transpose_out_edges = graph.sorted_out_edges(transpose, data=True)
        if len(pack_out_edges) != 1 or len(transpose_out_edges) != 1:
            continue

        pack_in_shapes = pack_obj.get_input_shapes()
        if len(pack_in_shapes) < 1 \
                or any(s is None for shape in pack_in_shapes for s in shape):
            continue

        pack_out_shapes = pack_obj.get_output_shapes()
        if len(pack_out_shapes) < 1 \
                or any(s is None for shape in pack_out_shapes for s in shape) \
                or len(pack_out_shapes[0]) <= 2:
            continue
        pack_out_shape = pack_obj.get_output_shapes()[0]

        reshape_out_shapes = reshape_obj.get_output_shapes()
        if len(reshape_out_shapes) < 1 \
                or reshape_out_shapes[0] is None \
                or None in reshape_out_shapes[0]:
            continue

        reshape_out_shape = list(reshape_obj.get_output_shapes()[0])

        concat_output_shape = list(pack_in_shapes[0])
        concat_output_shape[pack_obj.axis] *= len(pack_in_shapes)
        if concat_output_shape != reshape_out_shape:
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
        _, _, pack_out_attr = pack_out_edges[0]
        reshape_in_edges = graph.sorted_in_edges(reshape, data=True)
        graph.remove_edges_from(pack_out_edges + reshape_in_edges)
        pack_out_attr['tensor'].value = None
        pack_out_attr['tensor'].shape = tuple(concat_output_shape)
        graph.add_edge(pack, reshape, **pack_out_attr)
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


def merge_divmod(graph):
    '''Merge x->Div(B=i0)->Mul(B=i0)->Sub(A=x) and x->Div(B=i0) to DivMod op.
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('div', {'op': 'Div'}),
                                   ('divc', {'op': 'Constant'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('mulc', {'op': 'Constant'}),
                                   ('sub', {'op': 'Sub'}),
                                   ('div1', {'op': 'Div'}),
                                   ('div1c', {'op': 'Constant'}),
                               ],
                               edges=[
                                   ('divc', 'div', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                   ('div', 'mul'),
                                   ('mulc', 'mul'),
                                   ('mul', 'sub', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                   ('div1c', 'div1', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                               ]
                               )
    for m in matches:
        key_names = ['div', 'divc', 'mul', 'mulc', 'sub', 'div1', 'div1c']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if any(obj is None for obj in node_objs.values()):
            ERROR('[Parser]: Meets invalid nodes in merge_divmod!')
            continue
        dividend = node_objs['divc'].value
        if not np.issubdtype(dividend.dtype, np.integer) \
                or node_objs['mulc'].value != dividend \
                or node_objs['div1c'].value != dividend:
            continue
        sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
        div_in_edges = graph.sorted_in_edges(m['div'], data=True)
        div1_in_edges = graph.sorted_in_edges(m['div'], data=True)
        if len(sub_in_edges) < 2 or len(div_in_edges) < 2 or len(div1_in_edges) < 2:
            ERROR('[Parser]: Meets invalid in edges of nodes in merge_divmod!')
            continue
        inp = sub_in_edges[0][0]
        if div_in_edges[0][0] != inp or div1_in_edges[0][0] != inp:
            continue
        inp_out_port = sub_in_edges[0][2]['src_out_port']
        if div_in_edges[0][2]['src_out_port'] != inp_out_port \
                or div1_in_edges[0][2]['src_out_port'] != inp_out_port:
            continue
        matched = True
        sub_out_edges = graph.sorted_out_edges(m['sub'], data=True)
        div1_out_edges = graph.sorted_out_edges(m['div1'], data=True)
        graph.remove_edges_from(sub_in_edges[1:] + div1_out_edges)
        divc_out_attr = div_in_edges[1][2]
        graph.add_edge(m['divc'], m['sub'], **divc_out_attr)
        for _, _, out_attr in sub_out_edges:
            out_attr['src_out_port'] = 1
        for _, dst, out_attr in div1_out_edges:
            graph.add_edge(m['sub'], dst, **out_attr)

        divmod_attr = node_objs['sub'].copied_attr()
        divmod_attr.update({'mode': 'trunc'})
        NodeWrap(graph, m['sub']).replace_obj('DivMod', divmod_attr)
        if m['div1'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['div1'])
            if m['sub'] in graph._attr['output_names']:
                graph._attr['output_names'].pop(index)
            else:
                graph._attr['output_names'][index] = m['sub']
    if matched:
        clear_redundant_nodes(graph)


def merge_divmod2(graph):
    '''Merge Cast(int->float)+Div(B=float)+Cast(float->int) to DivMod+Cast(->int).
    Keeping the Cast after Div is because the input 'int' could be different with
    the output 'int'.
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('cast_in', {'op': 'Cast'}),
                                   ('divc', {'op': 'Constant'}),
                                   ('div', {'op': 'Div'}),
                                   ('cast_out', {'op': 'Cast'})],
                               edges=[
                                   ('cast_in', 'div', {'dst_in_port': 0}),
                                   ('divc', 'div', {'dst_in_port': 1}),
                                   ('div', 'cast_out')])
    for m in matches:
        key_names = ['div', 'divc', 'cast_in', 'cast_out']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if any(obj is None for obj in node_objs.values()):
            ERROR('[Parser]: Meets invalid nodes in merge_divmod2!')
            continue
        cast_in_to_type = node_objs['cast_in'].to
        cast_out_to_type = node_objs['cast_out'].to
        if 'int' in cast_in_to_type or 'int' not in cast_out_to_type:
            continue
        cast_in_in_edges = graph.sorted_in_edges(m['cast_in'], data=True)
        if len(cast_in_in_edges) < 1 or cast_in_in_edges[0][2].get('tensor', None) is None:
            continue
        input_type = cast_in_in_edges[0][2]['tensor'].get_dtype()
        if input_type is None or 'int' not in input_type:
            continue
        dividend = node_objs['divc'].value
        if not FLOAT_EQUAL(dividend, dividend.astype(input_type)):
            continue
        cast_in_src, _, src_in_attr = cast_in_in_edges[0]
        div_out_edges = graph.sorted_out_edges(m['div'], data=True)
        if len(div_out_edges) != 1:
            continue
        matched = True
        div_in_edges = graph.sorted_in_edges(m['div'])
        graph.remove_edges_from(div_in_edges + div_out_edges)
        graph.add_edge(cast_in_src, m['div'], **src_in_attr)
        insert_constant(graph, m['div'] + '_b', np.array(dividend, dtype=input_type), m['div'], in_port=1)
        cast_out_in_attr = div_out_edges[0][2]
        if cast_out_in_attr['tensor'].value is not None:
            cast_out_in_attr['tensor'].value = cast_out_in_attr['tensor'].value.astype(input_type)
        cast_out_in_attr['tensor'].dtype = input_type
        graph.add_edge(m['div'], m['cast_out'], **cast_out_in_attr)
        divmod_out1 = get_valid_node_name(graph, m['div'] + '_out1')
        graph.add_edge(m['div'], divmod_out1, **{'src_out_port': 1})
        NodeWrap(graph, divmod_out1).replace_obj('Out', {'name': divmod_out1})
        divmod_attr = node_objs['div'].copied_attr()
        divmod_attr.update({'opset_version': 1, 'mode': 'trunc'})
        NodeWrap(graph, m['div']).replace_obj('DivMod', divmod_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_gather_slice(graph):
    matched = False
    matches = matched_patterns(graph, nodes=[
        ('indices', {'op': 'Constant', 'unique': False}),
        ('gather', {'op': 'Gather', 'unique': False}),
        ('add_const', {'op': 'Constant', 'unique': False}),
        ('add', {'op': 'Add'}),
        ('slice', {'op': 'Slice'}),
    ], edges=[
        ('indices', 'gather', {'dst_in_port': 1}),
        ('gather', 'add'),
        ('add_const', 'add'),
        ('gather', 'slice', {'dst_in_port': 1}),
        ('add', 'slice', {'dst_in_port': 2}),
    ])
    for m in matches:
        names = ['indices', 'gather', 'add_const', 'add', 'slice']
        node_objs = {name: NodeWrap(graph, m[name])['object'] for name in names}
        if any(obj is None for obj in node_objs.values()):
            ERROR('[Parser]: Meets invalid nodes in merge_gather_slice!')
            continue
        if node_objs['indices'].value.ndim != 0 or node_objs['indices'].value != 0 \
                or node_objs['add_const'].value.ndim != 0 or node_objs['add_const'].value != 1:
            continue
        gather_input_shapes = node_objs['gather'].get_input_shapes()
        if len(gather_input_shapes) < 1 or gather_input_shapes[0] is None \
                or None in gather_input_shapes[0] or int(np.prod(gather_input_shapes[0])) != 1:
            continue
        gather_output_shapes = node_objs['gather'].get_output_shapes()
        if len(gather_output_shapes) < 1 or gather_output_shapes[0] is None \
                or gather_output_shapes[0] != [1]:
            continue
        slice_input_shapes = node_objs['slice'].get_input_shapes()
        if len(slice_input_shapes) < 1 or slice_input_shapes[0] is None:
            continue
        slice_in_edges = graph.sorted_in_edges(m['slice'], data=True)
        if len(slice_in_edges) > 3:
            axes_in_attr = slice_in_edges[3][2]['tensor']
            if axes_in_attr is None or not axes_in_attr.is_const \
                    or axes_in_attr.value is None or axes_in_attr.value.size != 1:
                continue
        if len(slice_in_edges) > 4:
            steps_in_attr = slice_in_edges[4][2]['tensor']
            if steps_in_attr is None or not steps_in_attr.is_const \
                    or steps_in_attr.value is None or steps_in_attr.value.size != 1 \
                    or steps_in_attr.value.item(0) != 1:
                continue
        matched = True
        axis = axes_in_attr.value.item(0)
        axis = (axis + len(slice_input_shapes[0])) if axis < 0 else axis
        graph.remove_edges_from(slice_in_edges[2:])
        gather_attr = node_objs['slice'].copied_attr()
        gather_attr.update({'opset_version': 13, 'axis': axis, 'axes': None})
        NodeWrap(graph, m['slice']).replace_obj('Gather', gather_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_gelu_1(graph):
    matched = False
    # expression = 'Mul(Mul(x, Erf(x/1.41421356)+1), 0.5)'
    # matches = match_patterns_from_expression(graph, expression)
    matches = matched_patterns(graph,
                               nodes=[
                                   ('inp', {}),
                                   ('div', {'op': 'Div'}),
                                   ('divc', {'op': 'Constant', 'unique': False}),
                                   ('erf', {'op': 'Erf'}),
                                   ('add_1', {'op': 'Add'}),
                                   ('addc', {'op': 'Constant', 'unique': False}),
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
                    or len(erf_out_edges) != 1 \
                    or div_in_edges[0][2]['src_out_port'] != mul_1_in_edges[0][2]['src_out_port'] \
                    or len(node_objs['div'].sorted_in_consts()) != 1 \
                    or len(node_objs['add_1'].sorted_in_consts()) != 1 \
                    or len(node_objs['mul_2'].sorted_in_consts()) != 1 \
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
            gelu_attr.update({'opset_version': 20, 'approximate': 'none'})
            NodeWrap(graph, m['mul_2']).replace_obj('Gelu', gelu_attr)
        else:
            ERROR('[Parser]: Meets invalid nodes in merge_gelu!')
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
            add_1_in_edges = graph.sorted_in_edges(m['add_1'], data=True)
            mul_4_in_edges = graph.sorted_in_edges(m['mul_4'], data=True)
            if len(pow_in_edges) != 2 \
                    or len(add_1_in_edges) != 2 \
                    or len(mul_4_in_edges) != 2:
                continue
            add_1_main_in_port = [e[2]['dst_in_port'] for e in add_1_in_edges if e[0] != m['mul_1']][0]
            mul_4_main_in_port = [e[2]['dst_in_port'] for e in mul_4_in_edges if e[0] != m['mul_4c']][0]
            pow_src, _, in_attr1 = pow_in_edges[0]
            add_1_src, _, in_attr3 = add_1_in_edges[add_1_main_in_port]
            mul_4_src, _, in_attr2 = mul_4_in_edges[mul_4_main_in_port]

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
                    or len(mul_4_out_edges) != 1 \
                    or len(node_objs['pow'].sorted_in_consts()) != 1 \
                    or len(node_objs['add_2'].sorted_in_consts()) != 1 \
                    or len(node_objs['mul_2'].sorted_in_consts()) != 1 \
                    or len(node_objs['mul_1'].sorted_in_consts()) != 1 \
                    or len(node_objs['mul_4'].sorted_in_consts()) != 1 \
                    or FLOAT_EQUAL(node_objs['pow'].sorted_in_consts()[0][2], 3.0) is False \
                    or FLOAT_EQUAL(node_objs['mul_1'].sorted_in_consts()[0][2], 0.044714998453855515) is False \
                    or FLOAT_EQUAL(node_objs['mul_2'].sorted_in_consts()[0][2], 0.7978845834732056) is False \
                    or FLOAT_EQUAL(node_objs['mul_4'].sorted_in_consts()[0][2], 0.5) is False \
                    or FLOAT_EQUAL(node_objs['add_2'].sorted_in_consts()[0][2], 1.0) is False:
                continue

            matched = True
            graph.remove_edge(pow_src, m['pow'])
            graph.remove_edge(pow_src, m['mul_4'])
            graph.remove_edges_from(mul_3_in_edges)
            graph.add_edge(pow_src, m['mul_3'], **in_attr1)
            gelu_attr = node_objs['mul_3'].copied_attr()
            gelu_attr.update({'opset_version': 20, 'approximate': 'tanh'})
            NodeWrap(graph, m['mul_3']).replace_obj('Gelu', gelu_attr)
        else:
            ERROR('[Parser]: Meets invalid nodes in merge_gelu!')
    if matched:
        clear_redundant_nodes(graph)


def merge_gelu_3(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('div', {'op': 'Div'}),
                                   ('divc', {'op': 'Constant', 'unique': False}),
                                   ('erf', {'op': 'Erf'}),
                                   ('add', {'op': 'Add'}),
                                   ('addc2', {'op': 'Constant', 'unique': False}),
                                   ('mul_1', {'op': 'Mul'}),
                                   ('mulc', {'op': 'Constant', 'unique': False}),
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
        if all(obj is not None for obj in node_objs.values()):

            mul_1_in_edges = graph.sorted_in_edges(m['mul_1'], data=True)
            mul_1_in_edges = [(src, _, in_attr) for src, _, in_attr in mul_1_in_edges if src != m['mulc']]
            div_in_edges = graph.sorted_in_edges(m['div'], data=True)

            if len(mul_1_in_edges) != 1 \
                    or len(div_in_edges) != 2:
                continue
            mul_1_src, _, in_attr1 = mul_1_in_edges[0]
            div_src, _, in_attr2 = div_in_edges[0]

            if mul_1_src != div_src \
                    or in_attr1['src_out_port'] != in_attr2['src_out_port']:
                continue

            if not FLOAT_EQUAL(node_objs['divc'].value, 1.4142135381698608) \
                    or not FLOAT_EQUAL(node_objs['addc2'].value, 1.0) \
                    or not FLOAT_EQUAL(node_objs['mulc'].value, 0.5):
                continue

            matched = True
            mul_2_in_edges = graph.sorted_in_edges(m['mul_2'])
            graph.remove_edges_from(mul_2_in_edges)
            gelu_in_attr = copy.deepcopy(in_attr1)
            gelu_in_attr.update({'dst_in_port': 0})
            graph.add_edge(div_src, m['mul_2'], **gelu_in_attr)
            gelu_attr = node_objs['mul_2'].copied_attr()
            gelu_attr.update({'opset_version': 20, 'approximate': 'none'})
            NodeWrap(graph, m['mul_2']).replace_obj('Gelu', gelu_attr)
        else:
            ERROR('[Parser]: Meets invalid nodes in merge_gelu!')
    if matched:
        clear_redundant_nodes(graph)


def merge_gelu_4(graph):
    '''Merge Gelu according to this formula:
    y = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))) if approximate is True, so
    y = (0.5 * x) * (1 + tanh(z)), in which
    z = sqrt(2 / pi) * (x + 0.044715 * x^3)
      = 0.797884 * x * (1 + 0.044715 * x^2)
      = (0.797884 * x) * (1 + x * (x * 0.044715))
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('const_mul', {'op': 'Constant'}),
                                   ('mul_x', {'op': 'Mul'}),
                                   ('mul_xx', {'op': 'Mul'}),
                                   ('const_add_1', {'op': 'Constant'}),
                                   ('add_1', {'op': 'Add'}),
                                   ('const_sqrt', {'op': 'Constant'}),
                                   ('mul_sqrt', {'op': 'Mul'}),
                                   ('mul_for_tanh', {'op': 'Mul'}),
                                   ('tanh', {'op': 'Tanh'}),
                                   ('const_add_1_tanh', {'op': 'Constant'}),
                                   ('add_tanh', {'op': 'Add'}),
                                   ('const_half', {'op': 'Constant'}),
                                   ('mul_half', {'op': 'Mul'}),
                                   ('mul_out', {'op': 'Mul'}),
                               ],
                               edges=[
                                   ('const_mul', 'mul_x'),
                                   ('mul_x', 'mul_xx'),
                                   ('const_add_1', 'add_1'),
                                   ('mul_xx', 'add_1'),
                                   ('const_sqrt', 'mul_sqrt'),
                                   ('add_1', 'mul_for_tanh'),
                                   ('mul_sqrt', 'mul_for_tanh'),
                                   ('mul_for_tanh', 'tanh'),
                                   ('const_add_1_tanh', 'add_tanh'),
                                   ('tanh', 'add_tanh'),
                                   ('const_half', 'mul_half'),
                                   ('add_tanh', 'mul_out'),
                                   ('mul_half', 'mul_out'),
                               ]
                               )
    for m in matches:
        key_names = ['const_mul', 'const_add_1', 'const_sqrt', 'const_add_1_tanh',
                     'const_half', 'mul_x', 'mul_xx', 'mul_sqrt', 'mul_half', 'mul_out']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if any(obj is None for obj in node_objs.values()):
            ERROR('[Parser]: Meets invalid nodes in merge_gelu_4!')
            continue
        if not FLOAT_EQUAL(node_objs['const_mul'].value, 0.044715) \
                or not FLOAT_EQUAL(node_objs['const_sqrt'].value, 0.797884) \
                or not FLOAT_EQUAL(node_objs['const_half'].value, 0.5) \
                or not FLOAT_EQUAL(node_objs['const_add_1'].value, 1) \
                or not FLOAT_EQUAL(node_objs['const_add_1_tanh'].value, 1):
            continue
        mul_xx_in_edges = graph.sorted_in_edges(m['mul_xx'], data=True)
        inp_in_edges = [(src, in_attr) for src, _, in_attr in mul_xx_in_edges if src != m['mul_x']]
        if len(inp_in_edges) != 1:
            continue
        gelu_inp, inp_in_attr = inp_in_edges[0]
        has_same_input = True
        for name in ['mul_x', 'mul_half', 'mul_sqrt']:
            mul_in_edges = graph.sorted_in_edges(m[name], data=True)
            src_out_ports = [in_attr['src_out_port'] for src, _, in_attr in mul_in_edges if src == gelu_inp]
            if len(src_out_ports) != 1 or src_out_ports[0] != inp_in_attr['src_out_port']:
                has_same_input = False
                break
        if not has_same_input:
            continue
        matched = True
        mul_out_in_edges = graph.sorted_in_edges(m['mul_out'])
        graph.remove_edges_from(mul_out_in_edges)
        gelu_in_attr = copy.deepcopy(inp_in_attr)
        gelu_in_attr.update({'dst_in_port': 0})
        graph.add_edge(gelu_inp, m['mul_out'], **gelu_in_attr)
        gelu_attr = node_objs['mul_out'].copied_attr()
        gelu_attr.update({'approximate': 'tanh', 'opset_version': 20})
        NodeWrap(graph, m['mul_out']).replace_obj('Gelu', gelu_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_gelu_5(graph):
    '''Merge Gelu according to this formula:
    y = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))) if approximate is True, so
    y = 0.5 * x * (1 + tanh(z)), in which
    z = sqrt(2 / pi) * (x + 0.044715 * x^3)
      = 0.797884 * (x + 0.044715 * x^3)
      = beta * (x + kappa * x^3)
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('square', {'op': 'Mul'}),
                                   ('cube', {'op': 'Mul'}),
                                   ('const_kappa', {'op': 'Constant'}),
                                   ('mul_cube_kappa', {'op': 'Mul'}),
                                   ('add_x', {'op': 'Add'}),
                                   ('const_beta', {'op': 'Constant'}),
                                   ('mul_beta', {'op': 'Mul'}),
                                   ('tanh', {'op': 'Tanh'}),
                                   ('const_one', {'op': 'Constant'}),
                                   ('add_tanh', {'op': 'Add'}),
                                   ('mul_x', {'op': 'Mul'}),
                                   ('const_half', {'op': 'Constant'}),
                                   ('mul_half', {'op': 'Mul'}),
                               ],
                               edges=[
                                   ('square', 'cube'),
                                   ('const_kappa', 'mul_cube_kappa'),
                                   ('cube', 'mul_cube_kappa'),
                                   ('mul_cube_kappa', 'add_x'),
                                   ('const_beta', 'mul_beta'),
                                   ('add_x', 'mul_beta'),
                                   ('mul_beta', 'tanh'),
                                   ('const_one', 'add_tanh'),
                                   ('tanh', 'add_tanh'),
                                   ('add_tanh', 'mul_x'),
                                   ('const_half', 'mul_half'),
                                   ('mul_x', 'mul_half'),
                               ]
                               )
    for m in matches:
        key_names = ['const_kappa', 'const_beta', 'const_one', 'const_half',
                     'square', 'cube', 'add_x', 'mul_x', 'mul_half']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if any(obj is None for obj in node_objs.values()):
            ERROR('[Parser]: Meets invalid nodes in merge_gelu_5!')
            continue
        if not FLOAT_EQUAL(node_objs['const_kappa'].value, 0.044715) \
                or not FLOAT_EQUAL(node_objs['const_beta'].value, 0.797884) \
                or not FLOAT_EQUAL(node_objs['const_half'].value, 0.5) \
                or not FLOAT_EQUAL(node_objs['const_one'].value, 1):
            continue
        cube_in_edges = graph.sorted_in_edges(m['cube'], data=True)
        inp_in_edges = [(src, in_attr) for src, _, in_attr in cube_in_edges if src != m['square']]
        if len(inp_in_edges) != 1:
            continue
        gelu_inp, inp_in_attr = inp_in_edges[0]
        exp_src_out_port = inp_in_attr['src_out_port']
        square_in_edges = graph.sorted_in_edges(m['square'], data=True)
        if len(square_in_edges) != 2 \
            or any(
                (src != gelu_inp or in_attr['src_out_port'] != exp_src_out_port) for src, _, in_attr in square_in_edges):
            continue
        mul_x_in_edges = graph.sorted_in_edges(m['mul_x'], data=True)
        src_out_ports = [in_attr['src_out_port'] for src, _, in_attr in mul_x_in_edges if src == gelu_inp]
        if len(src_out_ports) != 1 or src_out_ports[0] != exp_src_out_port:
            continue
        add_x_in_edges = graph.sorted_in_edges(m['add_x'], data=True)
        src_out_ports = [in_attr['src_out_port'] for src, _, in_attr in add_x_in_edges if src == gelu_inp]
        if len(src_out_ports) != 1 or src_out_ports[0] != exp_src_out_port:
            continue
        matched = True
        mul_out_in_edges = graph.sorted_in_edges(m['mul_half'])
        graph.remove_edges_from(mul_out_in_edges)
        gelu_in_attr = copy.deepcopy(inp_in_attr)
        gelu_in_attr.update({'dst_in_port': 0})
        graph.add_edge(gelu_inp, m['mul_half'], **gelu_in_attr)
        gelu_attr = node_objs['mul_half'].copied_attr()
        gelu_attr.update({'approximate': 'tanh', 'opset_version': 20})
        NodeWrap(graph, m['mul_half']).replace_obj('Gelu', gelu_attr)
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
        if (exclude_pad and len(trans1_obj.get_input_shapes()) > 0 and all(
                s is not None for s in trans1_obj.get_input_shapes()[0])) \
                or ((not exclude_pad) and len(pad_obj.get_input_shapes()) > 0 and all(
                    s is not None for s in pad_obj.get_input_shapes()[0])):
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
            if len(trans4_obj.get_output_shapes()) > 0 and all(
                    s is not None for s in trans4_obj.get_output_shapes()[0]):
                true_out_shape = trans4_obj.get_output_shapes()[0]
                if new_perm is not None:
                    inverse_perm = Op.cal_inverse_perm(perm)
                    true_out_shape = np.array(true_out_shape)[np.array(inverse_perm)]
                    true_out_shape = true_out_shape[np.array(ref_perm)].tolist()
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
            if (exclude_slice and len(trans4_obj.get_output_shapes()) > 0 and all(
                    s is not None for s in trans4_obj.get_output_shapes()[0])) \
                    or ((not exclude_slice) and len(slice_obj.get_output_shapes()) > 0 and all(
                        s is not None for s in slice_obj.get_output_shapes()[0])):
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
            ERROR('[Parser]: Meets invalid Op in merge_dilated_conv!')
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
                ERROR('[Parser]: The length of in_edges of Pad(%s) is invalid in merge_dilated_conv_group!' % pad)
                continue

            matched = True
            block_size = s2d_obj.blocksize
            pad_pads = np.reshape(
                np.array(pad_obj.space_pads(), np.int64), newshape=(2, -1))

            sliced_pads_1 = type(slice_1_obj).cal_sliced(
                slice_1_obj.starts if len(slice_1_obj.starts) == 2 else slice_1_obj.starts[1:3],
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

            sliced_pads_2 = type(slice_2_obj).cal_sliced(
                slice_2_obj.starts if len(slice_2_obj.starts) == 2 else slice_2_obj.starts[1:3],
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
            ERROR('[Parser]: Meets invalid node in merge_dilated_conv_group!')
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
            ERROR('[Parser]: Meets invalid Op in merge_hardswish!')
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
            ERROR('[Parser]: Meets invalid Op in merge_hardswish!')
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
    # = max(clip_min', min(1, alpha' * x + beta')), in which
    # clip_min' = clip_min/clip_max, alpha' = alpha/clip_max, beta' = beta/clip_max
    # According to IR def, the latter format is used.
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
            ERROR('[Parser]: Meets invalid Op in merge_hargsigmoid2!')
            continue
        mul_div_in_edges = graph.sorted_in_edges(m['mul_or_div'], data=True)
        if obj_dict['mul_or_div'].type == 'Div' \
                and mul_div_in_edges[1][0] != m['const_2']:
            continue
        c1 = obj_dict['const_1'].value
        c2 = obj_dict['const_2'].value
        if FLOAT_EQUAL(c2, 0):
            continue
        if obj_dict['mul_or_div'].type == 'Mul':
            c2 = 1 / c2
        clip_min, clip_max = obj_dict['clip'].min, obj_dict['clip'].max
        if not FLOAT_EQUAL(c2, clip_max):
            continue
        add_in_edges = graph.sorted_in_edges(m['add'], data=True)
        if len(add_in_edges) != 2:
            ERROR(
                '[Parser]: Invalid number of inputs of Add Op(%s) in merge_hargsigmoid2!' % m['add'])
            continue
        inp, in_attr = None, None
        for src, _, attr in add_in_edges:
            if src != m['const_1']:
                inp = src
                in_attr = copy.deepcopy(attr)
                break
        if inp is None or in_attr is None:
            ERROR('[Parser]: Meets invalid pattern of hargsigmoid!')
            continue
        matched = True
        in_attr['dst_in_port'] = 0
        graph.remove_edges_from(add_in_edges + mul_div_in_edges)
        graph.add_edge(inp, m['mul_or_div'], **in_attr)
        hs_attr = obj_dict['mul_or_div'].copied_attr()
        hs_attr.update({'alpha': 1. / c2,
                        'beta': float(c1) / c2,
                        'clip_min': float(clip_min) / c2,
                        'clip_max': 1.,
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
            ERROR('[Parser]: Meets invalid Op in merge_l2pool!')
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
            ERROR(
                '[Parser]: Node (%s or %s or %s or %s or %s) cannot be found, graph maybe has been changed in merge_leaky_relu!' % (
                    inp, relu, neg, const, end))
            continue

        relu_in_edges = graph.sorted_in_edges(relu, data=True)
        neg_in_edges = graph.sorted_in_edges(neg, data=True)
        if len(relu_in_edges) != 1 \
                or len(neg_in_edges) != 1:
            ERROR('[Parser]: Meets invalid in_edges in merge_leaky_relu!')
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
            ERROR('[Parser]: Meets invalid Op in merge_logical_xor!')
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


def merge_double_reduce(graph):
    matched = False
    matches = [matched_patterns(graph,
                                nodes=[
                                    ('reduce0', {'op': reduceop}),
                                    ('reduce1', {'op': reduceop}),
                                ],
                                edges=[
                                    ('reduce0', 'reduce1')
                                ]) for reduceop in ['ReduceMean', 'ReduceMax', 'ReduceMin', 'ReduceSum']]
    for m in extend_lists(matches):
        reduce0, reduce1 = m['reduce0'], m['reduce1']
        reduce0_obj = NodeWrap(graph, reduce0)['object']
        reduce1_obj = NodeWrap(graph, reduce1)['object']
        if reduce0_obj is None or reduce1_obj is None:
            ERROR('[Parser]: Meets invalid Op in merge_double_reduce!')
            continue
        if reduce0_obj.keepdims != reduce1_obj.keepdims:
            continue
        reduce1_in_edges = graph.sorted_in_edges(reduce1, data=True)
        if len(reduce1_in_edges) != 1:
            continue
        matched = True
        reduce0_in_edges = graph.sorted_in_edges(reduce0, data=True)
        graph.remove_edges_from(reduce0_in_edges)
        graph.remove_edges_from(reduce1_in_edges)
        for src, _, in_attr in reduce0_in_edges:
            new_in_attr = copy.deepcopy(in_attr)
            graph.add_edge(src, reduce1, **new_in_attr)
        if reduce0_obj.keepdims:
            new_axes = reduce0_obj.axes + reduce1_obj.axes
        else:
            reduce1_new_axes = []
            for axis in reduce1_obj.axes:
                if axis < min(reduce0_obj.axes):
                    reduce1_new_axes.append(axis)
                else:
                    reduce1_new_axes.append(axis + len(reduce0_obj.axes))
            new_axes = reduce0_obj.axes + reduce1_new_axes
        reduce1_obj.axes = new_axes.copy()
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
            ERROR('[Parser]: Meets invalid Op in merge_meshgrid!')
            continue
        reshape1_in_edges = graph.sorted_in_edges(
            m['reshape1'], keys=True, data=True)
        reshape2_in_edges = graph.sorted_in_edges(
            m['reshape2'], keys=True, data=True)
        if len(reshape1_in_edges) < 1 or len(reshape2_in_edges) < 1:
            ERROR('[Parser]: Meets invalid Reshape Op(%s or %s) in merge_meshgrid!' % (
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
    # matches = match_patterns_from_expression(graph,
    #           'slope = Constant(); half = Constant(); y = Relu(x) + (Abs(x) - x) * half * slope')
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


def merge_special_concat_split_concat(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('concat0', {'op': 'Concat'}),
                                   ('split', {'op': 'Split'}),
                                   ('concat1', {'op': 'Concat'}),
                               ],
                               edges=[
                                   ('concat0', 'split'),
                                   ('split', 'concat1'),
                               ])
    matched = False
    for m in matches:
        concat0, split, concat1 = m['concat0'], m['split'], m['concat1']
        concat0_obj = NodeWrap(graph, concat0)['object']
        split_obj = NodeWrap(graph, split)['object']
        concat1_obj = NodeWrap(graph, concat1)['object']
        if all([obj is not None for obj in [concat0_obj, split_obj, concat1_obj]]):
            concat0_in_edges = graph.sorted_in_edges(concat0, data=True)
            concat0_axis = concat0_obj.axis
            split_axis = split_obj.axis
            split_out_edges = graph.sorted_out_edges(split, data=True)
            split_size = split_obj.split
            concat1_in_edges = graph.sorted_in_edges(concat1, data=True)
            concat1_axis = concat1_obj.axis
            if concat0_axis == concat1_axis and concat0_axis == split_axis and len(split_out_edges) == 1:
                if len(concat0_in_edges) == 2 and \
                        len(concat1_in_edges) == 2 and \
                        len(split_size) == 2:
                    concat0_input_shapes = concat0_obj.get_input_shapes()
                    if concat0_in_edges[0][0] == concat1_in_edges[0][0] or \
                            concat0_in_edges[1][0] == concat1_in_edges[1][0]:
                        matched = True
                        if concat0_in_edges[0][0] == concat1_in_edges[0][0]:
                            if concat0_input_shapes[0] and None not in concat0_input_shapes[0] and \
                                    concat0_input_shapes[0][concat0_axis] == split_size[0]:
                                graph.remove_edges_from(concat1_in_edges[1:])
                                src, _, in_attr = concat0_in_edges[1]
                                graph.add_edge(src, concat1_in_edges[0][1], **in_attr)
                        else:
                            if concat0_input_shapes[1] and None not in concat0_input_shapes[1] and \
                                    concat0_input_shapes[1][concat0_axis] == split_size[1]:
                                graph.remove_edges_from(concat1_in_edges[:1])
                                src, _, in_attr = concat0_in_edges[0]
                                graph.add_edge(src, concat1_in_edges[1][1], **in_attr)
                    else:
                        split_out_port = split_out_edges[0][2]['src_out_port']
                        if concat0_input_shapes[split_out_port] and \
                                None not in concat0_input_shapes[split_out_port] and \
                                split_size[split_out_port] == concat0_input_shapes[split_out_port][concat0_axis]:
                            matched = True
                            graph.remove_edges_from(concat0_in_edges)
                            graph.remove_edges_from(split_out_edges)
                            src, _, in_attr = concat0_in_edges[split_out_port]
                            graph.add_edge(src, split_out_edges[0][1], **in_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_special_thresholdedrelu_to_relu(graph):
    '''Merge thresholdedrelu(alpha=0) to Relu.
    '''
    matches = single_node_matcher(graph, 'ThresholdedRelu')
    for m in matches:
        thres_relu = m['target']
        thres_relu_obj = NodeWrap(graph, thres_relu)['object']
        if thres_relu_obj is None:
            ERROR(
                '[Parser]: Meets invalid ThresholdedRelu node(%s) in convert_special_thresholdedrelu_to_relu!' % thres_relu)
            continue
        if thres_relu_obj.quantize:
            continue
        alpha = thres_relu_obj.alpha
        if not FLOAT_EQUAL(alpha, 0.0):
            continue
        relu_attr = thres_relu_obj.copied_attr()
        relu_attr.update({'opset_version': 14})
        NodeWrap(graph, thres_relu).replace_obj('Relu', relu_attr)


def merge_clip(graph):
    '''Merge Relu+Min(x/y=const) to Clip.
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('relu', {'op': 'Relu', 'unique': False}),
                                   ('const', {'op': 'Constant', 'unique': False}),
                                   ('min', {'op': 'Min'})],
                               edges=[
                                   ('relu', 'min'),
                                   ('const', 'min')])
    for m in matches:
        names = ['relu', 'const', 'min']
        obj_dict = {k: NodeWrap(graph, m[k])['object'] for k in names}
        relu_in_edges = graph.sorted_in_edges(m['relu'], data=True)
        if any(obj is None for obj in obj_dict) or len(relu_in_edges) < 1:
            ERROR('[Parser]: Meets invalid nodes in merge_clip!')
            continue
        if obj_dict['relu'].quantize:
            continue
        const_val = obj_dict['const'].value
        if not FLOAT_EQUAL(const_val.item(0), const_val) or const_val.item(0) < 0:
            continue
        matched = True
        min_val = np.array(0.0, dtype=np.float32)
        max_val = np.array(const_val.item(0), dtype=np.float32)

        min_in_edges = graph.sorted_in_edges(m['min'])
        graph.remove_edges_from(min_in_edges)
        src, _, in_attr = relu_in_edges[0]
        graph.add_edge(src, m['min'], **in_attr)
        insert_constant(graph, m['min'] + '_min', min_val, m['min'], in_port=1)
        insert_constant(graph, m['min'] + '_max', max_val, m['min'], in_port=2)

        clip_attr = obj_dict['min'].copied_attr()
        clip_attr.update({'opset_version': 13})
        NodeWrap(graph, m['min']).replace_obj('Clip', clip_attr)
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
                and len(sum_in_edges) >= 1 \
                and len(sum_out_edges) == 1 \
                and len(obj_dict['sum'].axes) == 1:
            matched = True
            src, _, in_attr = in_edges[0]
            graph.remove_edges_from(in_edges + div_in_edges)
            graph.add_edge(src, m['div'], **in_attr)
            softmax_attr = obj_dict['div'].copied_attr()
            softmax_attr.update(
                {'opset_version': 13, 'axis': obj_dict['sum'].axes[0]})
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
                mish_attr.update({'opset_version': 18})
                NodeWrap(graph, m['mul']).replace_obj(
                    'Mish', mish_attr)
        else:
            ERROR('[Parser]: Meets invalid Node in merge_mish!')
    if matched:
        clear_redundant_nodes(graph)


def merge_query_rebatch(graph):
    '''
    This pass is used to merge QueryRebatch op defined in bevformer model
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('gc00', {'op': 'Constant', 'unique': False}),
                                   ('g00', {'op': 'Gather'}),

                                   ('g0', {'op': 'Gather'}),
                                   ('e0', {'op': 'Expand'}),
                                   ('r0', {'op': 'Reshape'}),
                                   ('scc0', {'op': 'Constant', 'unique': False}),
                                   ('sccc0', {'op': 'Constant', 'unique': False}),
                                   ('sc0', {'op': 'ScatterND'}),

                                   ('g1', {'op': 'Gather'}),
                                   ('e1', {'op': 'Expand'}),
                                   ('r1', {'op': 'Reshape'}),
                                   ('scc1', {'op': 'Constant', 'unique': False}),
                                   ('sc1', {'op': 'ScatterND'}),

                                   ('g2', {'op': 'Gather'}),
                                   ('e2', {'op': 'Expand'}),
                                   ('r2', {'op': 'Reshape'}),
                                   ('scc2', {'op': 'Constant', 'unique': False}),
                                   ('sc2', {'op': 'ScatterND'}),

                                   ('g3', {'op': 'Gather'}),
                                   ('e3', {'op': 'Expand'}),
                                   ('r3', {'op': 'Reshape'}),
                                   ('scc3', {'op': 'Constant', 'unique': False}),
                                   ('sc3', {'op': 'ScatterND'}),

                                   ('g4', {'op': 'Gather'}),
                                   ('e4', {'op': 'Expand'}),
                                   ('r4', {'op': 'Reshape'}),
                                   ('scc4', {'op': 'Constant', 'unique': False}),
                                   ('sc4', {'op': 'ScatterND'}),

                                   ('g5', {'op': 'Gather'}),
                                   ('e5', {'op': 'Expand'}),
                                   ('r5', {'op': 'Reshape'}),
                                   ('scc5', {'op': 'Constant', 'unique': False}),
                                   ('sc5', {'op': 'ScatterND'}),
                               ],
                               edges=[
                                   ('gc00', 'g00', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g00', 'g0', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('g0', 'e0', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('e0', 'r0', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('scc0', 'sc0', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('sccc0', 'sc0', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('r0', 'sc0', {'src_out_port': 0, 'dst_in_port': 2}),

                                   ('g00', 'g1', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('g1', 'e1', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('e1', 'r1', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('sc0', 'sc1', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('scc1', 'sc1', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('r1', 'sc1', {'src_out_port': 0, 'dst_in_port': 2}),

                                   ('g00', 'g2', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('g2', 'e2', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('e2', 'r2', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('sc1', 'sc2', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('scc2', 'sc2', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('r2', 'sc2', {'src_out_port': 0, 'dst_in_port': 2}),

                                   ('g00', 'g3', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('g3', 'e3', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('e3', 'r3', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('sc2', 'sc3', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('scc3', 'sc3', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('r3', 'sc3', {'src_out_port': 0, 'dst_in_port': 2}),

                                   ('g00', 'g4', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('g4', 'e4', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('e4', 'r4', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('sc3', 'sc4', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('scc4', 'sc4', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('r4', 'sc4', {'src_out_port': 0, 'dst_in_port': 2}),

                                   ('g00', 'g5', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('g5', 'e5', {'src_out_port': 0, 'dst_in_port': 0}),

                                   ('e5', 'r5', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('sc4', 'sc5', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('scc5', 'sc5', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('r5', 'sc5', {'src_out_port': 0, 'dst_in_port': 2}),
                               ])
    matches_2 = matched_patterns(graph,
                                 nodes=[
                                     ('gc00', {'op': 'Constant', 'unique': False}),
                                     ('g00', {'op': 'Gather'}),

                                     ('g0', {'op': 'Gather'}),
                                     ('r0', {'op': 'Reshape'}),
                                     ('scc0', {'op': 'Constant', 'unique': False}),
                                     ('sccc0', {'op': 'Constant', 'unique': False}),
                                     ('sc0', {'op': 'ScatterND'}),

                                     ('g1', {'op': 'Gather'}),
                                     ('r1', {'op': 'Reshape'}),
                                     ('scc1', {'op': 'Constant', 'unique': False}),
                                     ('sc1', {'op': 'ScatterND'}),

                                     ('g2', {'op': 'Gather'}),
                                     ('r2', {'op': 'Reshape'}),
                                     ('scc2', {'op': 'Constant', 'unique': False}),
                                     ('sc2', {'op': 'ScatterND'}),

                                     ('g3', {'op': 'Gather'}),
                                     ('r3', {'op': 'Reshape'}),
                                     ('scc3', {'op': 'Constant', 'unique': False}),
                                     ('sc3', {'op': 'ScatterND'}),

                                     ('g4', {'op': 'Gather'}),
                                     ('r4', {'op': 'Reshape'}),
                                     ('scc4', {'op': 'Constant', 'unique': False}),
                                     ('sc4', {'op': 'ScatterND'}),

                                     ('g5', {'op': 'Gather'}),
                                     ('r5', {'op': 'Reshape'}),
                                     ('scc5', {'op': 'Constant', 'unique': False}),
                                     ('sc5', {'op': 'ScatterND'}),
                                 ],
                                 edges=[
                                     ('gc00', 'g00', {'src_out_port': 0, 'dst_in_port': 1}),
                                     ('g00', 'g0', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('g0', 'r0', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('scc0', 'sc0', {'src_out_port': 0, 'dst_in_port': 0}),
                                     ('sccc0', 'sc0', {'src_out_port': 0, 'dst_in_port': 1}),
                                     ('r0', 'sc0', {'src_out_port': 0, 'dst_in_port': 2}),

                                     ('g00', 'g1', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('g1', 'r1', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('sc0', 'sc1', {'src_out_port': 0, 'dst_in_port': 0}),
                                     ('scc1', 'sc1', {'src_out_port': 0, 'dst_in_port': 1}),
                                     ('r1', 'sc1', {'src_out_port': 0, 'dst_in_port': 2}),

                                     ('g00', 'g2', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('g2', 'r2', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('sc1', 'sc2', {'src_out_port': 0, 'dst_in_port': 0}),
                                     ('scc2', 'sc2', {'src_out_port': 0, 'dst_in_port': 1}),
                                     ('r2', 'sc2', {'src_out_port': 0, 'dst_in_port': 2}),

                                     ('g00', 'g3', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('g3', 'r3', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('sc2', 'sc3', {'src_out_port': 0, 'dst_in_port': 0}),
                                     ('scc3', 'sc3', {'src_out_port': 0, 'dst_in_port': 1}),
                                     ('r3', 'sc3', {'src_out_port': 0, 'dst_in_port': 2}),

                                     ('g00', 'g4', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('g4', 'r4', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('sc3', 'sc4', {'src_out_port': 0, 'dst_in_port': 0}),
                                     ('scc4', 'sc4', {'src_out_port': 0, 'dst_in_port': 1}),
                                     ('r4', 'sc4', {'src_out_port': 0, 'dst_in_port': 2}),

                                     ('g00', 'g5', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('g5', 'r5', {'src_out_port': 0, 'dst_in_port': 0}),

                                     ('sc4', 'sc5', {'src_out_port': 0, 'dst_in_port': 0}),
                                     ('scc5', 'sc5', {'src_out_port': 0, 'dst_in_port': 1}),
                                     ('r5', 'sc5', {'src_out_port': 0, 'dst_in_port': 2}),
                                 ])
    matches += matches_2
    for m in matches:
        node_list = ['gc00', 'g00',
                     'g0', 'r0', 'scc0', 'sccc0', 'sc0',
                     'g1', 'r1', 'scc1', 'sc1',
                     'g2', 'r2', 'scc2', 'sc2',
                     'g3', 'r3', 'scc3', 'sc3',
                     'g4', 'r4', 'scc4', 'sc4',
                     'g5', 'r5', 'scc5', 'sc5',
                     ]
        for n in node_list:
            n_obj = NodeWrap(graph, m[n])['object']
            if n_obj is None:
                ERROR(
                    f'[Parser]: Meets invalid node({m[n]}) in merge_query_rebatch!')
                continue

        if 'e0' in m:
            expand_reshape_const_in = True
            for i in range(6):
                expand_in_edges = graph.sorted_in_edges(m[f'e{i}'], data=True)
                if not expand_in_edges[1][-1]['tensor'].is_const:
                    expand_reshape_const_in = False
                    break
                reshape_in_edges = graph.sorted_in_edges(m[f'r{i}'], data=True)
                if not reshape_in_edges[1][-1]['tensor'].is_const:
                    expand_reshape_const_in = False
                    break

            if not expand_reshape_const_in:
                continue

        sc0_input_shapes = NodeWrap(graph, m['sc0'])['object'].get_input_shapes()
        g00_input_shapes = NodeWrap(graph, m['g00'])['object'].get_input_shapes()

        max_len = sc0_input_shapes[0][2]
        if max_len != g00_input_shapes[0][1]:
            continue

        matched = True
        query_in_edges = graph.sorted_in_edges(m['g00'], data=True)
        idx0_in_edges = graph.sorted_in_edges(m['g0'], data=True)
        idx1_in_edges = graph.sorted_in_edges(m['g1'], data=True)
        idx2_in_edges = graph.sorted_in_edges(m['g2'], data=True)
        idx3_in_edges = graph.sorted_in_edges(m['g3'], data=True)
        idx4_in_edges = graph.sorted_in_edges(m['g4'], data=True)
        idx5_in_edges = graph.sorted_in_edges(m['g5'], data=True)

        graph.remove_edges_from(query_in_edges)
        graph.remove_edges_from(idx0_in_edges)
        graph.remove_edges_from(idx1_in_edges)
        graph.remove_edges_from(idx2_in_edges)
        graph.remove_edges_from(idx3_in_edges)
        graph.remove_edges_from(idx4_in_edges)
        graph.remove_edges_from(idx5_in_edges)

        in_edges = graph.sorted_in_edges(m['sc5'], data=True)
        graph.remove_edges_from(in_edges)

        query, _, in_attr = query_in_edges[0]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 0
        graph.add_edge(query, m['sc5'], **new_in_attr)

        idx0, _, in_attr = idx0_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 1
        graph.add_edge(idx0, m['sc5'], **new_in_attr)
        idx1, _, in_attr = idx1_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 2
        graph.add_edge(idx1, m['sc5'], **new_in_attr)
        idx2, _, in_attr = idx2_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 3
        graph.add_edge(idx2, m['sc5'], **new_in_attr)
        idx3, _, in_attr = idx3_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 4
        graph.add_edge(idx3, m['sc5'], **new_in_attr)
        idx4, _, in_attr = idx4_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 5
        graph.add_edge(idx4, m['sc5'], **new_in_attr)
        idx5, _, in_attr = idx5_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 6
        graph.add_edge(idx5, m['sc5'], **new_in_attr)
        NodeWrap(graph, m['sc5']).replace_obj('QueryRebatch',
                                              {'name': m['sc5'], 'opset_version': 1})
    if matched:
        clear_redundant_nodes(graph)


def merge_slot_update(graph):
    '''
    This pass is used to merge SlotUpdate op defined in bevformer model
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('gc0', {'op': 'Constant', 'unique': False}),
                                   ('g0', {'op': 'Gather'}),
                                   # 0

                                   ('g00', {'op': 'Gather'}),
                                   ('slice0', {'op': 'Slice'}),
                                   ('g01c0', {'op': 'Constant', 'unique': False}),
                                   ('g01', {'op': 'Gather', 'unique': False}),
                                   ('add00', {'op': 'Add'}),
                                   ('concat0', {'op': 'Concat', 'unique': False}),
                                   ('r0', {'op': 'Reshape'}),
                                   ('scc0', {'op': 'Constant', 'unique': False}),
                                   ('sc0', {'op': 'ScatterND'}),
                                   # 1

                                   ('g10', {'op': 'Gather'}),
                                   ('g11', {'op': 'Gather'}),
                                   ('g12c0', {'op': 'Constant', 'unique': False}),
                                   ('g12', {'op': 'Gather'}),
                                   ('slice1', {'op': 'Slice'}),
                                   ('add10', {'op': 'Add'}),
                                   ('concat1', {'op': 'Concat', 'unique': False}),
                                   ('r1', {'op': 'Reshape'}),
                                   ('sc1', {'op': 'ScatterND'}),
                                   # 2

                                   ('g20', {'op': 'Gather'}),
                                   ('g21', {'op': 'Gather'}),
                                   ('g22c0', {'op': 'Constant', 'unique': False}),
                                   ('g22', {'op': 'Gather'}),
                                   ('slice2', {'op': 'Slice'}),
                                   ('add20', {'op': 'Add'}),
                                   ('concat2', {'op': 'Concat', 'unique': False}),
                                   ('r2', {'op': 'Reshape'}),
                                   ('sc2', {'op': 'ScatterND'}),
                                   # 3

                                   ('g30', {'op': 'Gather'}),
                                   ('g31', {'op': 'Gather'}),
                                   ('g32c0', {'op': 'Constant', 'unique': False}),
                                   ('g32', {'op': 'Gather'}),
                                   ('slice3', {'op': 'Slice'}),
                                   ('add30', {'op': 'Add'}),
                                   ('concat3', {'op': 'Concat', 'unique': False}),
                                   ('r3', {'op': 'Reshape'}),
                                   ('sc3', {'op': 'ScatterND'}),
                                   # 4

                                   ('g40', {'op': 'Gather'}),
                                   ('g41', {'op': 'Gather'}),
                                   ('g42c0', {'op': 'Constant', 'unique': False}),
                                   ('g42', {'op': 'Gather'}),
                                   ('slice4', {'op': 'Slice'}),
                                   ('add40', {'op': 'Add'}),
                                   ('concat4', {'op': 'Concat', 'unique': False}),
                                   ('r4', {'op': 'Reshape'}),
                                   ('sc4', {'op': 'ScatterND'}),
                                   # 5

                                   ('g50', {'op': 'Gather'}),
                                   ('g51', {'op': 'Gather'}),
                                   ('g52c0', {'op': 'Constant', 'unique': False}),
                                   ('g52', {'op': 'Gather'}),
                                   ('slice5', {'op': 'Slice'}),
                                   ('add50', {'op': 'Add'}),
                                   ('concat5', {'op': 'Concat', 'unique': False}),
                                   ('r5', {'op': 'Reshape'}),
                                   ('sc5', {'op': 'ScatterND'}),
                               ],
                               edges=[
                                   ('gc0', 'g0', {'src_out_port': 0, 'dst_in_port': 1}),
                                   # 0

                                   ('g0', 'g00', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('gc0', 'g00', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g00', 'slice0', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g01c0', 'g01', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g01', 'add00', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('slice0', 'add00', {'src_out_port': 0, 'dst_in_port': 1}),

                                   ('add00', 'r0', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('scc0', 'sc0', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('concat0', 'sc0', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('r0', 'sc0', {'src_out_port': 0, 'dst_in_port': 2}),
                                   # 1
                                   ('sc0', 'g10', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('gc0', 'g10', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g10', 'g11', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g0', 'g12', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g12c0', 'g12', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g12', 'slice1', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g11', 'add10', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('slice1', 'add10', {'src_out_port': 0, 'dst_in_port': 1}),

                                   ('add10', 'r1', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('r1', 'sc1', {'src_out_port': 0, 'dst_in_port': 2}),
                                   ('concat1', 'sc1', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('sc0', 'sc1', {'src_out_port': 0, 'dst_in_port': 0}),
                                   # 2
                                   ('sc1', 'g20', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('gc0', 'g20', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g20', 'g21', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g0', 'g22', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g22c0', 'g22', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g22', 'slice2', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g21', 'add20', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('slice2', 'add20', {'src_out_port': 0, 'dst_in_port': 1}),

                                   ('add20', 'r2', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('r2', 'sc2', {'src_out_port': 0, 'dst_in_port': 2}),
                                   ('concat2', 'sc2', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('sc1', 'sc2', {'src_out_port': 0, 'dst_in_port': 0}),
                                   # 3

                                   ('sc2', 'g30', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('gc0', 'g30', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g30', 'g31', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g0', 'g32', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g32c0', 'g32', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g32', 'slice3', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g31', 'add30', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('slice3', 'add30', {'src_out_port': 0, 'dst_in_port': 1}),

                                   ('add30', 'r3', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('r3', 'sc3', {'src_out_port': 0, 'dst_in_port': 2}),
                                   ('concat3', 'sc3', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('sc2', 'sc3', {'src_out_port': 0, 'dst_in_port': 0}),
                                   # 4

                                   ('sc3', 'g40', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('gc0', 'g40', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g40', 'g41', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g0', 'g42', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g42c0', 'g42', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g42', 'slice4', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g41', 'add40', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('slice4', 'add40', {'src_out_port': 0, 'dst_in_port': 1}),

                                   ('add40', 'r4', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('r4', 'sc4', {'src_out_port': 0, 'dst_in_port': 2}),
                                   ('concat4', 'sc4', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('sc3', 'sc4', {'src_out_port': 0, 'dst_in_port': 0}),
                                   # 5

                                   ('sc4', 'g50', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('gc0', 'g50', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g50', 'g51', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g0', 'g52', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g52c0', 'g52', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('g52', 'slice5', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('g51', 'add50', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('slice5', 'add50', {'src_out_port': 0, 'dst_in_port': 1}),

                                   ('add50', 'r5', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('r5', 'sc5', {'src_out_port': 0, 'dst_in_port': 2}),
                                   ('concat5', 'sc5', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('sc4', 'sc5', {'src_out_port': 0, 'dst_in_port': 0}),
                               ])
    for m in matches:
        node_list = [
            'gc0', 'g0',
            'g00', 'slice0', 'g01', 'g01c0', 'add00', 'concat0', 'r0', 'scc0', 'sc0',
            'g10', 'g11', 'g12c0', 'g12', 'slice1', 'add10', 'concat1', 'r1', 'sc1',
            'g20', 'g21', 'g22c0', 'g22', 'slice2', 'add20', 'concat2', 'r2', 'sc2',
            'g30', 'g31', 'g32c0', 'g32', 'slice3', 'add30', 'concat3', 'r3', 'sc3',
            'g40', 'g41', 'g42c0', 'g42', 'slice4', 'add40', 'concat4', 'r4', 'sc4',
            'g50', 'g51', 'g52c0', 'g52', 'slice5', 'add50', 'concat5', 'r5', 'sc5',
        ]
        for n in node_list:
            n_obj = NodeWrap(graph, m[n])['object']
            if n_obj is None:
                ERROR(
                    f'[Parser]: Meets invalid node({m[n]}) in merge_slot_update!')
                continue

        slice_const_in = True
        for i in range(6):
            slice_in_edges = graph.sorted_in_edges(m[f'slice{i}'], data=True)
            if len(slice_in_edges) < 5:
                slice_const_in = False
                break
            for j in range(4):
                if not slice_in_edges[j + 1][-1]['tensor'].is_const:
                    slice_const_in = False
                    break

        if not slice_const_in:
            continue

        # check gather indices
        g00_in_edges = graph.sorted_in_edges(m['g00'], data=True)
        g12_in_edges = graph.sorted_in_edges(m['g12'], data=True)
        g22_in_edges = graph.sorted_in_edges(m['g22'], data=True)
        g32_in_edges = graph.sorted_in_edges(m['g32'], data=True)
        g42_in_edges = graph.sorted_in_edges(m['g42'], data=True)
        g52_in_edges = graph.sorted_in_edges(m['g52'], data=True)

        if not (g00_in_edges[1][-1]['tensor'].is_const == True and
                np.all(g00_in_edges[1][-1]['tensor'].value == 0)) or \
                not (g12_in_edges[1][-1]['tensor'].is_const == True and
                     np.all(g12_in_edges[1][-1]['tensor'].value == 1)) or \
                not (g22_in_edges[1][-1]['tensor'].is_const == True and
                     np.all(g22_in_edges[1][-1]['tensor'].value == 2)) or \
                not (g32_in_edges[1][-1]['tensor'].is_const == True and
                     np.all(g32_in_edges[1][-1]['tensor'].value == 3)) or \
                not (g42_in_edges[1][-1]['tensor'].is_const == True and
                     np.all(g42_in_edges[1][-1]['tensor'].value == 4)) or \
                not (g52_in_edges[1][-1]['tensor'].is_const == True and
                     np.all(g52_in_edges[1][-1]['tensor'].value == 5)):
            continue

        if not np.all(NodeWrap(graph, m['g01c0'])['object'].value == 0):
            continue

        g0_in_edges = graph.sorted_in_edges(m['g0'], data=True)
        sc5_in_edges = graph.sorted_in_edges(m['sc5'], data=True)

        g01_in_edges = graph.sorted_in_edges(m['g01'], data=True)
        g11_in_edges = graph.sorted_in_edges(m['g11'], data=True)
        g21_in_edges = graph.sorted_in_edges(m['g21'], data=True)
        g31_in_edges = graph.sorted_in_edges(m['g31'], data=True)
        g41_in_edges = graph.sorted_in_edges(m['g41'], data=True)
        g51_in_edges = graph.sorted_in_edges(m['g51'], data=True)

        if not has_path(graph, g01_in_edges[1][0], m['concat0']) or \
                not has_path(graph, g11_in_edges[1][0], m['concat1']) or \
                not has_path(graph, g21_in_edges[1][0], m['concat2']) or \
                not has_path(graph, g31_in_edges[1][0], m['concat3']) or \
                not has_path(graph, g41_in_edges[1][0], m['concat4']) or \
                not has_path(graph, g51_in_edges[1][0], m['concat5']):
            continue

        matched = True

        graph.remove_edges_from(g0_in_edges)
        graph.remove_edges_from(sc5_in_edges)

        src, _, in_attr = g0_in_edges[0]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 0
        graph.add_edge(src, m['sc5'], **new_in_attr)
        src, _, in_attr = g01_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 1
        graph.add_edge(src, m['sc5'], **new_in_attr)
        src, _, in_attr = g11_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 2
        graph.add_edge(src, m['sc5'], **new_in_attr)
        src, _, in_attr = g21_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 3
        graph.add_edge(src, m['sc5'], **new_in_attr)
        src, _, in_attr = g31_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 4
        graph.add_edge(src, m['sc5'], **new_in_attr)
        src, _, in_attr = g41_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 5
        graph.add_edge(src, m['sc5'], **new_in_attr)
        src, _, in_attr = g51_in_edges[1]
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr['dst_in_port'] = 6
        graph.add_edge(src, m['sc5'], **new_in_attr)

        NodeWrap(graph, m['sc5']).replace_obj('SlotUpdate',
                                              {'name': m['sc5'], 'opset_version': 1})
    if matched:
        clear_redundant_nodes(graph)


def convert_simplified_layernorm(graph):
    matches = single_node_matcher(graph, 'SimplifiedLayerNormalization')
    for m in matches:
        simple_ln = m['target']
        simple_ln_obj = NodeWrap(graph, simple_ln)['object']
        if simple_ln_obj is None:
            ERROR(
                '[Parser]: Meets invalid SimplifiedLayerNormalization node(%s) in convert_simplified_layernorm!' % simple_ln)
            continue
        if simple_ln_obj.quantize:
            continue
        sim_ln_in_edges = graph.sorted_in_edges(simple_ln, data=True)
        sim_ln_out_edges = graph.sorted_out_edges(simple_ln, data=True)
        outports = simple_ln_obj.get_out_ports()
        if len(outports) > 1:
            ERROR('[Parser]: outputs > 1 of SimplifiedLayerNormalization node(%s) not support yet!' % simple_ln)
            continue
        if len(sim_ln_in_edges) == 2 and NodeWrap(graph, sim_ln_in_edges[1][0])['object'].type == 'Constant':
            input_shapes = simple_ln_obj.get_input_shapes()
            norm_axes = OpHasAxis.make_axes_non_negative(simple_ln_obj.axes, len(input_shapes[0]))
            weights = OpHasAxis.align_axes(NodeWrap(graph, sim_ln_in_edges[1][0])['object'].value,
                                           norm_axes, input_shapes[0])
            if weights is None:
                continue
            epsilon = simple_ln_obj.epsilon
            graph.remove_edges_from(sim_ln_in_edges[1:])
            rms_norm_attr = simple_ln_obj.copied_attr()
            rms_norm_attr.update({'axes': norm_axes, 'weights': weights, 'epsilon': epsilon})
            NodeWrap(graph, simple_ln).replace_obj('ArmRMSNorm', rms_norm_attr)
            clear_redundant_nodes(graph)
        else:
            ERROR('[Parser]: non-const scale of SimplifiedLayerNormalization node(%s) not support yet!' % simple_ln)


def convert_rotary_embedding(graph):
    matches = single_node_matcher(graph, 'RotaryEmbeddingMs')
    for m in matches:
        embd = m['target']
        embd_obj = NodeWrap(graph, embd)['object']
        if embd_obj is None:
            ERROR(
                '[Parser]: Meets invalid RotaryEmbeddingMs node(%s) in convert_rotary_embedding!' % embd)
            continue
        if embd_obj.quantize:
            continue
        embd_in_edges = graph.sorted_in_edges(embd, data=True)
        embd_out_edges = graph.sorted_out_edges(embd, data=True)
        if len(embd_in_edges) == 4 and \
                NodeWrap(graph, embd_in_edges[2][0])['object'].type == 'Constant' and \
                NodeWrap(graph, embd_in_edges[3][0])['object'].type == 'Constant':
            input = embd_in_edges[0][0]
            position_ids = embd_in_edges[1][0]
            cos_cache = embd_in_edges[2][0]
            sin_cache = embd_in_edges[3][0]
            input_shapes = embd_obj.get_input_shapes()
            bs, seq_len, hidden_size = input_shapes[0][:3]
            if len(input_shapes[0]) == 4:
                seq_len = input_shapes[0][2]
                hidden_size = input_shapes[0][1] * input_shapes[0][3]
            max_seq_len = embd_obj.cos_cache.shape[0]
            head_size = embd_obj.cos_cache.shape[1] * \
                2 if embd_obj.rotary_embedding_dim == 0 else hidden_size // embd_obj.num_heads
            num_heads = int(np.prod(input_shapes[0])) // (bs * seq_len * head_size)
            rs0 = insert_reshape(graph, input, embd, embd_in_edges[0][-1], [bs, seq_len, num_heads, head_size])
            # tile cos_cache & sin_cache
            tile_cos = insert_tile(graph, cos_cache, embd, embd_in_edges[2][-1], [1, 2])
            tile_sin = insert_tile(graph, sin_cache, embd, embd_in_edges[3][-1], [1, 2])

            # get cos embedding
            tile_cos_out_edges = graph.sorted_out_edges(tile_cos, data=True)
            gather_cos = insert_gather(graph, tile_cos, embd, position_ids, axis=0, edge_attr=tile_cos_out_edges[0][-1])
            gather_cos_out_edges = graph.sorted_out_edges(gather_cos, data=True)
            gather_cos_output_shape = list(gather_cos_out_edges[0][-1]['tensor'].shape)
            gather_cos_output_shape.insert(2, 1)
            rs_cos = insert_reshape(graph, gather_cos, embd, gather_cos_out_edges[0][-1], gather_cos_output_shape)
            cos_mul = get_valid_node_name(graph, rs_cos + '_post_mul')
            graph.add_node(cos_mul)
            cos_mul_attr = {'name': cos_mul, 'opset_version': 13}
            graph.add_edge(rs_cos, cos_mul, **{'src_out_port': 0, 'dst_in_port': 1,
                                               'tensor': Tensor(shape=gather_cos_output_shape)})
            graph.add_edge(rs0, cos_mul, **{'src_out_port': 0, 'dst_in_port': 0,
                                            'tensor': Tensor(shape=(bs, seq_len, num_heads, head_size))})
            NodeWrap(graph, cos_mul).replace_obj('Mul', cos_mul_attr)

            # crop input & concat
            inp_crop0 = get_valid_node_name(graph, rs0 + '_post_slice')
            graph.add_node(inp_crop0)
            inp_crop0_attr = {'name': inp_crop0, 'opset_version': 1,
                              'axes': [-1], 'starts': [head_size // 2], 'ends': [head_size]}
            graph.add_edge(rs0, inp_crop0, **{'src_out_port': 0, 'dst_in_port': 0,
                                              'tensor': Tensor(shape=(bs, seq_len, num_heads, head_size))})
            NodeWrap(graph, inp_crop0).replace_obj('Slice', inp_crop0_attr)

            neg = get_valid_node_name(graph, inp_crop0 + '_post_neg')
            graph.add_node(neg)
            neg_attr = {'name': neg, 'opset_version': 13}
            graph.add_edge(inp_crop0, neg, **{'src_out_port': 0, 'dst_in_port': 0,
                                              'tensor': Tensor(shape=(bs, seq_len, num_heads, head_size // 2))})
            NodeWrap(graph, neg).replace_obj('Neg', neg_attr)

            inp_crop1 = get_valid_node_name(graph, rs0 + '_post_slice')
            graph.add_node(inp_crop1)
            inp_crop1_attr = {'name': inp_crop1, 'opset_version': 1, 'axes': [-1], 'ends': [head_size // 2],
                              'starts': [0]}
            graph.add_edge(rs0, inp_crop1, **{'src_out_port': 0, 'dst_in_port': 0,
                                              'tensor': Tensor(shape=(bs, seq_len, num_heads, head_size))})
            NodeWrap(graph, inp_crop1).replace_obj('Slice', inp_crop1_attr)

            concat = get_valid_node_name(graph, rs0 + '_post_concat')
            graph.add_node(concat)
            concat_attr = {'name': concat, 'opset_version': 13, 'axis': -1}
            graph.add_edge(neg, concat, **{'src_out_port': 0, 'dst_in_port': 0,
                                           'tensor': Tensor(shape=(bs, seq_len, num_heads, head_size // 2))})
            graph.add_edge(inp_crop1, concat, **{'src_out_port': 0, 'dst_in_port': 1,
                                                 'tensor': Tensor(shape=(bs, seq_len, num_heads, head_size // 2))})
            NodeWrap(graph, concat).replace_obj('Concat', concat_attr)

            # get sin embedding
            tile_sin_out_edges = graph.sorted_out_edges(tile_sin, data=True)
            gather_sin = insert_gather(graph, tile_sin, embd, position_ids, axis=0, edge_attr=tile_sin_out_edges[0][-1])
            gather_sin_out_edges = graph.sorted_out_edges(gather_sin, data=True)
            gather_sin_output_shape = list(gather_sin_out_edges[0][-1]['tensor'].shape)
            gather_sin_output_shape.insert(2, 1)
            rs_sin = insert_reshape(graph, gather_sin, embd, gather_sin_out_edges[0][-1], gather_sin_output_shape)
            sin_mul = get_valid_node_name(graph, rs_sin + '_post_mul')
            graph.add_node(sin_mul)
            sin_mul_attr = {'name': sin_mul, 'opset_version': 13}
            graph.add_edge(rs_sin, sin_mul, **{'src_out_port': 0, 'dst_in_port': 1})
            graph.add_edge(concat, sin_mul, **{'src_out_port': 0, 'dst_in_port': 0})
            NodeWrap(graph, sin_mul).replace_obj('Mul', sin_mul_attr)

            # Add cos_mul & sin_mul
            add = get_valid_node_name(graph, embd + '_post_add')
            graph.add_node(add)
            add_attr = {'name': add, 'opset_version': 13}
            graph.add_edge(sin_mul, add, **{'src_out_port': 0, 'dst_in_port': 1})
            graph.add_edge(cos_mul, add, **{'src_out_port': 0, 'dst_in_port': 0})
            NodeWrap(graph, add).replace_obj('Add', add_attr)

            rs1 = insert_reshape(graph, add, embd, {'src_out_port': 0, 'dst_in_port': 0}, input_shapes[0])
            graph.remove_edges_from(embd_in_edges)
            graph.remove_edges_from(embd_out_edges)

            for _, dst, out_attr in embd_out_edges:
                tmp_out_attr = copy.deepcopy(out_attr)
                tmp_out_attr.update({'src_out_port': 0})
                graph.add_edge(rs1, dst, **tmp_out_attr)

            clear_redundant_nodes(graph)
        else:
            ERROR(
                '[Parser]: non-const cos_cache/sin_cache of RotaryEmbeddingMs node(%s) not support yet!' % embd)


def convert_mha(graph):
    matches = single_node_matcher(graph, 'MultiHeadAttentionMs')
    for m in matches:
        mha = m['target']
        mha_obj = NodeWrap(graph, mha)['object']
        if mha_obj is None:
            ERROR(
                '[Parser]: Meets invalid MultiHeadAttentionMs node(%s) in convert_mha!' % mha)
            continue
        if mha_obj.quantize:
            continue
        mha_in_edges = graph.sorted_in_edges(mha, data=True)
        mha_out_edges = graph.sorted_out_edges(mha, data=True)
        outports = mha_obj.get_out_ports()
        # TODO, support more cases of MHA
        if len(outports) > 1:
            ERROR(
                '[Parser]: outputs > 1 of MultiHeadAttentionMs node(%s) not support yet!' % mha)
            continue
        input_shapes = mha_obj.get_input_shapes()
        input_dtypes = mha_obj.get_input_dtypes()
        q_shape = input_shapes[0]
        if len(q_shape) != 3:
            ERROR(
                '[Parser]: Only support 3-dims(batch_size, seq_len, hidden_size) of query in MHA node(%s)' % mha)
            continue
        head_dim = q_shape[-1] // mha_obj.num_heads
        bs, seq_len = q_shape[:2]
        query = mha_in_edges[0][0]
        key = mha_in_edges[1][0]
        value = mha_in_edges[2][0]
        attention_bias = mha_in_edges[5][0]

        # split query
        rs_q = insert_reshape(graph, query, mha, mha_in_edges[0][-1], [bs, seq_len, mha_obj.num_heads, head_dim])
        rs_q_in_attr = copy.deepcopy(mha_in_edges[0][-1])
        rs_q_in_attr['tensor'].shape = [bs, seq_len, mha_obj.num_heads, head_dim]
        trans_q = insert_transpose(graph, rs_q, mha, rs_q_in_attr, perm=[0, 2, 1, 3])

        # split key
        rs_k = insert_reshape(graph, key, mha, mha_in_edges[1][-1], [bs, seq_len, mha_obj.num_heads, head_dim])
        rs_k_in_attr = copy.deepcopy(mha_in_edges[1][-1])
        rs_k_in_attr['tensor'].shape = [bs, seq_len, mha_obj.num_heads, head_dim]
        trans_k = insert_transpose(graph, rs_k, mha, rs_k_in_attr, perm=[0, 2, 3, 1])

        trans_q_out_shape = [rs_q_in_attr['tensor'].shape[axis] for axis in [0, 2, 1, 3]]
        trans_k_out_shape = [rs_k_in_attr['tensor'].shape[axis] for axis in [0, 2, 3, 1]]

        # split_q @ split_k
        matmul = get_valid_node_name(graph, mha + '_matmul')
        graph.add_node(matmul)
        matmul_attr = {'name': matmul, 'opset_version': 13}
        graph.add_edge(trans_q, matmul, **{'src_out_port': 0, 'dst_in_port': 0,
                                           'tensor': Tensor(shape=tuple(trans_q_out_shape))})
        graph.add_edge(trans_k, matmul, **{'src_out_port': 0, 'dst_in_port': 1,
                                           'tensor': Tensor(shape=tuple(trans_k_out_shape))})
        NodeWrap(graph, matmul).replace_obj('MatMul', matmul_attr)

        matmul_out_shape = trans_q_out_shape[:-1] + trans_k_out_shape[-1:]

        # scores
        mul = get_valid_node_name(graph, mha + '_mul')
        graph.add_node(mul)
        mul_attr = {'name': mul, 'opset_version': 13}
        graph.add_edge(matmul, mul, **{'src_out_port': 0, 'dst_in_port': 0,
                                       'tensor': Tensor(shape=tuple(matmul_out_shape))})
        insert_constant(graph, mha + '_scale', np.array(mha_obj.scale, dtype=input_dtypes[0]), mul, in_port=1)
        NodeWrap(graph, mul).replace_obj('Mul', mul_attr)

        add = get_valid_node_name(graph, mha + '_add')
        graph.add_node(add)
        add_attr = {'name': add, 'opset_version': 13}
        graph.add_edge(mul, add, **{'src_out_port': 0, 'dst_in_port': 0,
                                    'tensor': Tensor(shape=tuple(matmul_out_shape))})
        graph.add_edge(attention_bias, add, **{'src_out_port': 0, 'dst_in_port': 1})
        NodeWrap(graph, add).replace_obj('Add', add_attr)

        # Softmax
        softmax = get_valid_node_name(graph, mha + '_softmax')
        graph.add_node(softmax)
        softmax_attr = {'name': softmax, 'opset_version': 13, 'axis': -1}
        graph.add_edge(add, softmax, **{'src_out_port': 0, 'dst_in_port': 0})
        NodeWrap(graph, softmax).replace_obj('Softmax', softmax_attr)

        # split value
        rs_v = insert_reshape(graph, value, mha, mha_in_edges[2][-1], [bs, seq_len, mha_obj.num_heads, head_dim])
        rs_v_in_attr = copy.deepcopy(mha_in_edges[2][-1])
        rs_v_in_attr['tensor'].shape = [bs, seq_len, mha_obj.num_heads, head_dim]
        trans_v = insert_transpose(graph, rs_v, mha, rs_v_in_attr, perm=[0, 2, 1, 3])

        trans_v_out_shape = [rs_v_in_attr['tensor'].shape[axis] for axis in [0, 2, 1, 3]]

        matmul_v = get_valid_node_name(graph, mha + '_matmul_v')
        graph.add_node(matmul_v)
        matmul_v_attr = {'name': matmul_v, 'opset_version': 13}
        graph.add_edge(softmax, matmul_v, **{'src_out_port': 0, 'dst_in_port': 0,
                                             'tensor': Tensor(shape=tuple(matmul_out_shape))})
        graph.add_edge(trans_v, matmul_v, **{'src_out_port': 0, 'dst_in_port': 1,
                                             'tensor': Tensor(shape=tuple(trans_v_out_shape))})
        NodeWrap(graph, matmul_v).replace_obj('MatMul', matmul_v_attr)

        matmul_v_out_shape = matmul_out_shape[:-1] + trans_v_out_shape[-1:]

        # concat heads
        trans_out_in_attr = copy.deepcopy(mha_in_edges[2][-1])
        trans_out_in_attr['tensor'].shape = matmul_v_out_shape
        trans_out = insert_transpose(graph, matmul_v, mha, trans_out_in_attr, perm=[0, 2, 1, 3])
        rs_out_in_attr = copy.deepcopy(mha_in_edges[2][-1])
        rs_out_in_attr['tensor'].shape = [matmul_v_out_shape[axis] for axis in [0, 2, 1, 3]]
        rs_out = insert_reshape(graph, trans_out, mha, rs_out_in_attr, q_shape)
        graph.remove_edges_from(mha_in_edges)
        graph.remove_edges_from(mha_out_edges)

        for _, dst, out_attr in mha_out_edges:
            tmp_out_attr = copy.deepcopy(out_attr)
            tmp_out_attr.update({'src_out_port': 0})
            graph.add_edge(rs_out, dst, **tmp_out_attr)

        clear_redundant_nodes(graph)


def convert_skip_simplified_layernorm(graph):
    matches = single_node_matcher(graph, 'SkipSimplifiedLayerNormalizationMs')
    for m in matches:
        skip_simple_ln = m['target']
        skip_simple_ln_obj = NodeWrap(graph, skip_simple_ln)['object']
        if skip_simple_ln_obj is None:
            ERROR(
                '[Parser]: Meets invalid SkipSimplifiedLayerNormalizationMs node(%s) in convert_skip_simplified_layernorm!' % skip_simple_ln)
            continue
        if skip_simple_ln_obj.quantize:
            continue
        ln_in_edges = graph.sorted_in_edges(skip_simple_ln, data=True)
        ln_out_edges = graph.sorted_out_edges(skip_simple_ln, data=True)
        outports = skip_simple_ln_obj.get_out_ports()
        inports = skip_simple_ln_obj.get_in_ports()
        if len(outports) > 2:
            ERROR(
                '[Parser]: outputs > 2 of SkipSimplifiedLayerNormalizationMs node(%s) not support yet!' % skip_simple_ln)
            continue
        if len(ln_in_edges) in (3, 4) and NodeWrap(graph, ln_in_edges[2][0])['object'].type == 'Constant':
            input_shapes = skip_simple_ln_obj.get_input_shapes()
            norm_axes = OpHasAxis.make_axes_non_negative([-1], len(input_shapes[0]))
            weights = OpHasAxis.align_axes(NodeWrap(graph, ln_in_edges[2][0])['object'].value,
                                           norm_axes, input_shapes[0])
            if weights is None:
                continue
            epsilon = skip_simple_ln_obj.epsilon
            rms_norm_attr = skip_simple_ln_obj.copied_attr()
            rms_norm_attr.update({'axes': norm_axes, 'weights': weights, 'epsilon': epsilon})
            NodeWrap(graph, skip_simple_ln).replace_obj('ArmRMSNorm', rms_norm_attr)

            graph.remove_edges_from(ln_in_edges)
            graph.remove_edges_from(ln_out_edges)

            input = ln_in_edges[0][0]
            skip = ln_in_edges[1][0]
            if 3 in inports:
                bias = ln_in_edges[3][0]
            else:
                bias = None

            add = get_valid_node_name(graph, skip_simple_ln + '_add')
            graph.add_node(add)
            add_attr = {'name': add, 'opset_version': 13}
            graph.add_edge(input, add, **ln_in_edges[0][-1])
            graph.add_edge(skip, add, **ln_in_edges[1][-1])
            NodeWrap(graph, add).replace_obj('Add', add_attr)
            if bias is not None:
                add_bias = get_valid_node_name(graph, skip_simple_ln + '_add')
                graph.add_node(add_bias)
                add_bias_attr = {'name': add_bias, 'opset_version': 13}
                graph.add_edge(add, add_bias, **{'src_out_port': 0, 'dst_in_port': 0})
                graph.add_edge(bias, add_bias, **ln_in_edges[3][-1])
                NodeWrap(graph, add_bias).replace_obj('Add', add_bias_attr)
                graph.add_edge(add_bias, skip_simple_ln, **{'src_out_port': 0, 'dst_in_port': 0})
            else:
                graph.add_edge(add, skip_simple_ln, **{'src_out_port': 0, 'dst_in_port': 0})

            for _, dst, out_attr in ln_out_edges:
                tmp_out_attr = copy.deepcopy(out_attr)
                out_port = tmp_out_attr['src_out_port']
                if out_port == 0:
                    graph.add_edge(skip_simple_ln, dst, **tmp_out_attr)
                elif out_port == 3:
                    tmp_out_attr.update({'src_out_port': 0})
                    out_name = add if bias is None else add_bias
                    graph.add_edge(out_name, dst, **tmp_out_attr)
                else:
                    raise NotImplementedError('mean/var output still not support!')

            clear_redundant_nodes(graph)
        else:
            ERROR(
                '[Parser]: non-const gamma of SkipSimplifiedLayerNormalizationMs node(%s) not support yet!' % skip_simple_ln)


def merge_q_ln(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('dequant', {'op': ['DequantizeLinear', 'Slice']}),
                                   ('mean_1', {'op': 'ReduceMean'}),
                                   ('sub', {'op': 'Sub'}),
                                   ('pow_y', {'op': 'Constant', 'unique': False}),
                                   ('pow', {'op': 'Pow'}),
                                   ('mean_2', {'op': 'ReduceMean'}),
                                   ('quant_1', {'op': 'QuantizeLinear'}),
                                   ('eps', {'op': 'Constant', 'unique': False}),
                                   ('add_1', {'op': 'Add'}),
                                   ('dequant_1', {'op': 'DequantizeLinear'}),
                                   ('sqrt', {'op': 'Sqrt'}),
                                   ('div', {'op': 'Div'}),
                                   ('quant_2', {'op': 'QuantizeLinear'}),
                                   ('gamma', {'op': 'Constant'}),
                                   ('mul_1', {'op': 'Mul'}),
                                   ('beta', {'op': 'Constant'}),
                                   ('add_2', {'op': 'Add'}),
                               ],
                               edges=[
                                   ('dequant', 'mean_1'),
                                   ('dequant', 'sub', {'dst_in_port': 0}),
                                   ('mean_1', 'sub', {'dst_in_port': 1}),
                                   ('sub', 'pow', {'dst_in_port': 0}),
                                   ('pow_y', 'pow', {'dst_in_port': 1}),
                                   ('pow', 'mean_2'),
                                   ('mean_2', 'quant_1', {'dst_in_port': 0}),
                                   ('quant_1', 'add_1'),
                                   ('eps', 'add_1'),
                                   ('add_1', 'dequant_1', {'dst_in_port': 0}),
                                   ('dequant_1', 'sqrt'),
                                   ('sub', 'div', {'dst_in_port': 0}),
                                   ('sqrt', 'div', {'dst_in_port': 1}),
                                   ('div', 'quant_2', {'dst_in_port': 0}),
                                   ('quant_2', 'mul_1'),
                                   ('gamma', 'mul_1'),
                                   ('mul_1', 'add_2'),
                                   ('beta', 'add_2'),
                               ])
    matches2 = matched_patterns(graph,
                                nodes=[
                                    ('dequant', {'op': 'DequantizeLinear'}),
                                    ('mean_1', {'op': 'ReduceMean'}),
                                    ('sub', {'op': 'Sub'}),
                                    ('trans', {'op': 'Transpose'}),
                                    ('pow_y', {'op': 'Constant'}),
                                    ('pow', {'op': 'Pow'}),
                                    ('mean_2', {'op': 'ReduceMean'}),
                                    ('quant_1', {'op': 'QuantizeLinear'}),
                                    ('eps', {'op': 'Constant'}),
                                    ('add_1', {'op': 'Add'}),
                                    ('dequant_1', {'op': 'DequantizeLinear'}),
                                    ('sqrt', {'op': 'Sqrt'}),
                                    ('div', {'op': 'Div'}),
                                    ('quant_2', {'op': 'QuantizeLinear'}),
                                    ('gamma', {'op': 'Constant'}),
                                    ('mul_1', {'op': 'Mul'}),
                                    ('beta', {'op': 'Constant'}),
                                    ('add_2', {'op': 'Add'}),
                                ],
                                edges=[
                                    ('dequant', 'mean_1'),
                                    ('dequant', 'sub', {'dst_in_port': 0}),
                                    ('mean_1', 'sub', {'dst_in_port': 1}),
                                    ('sub', 'trans'),
                                    ('trans', 'pow', {'dst_in_port': 0}),
                                    ('pow_y', 'pow', {'dst_in_port': 1}),
                                    ('pow', 'mean_2'),
                                    ('mean_2', 'quant_1', {'dst_in_port': 0}),
                                    ('quant_1', 'add_1'),
                                    ('eps', 'add_1'),
                                    ('add_1', 'dequant_1', {'dst_in_port': 0}),
                                    ('dequant_1', 'sqrt'),
                                    ('trans', 'div', {'dst_in_port': 0}),
                                    ('sqrt', 'div', {'dst_in_port': 1}),
                                    ('div', 'quant_2', {'dst_in_port': 0}),
                                    ('quant_2', 'mul_1'),
                                    ('gamma', 'mul_1'),
                                    ('mul_1', 'add_2'),
                                    ('beta', 'add_2'),
                                ])
    for m in matches + matches2:
        names = ['dequant', 'mean_1', 'pow_y', 'mean_2', 'add_1',
                 'eps', 'gamma', 'mul_1', 'beta', 'add_2']
        names += (['trans'] if 'trans' in m else [])
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(obj is None for obj in obj_dict.values()):
            ERROR('[Parser]: Meets invalid Op in merge_q_ln!')
            continue
        if not all(obj_dict[name].quantize for name in ['add_1', 'mul_1', 'add_2']):
            continue
        slice_node_obj = None
        if obj_dict['dequant'].type == 'Slice':
            slice_node = m['dequant']
            slice_node_obj = obj_dict['dequant']
            slice_in_edges = graph.sorted_in_edges(slice_node, data=True)
            if len(slice_in_edges) < 1 \
                    or any(not in_attr['tensor'].is_const for _, _, in_attr in slice_in_edges[1:]):
                continue
            dequant = slice_in_edges[0][0]
            input_shapes = slice_node_obj.get_input_shapes()
        else:
            dequant = m['dequant']
            input_shapes = obj_dict['mean_1'].get_input_shapes()
        if len(input_shapes) < 1 or input_shapes[0] is None \
                or None in input_shapes[0]:
            ERROR('[Parser]: Meets invalid input shape of Node (%s) in merge_q_ln!' % m['mean_1'])
            continue
        dequant_in_edges = graph.sorted_in_edges(dequant, data=True)
        add1_in_edges = graph.sorted_in_edges(m['add_1'], data=True)
        mul1_in_edges = graph.sorted_in_edges(m['mul_1'], data=True)
        add2_in_edges = graph.sorted_in_edges(m['add_2'], data=True)
        if len(dequant_in_edges) < 2 or len(mul1_in_edges) < 2 \
                or len(add1_in_edges) < 2 or len(add2_in_edges) < 2:
            ERROR('[Parser]: Meets invalid inputs in merge_q_ln!')
            continue
        dequant_src, _, dequant_in_attr = dequant_in_edges[0]
        if dequant_in_attr['tensor'].scale_zp is None or len(dequant_in_attr['tensor'].scale_zp) != 2:
            continue
        if not dequant_in_edges[1][2]['tensor'].is_const \
                or not FLOAT_EQUAL(dequant_in_edges[1][2]['tensor'].value, dequant_in_attr['tensor'].scale_zp[0]):
            continue
        if len(dequant_in_edges) > 2:
            if not dequant_in_edges[2][2]['tensor'].is_const \
                    or not FLOAT_EQUAL(dequant_in_edges[2][2]['tensor'].value, dequant_in_attr['tensor'].scale_zp[1]):
                continue
        else:
            if not FLOAT_EQUAL(dequant_in_attr['tensor'].scale_zp[1], 0):
                continue
        mean1_axes = OpHasAxis.make_axes_non_negative(
            obj_dict['mean_1'].axes, len(input_shapes[0]))
        mean2_axes = OpHasAxis.make_axes_non_negative(
            obj_dict['mean_2'].axes, len(input_shapes[0]))
        if mean2_axes != [len(input_shapes[0]) - 1] \
                or not obj_dict['mean_1'].keepdims \
                or not obj_dict['mean_2'].keepdims:
            continue
        input_perm = None
        if 'trans' in m:
            perm = obj_dict['trans'].perm
            if len(mean1_axes) != 1 or perm[mean1_axes[0]] != mean2_axes[0]:
                continue
            input_perm = perm
            channel_size = input_shapes[0][mean1_axes[0]]
        else:
            if mean1_axes != mean2_axes:
                continue
            channel_size = input_shapes[0][-1]
        eps_quantized = obj_dict['eps'].value
        if eps_quantized is None or eps_quantized.size != 1:
            continue
        eps_out_tensor = [in_attr['tensor'] for src, _, in_attr in add1_in_edges if src == m['eps']]
        if len(eps_out_tensor) != 1 or eps_out_tensor[0].scale_zp is None \
                or len(eps_out_tensor[0].scale_zp) != 2:
            continue
        eps_scale, eps_zp = eps_out_tensor[0].scale_zp
        eps = float((eps_quantized - eps_zp) * eps_scale)
        gamma = obj_dict['gamma'].value
        beta = obj_dict['beta'].value
        if gamma is None or beta is None:
            continue
        gamma = np.atleast_1d(np.squeeze(gamma))
        beta = np.atleast_1d(np.squeeze(beta))
        if np.ndim(gamma) != 1 or np.ndim(beta) != 1 \
                or gamma.size != beta.size \
                or gamma.size != channel_size:
            continue
        gamma_out_attrs = [in_attr for src, _, in_attr in mul1_in_edges if src == m['gamma']]
        if len(gamma_out_attrs) != 1 or gamma_out_attrs[0]['tensor'].scale_zp is None \
                or len(gamma_out_attrs[0]['tensor'].scale_zp) != 2:
            continue
        gamma_out_attr = gamma_out_attrs[0]
        beta_out_attrs = [in_attr for src, _, in_attr in add2_in_edges if src == m['beta']]
        if len(beta_out_attrs) != 1 or beta_out_attrs[0]['tensor'].scale_zp is None \
                or len(beta_out_attrs[0]['tensor'].scale_zp) != 2:
            continue
        beta_out_attr = beta_out_attrs[0]

        matched = True
        graph.remove_edges_from(add2_in_edges)
        graph.add_edge(dequant_src, m['add_2'], **dequant_in_attr)
        insert_constant(graph, m['add_2'] + '_weight', gamma,
                        m['add_2'], in_port=1, scale_zp=gamma_out_attr['tensor'].scale_zp, quantize=True)
        # bias should be int32 dtype, no matter input is int8 or uint8
        insert_constant(graph, m['add_2'] + '_bias', np.array(beta, dtype=np.int32),
                        m['add_2'], in_port=2, scale_zp=beta_out_attr['tensor'].scale_zp, quantize=True)

        ln_attr = obj_dict['add_2'].copied_attr()
        ln_attr.update({'axis': len(input_shapes[0]) - 1,
                        'epsilon': eps,
                        'opset_version': 17,
                        })
        NodeWrap(graph, m['add_2']).replace_obj('LayerNormalization', ln_attr)
        dst = m['add_2']
        if input_perm is not None:
            dst = insert_transpose(graph, dequant_src, dst, dequant_in_attr, input_perm, quantize=True)
        if slice_node_obj is not None:
            slice_axes = [(axis + len(input_shapes[0])) if axis < 0 else axis for axis in slice_node_obj.axes]
            starts = []
            sizes = []
            for axis, shape in enumerate(input_shapes[0]):
                if axis in slice_axes:
                    index = slice_axes.index(axis)
                    start = slice_node_obj.starts[index]
                    size = int((slice_node_obj.ends[index] - start) / slice_node_obj.steps[index])
                    starts.append(start)
                    sizes.append(size)
                else:
                    starts.append(0)
                    sizes.append(shape)
            insert_slice(graph, dequant_src, dst, dequant_in_attr, starts, sizes, quantize=True)

    if matched:
        clear_redundant_nodes(graph)


def merge_q_ln_partial(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('dequant', {'op': 'DequantizeLinear'}),
                                   ('mean_1', {'op': 'ReduceMean'}),
                                   ('sub', {'op': 'Sub'}),
                                   ('pow_y', {'op': 'Constant', 'unique': False}),
                                   ('pow', {'op': 'Pow'}),
                                   ('mean_2', {'op': 'ReduceMean'}),
                                   ('quant_1', {'op': 'QuantizeLinear'}),
                                   ('eps', {'op': 'Constant', 'unique': False}),
                                   ('add_1', {'op': 'Add'}),
                                   ('dequant_1', {'op': 'DequantizeLinear'}),
                                   ('sqrt', {'op': 'Sqrt'}),
                                   ('div', {'op': 'Div'}),
                                   ('quant_2', {'op': 'QuantizeLinear'}),
                                   ('gamma', {'op': 'Constant'}),
                                   ('mul_1', {'op': 'Mul'}),
                                   ('beta', {'op': 'Constant'}),
                                   ('add_2', {'op': 'Add'}),
                               ],
                               edges=[
                                   ('dequant', 'mean_1'),
                                   ('dequant', 'sub', {'dst_in_port': 0}),
                                   ('mean_1', 'sub', {'dst_in_port': 1}),
                                   ('sub', 'pow', {'dst_in_port': 0}),
                                   ('pow_y', 'pow', {'dst_in_port': 1}),
                                   ('pow', 'mean_2'),
                                   ('mean_2', 'quant_1', {'dst_in_port': 0}),
                                   ('quant_1', 'add_1'),
                                   ('eps', 'add_1'),
                                   ('add_1', 'dequant_1', {'dst_in_port': 0}),
                                   ('dequant_1', 'sqrt'),
                                   ('sub', 'div', {'dst_in_port': 0}),
                                   ('sqrt', 'div', {'dst_in_port': 1}),
                                   ('div', 'quant_2', {'dst_in_port': 0}),
                                   ('quant_2', 'mul_1'),
                                   ('gamma', 'mul_1'),
                                   ('mul_1', 'add_2'),
                                   ('beta', 'add_2'),
                               ])
    for m in matches:
        names = ['dequant', 'mean_1', 'pow_y', 'mean_2', 'quant_1', 'eps', 'add_1', 'dequant_1']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(obj is None for obj in obj_dict.values()):
            ERROR('[Parser]: Meets invalid Op in merge_q_ln_partial!')
            continue
        if not all(obj_dict[name].quantize for name in ['add_1']):
            continue

        quant_1_in_edges = graph.sorted_in_edges(m['quant_1'], data=True)
        dequant_1_in_edges = graph.sorted_in_edges(m['dequant_1'], data=True)
        eps_out_edges = graph.sorted_out_edges(m['eps'], data=True)
        if len(quant_1_in_edges) < 2 or len(dequant_1_in_edges) < 2 or len(eps_out_edges) < 1:
            ERROR('[Parser]: Meets invalid inputs/outputs in merge_q_ln_partial!')
            continue
        if not FLOAT_EQUAL(obj_dict['quant_1'].y_scale, obj_dict['dequant_1'].x_scale) \
                or not FLOAT_EQUAL(obj_dict['quant_1'].y_zero_point, obj_dict['dequant_1'].x_zero_point):
            continue
        matched = True
        eps_scale, eps_zp = eps_out_edges[0][2]['tensor'].scale_zp
        eps_float = (obj_dict['eps'].value - eps_zp) * eps_scale
        src, _, in_attr = quant_1_in_edges[0]
        graph.remove_edges_from(quant_1_in_edges)
        graph.remove_edge(m['quant_1'], m['add_1'])
        graph.add_edge(src, m['add_1'], **in_attr)
        for _, dst, out_attr in graph.sorted_out_edges(m['dequant_1'], data=True):
            graph.remove_edge(m['dequant_1'], dst)
            graph.add_edge(m['add_1'], dst, **out_attr)
        obj_dict['eps'].value = eps_float
        obj_dict['eps'].quantize = False
        obj_dict['add_1'].quantize = False

    if matched:
        clear_redundant_nodes(graph)


def merge_q_gelu(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('dequant', {'op': 'DequantizeLinear'}),
                                   ('div', {'op': 'Div'}),
                                   ('divc', {'op': 'Constant'}),
                                   ('erf', {'op': 'Erf'}),
                                   ('quant', {'op': 'QuantizeLinear'}),
                                   ('add', {'op': 'Add'}),
                                   ('addc', {'op': 'Constant'}),
                                   ('mul_1', {'op': 'Mul'}),
                                   ('mulc', {'op': 'Constant'}),
                                   ('mul_2', {'op': 'Mul'}),
                               ],
                               edges=[
                                   ('dequant', 'div', {'dst_in_port': 0}),
                                   ('divc', 'div', {'dst_in_port': 1}),
                                   ('div', 'erf'),
                                   ('erf', 'quant', {'dst_in_port': 0}),
                                   ('quant', 'add'),
                                   ('addc', 'add'),
                                   ('add', 'mul_1'),
                                   ('mul_1', 'mul_2'),
                                   ('mulc', 'mul_2'),
                               ]
                               )
    for m in matches:
        key_names = ['dequant', 'div', 'divc', 'erf', 'add', 'addc',
                     'mulc', 'mul_2']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if any(obj is None for obj in node_objs.values()):
            ERROR('[Parser]: Meets invalid nodes in merge_q_gelu!')
            continue
        if not FLOAT_EQUAL(node_objs['divc'].value, 1.4142135381698608):
            continue
        if not all(node_objs[name].quantize for name in ['add', 'mul_2']):
            continue
        dequant_in_edges = graph.sorted_in_edges(m['dequant'], data=True)
        add_in_edges = graph.sorted_in_edges(m['add'], data=True)
        mul1_in_edges = graph.sorted_in_edges(m['mul_1'], data=True)
        mul2_in_edges = graph.sorted_in_edges(m['mul_2'], data=True)
        if len(dequant_in_edges) < 2 or len(add_in_edges) < 2 \
                or len(mul1_in_edges) < 2 or len(mul2_in_edges) < 2:
            ERROR('[Parser]: Meets invalid inputs in merge_q_gelu!')
            continue
        dequant_src, _, dequant_in_attr = dequant_in_edges[0]
        mul1_in_attrs = [in_attr for src, _, in_attr in mul1_in_edges if src == dequant_src]
        if len(mul1_in_attrs) != 1 or mul1_in_attrs[0]['src_out_port'] != dequant_in_attr['src_out_port']:
            continue
        if dequant_in_attr['tensor'].scale_zp is None or len(dequant_in_attr['tensor'].scale_zp) != 2:
            continue
        if not dequant_in_edges[1][2]['tensor'].is_const \
                or not FLOAT_EQUAL(dequant_in_edges[1][2]['tensor'].value, dequant_in_attr['tensor'].scale_zp[0]):
            continue
        if len(dequant_in_edges) > 2:
            if not dequant_in_edges[2][2]['tensor'].is_const \
                    or not FLOAT_EQUAL(dequant_in_edges[2][2]['tensor'].value, dequant_in_attr['tensor'].scale_zp[1]):
                continue
        else:
            if not FLOAT_EQUAL(dequant_in_attr['tensor'].scale_zp[1], 0):
                continue
        addc_out_attrs = [in_attr for src, _, in_attr in add_in_edges if src == m['addc']]
        if len(addc_out_attrs) != 1 or addc_out_attrs[0]['tensor'].scale_zp is None \
                or len(addc_out_attrs[0]['tensor'].scale_zp) != 2:
            continue
        addc_scale, addc_zp = addc_out_attrs[0]['tensor'].scale_zp
        addc_value = (node_objs['addc'].value - addc_zp) * addc_scale
        if not FLOAT_EQUAL(addc_value, 1.0):
            continue
        mulc_out_attrs = [in_attr for src, _, in_attr in mul2_in_edges if src == m['mulc']]
        if len(mulc_out_attrs) != 1 or mulc_out_attrs[0]['tensor'].scale_zp is None \
                or len(mulc_out_attrs[0]['tensor'].scale_zp) != 2:
            continue
        mulc_scale, mulc_zp = mulc_out_attrs[0]['tensor'].scale_zp
        mulc_value = (node_objs['mulc'].value - mulc_zp) * mulc_scale
        if not FLOAT_EQUAL(mulc_value, 0.5):
            continue

        matched = True
        mul_2_in_edges = graph.sorted_in_edges(m['mul_2'])
        graph.remove_edges_from(mul_2_in_edges)
        graph.add_edge(dequant_src, m['mul_2'], **dequant_in_attr)
        gelu_attr = node_objs['mul_2'].copied_attr()
        gelu_attr.update({'opset_version': 20, 'approximate': 'none'})
        NodeWrap(graph, m['mul_2']).replace_obj('Gelu', gelu_attr)

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
                in_shapes = broadcast_obj.get_input_shapes()
                if any([s is None or None in s for s in in_shapes]):
                    ERROR(
                        '[Parser]: Meets Broadcast op (%s) with empty inputs in multidirectional_broadcasting!' % broadcast)
                    continue
                if broadcast_obj.type == 'BitShift' and list(in_shapes[1]) == [1]:
                    DEBUG(
                        '[Parser]: Meets Broadcast op (%s) with shift shape is [1], no need to broadcast in multidirectional_broadcasting!' % broadcast)
                    continue
                if graph._attr.get('quantize', False) \
                        and broadcast_obj.quantize:
                    quantize = True
                else:
                    quantize = False
                try:
                    dims_and_reps = OpNeedBroadcast.cal_reshape_and_tile([s for s in in_shapes])
                except:
                    dims_and_reps = []
                if len(dims_and_reps) == len(in_edges):
                    for i, dr in enumerate(dims_and_reps):
                        if dr['reshape'] is not None:
                            src, _, k, in_attr = in_edges[i]
                            insert_reshape(graph, src, broadcast,
                                           in_attr, dr['reshape'], key=k, quantize=quantize)
                            in_edges = graph.sorted_in_edges(
                                broadcast, keys=True, data=True)
                        if dr['tile'] is not None:
                            src, _, k, in_attr = in_edges[i]
                            insert_tile(graph, src, broadcast,
                                        in_attr, dr['tile'], key=k, quantize=quantize)
                            in_edges = graph.sorted_in_edges(
                                broadcast, keys=True, data=True)
                else:
                    ERROR(
                        '[Parser]: Failed to calculate Broadcast op (%s) broadcast in multidirectional_broadcasting!' % broadcast)


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
                                       key=k,
                                       quantize=prelu_obj.quantize)
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
                    ERROR(
                        '[Parser]: Number of broadcast params is wrong in reshape_prelu_slope!')
        else:
            ERROR('[Parser]: Meets invalid PRelu Op(%s) in reshape_prelu_slope!' % prelu)


def deduplicate_mean_sub(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('input', {}),
                                   ('mean', {'op': 'ReduceMean'}),
                                   ('sub_1', {'op': 'Sub'}),
                                   ('sub_2', {'op': 'Sub'}),
                               ],
                               edges=[
                                   ('input', 'mean'),
                                   ('input', 'sub_1', {'dst_in_port': 0}),
                                   ('mean', 'sub_1', {'dst_in_port': 1}),
                                   ('input', 'sub_2', {'dst_in_port': 0}),
                                   ('mean', 'sub_2', {'dst_in_port': 1}),
                               ])
    for m in matches:
        names = ['mean', 'sub_1', 'sub_2']
        node_objs = {n: NodeWrap(graph, m[n])['object'] for n in names}
        mean_in_edges = graph.sorted_in_edges(m['mean'], data=True)
        sub_1_in_edges = graph.sorted_in_edges(m['sub_1'], data=True)
        sub_2_in_edges = graph.sorted_in_edges(m['sub_2'], data=True)
        if any(obj is None for obj in node_objs.values()) \
                or len(mean_in_edges) < 1 \
                or len(sub_1_in_edges) < 2 or len(sub_2_in_edges) < 2:
            ERROR('[Parser]: Meets invalid node in deduplicate_mean_sub!')
            continue
        mean_in_port = mean_in_edges[0][2]['src_out_port']
        sub_1_in_port = sub_1_in_edges[0][2]['src_out_port']
        sub_2_in_port = sub_2_in_edges[0][2]['src_out_port']
        if mean_in_port != sub_1_in_port or mean_in_port != sub_2_in_port:
            continue
        sub_2_out_edges = graph.sorted_out_edges(m['sub_2'], data=True)
        graph.remove_edges_from(sub_2_out_edges + sub_2_in_edges)
        for _, dst, out_attr in sub_2_out_edges:
            graph.add_edge(m['sub_1'], dst, **out_attr)
        if m['sub_2'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['sub_2'])
            if m['sub_1'] not in graph._attr['output_names']:
                graph._attr['output_names'][index] = m['sub_1']
            else:
                graph._attr['output_names'].pop(index)
        graph.remove_node(m['sub_2'])


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
            ERROR('[Parser]: Meets invalid node in duplicate_moments_mean!')
            continue
        if len(graph.sorted_out_edges(m['mean1'])) <= 1:
            continue
        if any([len(graph.sorted_out_edges(m[n])) > 1 for n in ['sub', 'pow']]):
            continue
        mean1_in_edges = graph.sorted_in_edges(m['mean1'], data=True)
        sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
        if len(mean1_in_edges) < 1 or len(sub_in_edges) != 2:
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
    matches2 = matched_patterns(graph,
                                nodes=[
                                    ('mean1', {'op': 'ReduceMean'}),
                                    ('sub', {'op': 'Sub'}),
                                    ('mul', {'op': 'Mul'}),
                                    ('mean2', {'op': 'ReduceMean'}),
                                ],
                                edges=[
                                    ('mean1', 'sub'),
                                    ('sub', 'mul', {'dst_in_port': 0}),
                                    ('sub', 'mul', {'dst_in_port': 1}),
                                    ('mul', 'mean2')
                                ])
    for m in matches + matches2:
        names = ['mean1', 'sub', 'mean2'] + (['pow'] if 'pow' in m else ['mul'])
        node_objs = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in node_objs.values()]):
            ERROR('[Parser]: Meets invalid node in merge_reduce_variance!')
            continue
        mean1_in_edges = graph.sorted_in_edges(m['mean1'], data=True)
        sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
        if len(mean1_in_edges) < 1 or len(sub_in_edges) != 2:
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
        if 'pow' in m and (len(node_objs['pow'].get_input_tensors()) != 2
                           or not FLOAT_EQUAL(node_objs['pow'].get_input_tensors()[1], 2)):
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
            ERROR('[Parser]: Meets invalid node in merge_reduce_unbiased_variance!')
            continue

        if node_objs['var'].unbiased:
            continue

        var_in_edges = graph.sorted_in_edges(m['var'], data=True)
        var_input_shapes = node_objs['var'].get_input_shapes()
        if len(var_in_edges) < 1 or len(var_input_shapes) < 1:
            ERROR(
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
            ERROR('[Parser]: Meets invalid nodes in merge_normalized_moments!')
            continue

        mul_1_in_edges = graph.sorted_in_edges(m['mul1'], data=True)
        mul_2_in_edges = graph.sorted_in_edges(m['mul2'], data=True)
        sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
        sub_out_edges = graph.sorted_out_edges(m['sub'], data=True)
        pow_out_edges = graph.sorted_out_edges(m['pow'], data=True)
        mul_1_out_edges = graph.sorted_out_edges(m['mul1'], data=True)
        mul_2_out_edges = graph.sorted_out_edges(m['mul2'], data=True)

        if len(pow_out_edges) != 1 \
                or len(mul_2_out_edges) != 1 \
                or len(node_objs['pow'].sorted_in_consts()) != 1 \
                or len(node_objs['mul1'].sorted_in_consts()) != 1 \
                or len(node_objs['mul2'].sorted_in_consts()) != 1 \
                or FLOAT_EQUAL(node_objs['mul1'].sorted_in_consts()[0][2],
                               node_objs['mul2'].sorted_in_consts()[0][2]) is False:
            continue

        count_value = np.round(1 / node_objs['mul1'].sorted_in_consts()[0][2], 5)
        if type(count_value) is np.ndarray:
            if np.all(count_value == np.mean(count_value)):
                count_value = np.mean(count_value)
            else:
                continue

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
            ERROR('[Parser]: Meets invalid nodes in merge_moments!')
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
            ERROR('[Parser]: Invalid node in merge_erosion!')
    if matched:
        clear_redundant_nodes(graph)


def merge_reducel1(graph):
    '''
    ReduceL1(x)=reducesum(abs(x))
    Merge Abs+ReduceSum to ReduceL1.
    '''
    matched = False
    matches = two_nodes_matcher(graph, begin_op='Abs', end_op='ReduceSum')
    for m in matches:
        abs_node, sum_node = m['begin'], m['end']
        sum_obj = NodeWrap(graph, sum_node)['object']
        if sum_obj is None:
            ERROR('[Parser]: Meets invalid ReduceSum Node (%s) in merge_reducel1!' % sum_node)
            continue
        abs_in_edges = graph.sorted_in_edges(abs_node, data=True)
        if len(abs_in_edges) < 1:
            ERROR('[Parser]: Meets invalid in_edges of Abs Node (%s) in merge_reducel1!' % abs_node)
            continue
        matched = True
        sum_in_edges = graph.sorted_in_edges(sum_node)
        graph.remove_edges_from(sum_in_edges)
        src, _, in_attr = abs_in_edges[0]
        graph.add_edge(src, sum_node, **in_attr)
        reduce_l1_attr = sum_obj.copied_attr()
        reduce_l1_attr.update({'opset_version': 13})
        NodeWrap(graph, sum_node).replace_obj('ReduceL1', reduce_l1_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_reducel2(graph):
    '''
    ReduceL2(x)=sqrt(reducesum(pow(x, 2)))
    Merge Mul(or Pow)+ReduceSum+Sqrt to ReduceL2.
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('mul', {'op': 'Mul'}),
                                   ('sum', {'op': 'ReduceSum'}),
                                   ('sqrt', {'op': 'Sqrt'}),
                               ],
                               edges=[
                                   ('mul', 'sum'),
                                   ('sum', 'sqrt'),
                               ])
    matches2 = matched_patterns(graph,
                                nodes=[
                                    ('pow_y', {'op': 'Constant'}),
                                    ('pow', {'op': 'Pow'}),
                                    ('sum', {'op': 'ReduceSum'}),
                                    ('sqrt', {'op': 'Sqrt'}),
                                ],
                                edges=[
                                    ('pow_y', 'pow', {'dst_in_port': 1}),
                                    ('pow', 'sum'),
                                    ('sum', 'sqrt'),
                                ])
    for m in matches + matches2:
        sum_node, sqrt_node = m['sum'], m['sqrt']
        if 'mul' in m:
            square_in_edges = graph.sorted_in_edges(m['mul'], data=True)
            if len(square_in_edges) != 2 \
                    or square_in_edges[0][0] != square_in_edges[1][0] \
                    or square_in_edges[0][2]['src_out_port'] != square_in_edges[1][2]['src_out_port']:
                continue
        else:
            pow_y_obj = NodeWrap(graph, m['pow_y'])['object']
            if not FLOAT_EQUAL(pow_y_obj.value, 2):
                continue
            square_in_edges = graph.sorted_in_edges(m['pow'], data=True)
            if len(square_in_edges) < 1:
                ERROR('[Parser]: Meets invalid Pow Node (%s) in merge_reducel2!' % m['pow'])
                continue
        sum_obj = NodeWrap(graph, sum_node)['object']
        if sum_obj is None:
            ERROR('[Parser]: Meets invalid ReduceSum Node (%s) in merge_reducel2!' % sum_node)
            continue
        sqrt_obj = NodeWrap(graph, sqrt_node)['object']
        if sqrt_obj is None:
            ERROR('[Parser]: Meets invalid Sqrt Node (%s) in merge_reducel2!' % sqrt_node)
            continue
        matched = True
        src, _, in_attr = square_in_edges[0]
        sqrt_in_edges = graph.sorted_in_edges(sqrt_node)
        graph.remove_edges_from(sqrt_in_edges)
        graph.add_edge(src, sqrt_node, **in_attr)
        reduce_l2_attr = sqrt_obj.copied_attr()
        reduce_l2_attr.update({'opset_version': 13, 'axes': sum_obj.axes, 'keepdims': sum_obj.keepdims})
        NodeWrap(graph, sqrt_node).replace_obj('ReduceL2', reduce_l2_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_reducel2_reshape(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('l2', {'op': 'ReduceL2'}),
                                   ('reshape', {'op': 'Reshape'}),
                                   ('shape', {'op': 'Constant', 'unique': False}),
                               ],
                               edges=[
                                   ('l2', 'reshape', {'dst_in_port': 0}),
                                   ('shape', 'reshape', {'dst_in_port': 1}),
                               ])
    for m in matches:
        l2_node, reshape_node, shape_node = m['l2'], m['reshape'], m['shape']
        l2_obj, reshape_obj, shape_obj = [NodeWrap(graph, name)['object'] for name in
                                          [l2_node, reshape_node, shape_node]]
        if l2_obj is None or reshape_obj is None or shape_obj is None:
            ERROR('[Parser]: Meets invalid node in merge_reducel2_reshape!')
            continue
        if l2_obj.keepdims:
            continue
        l2_output_shapes = l2_obj.get_output_shapes()
        new_shape = l2_output_shapes[0].copy()
        for axis in l2_obj.axes:
            new_shape.insert(axis, 1)
        if new_shape != shape_obj.value.tolist():
            continue
        l2_out_edges = graph.sorted_out_edges(l2_node, data=True)
        if len(l2_out_edges) != 1:
            continue
        matched = True
        reshape_out_edges = graph.sorted_out_edges(reshape_node, data=True)
        graph.remove_edges_from(l2_out_edges)
        l2_obj.keepdims = True
        for _, dst, out_attr in reshape_out_edges:
            new_out_attr = copy.deepcopy(out_attr)
            graph.add_edge(l2_node, dst, **new_out_attr)
        if reshape_node in graph._attr['output_names']:
            index = graph._attr['output_names'].index(reshape_node)
            graph._attr['output_names'][index] = l2_node
    if matched:
        clear_redundant_nodes(graph)


def merge_l1norm(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('reduce_l1', {'op': 'ReduceL1'}),
                                      ('div', {'op': 'Div'}),
                                      ],
                               edges=[('reduce_l1', 'div', {'dst_in_port': 1}),
                                      ])
    for m in matches:
        reduce_l1, div = m['reduce_l1'], m['div']
        reduce_l1_obj, div_obj = [NodeWrap(graph, name)['object'] for name in [reduce_l1, div]]
        reduce_l1_in_edges = graph.sorted_in_edges(reduce_l1, data=True)
        div_in_edges = graph.sorted_in_edges(div, data=True)
        if reduce_l1_obj is None \
                or div_obj is None \
                or len(reduce_l1_in_edges) < 1 \
                or len(div_in_edges) < 2:
            ERROR('[Parser]: Meets invalid node in merge_l1norm!')
            continue
        if not reduce_l1_obj.keepdims:
            continue
        div_inp0, _, div0_in_attr = div_in_edges[0]
        reduce_inp, _, reduce_in_attr = reduce_l1_in_edges[0]
        if div_inp0 != reduce_inp \
                or div0_in_attr['dst_in_port'] != 0 \
                or reduce_in_attr['dst_in_port'] != 0 \
                or reduce_in_attr['src_out_port'] != div0_in_attr['src_out_port']:
            continue
        matched = True
        graph.remove_edges_from(div_in_edges)
        graph.add_edge(reduce_inp, div, **reduce_in_attr)
        l1norm_attr = div_obj.copied_attr()
        l1norm_attr.update({'opset_version': 1, 'p': 1, 'axes': reduce_l1_obj.axes})
        NodeWrap(graph, div).replace_obj('LpNormalization', l1norm_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_l2norm(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('square', {'op': 'Mul'}),
                                   ('l2norm', {'op': 'Mul'}),
                                   ('sum', {'op': 'ReduceSum'}),
                                   ('eps', {'op': 'Constant'}),
                                   ('max', {'op': 'Max'}),
                                   ('sqrt', {'op': 'Sqrt'}),
                                   ('recip', {'op': 'Reciprocal'}),
                               ],
                               edges=[
                                   ('square', 'sum'),
                                   ('sum', 'max'),
                                   ('eps', 'max'),
                                   ('max', 'sqrt'),
                                   ('sqrt', 'recip'),
                                   ('recip', 'l2norm'),
                               ])
    matches2 = matched_patterns(graph,
                                nodes=[
                                    ('pow_y', {'op': 'Constant'}),
                                    ('pow', {'op': 'Pow'}),
                                    ('l2norm', {'op': 'Mul'}),
                                    ('sum', {'op': 'ReduceSum'}),
                                    ('eps', {'op': 'Constant'}),
                                    ('max', {'op': 'Max'}),
                                    ('sqrt', {'op': 'Sqrt'}),
                                    ('recip', {'op': 'Reciprocal'}),
                                ],
                                edges=[
                                    ('pow_y', 'pow', {'dst_in_port': 1}),
                                    ('pow', 'sum'),
                                    ('sum', 'max'),
                                    ('eps', 'max'),
                                    ('max', 'sqrt'),
                                    ('sqrt', 'recip'),
                                    ('recip', 'l2norm'),
                                ])
    for m in matches + matches2:
        if 'pow' in m:
            pow_y_obj = NodeWrap(graph, m['pow_y'])['object']
            if pow_y_obj is None:
                ERROR('[Parser]: Meets invalid node (%s) in merge_l2norm!' % m['pow_y'])
                continue
            if not FLOAT_EQUAL(pow_y_obj.value, 2.):
                continue
            square_in_edges = graph.sorted_in_edges(m['pow'], data=True)
            if len(square_in_edges) < 1:
                continue
        else:
            square_in_edges = graph.sorted_in_edges(m['square'], data=True)
            if len(square_in_edges) != 2 \
                    or square_in_edges[0][0] != square_in_edges[1][0] \
                    or square_in_edges[0][2]['src_out_port'] != square_in_edges[1][2]['src_out_port']:
                continue
        inp, _, in_attr = square_in_edges[0]
        l2norm = m['l2norm']
        sum_obj, l2norm_obj, eps_obj, max_obj = [NodeWrap(graph, m[name])['object'] for name in [
            'sum', 'l2norm', 'eps', 'max']]
        if sum_obj is None or l2norm_obj is None or eps_obj is None or max_obj is None:
            ERROR('[Parser]: Meets invalid node in merge_l2norm!')
            continue
        if not sum_obj.keepdims \
                or max_obj.get_in_ports() != [0, 1] \
                or eps_obj.value.size != 1:
            continue
        l2norm_in_edges = graph.sorted_in_edges(l2norm, data=True)
        if len(l2norm_in_edges) != 2 \
                or inp not in (l2norm_in_edges[0][0], l2norm_in_edges[1][0]):
            continue
        matched = True
        axes = sum_obj.axes
        for src, _, in_attr in l2norm_in_edges:
            if src == inp:
                new_in_attr = copy.deepcopy(in_attr)
                new_in_attr.update({'dst_in_port': 0})
                graph.remove_edge(src, l2norm)
                graph.add_edge(src, l2norm, **new_in_attr)
            else:
                graph.remove_edge(src, l2norm)
        l2norm_attr = l2norm_obj.copied_attr()
        l2norm_attr.update({'opset_version': 1, 'p': 2, 'axes': axes, 'epsilon': eps_obj.value.item()})
        NodeWrap(graph, l2norm).replace_obj('LpNormalization', l2norm_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_l2norm2(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('l2', {'op': 'ReduceL2'}),
                                   ('add', {'op': 'Add'}),
                                   ('eps', {'op': 'Constant', 'unique': False}),
                                   ('div', {'op': 'Div'}),
                               ],
                               edges=[
                                   ('l2', 'add', {'dst_in_port': 0}),
                                   ('eps', 'add', {'dst_in_port': 1}),
                                   ('add', 'div', {'dst_in_port': 1}),
                               ])
    for m in matches:
        l2_node, add_node, eps_node, div_node = m['l2'], m['add'], m['eps'], m['div']
        l2_obj, add_obj, eps_obj, div_obj = [NodeWrap(graph, name)['object'] for name in
                                             [l2_node, add_node, eps_node, div_node]]
        if any([obj is None for obj in [l2_obj, add_obj, eps_obj, div_obj]]):
            ERROR('[Parser]: Meets invalid node in merge_l2norm2!')
            continue
        if not l2_obj.keepdims:
            continue
        l2_in_edges = graph.sorted_in_edges(l2_node, data=True)
        div_in_edges = graph.sorted_in_edges(div_node, data=True)
        if len(graph.sorted_out_edges(l2_node)) == 1 and \
                len(graph.sorted_out_edges(add_node)) == 1 and \
                l2_in_edges[0][0] == div_in_edges[0][0] and \
                int(np.prod(eps_obj.value.shape)) == 1:
            matched = True
            l2_out_edges = graph.sorted_out_edges(l2_node, data=True)
            div_out_edges = graph.sorted_out_edges(div_node, data=True)
            graph.remove_edges_from(l2_out_edges)
            graph.remove_edges_from(div_out_edges)
            for _, dst, out_attr in div_out_edges:
                new_out_attr = copy.deepcopy(out_attr)
                graph.add_edge(l2_node, dst, **new_out_attr)
            l2norm_attr = l2_obj.copied_attr()
            l2norm_attr.update({'opset_version': 1, 'p': 2, 'axes': l2_obj.axes, 'epsilon': eps_obj.value.item()})
            NodeWrap(graph, l2_node).replace_obj('LpNormalization', l2norm_attr)
            if div_node in graph._attr['output_names']:
                index = graph._attr['output_names'].index(div_node)
                graph._attr['output_names'][index] = l2_node
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
            ln_attr.update({'epsilon': epsilon, 'opset_version': 17,
                            'axes': [-1]})
            NodeWrap(graph, m['add_2']).replace_obj('LayerNormalization', ln_attr)
            insert_constant(graph, m['add_2'] + '_scale', gamma, m['add_2'], in_port=1)
            insert_constant(graph, m['add_2'] + '_bias', beta, m['add_2'], in_port=2)
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
                                      ('epsilon', 'add_1'),
                                      ('add_1', 'sqrt'),
                                      ('sqrt', 'recip'),
                                      ('recip', 'mul_1'),
                                      ('mean_1', 'mul_2'),
                                      ('recip', 'mul_2'),
                                      ('beta', 'sub_2'),
                                      ('mul_2', 'sub_2', {'dst_in_port': 1}),
                                      ('mul_1', 'add_2'),
                                      ('sub_2', 'add_2')
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
                                       ('gamma', 'mul_gamma'),
                                       ('mul_gamma', 'mul_1'),
                                       ('mean_1', 'mul_2'),
                                       ('mul_gamma', 'mul_2'),
                                       ('beta', 'sub_2'),
                                       ('mul_2', 'sub_2', {'dst_in_port': 1}),
                                       ('mul_1', 'add_2'),
                                       ('sub_2', 'add_2')
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
                        biases, non_axes[1], in_shape)
                    in_weights = None
                    if gamma:
                        in_weights = OpHasAxis.align_axes(
                            weights, non_axes[1], in_shape)
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
                    biases = OpHasAxis.align_axes(biases, axes, in_shape)
                    weights = OpHasAxis.align_axes(
                        weights, axes, in_shape) if gamma else None
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
                ln_attr.update({'epsilon': eps})
                if is_in:
                    ln_attr.update(
                        {'opset_version': 6, 'non_channel_axes': axes, 'data_format': data_format,
                         'weights': weights, 'biases': biases})
                    NodeWrap(graph, add_2).replace_obj(
                        'InstanceNormalization', ln_attr)
                else:
                    ln_attr.update({'opset_version': 17, 'axes': axes})
                    NodeWrap(graph, add_2).replace_obj('LayerNormalization', ln_attr)
                    insert_constant(graph, add_2 + '_scale', weights, add_2, in_port=1)
                    insert_constant(graph, add_2 + '_bias', biases, add_2, in_port=2)
                if pre_perm is not None:
                    insert_transpose(graph, inp, add_2, inp_out_attr, pre_perm)
                    post_trans = insert_transpose_after(
                        graph, add_2, Op.cal_inverse_perm(pre_perm))
                    if add_2 in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(add_2)
                        graph._attr['output_names'][index] = post_trans
        else:
            ERROR('[Parser]: Meets invalid nodes in merge_ln2!')
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
                                            input_shapes[0])
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
                                'opset_version': 17})
                NodeWrap(graph, add_1).replace_obj('LayerNormalization', ln_attr)
                insert_constant(graph, add_1 + '_scale', gamma, add_1, in_port=1)
                insert_constant(graph, add_1 + '_bias', beta, add_1, in_port=2)
        else:
            ERROR('[Parser]: Meets invalid nodes in merge_ln3!')
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
                                   ('eps', 'add_1'),
                                   ('weight', 'mul'),
                                   ('bias', 'add_2'),
                               ]
                               )
    for m in matches:
        key_names = ['mean_1', 'sub', 'pow', 'pow_y',
                     'mean_2', 'add_1', 'sqrt', 'div', 'mul', 'add_2', 'eps', 'weight', 'bias']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if all(obj is not None for obj in node_objs.values()):
            mean_1_in_edges = graph.sorted_in_edges(m['mean_1'], data=True)
            sub_in_edges = graph.sorted_in_edges(m['sub'], data=True)
            if len(mean_1_in_edges) < 1 \
                    or len(sub_in_edges) != 2 \
                    or mean_1_in_edges[0][0] != sub_in_edges[0][0] \
                    or mean_1_in_edges[0][2]['src_out_port'] != sub_in_edges[0][2]['src_out_port']:
                continue
            input_shape = mean_1_in_edges[0][2]['tensor'].get_shape()
            if input_shape is None or any(s is None for s in input_shape):
                continue

            add_2_in_edges = graph.sorted_in_edges(m['add_2'])
            weight = node_objs['weight'].value
            bias = node_objs['bias'].value

            if node_objs['mean_1'].axes != node_objs['mean_2'].axes \
                    or FLOAT_EQUAL(node_objs['pow_y'].value, 2.0) is False:
                continue
            axes = OpHasAxis.make_axes_non_negative(
                node_objs['mean_1'].axes, len(input_shape))
            axes = sorted(axes)
            weight = OpHasAxis.align_axes(weight, axes, input_shape)
            bias = OpHasAxis.align_axes(bias, axes, input_shape)
            if weight is None or bias is None:
                continue
            if node_objs['eps'].value is None \
                    or (node_objs['eps'].value.size > 1 and np.any(
                        node_objs['eps'].value.flatten()[0] != node_objs['eps'].value)):
                continue

            matched = True
            if node_objs['eps'].value.size == 1:
                eps = float(node_objs['eps'].value)
            else:
                eps = float(node_objs['eps'].value.flatten()[0])
            inp, _, in_attr = mean_1_in_edges[0]
            graph.remove_edges_from(add_2_in_edges)
            graph.add_edge(inp, m['add_2'], **in_attr)
            ln_attr = node_objs['add_2'].copied_attr()
            ln_attr.update({'epsilon': eps, 'opset_version': 17,
                            'axes': node_objs['mean_2'].axes})
            NodeWrap(graph, m['add_2']).replace_obj('LayerNormalization', ln_attr)
            insert_constant(graph, m['add_2'] + '_scale', weight, m['add_2'], in_port=1)
            insert_constant(graph, m['add_2'] + '_bias', bias, m['add_2'], in_port=2)
        else:
            ERROR('[Parser]: Meets invalid nodes in merge_ln4!')
    if matched:
        clear_redundant_nodes(graph)


def merge_ln5(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
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
    matches2 = matched_patterns(graph,
                                nodes=[
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
                                    ('mean', 'sub_1', {'dst_in_port': 1}),
                                    ('sub_1', 'square'),
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
    matches += matches2
    for m in matches:
        names = ['mean', 'sub_1', 'exponent_1', 'square', 'mean_1',
                 'add', 'eps', 'exponent_2', 'pow', 'div', 'mul', 'gamma', 'add_1', 'beta']
        in_names = ['mean', 'sub_1']
        if 'square_diff' in m:
            names += ['square_diff']
            in_names += ['square_diff']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in obj_dict.values()]):
            ERROR('[Parser]: Meets invalid nodes in merge_ln5!')
            continue

        found_error = False
        edges_dict = {}
        for name in in_names:
            edges = graph.sorted_in_edges(m[name], data=True)
            if (name == 'mean' and len(edges) < 1) \
                    or (name != 'mean' and len(edges) != 2):
                ERROR('[Parser]: Meets invalid nodes(%s) in merge_ln5!' % name)
                found_error = True
                break
            edges_dict.update({name: edges})
        if found_error:
            continue

        if edges_dict['mean'][0][0] != edges_dict['sub_1'][0][0]:
            continue
        if 'square_diff' in edges_dict and edges_dict['mean'][0][0] != edges_dict['square_diff'][0][0]:
            continue
        mean_in_edges = edges_dict['mean']
        sub_1_in_edges = edges_dict['sub_1']
        inp = mean_in_edges[0][0]
        inp_out_attr = mean_in_edges[0][2]
        inp_out_port = inp_out_attr['src_out_port']
        found_invalid_port = False
        if 'square_diff' in edges_dict:
            for src, _, in_attr in edges_dict['square_diff']:
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
        input_shape = inp_out_attr['tensor'].get_shape()
        if input_shape is None:
            continue
        axes1 = sorted(OpHasAxis.make_axes_non_negative(obj_dict['mean'].axes, len(input_shape)))
        axes2 = sorted(OpHasAxis.make_axes_non_negative(obj_dict['mean_1'].axes, len(input_shape)))
        if axes1 != axes2:
            continue
        matched = True
        gamma = obj_dict['gamma'].value
        beta = obj_dict['beta'].value
        weights = OpHasAxis.align_axes(gamma, axes1, input_shape)
        biases = OpHasAxis.align_axes(beta, axes1, input_shape)
        if weights is None or biases is None:
            last_node = m['div']
            new_op_type = 'MeanVarianceNormalization'
            node_attr = obj_dict['div'].copied_attr()
            node_attr.update({'opset_version': 13})
        else:
            last_node = m['add_1']
            new_op_type = 'LayerNormalization'
            node_attr = obj_dict['add_1'].copied_attr()
            node_attr.update({'opset_version': 17})
        last_node_in_edges = graph.sorted_in_edges(last_node)
        graph.remove_edges_from(list(edges_dict.values()) + last_node_in_edges)
        new_edge_attr = copy.deepcopy(inp_out_attr)
        new_edge_attr.update({'src_out_port': inp_out_port, 'dst_in_port': 0})
        graph.add_edge(inp, last_node, **new_edge_attr)
        eps = float(obj_dict['eps'].value)
        node_attr.update({'epsilon': eps,
                          'axes': axes1})
        NodeWrap(graph, last_node).replace_obj(new_op_type, node_attr)
        if new_op_type == 'LayerNormalization':
            insert_constant(graph, m['add_1'] + '_scale', weights, m['add_1'], in_port=1)
            insert_constant(graph, m['add_1'] + '_bias', biases, m['add_1'], in_port=2)
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
            ERROR('[Parser]: Meets invalid nodes in merge_ln6!')
            continue
        if obj_dict['bn'].training_mode is False:
            continue
        inputs = obj_dict['bn'].get_input_tensors()
        scale = inputs[1]
        offset = inputs[2]
        if np.all(scale == 1) and np.all(offset == 0):
            reshape1_in_edges = graph.sorted_in_edges(m['reshape1'], data=True)
            reshape2_in_edges = graph.sorted_in_edges(m['reshape2'], data=True)

            if len(reshape2_in_edges) >= 1 \
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
                                'axes': axis,
                                'opset_version': 17})
                NodeWrap(graph, m['reshape2']).replace_obj(
                    'LayerNormalization', ln_attr)
                insert_constant(graph, m['reshape2'] + '_scale', gamma, m['reshape2'], in_port=1)
                insert_constant(graph, m['reshape2'] + '_bias', beta, m['reshape2'], in_port=2)
            else:
                ERROR('[Parser]: Meets invalid nodes in merge_ln6!')

    if matched:
        clear_redundant_nodes(graph)


def merge_ln_reshape(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('reshape_1', {'op': 'Reshape'}),
                                   ('ln', {'op': 'LayerNormalization'}),
                                   ('reshape_2', {'op': 'Reshape'}),
                               ],
                               edges=[
                                   ('reshape_1', 'ln', {'dst_in_port': 0}),
                                   ('ln', 'reshape_2'),
                               ]
                               )
    for m in matches:
        obj_dict = {name: NodeWrap(graph, m[name])['object']
                    for name in ['reshape_1', 'ln', 'reshape_2']}
        ln_in_edges = graph.sorted_in_edges(m['ln'], data=True)
        if any(obj is None for obj in obj_dict.values()) \
                or len(graph.sorted_in_edges(m['reshape_1'])) < 1 \
                or len(ln_in_edges) < 2:
            ERROR('[Parser]: Meets invalid Node in merge_ln_reshape!')
            continue
        scale_in_attr = ln_in_edges[1][2]
        bias_in_attr = ln_in_edges[2][2]
        if scale_in_attr['tensor'] is None or not scale_in_attr['tensor'].is_const \
                or bias_in_attr['tensor'] is None or not bias_in_attr['tensor'].is_const:
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
        weights = np.reshape(scale_in_attr['tensor'].value, exp_wb_shape)
        biases = np.reshape(bias_in_attr['tensor'].value, exp_wb_shape)

        src, _, in_attr = reshape1_in_edges[0]
        reshape2_in_edges = graph.sorted_in_edges(m['reshape_2'])
        graph.remove_edges_from(reshape2_in_edges)
        graph.add_edge(src, m['reshape_2'], **in_attr)
        new_ln_attr = obj_dict['reshape_2'].copied_attr()
        new_ln_attr.update({'epsilon': obj_dict['ln'].epsilon,
                            'axes': np.array(new_ln_axes),
                            'opset_version': 17})
        NodeWrap(graph, m['reshape_2']).replace_obj('LayerNormalization', new_ln_attr)
        insert_constant(graph, m['reshape_2'] + '_scale', weights, m['reshape_2'], in_port=1)
        insert_constant(graph, m['reshape_2'] + '_bias', biases, m['reshape_2'], in_port=2)
    if matched:
        clear_redundant_nodes(graph)


def merge_ln_mul_add(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('ln', {'op': 'LayerNormalization'}),
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
        ln_in_edges = graph.sorted_in_edges(m['ln'], data=True)
        if any(obj is None for obj in obj_dict.values()) or \
                len(ln_in_edges) < 3:
            ERROR('[Parser]: Meets invalid Node in merge_ln_mul_add!')
            continue
        ln_in_shapes = obj_dict['ln'].get_input_shapes()
        if len(ln_in_shapes) < 1 \
                or ln_in_shapes[0] is None:
            continue
        add_out_shapes = obj_dict['add'].get_output_shapes()
        if len(add_out_shapes) < 1 \
                or add_out_shapes[0] is None \
                or ln_in_shapes[0] != add_out_shapes[0]:
            continue
        scale_in_attr = ln_in_edges[1][2]
        if scale_in_attr['tensor'] is None \
                or not scale_in_attr['tensor'].is_const \
                or not FLOAT_EQUAL(scale_in_attr['tensor'].value, 1.0):
            continue
        bias_in_attr = ln_in_edges[2][2]
        if bias_in_attr['tensor'] is None \
                or not bias_in_attr['tensor'].is_const \
                or not FLOAT_EQUAL(bias_in_attr['tensor'].value, 0.0):
            continue
        ln_axes = np.sort(OpHasAxis.make_axes_non_negative(
            obj_dict['ln'].axes, ln_in_shapes[0]))
        new_weights = OpHasAxis.align_axes(
            obj_dict['gamma'].value, ln_axes, ln_in_shapes[0])
        new_biases = OpHasAxis.align_axes(
            obj_dict['beta'].value, ln_axes, ln_in_shapes[0])
        if new_weights is None or new_biases is None:
            continue
        matched = True
        src, _, in_attr = ln_in_edges[0]
        add_in_edges = graph.sorted_in_edges(m['add'])
        graph.remove_edges_from(add_in_edges)
        graph.add_edge(src, m['add'], **in_attr)
        new_ln_attr = obj_dict['add'].copied_attr()
        new_ln_attr.update({'epsilon': obj_dict['ln'].epsilon,
                            'axes': np.array(ln_axes),
                            'opset_version': 17})
        NodeWrap(graph, m['add']).replace_obj('LayerNormalization', new_ln_attr)
        insert_constant(graph, m['add'] + '_scale', new_weights, m['add'], in_port=1)
        insert_constant(graph, m['add'] + '_bias', new_biases, m['add'], in_port=2)
    if matched:
        clear_redundant_nodes(graph)


def merge_sign_abs_relu(graph):
    '''Merge sign(x) * relu(abs(x)) to x.
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('sign', {'op': 'Sign'}),
                                   ('abs', {'op': 'Abs'}),
                                   ('relu', {'op': 'Relu'}),
                                   ('mul', {'op': 'Mul'})],
                               edges=[
                                   ('abs', 'relu'),
                                   ('relu', 'mul'),
                                   ('sign', 'mul')])
    for m in matches:
        abs_in_edges = graph.sorted_in_edges(m['abs'], data=True)
        sign_in_edges = graph.sorted_in_edges(m['sign'], data=True)
        if len(abs_in_edges) < 1 or len(sign_in_edges) < 1:
            ERROR('[Parser]: Meets invalid nodes in merge_sign_abs_relu!')
            continue
        src, _, in_attr = abs_in_edges[0]
        src_out_port = in_attr['src_out_port']
        sign_src, _, sign_in_attr = sign_in_edges[0]
        if src != sign_src or src_out_port != sign_in_attr['src_out_port']:
            continue
        matched = True
        mul_out_edges = graph.sorted_out_edges(m['mul'], data=True)
        graph.remove_edges_from(mul_out_edges)
        for _, dst, out_attr in mul_out_edges:
            out_attr.update({'src_out_port': src_out_port})
            graph.add_edge(src, dst, **out_attr)
        if m['mul'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['mul'])
            graph._attr['output_names'][index] = src
    if matched:
        clear_redundant_nodes(graph)


def merge_isinf(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('abs', {'op': 'Abs'}),
                                   ('cons', {'op': 'Constant', 'unique': False}),
                                   ('eq', {'op': 'Equal'})
                               ],
                               edges=[
                                   ('abs', 'eq'),
                                   ('cons', 'eq'),
                               ])
    for m in matches:
        abs_in_edges = graph.sorted_in_edges(m['abs'], data=True)
        eq_in_edges = graph.sorted_in_edges(m['eq'], data=True)
        if len(abs_in_edges) < 1 or len(eq_in_edges) < 2:
            ERROR('[Parser]: Meets invalid nodes in merge_isinf!')
            continue
        cons_value = NodeWrap(graph, m['cons'])['object'].value
        if np.not_equal(cons_value, np.array(np.inf)):
            continue

        matched = True
        graph.remove_edges_from(abs_in_edges)
        graph.remove_edges_from(eq_in_edges)
        src, _, in_attr = abs_in_edges[0]
        graph.add_edge(src, m['eq'], **in_attr)

        inf_attr = NodeWrap(graph, m['eq'])['object'].copied_attr()
        inf_attr.update({
            'detect_negative': 1,
            'detect_positive': 1,
            'opset_version': 10
        })
        NodeWrap(graph, m['eq']).replace_obj(
            'IsInf', inf_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_multi_matmuls(graph):
    '''
    This pass is used to merge 4 matmuls into a BatchMatMul in mobilebert
    '''
    matched = False
    mm_matches = matched_patterns(graph,
                                  nodes=[
                                      ('split0', {'op': 'Split'}),
                                      ('rs00', {'op': 'Reshape'}),
                                      ('rs01', {'op': 'Reshape'}),
                                      ('rs02', {'op': 'Reshape'}),
                                      ('rs03', {'op': 'Reshape'}),
                                      ('split1', {'op': 'Split'}),
                                      ('rs10', {'op': 'Reshape'}),
                                      ('rs11', {'op': 'Reshape'}),
                                      ('rs12', {'op': 'Reshape'}),
                                      ('rs13', {'op': 'Reshape'}),
                                      ('matmul0', {'op': 'MatMul'}),
                                      ('matmul1', {'op': 'MatMul'}),
                                      ('matmul2', {'op': 'MatMul'}),
                                      ('matmul3', {'op': 'MatMul'}),
                                      ('concat', {'op': 'ConcatFromSequence'}),
                                  ],
                                  edges=[
                                      ('split0', 'rs00', {'src_out_port': 0}),
                                      ('split0', 'rs01', {'src_out_port': 1}),
                                      ('split0', 'rs02', {'src_out_port': 2}),
                                      ('split0', 'rs03', {'src_out_port': 3}),
                                      ('split1', 'rs10', {'src_out_port': 0}),
                                      ('split1', 'rs11', {'src_out_port': 1}),
                                      ('split1', 'rs12', {'src_out_port': 2}),
                                      ('split1', 'rs13', {'src_out_port': 3}),
                                      ('rs00', 'matmul0'),
                                      ('rs10', 'matmul0'),
                                      ('rs01', 'matmul1'),
                                      ('rs11', 'matmul1'),
                                      ('rs02', 'matmul2'),
                                      ('rs12', 'matmul2'),
                                      ('rs03', 'matmul3'),
                                      ('rs13', 'matmul3'),
                                      ('matmul0', 'concat', {'dst_in_port': 0}),
                                      ('matmul1', 'concat', {'dst_in_port': 1}),
                                      ('matmul2', 'concat', {'dst_in_port': 2}),
                                      ('matmul3', 'concat', {'dst_in_port': 3}),
                                  ])
    for m in mm_matches:
        objs_dict = {m[name]: NodeWrap(graph, m[name])['object'] for name in [
            'split0', 'split1', 'matmul0', 'matmul1', 'matmul2', 'matmul3', 'concat']}
        if all([graph.has_node(n) for n in objs_dict.keys()]) and all([v is not None for v in objs_dict.values()]):
            sp0_obj = objs_dict[m['split0']]
            sp1_obj = objs_dict[m['split1']]
            concat_obj = objs_dict[m['concat']]
            if sp0_obj.axis == sp1_obj.axis == concat_obj.axis and sp0_obj.axis == 0:
                sp0_in_edges = graph.sorted_in_edges(m['split0'], data=True)
                sp1_in_edges = graph.sorted_in_edges(m['split1'], data=True)

                mm0_in_edges = graph.sorted_in_edges(m['matmul0'])

                is_split0_inp0 = mm0_in_edges[0][0] == m['rs00']

                for mm in ['matmul0', 'matmul1', 'matmul2', 'matmul3']:
                    mm_out_edges = graph.sorted_out_edges(m[mm])
                    if len(mm_out_edges) != 1:
                        return

                concat_in_edges = graph.sorted_in_edges(m['concat'])
                if len(concat_in_edges) != 4:
                    continue

                matched = True
                graph.remove_edges_from(concat_in_edges)
                graph.remove_edges_from(sp0_in_edges)
                graph.remove_edges_from(sp1_in_edges)

                if is_split0_inp0:
                    src_0, _, in0_attr = sp0_in_edges[0]
                    src_1, _, in1_attr = sp1_in_edges[0]
                else:
                    src_0, _, in0_attr = sp1_in_edges[0]
                    src_1, _, in1_attr = sp0_in_edges[0]

                in0_attr['dst_in_port'] = 0
                in1_attr['dst_in_port'] = 1

                graph.add_edge(src_0, m['concat'], **in0_attr)
                graph.add_edge(src_1, m['concat'], **in1_attr)

                mm_attr = objs_dict[m['concat']].copied_attr()
                mm_attr.update({'opset_version': 13})
                NodeWrap(graph, m['concat']).replace_obj('MatMul', mm_attr)
        else:
            ERROR('[Parser]: Meets invalid nodes in merge_multi_matmuls!')
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
            if pow_y_value is not None and eps_value is not None and FLOAT_EQUAL(pow_y_value, 2) and len(axes) == 1 and \
                    axes[0] == 2:
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
            ERROR('[Parser]: Meets invalid nodes in merge_mvn!')
    if matched:
        clear_redundant_nodes(graph)


def merge_mvn2(graph):
    matched = False
    ln_matches = matched_patterns(graph,
                                  nodes=[
                                      ('inp', {}),
                                      ('mean_1', {'op': 'ReduceMean'}),
                                      ('sub', {'op': 'Sub'}),
                                      ('pow', {'op': 'Pow'}),
                                      ('pow_y', {'op': 'Constant'}),
                                      ('mean_2', {'op': 'ReduceMean'}),
                                      ('add_1', {'op': 'Add'}),
                                      ('epsilon', {'op': 'Constant'}),
                                      ('sqrt', {'op': 'Sqrt'}),
                                      ('div', {'op': 'Div'}),
                                  ],
                                  edges=[
                                      ('inp', 'mean_1'),
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
                                      ('sqrt', 'div', {
                                          'src_out_port': 0, 'dst_in_port': 1}),
                                      ('sub', 'div', {
                                          'src_out_port': 0, 'dst_in_port': 0}),
                                  ])
    for m in ln_matches:
        objs_dict = {m[name]: NodeWrap(graph, m[name])['object'] for name in [
            'inp', 'mean_1', 'mean_2', 'sub', 'pow_y', 'epsilon', 'div']}
        if all([graph.has_node(n) for n in objs_dict.keys()]) and all([v is not None for v in objs_dict.values()]):
            pow_y_value = objs_dict[m['pow_y']].value
            eps_value = objs_dict[m['epsilon']].value
            axes = objs_dict[m['mean_1']].axes
            if pow_y_value is not None and eps_value is not None and FLOAT_EQUAL(pow_y_value, 2) and \
                    objs_dict[m['mean_1']].keepdims and objs_dict[m['mean_2']].keepdims and \
                    objs_dict[m['mean_1']].axes == objs_dict[m['mean_2']].axes:
                matched = True
                eps_value = float(eps_value)
                inp, mean_1, sub, div = [m[name] for name in [
                    'inp', 'mean_1', 'sub', 'div']]
                mean_1_in_edges = graph.sorted_in_edges(mean_1, data=True)
                div_in_edges = graph.sorted_in_edges(div)
                _, _, in_attr = mean_1_in_edges[0]
                graph.remove_edge(inp, mean_1)
                graph.remove_edge(inp, sub)
                graph.remove_edges_from(div_in_edges)
                graph.add_edge(inp, div, **in_attr)

                ln_attr = objs_dict[div].copied_attr()
                ln_attr.update({'axes': axes,
                                'epsilon': eps_value,
                                'opset_version': 13
                                })
                NodeWrap(graph, div).replace_obj(
                    'MeanVarianceNormalization', ln_attr)
        else:
            ERROR('[Parser]: Meets invalid nodes in merge_mvn2!')
    if matched:
        clear_redundant_nodes(graph)


def merge_mvn3(graph):
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
                    not all([FLOAT_EQUAL(val, objs_dict[epsilon].value.item(0)) for val in
                             objs_dict[epsilon].value.flatten()[1:]]):
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
            ERROR('[Parser]: Meets invalid nodes in merge_mvn3!')
    if matched:
        clear_redundant_nodes(graph)


def merge_mvn4(graph):
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
            ERROR('[Parser]: Meets invalid Node in merge_mvn4!')
            continue
        if any([len(graph.sorted_out_edges(m[name])) != 1 for name in names_check_out_edge_1]):
            continue
        if any([len(graph.sorted_out_edges(m[name])) != 2 for name in names_check_out_edge_2]):
            continue
        if any([len(graph.sorted_out_edges(m[name])) != 3 for name in names_check_out_edge_3]):
            continue
        if len(graph.sorted_in_edges(m['trans_1'])) < 1:
            ERROR(
                '[Parser]: Got invalid input edges of Transpose(%s) in merge_mvn4!' % m['trans_1'])
            continue
        if obj_dict['trans_1'].perm != [0, 2, 3, 1] \
                or obj_dict['trans_2'].perm != [0, 3, 1, 2]:
            continue
        if obj_dict['mean_1'].axes != [0, 1, 2] or obj_dict['mean_2'].axes != [0, 1, 2]:
            continue
        if len(obj_dict['mean_1'].get_input_shapes()) < 1 \
                or obj_dict['mean_1'].get_input_shapes()[0] is None \
                or len(obj_dict['mean_1'].get_input_shapes()[0]) != 4:
            ERROR(
                '[Parser]: Got invalid input shape of Mean(%s) in merge_mvn4!' % m['mean_1'])
            continue
        if len(obj_dict['add_1'].sorted_in_consts()) != 1 \
                or obj_dict['add_1'].sorted_in_consts()[0][2] is None \
                or obj_dict['add_1'].sorted_in_consts()[0][2].size != 1:
            continue
        if len(obj_dict['sub_2'].sorted_in_consts()) != 1 \
                or obj_dict['sub_2'].sorted_in_consts()[0][2] is None \
                or not FLOAT_EQUAL(obj_dict['sub_2'].sorted_in_consts()[0][2], 0) \
                or not FLOAT_EQUAL(obj_dict['pow'].get_input_tensors()[1], 2):
            continue
        matched = True
        trans1_in_edges = graph.sorted_in_edges(m['trans_1'], data=True)
        trans2_in_edges = graph.sorted_in_edges(m['trans_2'])
        src, _, in_attr = trans1_in_edges[0]
        graph.remove_edges_from(trans1_in_edges + trans2_in_edges)
        graph.add_edge(src, m['trans_2'], **in_attr)
        mean1_in_shape = obj_dict['mean_1'].get_input_shapes()[0]
        axes = [x for (i, x) in enumerate(
            obj_dict['mean_1'].axes) if mean1_in_shape[i] != 1]
        if not axes:
            axes = obj_dict['mean_1'].axes[:]
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
                                       'op': ['LayerNormalization', 'MeanVarianceNormalization']}),
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
        norm_in_edges = graph.sorted_in_edges(m['norm'], data=True)
        if any([obj is None for obj in obj_dict.values()]) \
                or len(graph.sorted_in_edges(m['reshape_1'])) < 1 \
                or len(norm_in_edges) < 1:
            ERROR('[Parser]: Meets invalid Node in merge_gn!')
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
                not np.array_equal(origin_shape[:channels_axis] + origin_shape[channels_axis + 1:],
                                   expand_shape[:channels_axis] + expand_shape[channels_axis + 2:]):
            continue

        weights_shape = [origin_shape[channels_axis]]
        if obj_dict['norm'].type == 'MeanVarianceNormalization':
            weights = np.ones(weights_shape, dtype=np.float32)
            biases = np.zeros(weights_shape, dtype=np.float32)
        else:
            if len(norm_in_edges) < 3:
                ERROR('[Parser]: Meets invalid LayerNormalization Node(%s) in merge_gn!' % m['norm'])
                continue
            scale_in_attr = norm_in_edges[1][2]
            if scale_in_attr['tensor'] is None or not scale_in_attr['tensor'].is_const:
                continue
            weights = scale_in_attr['tensor'].value
            bias_in_attr = norm_in_edges[2][2]
            if bias_in_attr['tensor'] is None or not bias_in_attr['tensor'].is_const:
                continue
            biases = bias_in_attr['tensor'].value
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
                ERROR(
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
                ERROR(
                    '[Parser]: Meets invalid biases of Node(%s) in merge_gn!' % m['norm'])
                continue

        input_shapes = obj_dict['reshape_1'].get_input_shapes()
        group = expand_shape[channels_axis]
        if input_shapes[0] and input_shapes[0][channels_axis] % group != 0:
            continue

        matched = True
        src, _, in_attr = graph.sorted_in_edges(m['reshape_1'], data=True)[0]
        graph.remove_edges_from(norm_in_edges)
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


def decompose_const_loop(graph, params):
    matched = False
    matches = single_node_matcher(graph, 'Loop')
    for m in matches:
        loop = m['target']
        loop_obj = NodeWrap(graph, loop)['object']
        in_edges = graph.sorted_in_edges(loop, data=True)
        if loop_obj is not None and len(in_edges) >= 2 + len(loop_obj.body._attr['root_in_ports']):
            if len(in_edges) != (2 + len(loop_obj.body._attr['root_in_ports'])) \
                    or not in_edges[0][2]['tensor'].is_const \
                    or in_edges[0][2]['tensor'].value is None \
                    or not in_edges[1][2]['tensor'].is_const \
                    or in_edges[1][2]['tensor'].value is None \
                    or not in_edges[1][2]['tensor'].value:
                continue

            subgraph_main_out = loop_obj.body._attr['output_names'][-1]
            subgraph_main_outport = loop_obj.body._attr['output_ports'][subgraph_main_out]
            subgraph_main_nodes = determined_sort(
                loop_obj.body, [subgraph_main_out])
            subgraph_main_nodes_objs = {n: NodeWrap(
                graph, n)['object'] for n in subgraph_main_nodes}
            if subgraph_main_out not in subgraph_main_nodes \
                    or any(obj is None for obj in subgraph_main_nodes_objs.values()):
                WARN('[Parser]: Meets invalid Subgraph Nodes in decompose_const_loop!')
                continue

            if len(subgraph_main_nodes_objs[subgraph_main_out].get_output_tensors()) < 1:
                continue

            matched = True
            main_out_tensor = subgraph_main_nodes_objs[subgraph_main_out].get_output_tensors()[
                0]
            main_out_shape = subgraph_main_nodes_objs[subgraph_main_out].get_output_shapes()[0]
            count = int(in_edges[0][2]['tensor'].value)
            out_edges = graph.sorted_out_edges(loop, data=True)
            stack = get_valid_node_name(graph, loop + '_stack')

            for n in loop_obj.body._filter_node:
                try:
                    NodeWrap(graph, n)['object'].in_subgraph = False
                except:
                    pass

            graph.remove_edges_from(in_edges)
            for i in range(count):
                if i == 0:
                    for n in subgraph_main_nodes:
                        n_in_edges = graph.sorted_in_edges(n, data=True)
                        for sub_src, _, in_attr in n_in_edges:
                            if graph.nodes[sub_src]['op'] in ['DummyInput', 'Constant'] \
                                and (in_attr['tensor'].name, in_attr['src_out_port']) in loop_obj.body._attr[
                                    'input_tensors']:
                                cur_count_value = np.array(
                                    i, np.dtype(in_attr['tensor'].dtype))
                                in_attr['tensor'].value = cur_count_value
                                if graph.nodes[sub_src]['op'] == 'DummyInput':
                                    NodeWrap(graph, sub_src).replace_obj('Constant', {
                                        'name': sub_src, 'opset_version': 9, 'value': cur_count_value})
                    graph.add_edge(subgraph_main_out,
                                   stack,
                                   **{'src_out_port': subgraph_main_outport,
                                      'dst_in_port': i,
                                      'tensor': Tensor(value=main_out_tensor, shape=main_out_shape)})

                else:
                    for n in subgraph_main_nodes:
                        name_suffix = '_loop_%s' % i
                        new_n = get_valid_node_name(graph, n + name_suffix)
                        n_obj = subgraph_main_nodes_objs[n]
                        n_in_edges = graph.sorted_in_edges(n, data=True)
                        for src, _, in_attr in n_in_edges:
                            if graph.nodes[src]['op'] in ['DummyInput', 'Constant'] \
                                and (in_attr['tensor'].name, in_attr['src_out_port']) in loop_obj.body._attr[
                                    'input_tensors']:
                                new_const = get_valid_node_name(
                                    graph, src + name_suffix)
                                cur_count_value = np.array(
                                    i, np.dtype(in_attr['tensor'].dtype))
                                new_in_attr = copy.deepcopy(in_attr)
                                new_in_attr['tensor'].value = cur_count_value
                                new_in_attr['tensor'].name = new_const
                                graph.add_edge(new_const, new_n, **new_in_attr)
                                NodeWrap(graph, new_const).replace_obj('Constant', {
                                    'name': new_const, 'opset_version': 9, 'value': cur_count_value})
                            elif src not in subgraph_main_nodes and not src.endswith(name_suffix):
                                new_in_attr = copy.deepcopy(in_attr)
                                graph.add_edge(src, new_n, **new_in_attr)
                            elif src in subgraph_main_nodes:
                                new_in_attr = copy.deepcopy(in_attr)
                                graph.add_edge(src + name_suffix,
                                               new_n, **new_in_attr)
                            else:
                                WARN(
                                    '[Parser]: Invalid in edges for Node(%s)!' % new_n)
                        cur_obj_attr = n_obj.copied_attr()
                        cur_obj_attr.update({'name': new_n})
                        NodeWrap(graph, new_n).replace_obj(
                            n_obj.type, cur_obj_attr)
                        if n == subgraph_main_out:
                            graph.add_edge(new_n,
                                           stack,
                                           **{'src_out_port': subgraph_main_outport,
                                              'dst_in_port': i,
                                              'tensor': Tensor(value=main_out_tensor, shape=main_out_shape)
                                              })

            for _, dst, out_attr in out_edges:
                graph.remove_edge(loop, dst)
                graph.add_edge(stack, dst, **out_attr)

            NodeWrap(graph, stack).replace_obj('ConcatFromSequence', {
                'name': stack, 'opset_version': 11, 'axis': 0, 'new_axis': 1})

        else:
            ERROR(
                '[Parser]: Meets invalid Loop Op (%s) in decompose_const_loop!' % loop)

    if matched:
        if graph._attr.get('subgraph_output_names', None) is not None:
            graph._attr['output_names'] = list(set(graph._attr['output_names']).difference(
                list(graph._attr['subgraph_output_names'])))
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
    gn_matches2 = matched_patterns(graph,
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
                                       ('tile_1', {'op': 'Tile'}),
                                       ('tile_1_reps', {'op': 'Constant'}),
                                       ('mul_1', {'op': 'Mul'}),
                                       ('tile_2', {'op': 'Tile'}),
                                       ('tile_2_reps', {'op': 'Constant'}),
                                       ('mul_2', {'op': 'Mul'}),
                                       ('sub_2', {'op': 'Sub'}),
                                       ('beta', {'op': 'Constant'}),
                                       ('tile_3', {'op': 'Tile'}),
                                       ('tile_3_reps', {'op': 'Constant'}),
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
                                       ('mul_gamma', 'tile_1'),
                                       ('tile_1_reps', 'tile_1', {'dst_in_port': 1}),
                                       ('tile_1', 'mul_1'),
                                       ('mean_1', 'mul_2'),
                                       ('mul_gamma', 'tile_2'),
                                       ('tile_2_reps', 'tile_2', {'dst_in_port': 1}),
                                       ('tile_2', 'mul_2'),
                                       ('beta', 'sub_2'),
                                       ('mul_2', 'sub_2', {'dst_in_port': 1}),
                                       ('mul_1', 'add_2'),
                                       ('sub_2', 'tile_3'),
                                       ('tile_3_reps', 'tile_3', {'dst_in_port': 1}),
                                       ('tile_3', 'add_2'),
                                       ('add_2', 'reshape_2'),
                                   ])
    for m in gn_matches + gn_matches2:
        key_names = ['reshape_1', 'mean_1', 'mean_2',
                     'pow_y', 'epsilon', 'gamma', 'beta', 'reshape_2']
        reshape_1, mean_1, mean_2, pow_y, epsilon, gamma, beta, reshape_2 = [
            m[name] for name in key_names]
        objs_dict = {m[name]: NodeWrap(graph, m[name])[
            'object'] for name in key_names}
        if any(obj is None for obj in objs_dict.values()):
            ERROR('[Parser]: Meets invalid nodes in merge_gn2!')
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
        exp_channel_axes = [idx for idx, shape in enumerate(input_shapes[0]) if shape != expanded_shape[idx]]
        exp_channel_axis = exp_channel_axes[0] if len(exp_channel_axes) > 0 else (len(input_shapes[0]) - 1)
        if channels_axis != exp_channel_axis:
            continue
        axes_after_reshape = [channels_axis, channels_axis + 1]
        biases = OpHasAxis.align_axes(
            objs_dict[beta].value, axes_after_reshape, expanded_shape)
        weights = OpHasAxis.align_axes(
            objs_dict[gamma].value, axes_after_reshape, expanded_shape)
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


def merge_gn3(graph):
    '''Merge the following pattern to GroupNorm:
    Reshape([N,C,H,W]->[N,G,...])->InstanceNorm->Reshape([N,G,...]->[N,C,H,W])->Mul(*[C,1,1])->Add(+[C,1,1])
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('reshape1', {'op': 'Reshape'}),
                                   ('norm', {'op': 'InstanceNormalization'}),
                                   ('reshape2', {'op': 'Reshape'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('add', {'op': 'Add'}),
                                   ('reshape1_dim', {'op': 'Constant'}),
                                   ('reshape2_dim', {'op': 'Constant'}),
                                   ('weights', {'op': 'Constant'}),
                                   ('biases', {'op': 'Constant'}),
                               ],
                               edges=[
                                   ('reshape1', 'norm'),
                                   ('reshape1_dim', 'reshape1', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                   ('norm', 'reshape2'),
                                   ('reshape2_dim', 'reshape2', {
                                       'src_out_port': 0, 'dst_in_port': 1}),
                                   ('reshape2', 'mul'),
                                   ('weights', 'mul'),
                                   ('mul', 'add'),
                                   ('biases', 'add'),
                               ]
                               )
    for m in matches:
        key_names = ['reshape1', 'norm', 'reshape2', 'mul', 'add',
                     'reshape1_dim', 'reshape2_dim', 'weights', 'biases']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if any(obj is None for obj in node_objs.values()):
            ERROR('[Parser]: Meets invalid nodes in merge_gn3!')
            continue
        if node_objs['norm'].data_format != 'NCHW' \
                or not FLOAT_EQUAL(node_objs['norm'].weights, 1.0) \
                or not FLOAT_EQUAL(node_objs['norm'].biases, 0.0):
            continue
        reshape1_in_edges = graph.sorted_in_edges(m['reshape1'], data=True)
        if len(reshape1_in_edges) < 1:
            continue
        input_shapes = node_objs['reshape1'].get_input_shapes()
        if len(input_shapes) < 1 \
                or input_shapes[0] is None \
                or None in input_shapes[0] \
                or len(input_shapes[0]) < 3:
            continue
        input_ndim = len(input_shapes[0])
        reshape1_dim = node_objs['reshape1_dim'].value
        reshape2_dim = node_objs['reshape2_dim'].value
        if reshape1_dim is None or reshape2_dim is None \
                or len(reshape1_dim) < 3 or reshape1_dim[0] not in (0, input_shapes[0][0]) \
                or reshape1_dim[1] == 0 \
                or reshape2_dim.tolist() != input_shapes[0]:
            continue
        channels = input_shapes[0][1]
        groups = reshape1_dim[1]
        weights = node_objs['weights'].value
        biases = node_objs['biases'].value
        if weights is None or biases is None \
                or weights.size not in (1, channels) or biases.size not in (1, channels) \
                or len(weights.shape) > input_ndim or len(biases.shape) > input_ndim:
            continue
        exp_weight_shape1 = [channels] + [1] * (input_ndim - 2)
        exp_weight_shape2 = [1] + exp_weight_shape1
        if weights.size == 1:
            weights = np.tile(np.reshape(weights, [1]), [channels])
        elif list(weights.shape) == exp_weight_shape1 or list(weights.shape) == exp_weight_shape2:
            weights = np.reshape(weights, [-1])
        else:
            continue
        if biases.size == 1:
            biases = np.tile(np.reshape(biases, [1]), [channels])
        elif list(biases.shape) == exp_weight_shape1 or list(biases.shape) == exp_weight_shape2:
            biases = np.reshape(biases, [-1])
        else:
            continue

        matched = True
        src, _, in_attr = reshape1_in_edges[0]
        add_in_edges = graph.sorted_in_edges(m['add'])
        graph.remove_edges_from(add_in_edges)
        graph.add_edge(src, m['add'], **in_attr)

        gn_attr = node_objs['add'].copied_attr()
        gn_attr.update({'group': groups, 'epsilon': node_objs['norm'].epsilon,
                        'weights': weights, 'biases': biases, 'axis': 1})
        NodeWrap(graph, m['add']).replace_obj('ArmGroupNorm', gn_attr)
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
            ERROR('[Parser]: Meets invalid Node in merge_in!')
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
            ERROR('[Parser]: Meets invalid input/output shape in merge_in!')
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
                                       'op': ['InstanceNormalization', 'MeanVarianceNormalization',
                                              'LayerNormalization']}),
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
            norm_in_edges = graph.sorted_in_edges(m['norm'], data=True)
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
                    if node_objs['norm'].type == 'LayerNormalization':
                        if len(norm_in_edges) < 3:
                            ERROR('[Parser]: Meets invalid LayerNormalization Node(%s) in merge_norm!' % m['norm'])
                            continue
                        scale_in_attr = norm_in_edges[1][2]
                        if scale_in_attr['tensor'] is None \
                                or not scale_in_attr['tensor'].is_const \
                                or scale_in_attr['tensor'].value.shape != weight.shape:
                            continue
                        norm_weights = scale_in_attr['tensor'].value
                        bias_in_attr = norm_in_edges[2][2]
                        if bias_in_attr['tensor'] is None \
                                or not bias_in_attr['tensor'].is_const \
                                or bias_in_attr['tensor'].value.shape != bias.shape:
                            continue
                        norm_biases = bias_in_attr['tensor'].value
                        bias = bias + norm_biases * weight
                        weight = weight * norm_weights
                matched = True

                graph.remove_edges_from(mul_in_edges)
                graph.remove_edges_from(add_out_edges)
                for _, dst, out_attr in add_out_edges:
                    graph.add_edge(m['reshape'], dst, **out_attr)
                graph.remove_edges_from(norm_in_edges[1:])

                ins_attr.update({'opset_version': 17})

                NodeWrap(graph, m['norm']).replace_obj(
                    'LayerNormalization', ins_attr)
                insert_constant(graph, m['norm'] + '_scale', weight, m['norm'], in_port=1)
                insert_constant(graph, m['norm'] + '_bias', bias, m['norm'], in_port=2)

                if m['add'] in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(m['add'])
                    graph._attr['output_names'][index] = m['reshape']
        else:
            ERROR('[Parser]: Meets invalid nodes in merge_norm!')
    if matched:
        clear_redundant_nodes(graph)


def merge_special_div_mul(graph):
    '''Convert Div(A=1, B=x1, C=div_out)+Mul(A=div_out, B=x2) to Div(A=x2, B=x1) because 1/x1*x2=x2/x1.'''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('div', {'op': 'Div'}),
                                   ('div_const', {'op': 'Constant'}),
                                   ('mul_inp', {}),
                                   ('mul', {'op': 'Mul'}),
                               ],
                               edges=[
                                   ('div_const', 'div', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('div', 'mul'),
                                   ('mul_inp', 'mul'),
                               ])
    for m in matches:
        key_names = ['div_const', 'mul']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        div_in_edges = graph.sorted_in_edges(m['div'], data=True)
        mul_in_edges = graph.sorted_in_edges(m['mul'], data=True)
        if any(obj is None for obj in node_objs.values()) \
                or len(div_in_edges) != 2 \
                or len(mul_in_edges) != 2:
            ERROR('[Parser]: Meets invalid nodes in merge_special_div_mul!')
            continue
        if not FLOAT_EQUAL(node_objs['div_const'].value, 1):
            continue
        divisor_in_attr = [in_attr for src, _, in_attr in mul_in_edges if src == m['mul_inp']]
        if len(divisor_in_attr) != 1:
            continue
        divisor_in_attr = divisor_in_attr[0]
        matched = True
        graph.remove_edges_from(mul_in_edges)
        divisor_in_attr.update({'dst_in_port': 0})
        graph.add_edge(m['mul_inp'], m['mul'], **divisor_in_attr)
        dividend, _, dividend_in_attr = div_in_edges[1]
        graph.add_edge(dividend, m['mul'], **dividend_in_attr)
        div_attr = node_objs['mul'].copied_attr()
        div_attr.update({'opset_version': 14})
        NodeWrap(graph, m['mul']).replace_obj('Div', div_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_rms_norm(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('pow', {'op': 'Pow'}),
                                   ('pow_y', {'op': 'Constant'}),
                                   ('mean', {'op': 'ReduceMean'}),
                                   ('add', {'op': 'Add'}),
                                   ('add_const', {'op': 'Constant'}),
                                   ('sqrt', {'op': 'Sqrt'}),
                                   ('div', {'op': 'Div'}),
                                   ('mul', {'op': 'Mul'}),
                                   ('mul_const', {'op': 'Constant'}),
                               ],
                               edges=[
                                   ('pow_y', 'pow', {'dst_in_port': 1}),
                                   ('pow', 'mean'),
                                   ('mean', 'add'),
                                   ('add_const', 'add'),
                                   ('add', 'sqrt'),
                                   ('sqrt', 'div', {'dst_in_port': 1}),
                                   ('div', 'mul'),
                                   ('mul_const', 'mul'),
                               ])
    matches2 = matched_patterns(graph,
                                nodes=[
                                    ('pow', {'op': 'Pow'}),
                                    ('pow_y', {'op': 'Constant'}),
                                    ('mean', {'op': 'ReduceMean'}),
                                    ('add', {'op': 'Add'}),
                                    ('add_const', {'op': 'Constant'}),
                                    ('sqrt', {'op': 'Sqrt'}),
                                    ('div', {'op': 'Div'}),
                                    ('cast1', {'op': 'Cast'}),
                                    ('cast2', {'op': 'Cast'}),
                                    ('mul', {'op': 'Mul'}),
                                    ('mul_const', {'op': 'Constant'}),
                                ],
                                edges=[
                                    ('pow_y', 'pow', {'dst_in_port': 1}),
                                    ('pow', 'mean'),
                                    ('mean', 'add'),
                                    ('add_const', 'add'),
                                    ('add', 'sqrt'),
                                    ('sqrt', 'div', {'dst_in_port': 1}),
                                    ('div', 'cast1'),
                                    ('cast1', 'cast2'),
                                    ('cast2', 'mul'),
                                    ('mul_const', 'mul'),
                                ])
    matches3 = matched_patterns(graph,
                                nodes=[
                                    ('pow', {'op': 'Pow'}),
                                    ('pow_y', {'op': 'Constant'}),
                                    ('mean', {'op': 'ReduceMean'}),
                                    ('add', {'op': 'Add'}),
                                    ('add_const', {'op': 'Constant'}),
                                    ('sqrt', {'op': 'Sqrt'}),
                                    ('div', {'op': 'Div'}),
                                    ('cast1', {'op': 'Cast'}),
                                    ('mul', {'op': 'Mul'}),
                                    ('mul_const', {'op': 'Constant'}),
                                ],
                                edges=[
                                    ('pow_y', 'pow', {'dst_in_port': 1}),
                                    ('pow', 'mean'),
                                    ('mean', 'add'),
                                    ('add_const', 'add'),
                                    ('add', 'sqrt'),
                                    ('sqrt', 'div', {'dst_in_port': 1}),
                                    ('div', 'cast1'),
                                    ('cast1', 'mul'),
                                    ('mul_const', 'mul'),
                                ])
    for m in matches + matches2 + matches3:
        cast_names = [] if 'cast1' not in m else (['cast1', 'cast2'] if 'cast2' in m else ['cast1'])
        key_names = ['pow', 'pow_y', 'mean', 'add_const',
                     'div', 'mul', 'mul_const'] + cast_names
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if any(obj is None for obj in node_objs.values()):
            ERROR('[Parser]: Meets invalid nodes in merge_rms_norm!')
            continue
        pow_in_edges = graph.sorted_in_edges(m['pow'], data=True)
        div_in_edges = graph.sorted_in_edges(m['div'], data=True)
        if len(pow_in_edges) != 2 or len(div_in_edges) != 2:
            ERROR('[Parser]: Meets invalid inputs of Pow(%s) or Div(%s) node in merge_rms_norm!' % (m['pow'], m['div']))
            continue
        if pow_in_edges[0][0] != div_in_edges[0][0] \
                or pow_in_edges[0][2]['dst_in_port'] != 0 \
                or div_in_edges[0][2]['dst_in_port'] != 0 \
                or pow_in_edges[0][2]['src_out_port'] != div_in_edges[0][2]['src_out_port']:
            continue
        if node_objs['pow_y'].value != 2 or not node_objs['mean'].keepdims:
            continue
        input_shapes = node_objs['pow'].get_input_shapes()
        if len(input_shapes) < 1 or input_shapes[0] is None or None in input_shapes[0]:
            ERROR('[Parser]: Meets invalid input shape of Pow(%s) node in merge_rms_norm!' % m['pow'])
            continue
        input_shape = input_shapes[0]
        input_dtypes = node_objs['pow'].get_input_dtypes()
        if len(input_dtypes) < 1 or input_dtypes[0] is None:
            ERROR('[Parser]: Meets invalid input dtype of Pow(%s) node in merge_rms_norm!' % m['pow'])
            continue
        input_dtype = input_dtypes[0]
        norm_axes = OpHasAxis.make_axes_non_negative(node_objs['mean'].axes, len(input_shape))
        weights = OpHasAxis.align_axes(node_objs['mul_const'].value, norm_axes, input_shape)
        if weights is None:
            continue
        epsilon = node_objs['add_const'].value
        if np.array(epsilon).size != 1:
            continue
        matched = True
        epsilon = np.array(epsilon).item()
        src, _, src_in_attr = pow_in_edges[0]
        mul_in_edges = graph.sorted_in_edges(m['mul'])
        graph.remove_edges_from(mul_in_edges)
        graph.add_edge(src, m['mul'], **src_in_attr)
        rms_norm_attr = node_objs['mul'].copied_attr()
        rms_norm_attr.update({'axes': norm_axes, 'weights': weights.astype(input_dtype), 'epsilon': epsilon})
        NodeWrap(graph, m['mul']).replace_obj('ArmRMSNorm', rms_norm_attr)
        out_node = m['mul']
        from_dtype = input_dtype
        for cast in cast_names:
            cast_to_dtype = node_objs[cast].to
            out_node = insert_cast_after(graph, out_node, from_dtype, cast_to_dtype)
            from_dtype = cast_to_dtype
        if out_node != m['mul'] and m['mul'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['mul'])
            graph._attr['output_names'][index] = out_node
    if matched:
        clear_redundant_nodes(graph)


def merge_rms_norm2(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('input', {}),
                                   ('pow', {'op': 'Pow'}),
                                   ('pow_y', {'op': 'Constant'}),
                                   ('mean', {'op': 'ReduceMean'}),
                                   ('add', {'op': 'Add'}),
                                   ('add_const', {'op': 'Constant'}),
                                   ('sqrt', {'op': 'Sqrt'}),
                                   ('div', {'op': 'Div'}),
                               ],
                               edges=[
                                   ('input', 'pow', {'dst_in_port': 0}),
                                   ('pow_y', 'pow', {'dst_in_port': 1}),
                                   ('pow', 'mean'),
                                   ('mean', 'add'),
                                   ('add_const', 'add'),
                                   ('add', 'sqrt'),
                                   ('sqrt', 'div', {'dst_in_port': 1}),
                                   ('input', 'div', {'dst_in_port': 0}),
                               ])
    for m in matches:
        key_names = ['pow', 'pow_y', 'mean', 'add_const',
                     'div']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if any(obj is None for obj in node_objs.values()):
            ERROR('[Parser]: Meets invalid nodes in merge_rms_norm2!')
            continue
        pow_in_edges = graph.sorted_in_edges(m['pow'], data=True)
        div_in_edges = graph.sorted_in_edges(m['div'], data=True)
        if len(pow_in_edges) != 2 or len(div_in_edges) != 2:
            ERROR(
                '[Parser]: Meets invalid inputs of Pow(%s) or Div(%s) node in merge_rms_norm2!' % (m['pow'], m['div']))
            continue
        if node_objs['pow_y'].value != 2 or not node_objs['mean'].keepdims:
            continue
        input_shapes = node_objs['pow'].get_input_shapes()
        if len(input_shapes) < 1 or input_shapes[0] is None or None in input_shapes[0]:
            ERROR('[Parser]: Meets invalid input shape of Pow(%s) node in merge_rms_norm2!' % m['pow'])
            continue
        input_shape = input_shapes[0]
        input_dtypes = node_objs['pow'].get_input_dtypes()
        if len(input_dtypes) < 1 or input_dtypes[0] is None:
            ERROR('[Parser]: Meets invalid input dtype of Pow(%s) node in merge_rms_norm2!' % m['pow'])
            continue
        input_dtype = input_dtypes[0]
        norm_axes = OpHasAxis.make_axes_non_negative(node_objs['mean'].axes, len(input_shape))
        weights = OpHasAxis.align_axes(np.ones([1], dtype=input_dtype), norm_axes, input_shape)
        if weights is None:
            continue
        epsilon = node_objs['add_const'].value
        if np.array(epsilon).size != 1:
            continue
        matched = True
        epsilon = np.array(epsilon).item()
        src, _, src_in_attr = pow_in_edges[0]
        graph.remove_edges_from(div_in_edges)
        graph.add_edge(src, m['div'], **src_in_attr)
        rms_norm_attr = node_objs['div'].copied_attr()
        rms_norm_attr.update({'axes': norm_axes, 'weights': weights.astype(input_dtype), 'epsilon': epsilon})
        NodeWrap(graph, m['div']).replace_obj('ArmRMSNorm', rms_norm_attr)
        out_node = m['div']
        if m['div'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['div'])
            graph._attr['output_names'][index] = out_node
    if matched:
        clear_redundant_nodes(graph)


def broadcast_ln_weights_biases(graph):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('ln', {'op': 'LayerNormalization'}),
                                   ('weights', {'op': 'Constant'}),
                                   ('biases', {'op': 'Constant'}),
                               ],
                               edges=[
                                   ('weights', 'ln', {'dst_in_port': 1}),
                                   ('biases', 'ln', {'dst_in_port': 2}),
                               ])
    for m in matches:
        ln, weights, biases = m['ln'], m['weights'], m['biases']
        ln_obj, weights_obj, biases_obj = [NodeWrap(graph, node)['object'] for node in [ln, weights, biases]]
        if ln_obj is not None and weights_obj is not None and biases_obj is not None:
            input_shapes = ln_obj.get_output_shapes()
            in_edges = graph.sorted_in_edges(ln, data=True)
            if len(input_shapes) >= 1 \
                    and input_shapes[0] \
                    and len(input_shapes[0]) >= 2 \
                    and len(in_edges) == 3:
                input_rank = len(input_shapes[0])
                weights_value = weights_obj.value
                biases_value = biases_obj.value
                if list(weights_value.shape) != list(biases_value.shape):
                    axes = OpHasAxis.make_axes_non_negative(
                        ln_obj.axes, len(input_shapes[0]))
                    axes = sorted(axes)
                    if list(weights_value.shape) != [input_shapes[0][d] for d in axes]:
                        w_rank = len(weights_value.shape)
                        reshape_dim = [
                            1] * (len(axes) - w_rank) + list(weights_value.shape)
                        weights_obj.value = np.reshape(
                            weights_value, reshape_dim)
                    if list(biases_value.shape) != [input_shapes[0][d] for d in axes]:
                        b_rank = len(biases_value.shape)
                        reshape_dim = [
                            1] * (len(axes) - b_rank) + list(biases_value.shape)
                        biases_obj.value = np.reshape(biases_value, reshape_dim)
                    if list(weights_value.shape) != list(biases_value.shape):
                        max_reps = np.maximum(
                            np.array(weights_value.shape), np.array(biases_value.shape))
                        weights_reps = max_reps // np.array(
                            weights_value.shape)
                        if np.any(weights_reps > 1):
                            weights_obj.value = np.tile(
                                weights_value, weights_reps.tolist())
                        biases_reps = max_reps // np.array(biases_value.shape)
                        if np.any(biases_reps > 1):
                            biases_obj.value = np.tile(
                                biases_value, biases_reps.tolist())
                    if not FLOAT_EQUAL(weights_value, weights_obj.value) and in_edges[1][2]['tensor'] is not None:
                        in_edges[1][2]['tensor'].value = weights_obj.value
                    if not FLOAT_EQUAL(biases_value, biases_obj.value) and in_edges[2][2]['tensor'] is not None:
                        in_edges[2][2]['tensor'].value = biases_obj.value


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
            ERROR(
                '[Parser]: Meets invalid FullyConnected Op (%s) in rearrange_fc_reshape_bn!' % fc)
            continue
        if reshape_obj is None:
            ERROR(
                '[Parser]: Meets invalid Reshape Op (%s) in rearrange_fc_reshape_bn!' % reshape)
            continue
        if bn_obj is None:
            ERROR(
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


def fuse_special_fc_reshape_transpose_div(graph):
    if graph._attr.get('quantize', False):
        return
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('fc', {'op': 'FullyConnected'}),
                                   ('reshape0', {'op': 'Reshape'}),
                                   ('trans', {'op': 'Transpose'}),
                                   ('reshape1', {'op': 'Reshape'}),
                                   ('cons', {'op': 'Constant', 'unique': False}),
                                   ('div', {'op': 'Div'}),
                               ],
                               edges=[
                                   ('fc', 'reshape0', {'dst_in_port': 0}),
                                   ('reshape0', 'trans', {'dst_in_port': 0}),
                                   ('trans', 'reshape1', {'src_out_port': 0}),
                                   ('reshape1', 'div', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('cons', 'div', {'src_out_port': 0, 'dst_in_port': 1}),
                               ])
    for m in matches:
        fc, reshape0, trans, reshape1, cons, div = m['fc'], m['reshape0'], m['trans'], m['reshape1'], m['cons'], m[
            'div']
        fc_obj = NodeWrap(graph, fc)['object']
        cons_obj = NodeWrap(graph, cons)['object']
        div_obj = NodeWrap(graph, div)['object']
        if fc_obj is None:
            ERROR(
                '[Parser]: Meets invalid FullyConnected Op (%s) in fuse_special_fc_reshape_transpose_div!' % fc)
            continue
        if cons_obj is None:
            ERROR(
                '[Parser]: Meets invalid Constant Op (%s) in fuse_special_fc_reshape_transpose_div!' % cons)
            continue
        if div_obj is None:
            ERROR(
                '[Parser]: Meets invalid Div Op (%s) in fuse_special_fc_reshape_transpose_div!' % div)
            continue
        const_value = cons_obj.value
        if const_value.min() != const_value.max():
            continue
        fc_out_edges = graph.sorted_out_edges(fc, data=True)
        reshape0_out_edges = graph.sorted_out_edges(reshape0)
        trans_out_edges = graph.sorted_out_edges(trans)
        reshape1_out_edges = graph.sorted_out_edges(reshape1, data=True)
        div_out_edges = graph.sorted_out_edges(div, data=True)
        if len(fc_out_edges) != 1 or \
                len(reshape0_out_edges) != 1 or \
                len(trans_out_edges) != 1 or \
                len(reshape1_out_edges) != 1:
            continue
        matched = True
        div_value = np.array(const_value.min(), dtype=const_value.dtype)
        new_weights = fc_obj.weights / div_value
        new_biases = fc_obj.biases / div_value
        fc_obj.weights = new_weights
        fc_obj.biases = new_biases
        div_in_edges = graph.sorted_in_edges(div)
        graph.remove_edges_from(div_in_edges)
        graph.remove_edges_from(div_out_edges)
        src, _, reshape1_out_attr = reshape1_out_edges[0]
        for _, div_dst, div_out_attr in div_out_edges:
            out_attr = copy.deepcopy(div_out_attr)
            graph.add_edge(src, div_dst, **out_attr)
        if div in graph._attr['output_names']:
            index = graph._attr['output_names'].index(div)
            graph._attr['output_names'][index] = reshape1
    if matched:
        clear_redundant_nodes(graph)


def lift_single_add_sub_mul_div(graph):
    matched = False
    matches0 = [matched_patterns(graph,
                                 nodes=[
                                     ('non_math_op', {'op': 'Reshape'}),
                                     ('const', {'op': 'Constant'}),
                                     ('math_op', {'op': math_op}),
                                 ],
                                 edges=[
                                     ('non_math_op', 'math_op',),
                                     ('const', 'math_op', )
                                 ]) for math_op in ['Add', 'Sub', 'Mul', 'Div']]
    matches1 = [matched_patterns(graph,
                                 nodes=[
                                     ('non_math_op', {'op': 'Transpose'}),
                                     ('const', {'op': 'Constant'}),
                                     ('math_op', {'op': math_op}),
                                 ],
                                 edges=[
                                     ('non_math_op', 'math_op',),
                                     ('const', 'math_op',)
                                 ]) for math_op in ['Add', 'Sub', 'Mul', 'Div']]
    matches = extend_lists(matches0) + extend_lists(matches1)
    for m in matches:
        non_math_op, const, math_op = m['non_math_op'], m['const'], m['math_op']
        non_math_obj = NodeWrap(graph, non_math_op)['object']
        const_obj = NodeWrap(graph, const)['object']
        math_obj = NodeWrap(graph, math_op)['object']
        if non_math_obj is None:
            ERROR(
                f'[Parser]: Meets invalid {non_math_obj.type} Op ({non_math_op}) in lift_single_add_sub_mul_div!')
            continue
        if math_obj is None:
            ERROR(
                f'[Parser]: Meets invalid {math_obj.type} Op ({math_op}) in lift_single_add_sub_mul_div!')
            continue
        if const_obj is None:
            ERROR(
                f'[Parser]: Meets invalid {const_obj.type} Op ({const}) in lift_single_add_sub_mul_div!')
            continue
        const_value = const_obj.value
        if const_value.min() != const_value.max():
            continue
        non_math_in_edges = graph.sorted_in_edges(non_math_op, data=True)
        non_math_out_edges = graph.sorted_out_edges(non_math_op, data=True)
        math_in_edges = graph.sorted_in_edges(math_op, data=True)
        math_out_edges = graph.sorted_out_edges(math_op, data=True)
        if len(non_math_out_edges) != 1:
            continue
        if math_obj.type in ['Sub', 'Div']:
            if math_in_edges[1][0] != const:
                continue
        lift_ok = True
        if non_math_obj.type == 'Reshape':
            try:
                out_t = np.reshape(np.ones(shape=non_math_in_edges[0][-1]['tensor'].shape,
                                           dtype=non_math_in_edges[0][-1]['tensor'].dtype) + const_value,
                                   non_math_obj.shape)
                if list(out_t.shape) != list(math_out_edges[0][-1]['tensor'].shape):
                    lift_ok = False
            except:
                lift_ok = False
        else:
            try:
                out_t = np.transpose(np.ones(shape=non_math_in_edges[0][-1]['tensor'].shape,
                                             dtype=non_math_in_edges[0][-1]['tensor'].dtype) + const_value,
                                     non_math_obj.perm)
                if list(out_t.shape) != list(math_out_edges[0][-1]['tensor'].shape):
                    lift_ok = False
            except:
                lift_ok = False
        if not lift_ok:
            continue
        matched = True
        non_math_src = non_math_in_edges[0][0]
        graph.remove_edge(non_math_src, non_math_op)
        graph.remove_edges_from(non_math_out_edges)
        graph.remove_edges_from(math_out_edges)

        math_scale_zp = math_out_edges[0][-1]['tensor'].scale_zp

        math_in_attr = copy.deepcopy(non_math_in_edges[0][-1])
        math_in_attr['dst_in_port'] = non_math_out_edges[0][-1]['dst_in_port']
        graph.add_edge(non_math_src, math_op, **math_in_attr)

        non_math_in_attr = copy.deepcopy(non_math_in_edges[0][-1])
        non_math_in_attr['tensor'].scale_zp = math_scale_zp
        graph.add_edge(math_op, non_math_op, **non_math_in_attr)

        for _, math_dst, math_out_attr in math_out_edges:
            out_attr = copy.deepcopy(math_out_attr)
            out_attr['tensor'].scale_zp = math_scale_zp
            graph.add_edge(non_math_op, math_dst, **out_attr)
        if math_op in graph._attr['output_names']:
            index = graph._attr['output_names'].index(math_op)
            graph._attr['output_names'][index] = non_math_op
    if matched:
        clear_redundant_nodes(graph)


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
            reshape_out_edges = graph.sorted_out_edges(reshape, data=True)
            if len(bias_obj_in_consts) >= 1 \
                    and len(reshape_out_edges) == 1 \
                    and bias_obj_in_consts[0][2] is not None \
                    and matmul_obj.num_output == bias_obj_in_consts[0][2].size:
                bias_in_edges = graph.sorted_in_edges(bias, data=True)
                bias_out_edges = graph.sorted_out_edges(bias, data=True)
                matmul_out_edges = graph.sorted_out_edges(matmul, data=True)
                if len(bias_in_edges) < 2 or len(bias_out_edges) < 1 or len(matmul_out_edges) < 1:
                    ERROR('[Parser]: Invalid Add/BatchNormalization(%s) in rearrange_matmul_reshape_bias!' % bias)
                    continue
                if bias_obj.type == 'BatchNormalization' and bias_in_edges[0][0] != reshape:
                    continue
                bias_main_in_port = 0
                if bias_obj.type == 'Add' and bias_in_edges[0][0] != reshape:
                    bias_main_in_port = 1
                graph.remove_edge(matmul, reshape)
                graph.remove_edge(reshape, bias)
                for _, dst, attr in bias_out_edges:
                    graph.remove_edge(bias, dst)
                    graph.add_edge(reshape, dst, **attr)
                bias_in_attr = copy.deepcopy(matmul_out_edges[0][2])
                bias_in_attr['dst_in_port'] = bias_main_in_port
                graph.add_edge(matmul, bias, **bias_in_attr)
                bias_out_attr = copy.deepcopy(bias_out_edges[0][2])
                bias_out_attr['dst_in_port'] = 0
                matmul_out_shape = matmul_out_edges[0][2]['tensor'].get_shape()
                if bias_out_attr['tensor'].value is not None:
                    if matmul_out_shape is not None and None not in matmul_out_shape:
                        bias_out_attr['tensor'].value = np.reshape(bias_out_attr['tensor'].value, matmul_out_shape)
                    else:
                        bias_out_attr['tensor'].value = None
                else:
                    bias_out_attr['tensor'].shape = matmul_out_shape
                graph.add_edge(bias, reshape, **bias_out_attr)
                if bias in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(bias)
                    if reshape not in graph._attr['output_names']:
                        graph._attr['output_names'][index] = reshape
                    else:
                        graph._attr['output_names'].pop(index)


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
            obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in ['linear', 'reshape', 'relu']}
            if any(obj is None for obj in obj_dict.values()):
                ERROR('[Parser]: Meets invalid Op in rearrange_linear_reshape_relu!')
                continue
            reshape_out_edges = graph.sorted_out_edges(reshape)
            if len(reshape_out_edges) != 1:
                continue
            reshape_in_shapes = obj_dict['reshape'].get_input_shapes()
            relu_in_edges = graph.sorted_in_edges(relu, data=True)
            relu_out_edges = graph.sorted_out_edges(relu, data=True)
            relu_out_tensor = None
            for _, dst, attr in relu_out_edges:
                graph.remove_edge(relu, dst)
                graph.add_edge(reshape, dst, **attr)
                if relu_out_tensor is None:
                    relu_out_tensor = copy.deepcopy(attr['tensor'])
            graph.add_edge(linear, relu, **relu_in_edges[0][2])
            if relu_out_tensor.value is not None \
                    and len(reshape_in_shapes) >= 1 \
                    and reshape_in_shapes[0] is not None \
                    and all(s is not None for s in reshape_in_shapes[0]):
                relu_out_tensor.value = np.reshape(relu_out_tensor.value, reshape_in_shapes[0])
            graph.add_edge(relu, reshape, **{'src_out_port': 0, 'dst_in_port': 0, 'tensor': relu_out_tensor})
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
        if bn_obj is not None and (
                transpose is None or (transpose_obj is not None and len(graph.sorted_out_edges(transpose)) == 1)):
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
            ERROR('[Parser]: Meets invalid Node in remove_preprocess!')
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


def remove_redundant_arithmetic(graph):
    if graph._attr.get('quantize', False):
        return
    matched = False
    matches0 = [matched_patterns(graph,
                                 nodes=[('const1', {'op': 'Constant'}),
                                        ('op1', {'op': op}),
                                        ('const2', {'op': 'Constant'}),
                                        ('op2', {'op': op})],
                                 edges=[('const1', 'op1'),
                                        ('const2', 'op2'),
                                        ('op1', 'op2')]) for op in ['Add', 'Mul']]
    matches1 = [matched_patterns(graph,
                                 nodes=[('const1', {'op': 'Constant'}),
                                        ('op1', {'op': op}),
                                        ('reshape', {'op': 'Reshape'}),
                                        ('const2', {'op': 'Constant'}),
                                        ('op2', {'op': op})],
                                 edges=[('const1', 'op1'),
                                        ('op1', 'reshape'),
                                        ('const2', 'op2'),
                                        ('reshape', 'op2')]) for op in ['Add', 'Mul']]
    matches = extend_lists(matches0)
    matches += extend_lists(matches1)
    for m in matches:
        key_names = ['const1', 'const2', 'op1', 'op2']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        op1_in_edges = graph.sorted_in_edges(m['op1'], data=True)
        op1_out_edges = graph.sorted_out_edges(m['op1'], data=True)
        op2_in_edges = graph.sorted_in_edges(m['op2'], data=True)
        op2_out_edges = graph.sorted_out_edges(m['op2'], data=True)
        if any(obj is None for obj in node_objs.values()) \
                or len(op1_in_edges) != 2 or len(op2_in_edges) != 2:
            ERROR('[Parser]: Meets invalid Nodes in remove_redundant_arithmetic!')
            continue
        src_to_op1_edge = [(src, in_attr) for (src, _, in_attr) in op1_in_edges if src != m['const1']]
        if len(src_to_op1_edge) < 1 or len(op1_out_edges) > 1:
            continue
        op_type = node_objs['op1'].type
        if op_type == 'Mul':
            const2_array = (node_objs['const2'].value *
                            np.ones(shape=op2_out_edges[0][-1]['tensor'].shape, dtype=node_objs['const2'].value.dtype))
        else:
            const2_array = (node_objs['const2'].value +
                            np.zeros(shape=op2_out_edges[0][-1]['tensor'].shape, dtype=node_objs['const2'].value.dtype))

        if 'reshape' in m:
            has_reshape = True
            rs_out_edges = graph.sorted_out_edges(m['reshape'], data=True)
            const2_array = np.reshape(const2_array, op1_out_edges[0][-1]['tensor'].shape)
            if len(rs_out_edges) > 1:
                continue
        else:
            has_reshape = False
        matched = True
        src, src_attr = src_to_op1_edge[0]
        if op_type == 'Mul':
            new_const_value = np.array(const2_array * node_objs['const1'].value)
        else:
            new_const_value = np.array(const2_array + node_objs['const1'].value)

        const_to_op2_in_port = 1 - src_attr['dst_in_port']

        graph.remove_edge(m['const1'], m['op1'])
        insert_constant(graph, m['op1'] + '_new_const', new_const_value, m['op1'], in_port=const_to_op2_in_port)
        graph.remove_edges_from(op2_out_edges)

        last_node_name = m['reshape'] if has_reshape else m['op1']

        for _, dst, out_attr in op2_out_edges:
            graph.add_edge(last_node_name, dst, **out_attr)

        if m['op2'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['op2'])
            graph._attr['output_names'][index] = last_node_name

    if matched:
        clear_redundant_nodes(graph)


def rename_single_mul_or_add_or_sub(graph):
    matched = False
    mas = ['Mul', 'Div', 'Add', 'Sub']
    mas_matches = [single_node_matcher(graph, op_type) for op_type in mas]
    mas_matches = extend_lists(mas_matches)
    for m in mas_matches:
        n = m['target']
        n_obj = NodeWrap(graph, n)['object']
        if n_obj is None:
            ERROR('[Parser]: Meets invalid Op(%s) in rename_single_mul_or_add_or_sub!' % n)
            continue
        if n_obj.quantize:
            continue
        in_tensors = n_obj.get_input_tensors()
        in_shapes = n_obj.get_input_shapes()
        in_consts = n_obj.sorted_in_consts()
        in_edges = graph.sorted_in_edges(n, keys=True, data=True)
        out_edges = graph.sorted_out_edges(n, data=True)
        if len(in_edges) == 2 \
                and len(in_tensors) == 2 \
                and len(in_shapes) == 2 \
                and len(in_consts) == 1 \
                and ((in_shapes[0] is not None and len(in_shapes[0]) in (0, 1, 2, 3, 4, 5) and in_consts[0][1] == 1)
                     or (in_shapes[1] is not None and len(in_shapes[1]) in (0, 1, 2, 3, 4, 5) and in_consts[0][1] == 0)) \
                and in_consts[0][2] is not None \
                and np.ndim(in_consts[0][2]) in (0, 1):
            const_in_port = in_consts[0][1]
            main_input_port = 1 - const_in_port
            main_input_shape = in_edges[main_input_port][3]['tensor'].get_shape()
            if in_tensors[const_in_port] is None \
                    or main_input_shape is None \
                    or None in main_input_shape:
                continue

            if n_obj.type == 'Div' and const_in_port == 0:
                continue

            if len(out_edges) == 1 \
                    and (NodeWrap(graph, out_edges[0][1])['object'].type in ('Relu', 'LeakyRelu')
                         or (NodeWrap(graph, out_edges[0][1])['object'].type == 'Clip'
                             and FLOAT_EQUAL(NodeWrap(graph, out_edges[0][1])['object'].min, 0)
                             and FLOAT_EQUAL(NodeWrap(graph, out_edges[0][1])['object'].max, 6)
                             )):
                continue

            matched = True

            if (n_obj.type in ('Mul', 'Div') and FLOAT_EQUAL(in_consts[0][2], 1.) and int(np.prod(main_input_shape)) >
                in_tensors[const_in_port].size) \
                    or (n_obj.type == 'Add' and FLOAT64_EQUAL(in_consts[0][2], 0.)) \
                    or (n_obj.type == 'Sub' and FLOAT64_EQUAL(in_consts[0][2], 0.) and const_in_port == 1):
                src, _, k, const_in_attr = in_edges[const_in_port]
                graph.remove_edge(src, n, key=k)
                remove_node_safely(graph, n)
            else:
                parent_node_type = NodeWrap(graph, in_edges[main_input_port][0])['object'].type
                linear_op_list = list(set(BaseLinearOp.get_concrete_subclass_names()).intersection(
                    OnnxOp.get_concrete_subclass_names())) + ['FullyConnected']
                if in_shapes[0] != in_shapes[1] and parent_node_type not in linear_op_list:
                    continue

                output_shapes = n_obj.get_output_shapes()
                if len(output_shapes) == 0 or output_shapes[0] is None:
                    ERROR(
                        '[Parser]: Meets invalid output shape of Node(%s) in rename_single_mul_or_add_or_sub!' % n)
                    continue

                src, _, k, in_attr = in_edges[main_input_port]
                input_dtype = in_attr['tensor'].get_dtype()
                if input_dtype is None or 'int' in input_dtype:
                    continue

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

                if tiled_const_value.shape[-1] > num_output:
                    num_output = tiled_const_value.shape[-1]
                    src, _, _, in_attr = in_edges[main_input_port]
                    insert_tile(graph, src, n, in_attr, [
                        1] * (len(in_shapes[main_input_port]) - 1) + [num_output])
                    in_edges = graph.sorted_in_edges(n, keys=True, data=True)

                if n_obj.type == 'Sub' and in_consts[0][1] == 0:
                    gamma_value = - np.ones((num_output,), input_dtype)
                    beta_value = tiled_const_value
                else:
                    gamma_value = tiled_const_value \
                        if n_obj.type == 'Mul' \
                        else (1 / tiled_const_value if n_obj.type == 'Div' else np.ones((num_output,), input_dtype))
                    beta_value = np.zeros((num_output,), input_dtype) \
                        if n_obj.type in ('Mul', 'Div') \
                        else (tiled_const_value if n_obj.type == 'Add' else -tiled_const_value)
                mean_value = np.zeros((num_output,), input_dtype)
                var_value = np.ones((num_output,), input_dtype)
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

                post_reshape = None
                if reshape_inserted:
                    post_reshape = insert_reshape_after(
                        graph, n, original_shape)

                if n in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(n)
                    if reshape_inserted and post_reshape is not None:
                        graph._attr['output_names'][index] = post_reshape

    if matched:
        clear_redundant_nodes(graph)


def convert_div_to_mul(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('div', {'op': 'Div'}),
                                   ('const', {'op': 'Constant'}),
                               ],
                               edges=[
                                   ('const', 'div', {'dst_in_port': 1}),
                               ])
    for m in matches:
        div = m['div']
        const = m['const']
        div_obj = NodeWrap(graph, div)['object']
        const_obj = NodeWrap(graph, const)['object']
        if div_obj is None or const_obj is None:
            ERROR('[Parser]: Meets invalid Nodes in convert_div_to_mul!')
            continue
        if div_obj.quantize:
            continue
        matched = True
        new_const_value = np.array(np.reciprocal(const_obj.value))
        graph.remove_edge(const, div)

        insert_constant(graph, div + '_new_const', new_const_value, div, in_port=1)
        NodeWrap(graph, div).replace_obj(
            'Mul', {'name': div, 'opset_version': 7})
    if matched:
        clear_redundant_nodes(graph)


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
            ERROR('[Parser]: Node (%s or %s or %s or %s) cannot be found, graph maybe has been changed!' % (
                const_1, const_2, add, sub))
            continue
        inp = add if sub in graph.children(add) else sub
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


def remove_special_gather(graph):
    '''Remove special gather whose indices are sorted from 0 to input_shape[axis].
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('indice', {'op': 'Constant'}),
                                   ('gather', {'op': 'Gather'}),
                               ],
                               edges=[
                                   ('indice', 'gather', {'dst_in_port': 1}),
                               ])
    for m in matches:
        indice, gather = m['indice'], m['gather']
        indice_obj, gather_obj = [NodeWrap(graph, name)['object'] for name in (indice, gather)]
        if indice_obj is None or gather_obj is None:
            ERROR('[Parser]: Meets invalid Nodes in remove_special_gather!')
            continue
        gather_in_edges = graph.sorted_in_edges(gather, data=True)
        input_shapes = gather_obj.get_input_shapes()
        if len(gather_in_edges) < 2 or len(input_shapes) < 2:
            ERROR('[Parser]: Meets invalid inputs of Gather (%s) in remove_special_gather!' % gather)
            continue
        input_shape = input_shapes[0]
        if input_shape is None or None in input_shape:
            continue
        exp_indices = list(range(input_shape[gather_obj.axis]))
        if not np.array_equal(exp_indices, indice_obj.value):
            continue
        matched = True
        src, _, in_attr = gather_in_edges[0]
        new_src_out_port = in_attr['src_out_port']
        gather_out_edges = graph.sorted_out_edges(gather, data=True)
        graph.remove_edges_from(gather_out_edges)
        for _, dst, out_attr in gather_out_edges:
            out_attr.update({'src_out_port': new_src_out_port})
            graph.add_edge(src, dst, **out_attr)
        if gather in graph._attr['output_names']:
            index = graph._attr['output_names'].index(gather)
            graph._attr['output_names'][index] = src
    if matched:
        clear_redundant_nodes(graph)


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


def split_special_gn(graph):
    '''Split GroupNormalization into GroupNormalization+Mul+Add if any one of scale and bias is not constant.
    '''
    matches = single_node_matcher(graph, 'GroupNormalization')
    for m in matches:
        gn = m['target']
        gn_obj = NodeWrap(graph, gn)['object']
        gn_in_edges = graph.sorted_in_edges(gn, data=True)
        gn_out_edges = graph.sorted_out_edges(gn, data=True)
        if gn_obj is None or len(gn_in_edges) < 3 or len(gn_out_edges) < 1:
            ERROR('[Parser]: Meets invalid GroupNormalization Node(%s) in split_special_gn!' % gn)
            continue
        if gn_obj.quantize:
            continue
        scale, _, scale_in_attr = gn_in_edges[1]
        bias, _, bias_in_attr = gn_in_edges[2]
        if scale_in_attr['tensor'] is not None and scale_in_attr['tensor'].is_const \
                and bias_in_attr['tensor'] is not None and bias_in_attr['tensor'].is_const:
            continue
        gn_input_shapes = gn_obj.get_input_shapes()
        if len(gn_input_shapes) < 1 or gn_input_shapes[0] is None or None in gn_input_shapes[0]:
            ERROR('[Parser]: Meets invalid input shape of GroupNormalization Node(%s) in split_special_gn!' % gn)
            continue
        gn_input_dtypes = gn_obj.get_input_dtypes()
        if len(gn_input_dtypes) < 3 or gn_input_dtypes[1] is None or gn_input_dtypes[2] is None:
            ERROR('[Parser]: Meets invalid input dtype of GroupNormalization Node(%s) in split_special_gn!' % gn)
            continue
        graph.remove_edges_from(gn_out_edges + gn_in_edges[1:])
        num_groups = gn_obj.num_groups
        scale_value = np.ones([num_groups], gn_input_dtypes[1])
        bias_value = np.zeros_like(scale_value, gn_input_dtypes[2])
        insert_constant(graph, scale + '_const', scale_value, gn, in_port=1)
        insert_constant(graph, bias + '_const', bias_value, gn, in_port=2)

        mul_node = get_valid_node_name(graph, gn + '_mul')
        graph.add_edge(gn, mul_node)
        graph.add_edge(scale, mul_node, **scale_in_attr)
        mul_attr = {'name': mul_node, 'opset_version': 13}
        NodeWrap(graph, mul_node).replace_obj('Mul', mul_attr)

        add_node = get_valid_node_name(graph, gn + '_add')
        graph.add_edge(mul_node, add_node)
        bias_in_attr.update({'dst_in_port': 1})
        graph.add_edge(bias, add_node, **bias_in_attr)
        add_attr = {'name': add_node, 'opset_version': 13}
        NodeWrap(graph, add_node).replace_obj('Add', add_attr)

        shape = None
        is_channels_last = (gn_obj.data_format[-1] == 'C')
        if not is_channels_last:
            shape = [1, -1] + [1] * (len(gn_input_shapes[0]) - 2)
            post_scale = insert_reshape(graph, scale, mul_node, scale_in_attr, shape)
            post_bias = insert_reshape(graph, bias, add_node, bias_in_attr, shape)
        else:
            post_scale = scale
            post_bias = bias

        channels = gn_input_shapes[0][-1] if is_channels_last else gn_input_shapes[0][1]
        if num_groups != channels:
            reps = [int(channels / num_groups)] * num_groups
            axis = (len(gn_input_shapes[0]) - 1) if is_channels_last else 1
            post_scale_out_attr = copy.deepcopy(scale_in_attr)
            if shape is not None and scale_in_attr['tensor'] is not None and scale_in_attr['tensor'].value is not None:
                post_scale_out_attr['tensor'].value = np.reshape(scale_in_attr['tensor'].value, shape)
            insert_repeat(graph, post_scale, mul_node, post_scale_out_attr, reps, axis)
            post_bias_out_attr = copy.deepcopy(bias_in_attr)
            if shape is not None and bias_in_attr['tensor'] is not None and bias_in_attr['tensor'].value is not None:
                post_bias_out_attr['tensor'].value = np.reshape(bias_in_attr['tensor'].value, shape)
            insert_repeat(graph, post_bias, add_node, post_bias_out_attr, reps, axis)

        for _, dst, out_attr in gn_out_edges:
            graph.add_edge(add_node, dst, **out_attr)
        if gn in graph._attr['output_names']:
            index = graph._attr['output_names'].index(gn)
            graph._attr['output_names'][index] = add_node


def split_special_ln(graph):
    '''Split LayerNormalization into MVN+Mul+Add if any one of scale and bias is not constant.
    '''
    matches = single_node_matcher(graph, 'LayerNormalization')
    for m in matches:
        ln = m['target']
        ln_obj = NodeWrap(graph, ln)['object']
        ln_in_edges = graph.sorted_in_edges(ln, data=True)
        ln_out_edges = graph.sorted_out_edges(ln, data=True)
        if ln_obj is None or len(ln_in_edges) < 3 or len(ln_out_edges) < 1:
            ERROR('[Parser]: Meets invalid LayerNormalization Node(%s) in split_special_ln!' % ln)
            continue
        if ln_obj.get_out_ports() != [0]:
            continue
        scale, _, scale_in_attr = ln_in_edges[1]
        bias, _, bias_in_attr = ln_in_edges[2]
        if scale_in_attr['tensor'] is not None and scale_in_attr['tensor'].is_const \
                and bias_in_attr['tensor'] is not None and bias_in_attr['tensor'].is_const:
            continue
        graph.remove_edges_from(ln_out_edges + ln_in_edges[1:])

        mul_node = get_valid_node_name(graph, ln + '_mul')
        graph.add_edge(ln, mul_node)
        graph.add_edge(scale, mul_node, **scale_in_attr)
        mul_attr = {'name': mul_node, 'opset_version': 13}
        NodeWrap(graph, mul_node).replace_obj('Mul', mul_attr)

        add_node = get_valid_node_name(graph, ln + '_add')
        graph.add_edge(mul_node, add_node)
        bias_in_attr.update({'dst_in_port': 1})
        graph.add_edge(bias, add_node, **bias_in_attr)
        add_attr = {'name': add_node, 'opset_version': 13}
        NodeWrap(graph, add_node).replace_obj('Add', add_attr)

        mvn_attr = ln_obj.copied_attr()
        mvn_attr.update({'opset_version': 13})
        NodeWrap(graph, ln).replace_obj('MeanVarianceNormalization', mvn_attr)

        for _, dst, out_attr in ln_out_edges:
            graph.add_edge(add_node, dst, **out_attr)
        if ln in graph._attr['output_names']:
            index = graph._attr['output_names'].index(ln)
            graph._attr['output_names'][index] = add_node


def split_special_ln2(graph):
    matches = single_node_matcher(graph, 'LayerNormalization')
    for m in matches:
        ln = m['target']
        ln_obj = NodeWrap(graph, ln)['object']
        ln_in_edges = graph.sorted_in_edges(ln, data=True)
        ln_out_edges = graph.sorted_out_edges(ln, data=True)
        if ln_obj is None or len(ln_in_edges) != 3 or len(ln_out_edges) < 1:
            ERROR('[Parser]: Meets invalid LayerNormalization Node(%s) in split_special_ln!' % ln)
            continue
        input_shapes = ln_obj.get_input_shapes()
        if any(s is None for s in input_shapes) or any(d is None for s in input_shapes for d in s):
            continue
        out_ports = ln_obj.get_out_ports()
        if out_ports != [0, 1] and out_ports != [0, 2] and out_ports != [0, 1, 2]:
            continue
        x, _, x_in_attr = ln_in_edges[0]
        scale, _, scale_in_attr = ln_in_edges[1]
        B, _, B_in_attr = ln_in_edges[2]
        if scale_in_attr['tensor'].is_const and B_in_attr['tensor'].is_const:
            continue
        if ln_obj.axis is None:
            continue
        x_shape = input_shapes[0]
        if ln_obj.axis < 0:
            ln_obj.axis += len(x_shape)
        if ln_obj.axes is None:
            ln_obj.axes = list(range(ln_obj.axis, len(x_shape)))
        mask = np.array(ln_obj.axes)
        mean_shape = np.array(x_shape)
        mean_shape[mask] = 1
        mean_shape = tuple(mean_shape.tolist())
        sub = get_valid_node_name(graph, ln + '_sub')
        add1 = get_valid_node_name(graph, ln + '_add1')
        sqrt = get_valid_node_name(graph, ln + '_sqrt')
        reciprocal = get_valid_node_name(graph, ln + '_reciprocal')
        mul1 = get_valid_node_name(graph, ln + '_mul1')
        mul2 = get_valid_node_name(graph, ln + '_mul2')
        add2 = get_valid_node_name(graph, ln + '_add2')

        graph.remove_edges_from(ln_in_edges[1:])
        graph.remove_edges_from(ln_out_edges)
        graph.add_edge(x, sub, **x_in_attr)
        graph.add_edge(ln, sub, **{'dst_in_port': 1, 'tensor': Tensor(shape=mean_shape)})
        graph.add_edge(ln, add1, **{'src_out_port': 1, 'tensor': Tensor(shape=mean_shape)})
        insert_constant(graph, ln + '_epsilon', np.array(ln_obj.epsilon, dtype=np.float32), add1, in_port=1)
        graph.add_edge(add1, sqrt, **{'tensor': Tensor(shape=mean_shape)})
        graph.add_edge(sqrt, reciprocal)
        graph.add_edge(sub, mul1)
        graph.add_edge(reciprocal, mul1, **{'dst_in_port': 1})
        graph.add_edge(mul1, mul2)
        graph.add_edge(scale, mul2, **scale_in_attr)
        graph.add_edge(mul2, add2)
        new_B_in_attr = copy.deepcopy(B_in_attr)
        new_B_in_attr['dst_in_port'] = 1
        graph.add_edge(B, add2, **new_B_in_attr)
        for _, dst, out_attr in ln_out_edges:
            if out_attr['src_out_port'] == 0:
                graph.add_edge(add2, dst, **out_attr)
            elif out_attr['src_out_port'] == 1:
                new_out_attr = copy.deepcopy(out_attr)
                new_out_attr['src_out_port'] = 0
                graph.add_edge(ln, dst, **new_out_attr)
            elif out_attr['src_out_port'] == 2:
                new_out_attr = copy.deepcopy(out_attr)
                new_out_attr['src_out_port'] = 0
                graph.add_edge(reciprocal, dst, **new_out_attr)

        mm_attr = ln_obj.copied_attr()
        mm_attr.update({'axes': ln_obj.axes, 'keepdims': 1})
        NodeWrap(graph, ln).replace_obj('Moments', mm_attr)
        NodeWrap(graph, sub).replace_obj('Sub', {'name': sub, 'opset_version': 7})
        NodeWrap(graph, add1).replace_obj('Add', {'name': add1, 'opset_version': 7})
        NodeWrap(graph, sqrt).replace_obj('Sqrt', {'name': sqrt, 'opset_version': 6})
        NodeWrap(graph, reciprocal).replace_obj('Reciprocal', {'name': reciprocal, 'opset_version': 6})
        NodeWrap(graph, mul1).replace_obj('Mul', {'name': mul1, 'opset_version': 7})
        NodeWrap(graph, mul2).replace_obj('Mul', {'name': mul2, 'opset_version': 7})
        NodeWrap(graph, add2).replace_obj('Add', {'name': add2, 'opset_version': 7})

        if ln in graph._attr['output_names']:
            index = graph._attr['output_names'].index(ln)
            graph._attr['output_names'].insert(index, add2)
            if 1 not in out_ports:
                graph._attr['output_names'].remove(ln)
                index += 1
            else:
                index += 2
            if 2 in out_ports:
                graph._attr['output_names'].insert(index, reciprocal)


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
            ERROR(
                '[Parser]: Meets invalid ConvTranspose (%s) in split_conv_transpose!' % conv_trans)
            continue
        if all(p == 0 for p in conv_trans_obj.output_padding):
            continue
        strides = np.array(conv_trans_obj.strides)
        dilations = np.array(conv_trans_obj.dilations)
        output_padding = np.array(conv_trans_obj.output_padding)
        if np.any(np.logical_and(output_padding >= strides, output_padding >= dilations)):
            ERROR('[Parser]: Onnx %s (%s) output_padding should be less than stride or dilation!'
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

        src, _, in_attr = in_edges[0]
        input_shape = in_attr['tensor'].shape
        padded_shape = []
        for i, s in enumerate(input_shape):
            padded_shape.append(s + full_pads[i] + full_pads[i + spatial_rank + 2])
        graph.remove_edges_from(in_edges)
        pre_pad = get_valid_node_name(graph, conv_trans + '_pre_pad')
        graph.add_edge(src, pre_pad, **in_attr)
        deconv_in_attr = copy.deepcopy(in_attr)
        deconv_in_attr['tensor'] = Tensor(shape=tuple(padded_shape))
        graph.add_edge(pre_pad, conv_trans, **deconv_in_attr)
        pad_attr = {'name': pre_pad,
                    'opset_version': 2,
                    'pads': full_pads,
                    'data_format': data_format}
        NodeWrap(graph, pre_pad).replace_obj('Pad', pad_attr)

        if ori_spatial_output_shape != new_spatial_output_shape:
            begin = [0] * (2 + spatial_rank)
            size = ori_output_shape
            new_output_shape = ori_output_shape[:2] + new_spatial_output_shape if data_format == 'NCHW' else (
                ori_output_shape[:1] + new_spatial_output_shape + ori_output_shape[-1:])
            post_slice = insert_slice_after(
                graph, conv_trans, begin, size, slice_before_shape=new_output_shape, data_format=data_format)
            if conv_trans in graph._attr['output_names']:
                index = graph._attr['output_names'].index(conv_trans)
                graph._attr['output_names'][index] = post_slice

        if conv_trans_obj.output_shape:
            assert (len(conv_trans_obj.output_shape) == spatial_rank), \
                '[Parser]: Meets invalid output_shape of ConvTranspose (%s) in split_conv_transpose!' % conv_trans
            conv_trans_obj.output_shape = new_spatial_output_shape
        conv_trans_obj.output_padding = [0] * spatial_rank


def split_deformable_conv(graph):
    matches = single_node_matcher(graph, 'DeformConv')
    for m in matches:
        deform_conv = m['target']
        deform_conv_obj = NodeWrap(graph, deform_conv)['object']
        in_edges = graph.sorted_in_edges(deform_conv, data=True)
        input_shapes = deform_conv_obj.get_input_shapes()
        if deform_conv_obj is None or len(in_edges) < 3 or len(input_shapes) < 3:
            ERROR(
                '[Parser]: Meets invalid DeformConv (%s) in split_deformable_conv!' % deform_conv)
            continue
        if deform_conv_obj.quantize:
            continue
        if any((shape is None or None in shape or len(shape) != 4) for shape in input_shapes[:3]):
            ERROR(
                '[Parser]: Meets unsupported input shapes of DeformConv (%s) in split_deformable_conv!' % deform_conv)
            continue
        input_dtypes = deform_conv_obj.get_input_dtypes()
        if len(input_dtypes) < 3 or input_dtypes[2] is None:
            ERROR(
                '[Parser]: Meets invalid input dtypes of DeformConv (%s) in split_deformable_conv!' % deform_conv)
            continue
        data_format = deform_conv_obj.data_format
        if not data_format.startswith('NC'):
            WARN('[Parser]: Meets unsupported data format of DeformConv (%s) in split_deformable_conv!' % deform_conv)
            continue
        output_shapes = deform_conv_obj.get_output_shapes()
        if len(output_shapes) < 1 or output_shapes[0] is None or None in output_shapes[0]:
            continue
        weights_tensor = in_edges[1][2]['tensor']
        if weights_tensor is None or (weights_tensor.is_const and weights_tensor.value is None):
            ERROR(
                '[Parser]: Meets invalid weights of DeformConv (%s) in split_deformable_conv!' % deform_conv)
            continue
        if len(in_edges) > 3 and in_edges[3][2]['tensor'] is None:
            ERROR(
                '[Parser]: Meets invalid biases of DeformConv (%s) in split_deformable_conv!' % deform_conv)
            continue
        input_shape = input_shapes[0]
        weights_shape = input_shapes[1]
        offset_shape = input_shapes[2]
        out_c = weights_shape[0]
        add_bias_shape = [1, out_c, 1, 1]
        kernel_shape = deform_conv_obj.kernel_shape
        group = deform_conv_obj.group
        offset_group = deform_conv_obj.offset_group
        k_h, k_w = kernel_shape
        k_hw = int(k_h * k_w)
        strides = deform_conv_obj.strides
        dilations = deform_conv_obj.dilations
        pads = deform_conv_obj.pads
        batch, in_c, in_h, in_w = input_shape
        out_h, out_w = output_shapes[0][2:]
        data_pre_reshape_dim = np.array([int(batch * offset_group), -1, in_h, in_w])
        data_repeats = np.array([k_hw] * int(batch * offset_group), np.int64)
        offset_reshape_dim = np.array([int(batch * offset_group * k_hw), 2, out_h, out_w], np.int64)
        mask_reshape_dim = np.array([int(batch * offset_group * k_hw), 1, out_h, out_w], np.int64)
        data_post_reshape_dim = np.array([int(batch * offset_group), k_hw, -1, out_h, out_w], np.int64)
        data_post_trans_perm = [0, 2, 1, 3, 4]  # from [NG, k_hw, C//G, H, W] to [NG, C//G, k_hw, H, W]
        data_post_reshape2_dim = np.array([batch, int(in_c * k_hw), out_h, out_w], np.int64)
        const_hw = np.array([[[2.0 / (in_h - 1)]], [[2.0 / (in_w - 1)]]], np.float32)
        const_neg1 = np.array(-1.0, np.float32)
        offset_base_data = DeformConvOp.gen_offset_base(weights_shape, offset_shape,
                                                        kernel_shape, strides, dilations, pads,
                                                        offset_group).astype(input_dtypes[2])

        graph.remove_edges_from(in_edges)
        data_src, _, data_in_attr = in_edges[0]
        offset_src, _, offset_in_attr = in_edges[2]

        # offset src -> reshape
        offset_reshape = get_valid_node_name(graph, deform_conv + '_offset_reshape')
        offset_reshape_in_attr = copy.deepcopy(offset_in_attr)
        offset_reshape_in_attr.update({'dst_in_port': 0})
        graph.add_edge(offset_src, offset_reshape, **offset_reshape_in_attr)
        insert_constant(graph, offset_reshape + '_shape', offset_reshape_dim, offset_reshape, in_port=1)
        NodeWrap(graph, offset_reshape).replace_obj('Reshape', {'name': offset_reshape, 'opset_version': 13})
        # reshape -> add base
        offset_add_base = get_valid_node_name(graph, deform_conv + '_offset_add')
        offset_add_base_in_attr = {'tensor': Tensor(shape=tuple(offset_reshape_dim))}
        graph.add_edge(offset_reshape, offset_add_base, **offset_add_base_in_attr)
        insert_constant(graph, offset_add_base + '_val', offset_base_data, offset_add_base, in_port=1)
        NodeWrap(graph, offset_add_base).replace_obj('Add', {'name': offset_add_base, 'opset_version': 13})
        # add base -> mul&add -1
        offset_mul = get_valid_node_name(graph, deform_conv + '_offset_mul')
        graph.add_edge(offset_add_base, offset_mul, **{'src_out_port': 0,
                                                       'tensor': Tensor(shape=tuple(offset_reshape_dim))})
        insert_constant(graph, offset_mul + '_hw', const_hw, offset_mul, in_port=1)
        NodeWrap(graph, offset_mul).replace_obj('Mul', {'name': offset_mul, 'opset_version': 11})
        offset_add_neg1 = get_valid_node_name(graph, deform_conv + '_offset_add_neg1')
        graph.add_edge(offset_mul, offset_add_neg1, **{'tensor': Tensor(shape=tuple(offset_reshape_dim))})
        insert_constant(graph, offset_add_neg1 + '_val', const_neg1, offset_add_neg1, in_port=1)
        NodeWrap(graph, offset_add_neg1).replace_obj('Add', {'name': offset_add_neg1, 'opset_version': 13})
        # mul&add -1 -> split
        offset_split = get_valid_node_name(graph, deform_conv + '_offset_split')
        graph.add_edge(offset_add_neg1, offset_split, **{'tensor': Tensor(shape=tuple(offset_reshape_dim))})
        NodeWrap(graph, offset_split).replace_obj(
            'Split', {'name': offset_split, 'opset_version': 11, 'split': [1, 1], 'axis': 1})
        # split -> concat(yx to xy)
        offset_concat = get_valid_node_name(graph, deform_conv + '_offset_concat')
        graph.add_edge(offset_split, offset_concat, **{'src_out_port': 1, 'dst_in_port': 0})  # yx to xy
        graph.add_edge(offset_split, offset_concat, **{'src_out_port': 0, 'dst_in_port': 1})
        NodeWrap(graph, offset_concat).replace_obj('Concat', {'name': offset_concat, 'opset_version': 13, 'axis': 1})
        # concat -> transpose to NHWC(grid format in onnx GridSample)
        offset_transpose = get_valid_node_name(graph, deform_conv + '_offset_tran')
        graph.add_edge(offset_concat, offset_transpose)
        NodeWrap(graph, offset_transpose).replace_obj('Transpose', {
            'name': offset_transpose, 'opset_version': 13, 'perm': [0, 2, 3, 1]})

        # data src -> reshape
        data_pre_reshape = get_valid_node_name(graph, deform_conv + '_pre_reshape')
        graph.add_edge(data_src, data_pre_reshape, **data_in_attr)
        insert_constant(graph, data_pre_reshape + '_shape', data_pre_reshape_dim, data_pre_reshape, in_port=1)
        NodeWrap(graph, data_pre_reshape).replace_obj('Reshape', {'name': data_pre_reshape, 'opset_version': 13})
        # reshape -> repeat
        data_repeat = get_valid_node_name(graph, deform_conv + '_repeat')
        graph.add_edge(data_pre_reshape, data_repeat)
        insert_constant(graph, data_repeat + '_reps', data_repeats, data_repeat, in_port=1)
        NodeWrap(graph, data_repeat).replace_obj('Repeat', {'name': data_repeat, 'opset_version': 1, 'axis': 0})
        # repeat -> grid_sample
        data_grid_sample = get_valid_node_name(graph, deform_conv + '_gridsample')
        graph.add_edge(data_repeat, data_grid_sample)
        graph.add_edge(offset_transpose, data_grid_sample, **{'dst_in_port': 1})
        NodeWrap(graph, data_grid_sample).replace_obj('GridSample', {
            'name': data_grid_sample, 'align_corners': 1, 'opset_version': 16, 'data_format': data_format})

        if len(in_edges) <= 4 \
                or (in_edges[4][2]['tensor'] is not None
                    and in_edges[4][2]['tensor'].is_const
                    and not in_edges[4][2]['tensor'].value):
            data_masked = data_grid_sample
        else:
            mask_src, _, mask_in_attr = in_edges[4]
            # mask src -> reshape
            mask_reshape = get_valid_node_name(graph, deform_conv + '_mask_reshape')
            mask_reshape_in_attr = copy.deepcopy(mask_in_attr)
            mask_reshape_in_attr.update({'dst_in_port': 0})
            graph.add_edge(mask_src, mask_reshape, **mask_reshape_in_attr)
            insert_constant(graph, mask_reshape + '_shape', mask_reshape_dim, mask_reshape, in_port=1)
            NodeWrap(graph, mask_reshape).replace_obj('Reshape', {'name': mask_reshape, 'opset_version': 13})
            # mask src -> reshape -> mul(*data)
            data_mask_mul = get_valid_node_name(graph, deform_conv + '_data_mask_mul')
            graph.add_edge(data_grid_sample, data_mask_mul)
            graph.add_edge(mask_reshape, data_mask_mul, **{'dst_in_port': 1,
                                                           'tensor': Tensor(shape=tuple(mask_reshape_dim))})
            NodeWrap(graph, data_mask_mul).replace_obj('Mul', {'name': data_mask_mul, 'opset_version': 13})
            data_masked = data_mask_mul

        # masked data -> reshape
        data_post_reshape = get_valid_node_name(graph, deform_conv + '_post_reshape')
        graph.add_edge(data_masked, data_post_reshape)
        insert_constant(graph, data_post_reshape + '_shape', data_post_reshape_dim, data_post_reshape, in_port=1)
        NodeWrap(graph, data_post_reshape).replace_obj('Reshape', {'name': data_post_reshape, 'opset_version': 13})
        # reshape -> transpose(data_post_trans_perm)
        data_post_trans = get_valid_node_name(graph, deform_conv + '_post_trans')
        graph.add_edge(data_post_reshape, data_post_trans, **{'tensor': Tensor(shape=data_post_reshape_dim)})
        NodeWrap(graph, data_post_trans).replace_obj('Transpose', {
            'name': data_post_trans, 'opset_version': 13, 'perm': data_post_trans_perm})
        # transpose -> reshape(data_post_reshape2_dim)
        data_post_reshape2 = get_valid_node_name(graph, deform_conv + '_post_reshape2')
        graph.add_edge(data_post_trans, data_post_reshape2)
        insert_constant(graph, data_post_reshape2 + '_shape', data_post_reshape2_dim, data_post_reshape2, in_port=1)
        NodeWrap(graph, data_post_reshape2).replace_obj('Reshape', {'name': data_post_reshape2, 'opset_version': 13})

        # data -> conv
        graph.add_edge(data_post_reshape2, deform_conv, **{'tensor': Tensor(shape=data_post_reshape2_dim)})
        conv_attr = deform_conv_obj.copied_attr()
        if weights_tensor.is_const:
            new_weights = np.reshape(weights_tensor.value, [out_c, -1, 1, 1])
            default_biases = np.zeros([out_c], dtype=new_weights.dtype)
            conv_attr.update({'weights': new_weights, 'biases': default_biases})
            if len(in_edges) > 3:
                biases_tensor = in_edges[3][2]['tensor']
                biases_shape = biases_tensor.get_shape()
                if biases_tensor.is_const:
                    if biases_tensor.value is not None and biases_tensor.value.size > 0:
                        conv_attr.update({'biases': biases_tensor.value})
                elif biases_shape is not None and len(biases_shape) > 0 and 0 not in biases_shape:
                    out_edges = graph.sorted_out_edges(deform_conv, data=True)
                    graph.remove_edges_from(out_edges)
                    add_bias = get_valid_node_name(graph, deform_conv + '_bias')
                    graph.add_edge(deform_conv, add_bias, **{'tensor': Tensor(shape=tuple(output_shapes[0]))})
                    bias, _, bias_in_attr = in_edges[3]
                    bias_in_attr.update({'dst_in_port': 1})
                    graph.add_edge(bias, add_bias, **bias_in_attr)
                    NodeWrap(graph, add_bias).replace_obj('Add', {'name': add_bias, 'opset_version': 14})
                    insert_reshape(graph, bias, add_bias, bias_in_attr, add_bias_shape)
                    for _, dst, out_attr in out_edges:
                        graph.add_edge(add_bias, dst, **out_attr)
                    if deform_conv in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(deform_conv)
                        graph._attr['output_names'][index] = add_bias
        else:
            conv_attr.update({'weights': None, 'biases': None})
            weight, _, weight_in_attr = in_edges[1]
            graph.add_edge(weight, deform_conv, **weight_in_attr)
            insert_reshape(graph, weight, deform_conv, weight_in_attr, data_post_reshape2_dim)
            if len(in_edges) > 3:
                bias, _, bias_in_attr = in_edges[3]
                biases_shape = bias_in_attr['tensor'].get_shape()
                if biases_shape is not None and len(biases_shape) > 0 and 0 not in biases_shape:
                    bias_in_attr.update({'dst_in_port': 2})
                    graph.add_edge(bias, deform_conv, **bias_in_attr)
                    insert_reshape(graph, bias, deform_conv, bias_in_attr, add_bias_shape)
        conv_attr.update({'opset_version': 11, 'kernel_shape': [1, 1],
                          'strides': [1, 1], 'pads': [0, 0, 0, 0], 'dilations': [1, 1], 'group': group})
        NodeWrap(graph, deform_conv).replace_obj('Conv', conv_attr)


def split_negative_pads(graph):
    possible_op_types = [op_type.get_concrete_subclass_names()
                         for op_type in [OnnxOp, CommonOp]]
    possible_op_types = extend_lists(possible_op_types)
    op_with_pads_list = list(set(
        OpHasPaddingStrides.get_concrete_subclass_names()).difference(['ConvTranspose']).intersection(
        possible_op_types))
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
            ERROR(
                '[Parser]: Meets invalid Node (%s) in split_negative_pads!' % node)


def split_reduce_logsumexp(graph):
    matches = single_node_matcher(graph, 'ReduceLogSumExp')
    for m in matches:
        rlse = m['target']
        rlse_obj = NodeWrap(graph, rlse)['object']
        in_edges = graph.sorted_in_edges(rlse, data=True)
        if rlse_obj is None or len(in_edges) < 1:
            ERROR(
                '[Parser]: Meets invalid ReduceLogSumExp (%s) in split_reduce_logsumexp!' % rlse)
            continue
        exp = get_valid_node_name(graph, rlse + '_exp')
        src, _, in_attr = in_edges[0]
        graph.remove_edges_from(in_edges)
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
        if rls_obj is None or len(in_edges) < 1:
            ERROR(
                '[Parser]: Meets invalid ReduceLogSum (%s) in split_reduce_logsum!' % rls)
            continue
        reduce_sum = get_valid_node_name(graph, rls + '_reduce_sum')
        src, _, in_attr = in_edges[0]
        graph.remove_edges_from(in_edges)
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
        if rss_obj is not None and len(in_edges) >= 1:
            pow = get_valid_node_name(graph, rss + '_pre_pow')
            src, _, in_attr = in_edges[0]
            graph.remove_edges_from(in_edges)
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
            roll_shift, start1, end1, steps1, axes1, start2, end2, steps2, axes2 = \
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
                ERROR(
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
            ERROR('[Parser]: Meets invalid Mean Op(%s) in split_mean!' % mean)


def split_sum_or_max_or_min(graph, op_type_list=['Sum', 'Max', 'Min']):
    # a dict of new op_type and its opset version
    op_type_name_and_ver_dict = {'Sum': ['Add', 7],
                                 'Max': ['Max', 8],
                                 'Min': ['Min', 8],
                                 'TfKerasMultiply': ['Mul', 7]}
    if not isinstance(op_type_list, list) \
            or any((op_type not in op_type_name_and_ver_dict
                    or len(op_type_name_and_ver_dict[op_type]) != 2) for op_type in op_type_list):
        ERROR('[Parser]: Meet invalid op_type %s in split_sum_or_max_or_min!' % str(op_type_list))
        return

    matches = single_node_matcher(graph, op_type_list)
    for single_match in matches:
        node = single_match['target']
        node_obj = NodeWrap(graph, node)['object']
        if node_obj is not None:
            node_input_shapes = node_obj.get_input_shapes()
            if len(node_input_shapes) < 2:
                continue
            new_op_type, new_op_version = op_type_name_and_ver_dict[node_obj.type]
            split_num = len(node_input_shapes) - 2
            if split_num > 0:
                in_edges = graph.sorted_in_edges(node, keys=True, data=True)
                out_edges = graph.sorted_out_edges(node, data=True)
                graph.remove_edges_from(in_edges[2:])

                nodes_list = [node]
                for i in range(split_num):
                    cur_src, _, _, cur_in_attr = in_edges[2 + i]
                    new_node = get_valid_node_name(graph, node + '_expand_' + str(i + 1))
                    new_node_in_tensor = Tensor()
                    last_node_obj = NodeWrap(graph, nodes_list[-1])['object']
                    if last_node_obj is not None and all(
                            [inp is not None for inp in last_node_obj.get_input_tensors()]):
                        node_result = reduce(
                            lambda x, y: x + y, last_node_obj.get_input_tensors())
                        new_node_in_tensor.value = node_result
                    else:
                        new_node_in_tensor.shape = tuple(node_input_shapes[0]) if node_input_shapes[0] is not None \
                            else tuple(node_input_shapes[1])
                    graph.add_edge(
                        nodes_list[-1], new_node, **{'src_out_port': 0, 'dst_in_port': 0, 'tensor': new_node_in_tensor})
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
            ERROR('[Parser]: Invalid Op(%s) for splitting in split_sum_or_max_or_min!' % node)


def split_hardmax(graph):
    matches = single_node_matcher(graph, 'Hardmax')
    for m in matches:
        hardmax = m['target']
        hardmax_obj = NodeWrap(graph, hardmax)['object']

        in_edges = graph.sorted_in_edges(hardmax, data=True)
        if hardmax_obj is None or len(in_edges) != 1:
            ERROR('[Parser]: Meet invalid Hardmax (%s) in split_hardmax!' % hardmax)
            continue

        input_shape = hardmax_obj.get_input_shapes()[0]
        hardmax_axis = hardmax_obj.axis + \
            len(input_shape) if hardmax_obj.axis < 0 else hardmax_obj.axis
        if hardmax_axis < 0 or hardmax_axis >= len(input_shape):
            ERROR('[Parser]: Meet invalid axis (%s) of Hardmax (%s) in split_hardmax!' % (
                str(hardmax_axis), hardmax))
            continue

        input_tensors = hardmax_obj.get_input_tensors()
        if len(input_tensors) != 1 or input_tensors[0] is None:
            ERROR(
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


def adjust_1d_matmul(graph):
    matches = single_node_matcher(graph, ['MatMul'])
    for m in matches:
        matmul = m['target']
        node_obj = NodeWrap(graph, matmul)['object']
        in_edges = graph.sorted_in_edges(matmul, keys=True, data=True)

        if node_obj is None \
                or len(in_edges) != 2:
            ERROR('[Parser]: Meets invalid Matmul Op(%s) in adjust_1d_matmul!' % matmul)
            continue

        in_shapes = node_obj.get_input_shapes()
        out_shapes = node_obj.get_output_shapes()

        if len(in_shapes) != 2 \
                or any((shape is None for shape in in_shapes)) \
                or len(out_shapes) < 1 \
                or any((shape is None for shape in out_shapes)) \
                or any((shape is None for shape in out_shapes[0])):
            ERROR('[Parser]: Meets invalid Matmul Op(%s) in adjust_1d_matmul!' % matmul)
            continue

        if len(in_shapes[0]) != 1 or len(in_shapes[1]) != 1:
            continue

        for in_port, (src, _, k, in_attr) in enumerate(in_edges):
            pre_reshape_dim = [1] * (1 - in_port) + \
                in_shapes[in_port] + [1] * (in_port)
            insert_reshape(graph, src, matmul, in_attr, pre_reshape_dim, key=k,
                           quantize=node_obj.quantize)

        post_reshape = insert_reshape_after(
            graph, matmul, old_dim=[1, 1], new_dim=out_shapes[0], quantize=node_obj.quantize)
        if matmul in graph._attr['output_names']:
            index = graph._attr['output_names'].index(matmul)
            graph._attr['output_names'][index] = post_reshape


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
            dim_diff = max_dim - min_dim
            if min_dim < 2 and edge_index == 1:
                reshape_dim = [1] * (dim_diff - 2 + min_dim) + list(input_shapes[edge_index]) + [1] * (2 - min_dim)
            else:
                reshape_dim = [1] * dim_diff + list(input_shapes[edge_index])
            src, _, k, in_attr = in_edges[edge_index]
            insert_reshape(graph, src, matmul, in_attr, reshape_dim, key=k, quantize=obj.quantize)
            out_shape = obj.get_output_shapes()[0]
            post_reshape = insert_reshape_after(
                graph, matmul, out_shape, old_dim=list(out_shape), quantize=obj.quantize)
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
                ERROR(
                    '[Parser]: Meets invalid Node (%) in adjust_scalar_to_1d!' % broadcast)


def adjust_1d_to_4d(graph):
    convert_types = ['BatchNormalization']
    for op_type in convert_types:
        matches = single_node_matcher(graph, op_type)
        for m in matches:
            node_name = m['target']
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is None:
                ERROR('[Parser]: Meets invalid node(%s) in adjust_1d_to_4d!' % node_name)
                continue
            in_edges = graph.sorted_in_edges(node_name, keys=True, data=True)
            in_shapes = node_obj.get_input_shapes()
            out_shapes = node_obj.get_output_shapes()
            if len(in_edges) < 1 \
                    or len(in_shapes) < 1 \
                    or len(in_shapes[0]) != 1 \
                    or len(out_shapes) < 1 \
                    or out_shapes[0] is None \
                    or None in out_shapes[0]:
                continue
            src, _, k, in_attr = in_edges[0]
            pre_reshape_dim = [1, 1, 1] + in_shapes[0]
            insert_reshape(graph, src, node_name, in_attr, pre_reshape_dim, key=k, quantize=node_obj.quantize)
            post_reshape = insert_reshape_after(graph, node_name, out_shapes[0], pre_reshape_dim,
                                                quantize=node_obj.quantize)

            if node_name in graph._attr['output_names']:
                index = graph._attr['output_names'].index(node_name)
                graph._attr['output_names'][index] = post_reshape


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
                    if graph._attr.get('quantize', False) \
                            and node_obj.quantize:
                        quantize = True
                    else:
                        quantize = False
                    for in_port, (src, _, k, in_attr) in enumerate(in_edges):
                        pre_reshape_dim = [1, 1] + in_shapes[in_port]
                        insert_reshape(graph, src, node_name,
                                       in_attr, pre_reshape_dim, key=k, quantize=quantize)
                    post_reshape_dim = out_shapes[0]
                    post_reshape = insert_reshape_after(graph, node_name, post_reshape_dim, quantize=quantize)

                    if node_name in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(node_name)
                        graph._attr['output_names'][index] = post_reshape


def adjust_3d_to_4d(graph):
    pure_inputs_types = ['InstanceNormalization', 'LRN', 'MatMul', 'Moments', 'ArmGroupNorm']
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
                        if op_type in ('InstanceNormalization', 'LRN', 'Resize', 'ArmGroupNorm'):
                            reshape1_dim = in_shapes[in_port][0:-
                                                              1] + [1] + in_shapes[in_port][-1:]
                        else:
                            reshape1_dim = [1] + in_shapes[in_port]
                        insert_reshape(graph, src, node_name,
                                       in_attr, reshape1_dim, key=k,
                                       quantize=node_obj.quantize)

                    ports_shape = OrderedDict()
                    for _, _, out_attr in out_edges:
                        if out_attr['src_out_port'] not in ports_shape:
                            out_shape = out_attr['tensor'].value.shape if out_attr['tensor'].value is not None else \
                                out_attr['tensor'].shape
                            ports_shape.update(
                                {out_attr['src_out_port']: list(out_shape)})

                    reshape2_nodes = []
                    for out_port in node_obj.get_out_ports():
                        reshape = insert_reshape_after(graph,
                                                       node_name,
                                                       ports_shape[out_port],
                                                       out_port=out_port,
                                                       quantize=node_obj.quantize)
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
                            and NodeWrap(graph, in_edges[0][0])['object'].type == 'Transpose':
                        pred = graph.predecessor
                        if NodeWrap(graph, pred[pred[node_name][0]][0])['object'].type == 'Input':
                            new_w = 1
                            for i in range(2, len(in_shapes[0]) - 1):
                                new_w *= in_shapes[0][i]
                            src, _, k, in_attr = in_edges[0]
                            pre_reshape_dim = [
                                in_shapes[0][0], in_shapes[0][1], new_w, in_shapes[0][-1]]
                            insert_reshape(graph, src, node_name,
                                           in_attr, pre_reshape_dim, key=k,
                                           quantize=node_obj.quantize)

                            src, dst, in_attr = out_edges[0]
                            post_reshape_dim = in_shapes[0]
                            new_out = insert_reshape(
                                graph, node_name, dst, in_attr, post_reshape_dim,
                                quantize=node_obj.quantize)
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
            in_shapes = [e[2]['tensor'].get_shape() for e in in_edges]
            if in_types.count('Constant') == 2:
                meta_ret = False
                ERROR(
                    '[Parser]: broadcast op (%s) with Constant inputs should be fused in broadcast_prelu!' % broadcastop)
            elif len(in_shapes) == 2:
                if in_shapes[0] is not None and in_shapes[1] is not None:
                    if in_shapes[0] and list(in_shapes[0]) == list(in_shapes[1]):
                        pass
                    else:
                        dim_1, dim_2 = len(in_shapes[0]), len(in_shapes[1])
                        if dim_1 == dim_2:
                            if in_shapes[1][0] == 1:
                                reshape_shape = in_shapes[1][1:]
                                insert_reshape(
                                    graph, in_edges[1][0], broadcastop, in_edges[1][2], reshape_shape,
                                    quantize=broadcastop_obj.quantize)
                            else:
                                ERROR(
                                    '[Parser]: Invalid inputs of Node(%s) for broadcasting in broadcast_prelu!' % broadcastop)
                else:
                    meta_ret = False
                    ERROR(
                        '[Parser]: Invalid inputs of Node(%s) for broadcasting in broadcast_prelu!' % broadcastop)
        else:
            ERROR(
                '[Parser]: Meets Invalid broadcast Op (%s) that cannot be converted in broadcast_prelu!' % broadcastop)


def middle_passes(graph, params):
    '''
    Pass is an optimization based on IR to remove redundant operators and perform hardware-friendly operator transformation.
    Among them, middle_pass focuses on operator splitting and merging,
    while back_pass focuses on converting onnx operators into Arm operators defined in IR def.
    '''

    decompose_const_if_loop(graph, params)
    convert_to_const(graph, ['Shape', 'ConstantOfShape',
                             'Range', 'NonZero', 'EyeLike'])

    fuse_const(graph)
    merge_same_op_at_out_port(graph, ['Cast'])

    merge_query_rebatch(graph)
    merge_slot_update(graph)

    # LLM related contributed ops
    convert_simplified_layernorm(graph)
    convert_rotary_embedding(graph)
    convert_mha(graph)
    convert_skip_simplified_layernorm(graph)

    # merge_q_ln(graph)
    merge_q_ln_partial(graph)
    merge_q_gelu(graph)
    convert_qadd(graph)
    convert_qavgpool(graph)
    convert_qconcat(graph)
    convert_qconv(graph)
    convert_qgemm(graph)
    convert_qglobal_avgpool(graph)
    convert_qleakyrelu(graph)
    convert_qmatmul(graph)
    convert_qsigmoid(graph)
    if params.get('force_fp_norm', False):
        convert_qnorm_to_float(graph)
    convert_special_conv_to_mul(graph)

    convert_abnormal_reshape(graph)
    convert_center_crop_pad(graph)
    convert_fill(graph)
    convert_dequantizelinear(graph)
    convert_quantizelinear(graph)
    convert_bn_train(graph)
    clear_useless_concat_input(graph)
    remove_redundant_transpose(graph)
    merge_dilated_conv_group(graph)
    merge_dilated_conv(graph)
    remove_useless_op(graph, ['Concat', 'Pad', 'Pow', 'Sum',
                              'Dropout', 'Expand', 'Reshape', 'Slice', 'Transpose', 'Roll']
                      + OnnxReduceOp.get_concrete_subclass_names())
    remove_redundant_transpose(graph)

    rename_reshape_like(graph)

    merge_multi_matmuls(graph)
    merge_isinf(graph)
    merge_sign_abs_relu(graph)
    split_deformable_conv(graph)
    split_negative_pads(graph)
    split_conv_transpose(graph)
    convert_to_const(graph, ['Concat', 'Mul', 'Shape',
                             'Slice', 'Tile', 'Unsqueeze'])
    convert_multi_outputs_to_const(graph, ['Split', 'Unique'])
    merge_gelu_1(graph)
    merge_gelu_2(graph)
    merge_gelu_3(graph)
    merge_gelu_4(graph)
    merge_gelu_5(graph)

    split_special_bn(graph)
    split_special_gn(graph)
    split_special_ln(graph)
    split_special_ln2(graph)
    split_hardmax(graph)
    split_reduce_logsumexp(graph)
    split_reduce_logsum(graph)
    split_reduce_sumsq(graph)
    split_roll(graph)
    convert_special_pow(graph)
    convert_special_mul(graph)
    convert_upsample_to_resize(graph)
    convert_special_resize(graph)
    convert_multi_scatternd_to_concat(graph)
    convert_special_scatternd(graph)
    convert_special_scatternd2(graph)
    convert_special_cast(graph)
    merge_special_concat_split_concat(graph)
    convert_global_pool(graph)
    # merge_l2pool(graph)
    convert_special_clip_to_relu(graph)
    convert_special_thresholdedrelu_to_relu(graph)
    convert_sigmoid_mul_to_silu(graph)
    convert_sigmoid_mul_to_swish(graph)
    remove_useless_op(graph, ['Cast', 'Concat', 'Identity', 'Pad', 'Slice',
                              'Transpose', 'Reshape', 'AveragePool', 'MaxPool', 'Resize'])
    remove_sub_add_pair(graph)
    merge_divmod(graph)
    merge_divmod2(graph)
    merge_clip(graph)
    merge_leaky_relu(graph)
    merge_prelu(graph)
    reshape_prelu_slope(graph)

    merge_logical_xor(graph)
    merge_erosion(graph)
    merge_reducel1(graph)
    merge_reducel2(graph)
    merge_reducel2_reshape(graph)
    merge_l1norm(graph)
    merge_l2norm(graph)
    merge_l2norm2(graph)
    merge_hardswish(graph)
    merge_hardswish2(graph)
    merge_hardsigmoid(graph)
    merge_hargsigmoid2(graph)
    merge_softplus(graph)
    merge_double_reduce(graph)
    merge_softmax(graph)
    merge_mish(graph)
    merge_batchnorm(graph)
    merge_channel_shuffle(graph)
    merge_channel_shuffle_with_pack(graph)
    deduplicate_mean_sub(graph)
    merge_ln(graph)
    merge_ln2(graph)
    merge_ln3(graph)
    merge_ln4(graph)
    merge_ln5(graph)
    merge_ln6(graph)
    merge_mvn(graph)
    merge_mvn2(graph)
    merge_mvn3(graph)
    merge_mvn4(graph)
    merge_gn(graph)
    merge_gn2(graph)
    merge_gn3(graph)
    merge_in(graph)
    broadcast_ln_weights_biases(graph)
    merge_norm(graph)
    merge_special_div_mul(graph)
    merge_rms_norm(graph)
    merge_rms_norm2(graph)
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
    merge_gather_slice(graph)
    remove_special_gather(graph)
    fuse_gather_const_mul(graph)
    if not params.get('ds_compat', False):
        convert_gather_to_slice(graph)
    for i in range(3):
        lift_single_add_sub_mul_div(graph)
    rearrange_matmul_reshape_bias(graph)
    fuse_bias(graph)
    rename_single_mul_or_add_or_sub(graph)
    rearrange_fc_reshape_bn(graph)
    fuse_pad(graph)
    fuse_linear_bn(graph)
    convert_1d_conv(graph)
    convert_reducemean_to_avgpool(graph)
    convert_1d_pooling(graph)
    convert_div_to_mul(graph)
    remove_redundant_arithmetic(graph)

    decompose_pack(graph)
    remove_useless_op(graph, ['ChannelShuffle', 'Concat', 'Split', 'Slice'])
    rearrange_pack_concat(graph)
    convert_min_max_to_clip(graph)
    remove_redundant_reshape(graph)
    fuse_special_fc_reshape_transpose_div(graph)
    rearrange_linear_reshape_relu(graph)
    rearrange_linear_concat_relu(graph)
    convert_special_transpose(graph)
    merge_meshgrid(graph)

    convert_einsum(graph)
    convert_nms(graph)
    align_matmul_input(graph)
    adjust_1d_matmul(graph)
    adjust_scalar_to_1d(graph)
    adjust_1d_to_4d(graph)
    adjust_2d_to_4d(graph)
    adjust_3d_to_4d(graph)
