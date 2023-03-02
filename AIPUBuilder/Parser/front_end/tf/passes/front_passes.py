# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
import numpy as np
import re
import copy
from ....ops.op import TfOp, Op, OpHasWeights, OpHasPaddingStrides, TfHasPaddingStrides
from ....graph.node_wrap import NodeWrap
from ....graph.graph_algo import get_valid_node_name, clear_redundant_nodes, cal_path_length, has_path
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from ...onnx.passes.common_passes import insert_constant, insert_reshape, insert_reshape_after, \
    insert_transpose, insert_transpose_after, remove_node_safely, insert_cast, place_reshape, \
    insert_cast_after
from ....common.defs import Tensor, FLOAT_EQUAL, INT_MAX
from ....common.utils import extend_lists
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL


def convert_conv_backpropinput(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('const', {'op': ['Constant', 'TfConst']}),
                                   ('conv_back', {
                                       'op': ['TfConv2DBackpropInput', 'TfConv3DBackpropInputV2']})
                               ],
                               edges=[
                                   ('const', 'conv_back', {
                                       'src_out_port': 0, 'dst_in_port': 0})
                               ])
    for m in matches:
        const, conv_back = m['const'], m['conv_back']
        const_obj = NodeWrap(graph, const)['object']
        conv_back_obj = NodeWrap(graph, conv_back)['object']
        in_edges = graph.sorted_in_edges(conv_back, data=True)
        if conv_back_obj is not None and const_obj is not None and len(in_edges) == 2:
            if conv_back_obj.weights is None:
                WARN('[Parser]: TfConv2DBackpropInput/TfConv3DBackpropInputV2 Node(%s) does not contain weights!' %
                     conv_back)
                continue
            in_tensors = conv_back_obj.get_input_tensors()
            if len(in_tensors) < 2 \
                    or in_tensors[1] is None:
                continue

            matched = True
            graph.remove_edges_from(in_edges)

            src, _, in_attr = in_edges[1]
            new_in_attr = copy.deepcopy(in_attr)
            new_in_attr['dst_in_port'] = 0
            graph.add_edge(src, conv_back, **new_in_attr)

            conv_attr = conv_back_obj.copied_attr()
            new_weights = np.transpose(
                conv_back_obj.weights, axes=type(conv_back_obj).perm_tf_to_onnx())
            if conv_back_obj.data_format[:2] == 'NC':
                input_spatial_shape = in_tensors[1].shape[2:]
                output_spatial_shape = const_obj.value.tolist()[2:]
                data_format = 'NCHW'
            else:
                input_spatial_shape = in_tensors[1].shape[1:-1]
                output_spatial_shape = const_obj.value.tolist()[1:-1]
                data_format = 'NHWC'
            # When padding is explictly provided, do not set output_shape so that pads won't
            # be re-calculated in function update_pads.
            if conv_back_obj.auto_pad != 'NOTSET':
                conv_attr.update({'output_shape': output_spatial_shape})
            else:
                full_len = len(input_spatial_shape) + 2
                pad_slice = slice(1, full_len - 1) if data_format == 'NHWC' else slice(2, full_len)
                pads = np.transpose(np.reshape(np.array(conv_back_obj.explicit_paddings),
                                    (full_len, 2))[pad_slice, :]).flatten().tolist()
                conv_attr.update({'pads': pads})
            conv_attr.update({'opset_version': 11,
                              'weights': new_weights,
                              'strides': conv_back_obj.strides,
                              'dilations': conv_back_obj.dilations,
                              'data_format': data_format
                              })
            NodeWrap(graph, conv_back).replace_obj('ConvTranspose', conv_attr)
            NodeWrap(graph, conv_back)['object'].update_pads(input_spatial_shape)
        else:
            WARN(
                '[Parser]: Meets invalid Conv2DBackpropInput/Conv3DBackpropInputV2 Op (%s) in convert_conv_backpropinput!' % conv_back)
    if matched:
        clear_redundant_nodes(graph)


def convert_depth_to_space(graph, op_type='TfDepthToSpace'):
    '''Convert tf DepthToSpace/depth_to_space to onnx DepthToSpace.
    For data format NCHW_VECT_C, convert it to NHWC by inserting
    transpose and reshape nodes before and after onnx DepthToSpace.
    '''
    if op_type not in ('TfDepthToSpace', 'Tfdepth_to_space'):
        WARN('[Parser]: Meets invalid Op type (%s) in convert_depth_to_space!' % op_type)
        return
    matched = False
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        d2s = m['target']
        d2s_obj = NodeWrap(graph, d2s)['object']
        in_edges = graph.sorted_in_edges(d2s, data=True)
        if d2s_obj is None or len(in_edges) < 1:
            WARN('[Parser]: Meets invalid Op (%s) in convert_depth_to_space!' % d2s)
            continue
        if d2s_obj.type == 'Tfdepth_to_space' and len(d2s_obj.sorted_in_consts()) < 2:
            WARN('[Parser]: Meets non-constant block_size or data_format in Tfdepth_to_space Op (%s) in convert_depth_to_space!' % d2s)
            continue
        block_size = d2s_obj.block_size
        data_format = d2s_obj.data_format
        if data_format == 'NCHW_VECT_C':
            input_shapes = d2s_obj.get_input_shapes()
            if len(input_shapes) < 1 \
                    or input_shapes[0] is None \
                    or None in input_shapes[0] \
                    or len(input_shapes[0]) != 5 \
                    or input_shapes[0][-1] != 4:
                WARN('[Parser]: Meets invalid input shape in Tfdepth_to_space Op (%s) in convert_depth_to_space!' % d2s)
                continue
            data_format = 'NHWC'
            input_shape = input_shapes[0]
            batch, channels_div_4, height, width, _ = input_shape
            src, _, in_attr = in_edges[0]
            pre_perm = TfOp.perm_nchw_vect_c_to_nhwc()
            pre_trans = insert_transpose(graph, src, d2s, in_attr, pre_perm)
            pre_trans_out_attr = copy.deepcopy(in_attr)
            pre_trans_out_attr.update({'src_out_port': 0})
            if pre_trans_out_attr['tensor'] is not None:
                val = pre_trans_out_attr['tensor'].value
                pre_trans_out_attr['tensor'].value = None if val is None else np.transpose(val, pre_perm)
            nhwc_shape = [input_shape[idx] for idx in pre_perm[:-1]]
            nhwc_shape[-1] *= input_shape[-1]
            insert_reshape(graph, pre_trans, d2s, pre_trans_out_attr, nhwc_shape, data_format='NHWC')
            post_dim = [batch, height * block_size, width * block_size, -1, 4]
            post_old_dim = [batch, height * block_size, width * block_size,
                            int(channels_div_4 * 4 / (block_size * block_size))]
            post_reshape = insert_reshape_after(graph, d2s, post_dim, post_old_dim)
            post_perm = TfOp.perm_nhwc_to_nchw_vect_c()
            post_trans = insert_transpose_after(graph, post_reshape, post_perm)
            if d2s in graph._attr['output_names']:
                index = graph._attr['output_names'].index(d2s)
                graph._attr['output_names'][index] = post_trans
        matched = True
        graph.remove_edges_from(in_edges[1:])
        new_node_attr = d2s_obj.copied_attr()
        new_node_attr.update({'blocksize': block_size, 'mode': 'DCR',
                              'data_format': data_format, 'opset_version': 13})
        NodeWrap(graph, d2s).replace_obj('DepthToSpace', new_node_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_invert_permutation(graph):
    matches = single_node_matcher(graph, 'TfInvertPermutation')
    for m in matches:
        invperm = m['target']
        invperm_obj = NodeWrap(graph, invperm)['object']
        in_edges = graph.sorted_in_edges(invperm, data=True)
        out_edges = graph.sorted_out_edges(invperm, data=True)
        if invperm_obj is None or len(in_edges) != 1 or len(out_edges) < 1:
            WARN(
                '[Parser]: Meets invalid Node(%s) in convert_invert_permutation!' % invperm)
            continue
        input_shapes = invperm_obj.get_input_shapes()
        if len(input_shapes) < 1 or input_shapes[0] is None or len(input_shapes[0]) != 1:
            continue
        input_num = input_shapes[0][0]
        src, _, in_attr = in_edges[0]
        neg = get_valid_node_name(graph, invperm + '_neg')
        graph.remove_edges_from(in_edges)
        graph.add_edge(src, neg, **in_attr)
        topk_in_attr = copy.deepcopy(in_attr)
        topk_in_attr.update({'src_out_port': 0})
        graph.add_edge(neg, invperm, **topk_in_attr)

        topk_out_val = get_valid_node_name(graph, invperm + '_topk_out_val')
        graph.add_edge(invperm, topk_out_val, **{'src_out_port': 0, 'dst_in_port': 0})
        for _, dst, out_attr in out_edges:
            graph.remove_edge(invperm, dst)
            new_out_attr = copy.deepcopy(out_attr)
            new_out_attr.update({'src_out_port': 1})
            graph.add_edge(invperm, dst, **new_out_attr)

        neg_attr = {'name': neg, 'opset_version': 13}
        NodeWrap(graph, neg).replace_obj('Neg', neg_attr)
        topk_attr = invperm_obj.copied_attr()
        topk_attr.update({'k': input_num, 'opset_version': 1})
        NodeWrap(graph, invperm).replace_obj('TopK', topk_attr)
        NodeWrap(graph, topk_out_val).replace_obj('Out', {'name': topk_out_val})


def convert_matmul(graph):
    need_clear = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('matmul', {
                                    'op': ['TfMatMul', 'TfBatchMatMulV2', 'TfBatchMatMul', 'Tfmatmul']}),
                               ],
                               edges=[])
    for m in matches:
        matmul = m['matmul']
        matmul_obj = NodeWrap(graph, matmul)['object']
        if matmul_obj is None:
            WARN('[Parser]: Meets invalid MatMul Op (%s) in convert_matmul!' % matmul)
            continue
        in_edges = graph.sorted_in_edges(matmul, keys=True, data=True)
        input_shapes = matmul_obj.get_input_shapes()
        output_type = None
        if matmul_obj.type == 'Tfmatmul':
            output_type = matmul_obj.output_type
            if len(in_edges) < 2 or len(input_shapes) < 2 \
                    or in_edges[0][3]['tensor'] is None \
                    or in_edges[0][3]['tensor'].value is None:
                ERROR('[Parser]: Meets invalid inputs of Tfmatmul Op (%s) in convert_matmul!' % matmul)
                continue
        elif len(in_edges) != 2 or len(input_shapes) != 2 \
                or in_edges[0][3]['tensor'] is None \
                or in_edges[0][3]['tensor'].value is None:
            ERROR('[Parser]: Meets invalid inputs of MatMul Op (%s) in convert_matmul!' % matmul)
            continue
        input_type = str(in_edges[0][3]['tensor'].value.dtype)
        if 'complex' in input_type:
            WARN('[Parser]: Meets unsupported complex input type of MatMul Op (%s) in convert_matmul!' % matmul)
            continue
        if input_shapes[0] is None \
                or len(input_shapes[0]) < 2 \
                or input_shapes[1] is None \
                or len(input_shapes[1]) < 2:
            ERROR('[Parser]: Meets invalid MatMul Op (%s) in convert_matmul!' % matmul)
            continue
        if matmul_obj.type == 'TfMatMul':
            transpose_a = matmul_obj.transpose_a
            transpose_b = matmul_obj.transpose_b
        elif matmul_obj.type == 'Tfmatmul':
            need_clear = True
            graph.remove_edges_from(in_edges[2:])
            transpose_a = (matmul_obj.transpose_a or matmul_obj.adjoint_a)
            transpose_b = (matmul_obj.transpose_b or matmul_obj.adjoint_b)
        else:
            transpose_a = matmul_obj.adj_x
            transpose_b = matmul_obj.adj_y
        if transpose_a:
            in_dim1 = len(input_shapes[0])
            perm1 = list(range(in_dim1 - 2)) + [in_dim1 - 1, in_dim1 - 2]
            src1, _, k1, in_attr1 = in_edges[0]
            insert_transpose(graph, src1, matmul, in_attr1, perm1, key=k1)
        if transpose_b:
            in_dim2 = len(input_shapes[1])
            perm2 = list(range(in_dim2 - 2)) + [in_dim2 - 1, in_dim2 - 2]
            src2, _, k2, in_attr2 = in_edges[1]
            insert_transpose(graph, src2, matmul, in_attr2, perm2, key=k2)
        matmul_attr = matmul_obj.copied_attr()
        matmul_attr.update({'opset_version': 9})
        NodeWrap(graph, matmul).replace_obj('MatMul', matmul_attr)
        if input_type is not None and output_type is not None and input_type != output_type:
            post_cast = insert_cast_after(graph, matmul, input_type, output_type)
            if matmul in graph._attr['output_names']:
                index = graph._attr['output_names'].index(matmul)
                graph._attr['output_names'][index] = post_cast
    if need_clear:
        clear_redundant_nodes(graph)


def convert_maxpoolwithargmax(graph, op_type='TfMaxPoolWithArgmax'):
    if op_type not in ('TfMaxPoolWithArgmax', 'Tfmax_pool_with_argmax'):
        ERROR('[Parser]: Meets invalid Op type (%s) in convert_maxpoolwithargmax!' % op_type)
        return
    matched = False
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        argmaxpool = m['target']
        argmaxpool_obj = NodeWrap(graph, argmaxpool)['object']
        in_edges = graph.sorted_in_edges(argmaxpool, data=True)
        out_edges = graph.sorted_out_edges(argmaxpool, keys=True, data=True)
        if argmaxpool_obj is None \
                or len(in_edges) < 1 \
                or len(out_edges) < 1 \
                or len(argmaxpool_obj.get_input_shapes()) < 1 \
                or len(argmaxpool_obj.get_output_shapes()) < 1 \
                or any(s is None for s in argmaxpool_obj.get_input_shapes()[0]) \
                or any(s is None for s in argmaxpool_obj.get_output_shapes()[0]):
            WARN(
                '[Parser]: Meets invalid Node(%s) in convert_maxpoolwithargmax!' % argmaxpool)
            continue
        matched = True
        argmaxpool_obj.update_pads(argmaxpool_obj.get_input_shapes()[0],
                                   argmaxpool_obj.get_output_shapes()[0])
        if not bool(argmaxpool_obj.include_batch_in_index):
            # Convert output indices from NHWC to HWC
            sub = get_valid_node_name(graph, argmaxpool + '_indices_sub')
            graph.add_edge(argmaxpool, sub, **{'src_out_port': 1,
                                               'dst_in_port': 0, 'tensor': out_edges[0][3]['tensor']})
            cast_to_int = get_valid_node_name(
                graph, argmaxpool + '_indices_to_int')
            graph.add_edge(sub, cast_to_int)
            for _, dst, k, out_attr in out_edges:
                if out_attr['src_out_port'] == 1:
                    graph.remove_edge(argmaxpool, dst, key=k)
                    new_out_attr = copy.deepcopy(out_attr)
                    new_out_attr.update({'src_out_port': 0})
                    graph.add_edge(cast_to_int, dst, **new_out_attr)

            input_shapes = argmaxpool_obj.get_input_shapes()
            output_shapes = argmaxpool_obj.get_output_shapes()
            in_n, in_h, in_w, in_c = input_shapes[0]
            _, out_h, out_w, out_c = output_shapes[0]
            sub_oprand = np.reshape(
                np.arange(0, in_n), (in_n, 1, 1, 1)) * in_h * in_w * in_c
            sub_oprand = np.tile(
                sub_oprand, [1, out_h, out_w, out_c]).astype(np.float32)
            insert_constant(graph, sub + '_oprand', sub_oprand, sub, in_port=1)

            NodeWrap(graph, sub).replace_obj(
                'Sub', {'name': sub, 'opset_version': 7})
            NodeWrap(graph, cast_to_int).replace_obj(
                'Cast', {'name': cast_to_int, 'opset_version': 1, 'to': 'int32'})

            if argmaxpool in graph._attr['output_names']:
                index = graph._attr['output_names'].index(argmaxpool)
                graph._attr['output_names'].insert(index, cast_to_int)
        graph.remove_edges_from(in_edges[1:])
        maxpool_attr = argmaxpool_obj.copied_attr()
        maxpool_attr.update({'opset_version': 12})
        NodeWrap(graph, argmaxpool).replace_obj('MaxPool', maxpool_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_nms(graph, params):
    def get_image_size(nms):
        hw_sizes = list()
        for item in ('image_height', 'image_width'):
            size = params.get(
                item, graph._attr['input_tensors'].get(item, None))
            if size is None:
                WARN('[Parser]: %s is not set! Set to default value 300 for NMS Op (%s)!' % (
                    item, nms))
                size = 300
            hw_sizes.append(size)
        return hw_sizes

    nms_output_num_dict = {
        'TfNonMaxSuppressionV3': 1,
        'TfNonMaxSuppressionV4': 2,
        'TfNonMaxSuppressionV5': 3,
        'LiteNON_MAX_SUPPRESSION_V4': 2,
        'LiteNON_MAX_SUPPRESSION_V5': 3
    }
    matched = False
    for nms_type in nms_output_num_dict.keys():
        matches = single_node_matcher(graph, nms_type)
        for m in matches:
            nms = m['target']
            nms_obj = NodeWrap(graph, nms)['object']
            in_edges = graph.sorted_in_edges(nms, keys=True, data=True)
            out_edges = graph.sorted_out_edges(nms, data=True)

            if nms_obj is None or len(in_edges) < 5 or len(nms_obj.get_input_shapes()) < 5 or \
                    len(nms_obj.get_out_ports()) > nms_output_num_dict[nms_type]:
                WARN('[Parser]: Meets invalid Node(%s) in convert_nms!' % nms)
                continue

            matched = True
            in_shapes = nms_obj.get_input_shapes()
            box_num = in_shapes[0][0]
            class_num = 1
            # Get attributes before modifying nms's inputs.
            max_output_size = nms_obj.max_output_size
            iou_threshold = nms_obj.iou_threshold
            score_threshold = nms_obj.score_threshold
            soft_nms_sigma = nms_obj.soft_nms_sigma if nms_type in ('TfNonMaxSuppressionV5',
                                                                    'LiteNON_MAX_SUPPRESSION_V5') else 0
            method = 'HARD' if FLOAT_EQUAL(soft_nms_sigma, 0.) else 'GAUSSIAN'

            # Align with COMPASS inputs: boxes, box_num_per_class, class_num, scores
            # Add reshape node in front of boxes and scores, move scores to input3, and add const nodes.
            in_edges[1][3]['dst_in_port'] = 3
            for idx, in_edge in enumerate(in_edges[:2]):
                src, _, key, in_attr = in_edge
                insert_reshape(graph, src, nms, in_attr,
                               [1] + in_shapes[idx], key)
            graph.remove_edges_from(in_edges[1:])
            insert_constant(graph, 'box_num_per_class', np.array(
                [[box_num]], dtype=np.int32), nms, 1)
            insert_constant(graph, 'class_num', np.array(
                [[class_num]], dtype=np.int32), nms, 2)

            # Comparing with original outputs shape, COMPASS outputs shape expand dims at axis 0.
            # Add reshape node after original outputs before updating src_output_port for nms.
            new_outs = []
            for idx in range(nms_output_num_dict[nms_type]):
                for _, dst, out_attr in out_edges:
                    if out_attr['src_out_port'] == idx:
                        new_outs.append(insert_reshape_after(
                            graph, nms, out_attr['tensor'].value.shape, out_port=idx))
                        break

            # out_edges have been updated after inserting reshape so need to get a new one.
            out_edges = graph.sorted_out_edges(nms, data=True)
            graph.remove_edges_from(out_edges)
            # Align with COMPASS outputs: nms_boxes, nms_box_num_per_class, nms_scores, nms_indices
            out_boxes = get_valid_node_name(graph, nms + '_boxes')
            out_box_num_per_class = get_valid_node_name(
                graph, nms + '_box_num_per_class')
            out_scores = get_valid_node_name(graph, nms + '_scores')

            for _, dst, out_attr in out_edges:
                if out_attr['src_out_port'] == 0:
                    graph.add_edge(nms, dst, **{'src_out_port': 3})
            if nms_type == 'TfNonMaxSuppressionV3':
                # Tf NMSV3 outputs: selected_indices
                graph.add_edge(nms, out_boxes, **{'src_out_port': 0})
                graph.add_edge(nms, out_box_num_per_class,
                               **{'src_out_port': 1})
                NodeWrap(graph, out_box_num_per_class).replace_obj(
                    'Out', {'name': out_box_num_per_class})
                graph.add_edge(nms, out_scores, **{'src_out_port': 2})
                NodeWrap(graph, out_scores).replace_obj(
                    'Out', {'name': out_scores})
            elif nms_type in ('TfNonMaxSuppressionV4', 'LiteNON_MAX_SUPPRESSION_V4'):
                # Tf NMSV4 outputs: selected_indices, valid_outputs(nms_box_num_per_class)
                for _, dst, out_attr in out_edges:
                    if out_attr['src_out_port'] == 1:
                        graph.add_edge(nms, dst, **{'src_out_port': 1})
                graph.add_edge(nms, out_boxes, **{'src_out_port': 0})
                graph.add_edge(nms, out_scores, **{'src_out_port': 2})
                NodeWrap(graph, out_scores).replace_obj(
                    'Out', {'name': out_scores})
            else:
                # Tf NMSV5 outputs: selected_indices, selected_scores, valid_outputs
                for _, dst, out_attr in out_edges:
                    if out_attr['src_out_port'] == 2:
                        graph.add_edge(nms, dst, **{'src_out_port': 1})
                for _, dst, out_attr in out_edges:
                    if out_attr['src_out_port'] == 1:
                        graph.add_edge(nms, dst, **{'src_out_port': 2})
                graph.add_edge(nms, out_boxes, **{'src_out_port': 0})
            NodeWrap(graph, out_boxes).replace_obj('Out', {'name': out_boxes})

            if nms in graph._attr['output_names']:
                index = graph._attr['output_names'].index(nms)
                graph._attr['output_names'].remove(nms)
                for new_out in new_outs:
                    if new_out in graph._attr['output_names']:
                        continue
                    graph._attr['output_names'].insert(index, new_out)
                    index += 1

            height, weight = get_image_size(nms)
            nms_attr = nms_obj.copied_attr()
            nms_attr.update(
                {'image_height': height,
                 'image_width': weight,
                 'max_box_num': max_output_size,
                 'iou_threshold': iou_threshold,
                 'center_point_box': 0,
                 'score_threshold': score_threshold,
                 'soft_nms_sigma': soft_nms_sigma,
                 'method': method})
            NodeWrap(graph, nms).replace_obj('ArmNMS', nms_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_resize_bilinear_nearest(graph):
    matches = extend_lists([single_node_matcher(graph, op_type)
                            for op_type in ['TfResizeBilinear', 'TfResizeNearestNeighbor']])
    for m in matches:
        resize_bili_near = m['target']
        resize_bili_near_obj = NodeWrap(graph, resize_bili_near)['object']
        in_edges = graph.sorted_in_edges(resize_bili_near, data=True)
        if resize_bili_near_obj is not None and len(in_edges) == 2:
            input_tensors = resize_bili_near_obj.get_input_tensors()
            if len(input_tensors) != 2 or input_tensors[0] is None or input_tensors[1] is None or \
                    len(input_tensors[0].shape) != 4 or len(input_tensors[1].shape) != 1 or \
                    input_tensors[1].size != 2:
                ERROR(
                    '[Parser]: Meets invalid inputs for Op (%s) in convert_resize_bilinear_nearest!' % resize_bili_near)
                continue

            graph.remove_edges_from(in_edges[1:])
            # insert constant roi
            insert_constant(graph, resize_bili_near + '_roi',
                            np.array([], np.float32), resize_bili_near, in_port=1)
            size_value = [input_tensors[0].shape[0], input_tensors[1][0],
                          input_tensors[1][1], input_tensors[0].shape[-1]]
            # insert constant empty scale
            insert_constant(graph, resize_bili_near + '_scale',
                            np.array([], np.float32), resize_bili_near, in_port=2)
            # insert constant size
            insert_constant(graph, resize_bili_near + '_size',
                            np.array(size_value, np.int64), resize_bili_near, in_port=3)
            mode = 'linear' if resize_bili_near_obj.type == 'TfResizeBilinear' else 'nearest'
            if resize_bili_near_obj.align_corners:
                nearest_mode = 'round_prefer_ceil'
                transform_mode = 'align_corners'
            else:
                nearest_mode = 'floor'
                if resize_bili_near_obj.half_pixel_centers:
                    if mode == 'nearest':
                        transform_mode = 'tf_half_pixel_for_nn'
                    else:
                        transform_mode = 'half_pixel'
                else:
                    transform_mode = 'asymmetric'
            resize_attr = resize_bili_near_obj.copied_attr()
            resize_attr.update(
                {'opset_version': 11, 'coordinate_transformation_mode': transform_mode, 'mode': mode, 'nearest_mode': nearest_mode})
            NodeWrap(graph, resize_bili_near).replace_obj(
                'Resize', resize_attr)
        else:
            ERROR(
                '[Parser]: Meets invalid Op (%s) in convert_resize_bilinear_nearest!' % resize_bili_near)


def convert_onehot(graph, op_type='Tfone_hot'):
    # need support Tf1/Tflite onehot
    if op_type not in ('Tfone_hot', 'TfOneHot'):
        ERROR('[Parser]: Meets invalid Op type (%s) in convert_onehot!' % op_type)
        return
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        onehot = m['target']
        onehot_obj = NodeWrap(graph, onehot)['object']
        in_edges = graph.sorted_in_edges(onehot, data=True)
        if onehot_obj is None or len(in_edges) < 4:
            ERROR(
                '[Parser]: Meets invalid Node(%s) in convert_onehot!' % argmaxpool)
            continue
        if not in_edges[2][2]['tensor'].is_const \
                or not in_edges[3][2]['tensor'].is_const:
            continue

        if len(onehot_obj.get_input_shapes()) < 1 \
                or len(onehot_obj.get_output_shapes()) < 1 \
                or len(onehot_obj.get_input_tensors()) < 1:
            ERROR(
                '[Parser]: Meets invaild Node for Onehot Op(%s)' % onehot)
            continue

        # about onehot with scalar
        indices = onehot_obj.get_input_tensors()[0]
        indices_shape = onehot_obj.get_input_shapes()[0]
        out_shapes_0 = onehot_obj.get_output_shapes()[0]
        if np.ndim(indices) == 0 and (any(d is None for d in indices_shape)
                                      or any(d is None for d in out_shapes_0)):
            ERROR(
                '[Parser]: Meets invalid Node for Onehot Op(%s)' % onehot)
            continue

        graph.remove_edges_from(in_edges[2:])
        values = np.array([onehot_obj.off_value, onehot_obj.on_value])
        insert_constant(graph, onehot + '_values',
                        values, onehot, in_port=2)

        if np.ndim(indices) == 0:
            in_edges = graph.sorted_in_edges(
                onehot, keys=True, data=True)
            src, _, k, in_attr = in_edges[0]
            pre_reshape_dim = [1] + indices_shape
            insert_reshape(graph, src, onehot,
                           in_attr, pre_reshape_dim, key=k)
            post_reshape = insert_reshape_after(
                graph, onehot, out_shapes_0)
            if onehot in graph._attr['output_names']:
                index = graph._attr['output_names'].index(onehot)
                graph._attr['output_names'][index] = post_reshape

        new_node_attr = onehot_obj.copied_attr()
        new_node_attr.update({'opset_version': 11})
        NodeWrap(graph, onehot).replace_obj('OneHot', new_node_attr)


def convert_reverse(graph, op_type='TfReverseV2'):
    if op_type not in ('TfReverseV2', 'LiteREVERSE_V2'):
        ERROR('[Parser]: Meets invalid Op type (%s) in convert_reverse!' % op_type)
        return
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        rev = m['target']
        rev_obj = NodeWrap(graph, rev)['object']
        in_edges = graph.sorted_in_edges(rev, data=True)
        if rev_obj is None or len(in_edges) < 1 \
                or len(rev_obj.get_input_shapes()) < 1:
            ERROR(
                '[Parser]: Meets invalid Op (%s) in convert_reverse!' % rev)
            continue
        input_shape = rev_obj.get_input_shapes()[0]
        if input_shape is None \
                or any(d is None for d in input_shape) \
                or len(rev_obj.axes) != 1:
            continue
        out_node = rev
        src, _, in_attr = in_edges[0]
        rev_axis = (rev_obj.axes[0] + len(input_shape)) if rev_obj.axes[0] < 0 else rev_obj.axes[0]
        if len(input_shape) < 2:
            insert_reshape(graph, src, rev, in_attr, [-1, 1])
            out_node = insert_reshape_after(graph, rev, input_shape, input_shape + [1])
            time_axis = 0
            batch = 1
            seq_length = input_shape[0]
        elif rev_axis in (0, 1):
            time_axis = rev_axis
            batch = input_shape[1 - time_axis]
            seq_length = input_shape[time_axis]
        else:
            pre_perm = [rev_axis] + [idx for idx in range(len(input_shape)) if idx != rev_axis]
            insert_transpose(graph, src, rev, in_attr, pre_perm)
            post_perm = Op.cal_inverse_perm(pre_perm)
            out_node = insert_transpose_after(graph, rev, post_perm)
            time_axis = 0
            batch = input_shape[0]
            seq_length = input_shape[rev_axis]
        seq_len = np.array([seq_length] * batch, np.int32)
        graph.remove_edges_from(in_edges[1:])
        insert_constant(graph, rev + '_seq_len', seq_len, rev, in_port=1)
        rev_seq_attr = rev_obj.copied_attr()
        rev_seq_attr.update({'time_axis': time_axis, 'batch_axis': 1 - time_axis,
                             'opset_version': 10})
        NodeWrap(graph, rev).replace_obj('ReverseSequence', rev_seq_attr)
        if rev in graph._attr['output_names'] and out_node != rev:
            index = graph._attr['output_names'].index(rev)
            graph._attr['output_names'][index] = out_node


def remove_identity_n(graph):
    matched = False
    matches = single_node_matcher(graph, 'TfIdentityN')
    for m in matches:
        identity_n = m['target']
        identity_n_obj = NodeWrap(graph, identity_n)['object']
        in_edges = graph.sorted_in_edges(identity_n, data=True)

        if identity_n_obj is not None and len(in_edges) >= 1:
            matched = True
            out_ports = identity_n_obj.get_out_ports()
            out_edges = graph.sorted_out_edges(identity_n, data=True)
            graph.remove_edges_from(in_edges)

            identity_n_src = []
            for src, _, in_attr in in_edges:
                if src not in identity_n_src:
                    identity_n_src.append(src)
                in_port = in_attr['dst_in_port']
                if in_port in out_ports:
                    for _, dst, out_attr in out_edges:
                        if out_attr['src_out_port'] == in_attr['dst_in_port']:
                            new_attr = copy.deepcopy(in_attr)
                            new_attr.update(
                                {'dst_in_port': out_attr['dst_in_port']})
                            graph.remove_edge(identity_n, dst)
                            graph.add_edge(src, dst, **new_attr)
                else:
                    src_out_port = in_attr['src_out_port']
                    src_out_edges = graph.sorted_out_edges(src, data=True)
                    if len([src_out_edge for src_out_edge in src_out_edges
                            if src_out_edge[2]['src_out_port'] == src_out_port]) == 0:
                        out_op_name = get_valid_node_name(
                            graph, src + '_out_' + str(in_attr['src_out_port']))
                        new_in_attr = copy.deepcopy(in_attr)
                        new_in_attr['dst_in_port'] = 0
                        graph.add_edge(src, out_op_name, **new_in_attr)
                        NodeWrap(graph, out_op_name).replace_obj(
                            'Out', {'name': out_op_name})

            if identity_n in graph._attr['output_names']:
                index = graph._attr['output_names'].index(
                    identity_n)
                graph._attr['output_names'].remove(
                    identity_n)
                for new_out in identity_n_src:
                    if new_out not in graph._attr['output_names']:
                        graph._attr['output_names'].insert(
                            index, new_out)
                        index += 1

    if matched:
        clear_redundant_nodes(graph)


def remove_switch(graph):
    matched = False
    matches = single_node_matcher(graph, 'TfSwitch')
    for m in matches:
        switch = m['target']
        switch_obj = NodeWrap(graph, switch)['object']
        switch_in_edges = graph.sorted_in_edges(switch, data=True)
        if switch_obj is None or len(switch_in_edges) != 2:
            ERROR('[Parser]: Meets invalid Node(%s) in remove_switch!' % switch)
            continue
        data_src, _, data_in_attr = switch_in_edges[0]
        _, _, pred_in_attr = switch_in_edges[1]
        if pred_in_attr.get('tensor', None) is None or \
                not pred_in_attr['tensor'].is_const:
            WARN(
                '[Parser]: Meets unsupported non-constant pre of Switch Node(%s) in remove_switch!' % switch)
            continue
        matched = True
        condition = pred_in_attr['tensor'].value
        valid_out_port = 1 if condition else 0
        invalid_nodes = []
        switch_out_edges = graph.sorted_out_edges(switch, keys=True, data=True)
        for _, dst, k, out_attr in switch_out_edges:
            graph.remove_edge(switch, dst, key=k)
            if out_attr['src_out_port'] == valid_out_port:
                new_attr = copy.deepcopy(data_in_attr)
                new_attr.update({'dst_in_port': out_attr['dst_in_port']})
                graph.add_edge(data_src, dst, **new_attr)
        graph.remove_edges_from(switch_in_edges)
    if matched:
        clear_redundant_nodes(graph)


def remove_merge(graph):
    matched = False
    matches = single_node_matcher(graph, 'TfMerge')
    for m in matches:
        merge = m['target']
        merge_obj = NodeWrap(graph, merge)['object']
        merge_in_edges = graph.sorted_in_edges(merge, data=True)
        if merge_obj is None or merge_obj.value_index is None or \
                len(merge_in_edges) < merge_obj.value_index:
            ERROR('[Parser]: Meets invalid Node(%s) in remove_merge!' % merge)
            continue
        matched = True
        src, _, in_attr = merge_in_edges[merge_obj.value_index]
        for _, dst, out_attr in graph.sorted_out_edges(merge, data=True):
            graph.remove_edge(merge, dst)
            new_out_attr = copy.deepcopy(in_attr)
            new_out_attr.update({'dst_in_port': out_attr['dst_in_port']})
            graph.add_edge(src, dst, **new_out_attr)
        graph.remove_edge(src, merge)
    if matched:
        clear_redundant_nodes(graph)


def remove_isfinite_select(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('is_finite', {'op': 'TfIsFinite'}),
                                      ('zeros_like', {}),
                                      ('select', {'op': 'TfSelect'}),
                                      ],
                               edges=[('is_finite', 'select', {'dst_in_port': 0}),
                                      ('zeros_like', 'select',
                                       {'dst_in_port': 2})
                                      ]
                               )
    for m in matches:
        names = ['is_finite', 'zeros_like', 'select']
        obj_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any([obj is None for obj in obj_dict.values()]):
            ERROR('[Parser]: Meets invalid Op in remove_isfinite_select!')
            continue
        is_finite_in_edges = graph.sorted_in_edges(
            m['is_finite'], keys=True, data=True)
        zeros_like_in_edges = graph.sorted_in_edges(
            m['zeros_like'], keys=True, data=True)
        select_in_edges = graph.sorted_in_edges(
            m['select'], keys=True, data=True)
        if len(is_finite_in_edges) != 1 \
                or len(zeros_like_in_edges) != 1 \
                or len(select_in_edges) != 3:
            continue
        is_finite_src, _, k1, in_attr1 = is_finite_in_edges[0]
        zeros_like_src, _, k2, in_attr2 = zeros_like_in_edges[0]
        select_src, _, k3, in_attr3 = select_in_edges[1]
        if is_finite_src != zeros_like_src \
                or is_finite_src != select_src \
                or in_attr1['src_out_port'] != in_attr2['src_out_port'] \
                or in_attr1['src_out_port'] != in_attr3['src_out_port'] \
                or in_attr3['dst_in_port'] != 1:
            continue
        is_finite_out_tensor = obj_dict['is_finite'].get_output_tensors()[0]
        if is_finite_out_tensor is None \
                or not np.all(is_finite_out_tensor):
            continue
        matched = True
        src = is_finite_src
        src_out_port = in_attr1['src_out_port']
        graph.remove_edge(src, m['is_finite'], key=k1)
        graph.remove_edge(src, m['zeros_like'], key=k2)
        graph.remove_edge(src, m['select'], key=k3)
        for _, dst, out_attr in graph.sorted_out_edges(m['select'], data=True):
            graph.remove_edge(m['select'], dst)
            new_out_attr = copy.deepcopy(out_attr)
            new_out_attr['src_out_port'] = src_out_port
            graph.add_edge(src, dst, **new_out_attr)
        if m['select'] in graph._attr['output_names']:
            index = graph._attr['output_names'].index(m['select'])
            if src not in graph._attr['output_names']:
                graph._attr['output_names'][index] = src
            else:
                graph._attr['output_names'].pop(index)
    if matched:
        clear_redundant_nodes(graph)


def convert_special_fakequantminmaxvars(graph):
    matched = False
    matches = single_node_matcher(graph, 'TfFakeQuantWithMinMaxVars')
    for m in matches:
        fake_quant = m['target']
        fake_quant_obj = NodeWrap(graph, fake_quant)['object']
        fake_quant_in_edges = graph.sorted_in_edges(fake_quant, data=True)
        if fake_quant_obj is not None \
                and len(fake_quant_in_edges) == 3 \
                and len(fake_quant_obj.get_input_tensors()) == 3 \
                and all([inp is not None for inp in fake_quant_obj.get_input_tensors()]) \
                and fake_quant_in_edges[1][2]['tensor'].is_const \
                and fake_quant_in_edges[2][2]['tensor'].is_const:
            inputs, min_val, max_val = fake_quant_obj.get_input_tensors()
            if np.ndim(min_val) in (0, 1) and np.ndim(max_val) in (0, 1):
                matched = True
                graph.remove_edges_from(fake_quant_in_edges[1:])
                fake_quant_attr = fake_quant_obj.copied_attr()
                fake_quant_attr.update({'min_val': float(min_val),
                                        'max_val': float(max_val),
                                        })
                NodeWrap(graph, fake_quant).replace_obj(
                    'ArmFakeQuantWithMinMaxVars', fake_quant_attr)
        else:
            ERROR(
                '[Parser]: Meets invalid Node(%s) in remove_special_fakequantminmaxvars!'
                % (fake_quant))
    if matched:
        clear_redundant_nodes(graph)


def convert_fusebatchnormv3(graph):
    matched = False
    matches = single_node_matcher(graph, 'TfFusedBatchNormV3')
    for m in matches:
        fusebnv3 = m['target']
        fusebnv3_obj = NodeWrap(graph, fusebnv3)['object']
        fusebnv3_in_edges = graph.sorted_in_edges(fusebnv3, data=True)
        if fusebnv3_obj is not None\
                and len(fusebnv3_in_edges) == 5 \
                and len(fusebnv3_obj.get_input_tensors()) == 5 \
                and all(inp is not None for inp in fusebnv3_obj.get_input_tensors()):
            if not all(fusebnv3_in_edges[idx][2]['tensor'].is_const for idx in range(1, 5)):
                WARN('[Parser]: Meets unsupported non-constant inputs(1-4) of Node(%s) in convert_fusebatchnormv3!' % fusebnv3)
                continue
            if fusebnv3_obj.is_training \
                    or fusebnv3_in_edges[3][2]['tensor'].value.size != 0 \
                    or fusebnv3_in_edges[4][2]['tensor'].value.size != 0:
                continue
            matched = True
            num_output = fusebnv3_obj.get_input_tensors()[1].shape
            new_mean = np.zeros(num_output, np.float32)
            new_var = np.ones(num_output, np.float32)
            graph.remove_edges_from(fusebnv3_in_edges[3:])
            fusebnv3_attr = fusebnv3_obj.copied_attr()
            insert_constant(graph, fusebnv3 + '_mean',
                            new_mean, fusebnv3, in_port=3, data_format='NHWC')
            insert_constant(graph, fusebnv3 + '_var',
                            new_var, fusebnv3, in_port=4, data_format='NHWC')
            fusebnv3_attr.update({'opset_version': 9})
            NodeWrap(graph, fusebnv3).replace_obj(
                'BatchNormalization', fusebnv3_attr)
        else:
            ERROR(
                '[Parser]: Meets invalid Node(%s) in convert_fusebatchnormv3!'
                % (fusebnv3))
    if matched:
        clear_redundant_nodes(graph)


def split_s2b(graph):
    pad_version, transpose_version, s2d_version, reshape_version = 2, 1, 1, 5
    matches = single_node_matcher(graph, 'TfSpaceToBatchND')
    for m in matches:
        s2b = m['target']
        s2b_obj = NodeWrap(graph, s2b)['object']
        in_edges = graph.sorted_in_edges(s2b, data=True)
        out_edges = graph.sorted_out_edges(s2b, data=True)
        if s2b_obj is not None and len(in_edges) >= 1 and len(out_edges) >= 1:
            block_shape, paddings = [c[2] for c in s2b_obj.sorted_in_consts()]
            in_shape = s2b_obj.get_input_shapes()[0]
            if in_shape is None or None in in_shape:
                continue
            if block_shape is None or len(in_shape) not in (3, 4) \
                    or (len(in_shape) == 3 and block_shape.size != 1) \
                    or (len(in_shape) == 4 and block_shape.size != 2):
                WARN(
                    '[Parser]: Only support 4D inputs(block_shape shape=[2]) or 3D inputs(block_shape shape=[1])'
                    ' for TfSpaceToBatchND Op (%s) for now!' % s2b)
                continue
            pads = OpHasPaddingStrides.tf_to_onnx(paddings, as_full=True)
            half_pads_len = int(len(pads) / 2)
            full_pads = [0] + pads[:half_pads_len] + [0, 0] + pads[half_pads_len:] + [0]
            is_4d_input = (len(in_shape) == 4)
            if is_4d_input and block_shape[0] == block_shape[1]:
                block_size = block_shape[0]
                pad = get_valid_node_name(graph, s2b + '_pad')
                trans1 = get_valid_node_name(graph, s2b + '_transpose1')
                trans2 = get_valid_node_name(graph, s2b + '_transpose2')

                graph.remove_edges_from(in_edges[1:])
                for src, _, in_attr in in_edges[:1]:
                    graph.remove_edge(src, s2b)
                    graph.add_edge(src, pad, **in_attr)
                graph.add_edges_from(
                    [(pad, trans1), (trans1, s2b), (s2b, trans2)])
                for _, dst, out_attr in out_edges:
                    graph.remove_edge(s2b, dst)
                    graph.add_edge(trans2, dst, **out_attr)

                pad_attr = {'name': pad,
                            'opset_version': pad_version, 'pads': full_pads}
                trans1_attr = {
                    'name': trans1, 'opset_version': transpose_version, 'perm': [3, 1, 2, 0]}
                s2d_attr = {'name': s2b, 'opset_version': s2d_version,
                            'blocksize': block_size}
                trans2_attr = {
                    'name': trans2, 'opset_version': transpose_version, 'perm': [3, 1, 2, 0]}
                NodeWrap(graph, pad).replace_obj('Pad', pad_attr)
                NodeWrap(graph, trans1).replace_obj('Transpose', trans1_attr)
                NodeWrap(graph, s2b).replace_obj('SpaceToDepth', s2d_attr)
                NodeWrap(graph, trans2).replace_obj('Transpose', trans2_attr)
                last_name = trans2
            else:
                need_pad = np.any(paddings != 0)
                if is_4d_input:
                    block_size_y, block_size_x = block_shape.tolist()
                    padded_in_shape = (np.array(in_shape, np.int64) + np.array(
                        [0, np.sum(paddings[0, :]), np.sum(paddings[1, :]), 0], np.int64)).tolist()
                    dim1 = [padded_in_shape[0],
                            padded_in_shape[1] // block_size_y,
                            block_size_y,
                            padded_in_shape[2] // block_size_x,
                            block_size_x,
                            padded_in_shape[-1]]
                    dim2 = [padded_in_shape[0] * block_size_y * block_size_x,
                            padded_in_shape[1] // block_size_y,
                            padded_in_shape[2] // block_size_x,
                            padded_in_shape[-1]]
                    trans_perm = [2, 4, 0, 1, 3, 5]
                else:
                    block_size = block_shape.item()
                    padded_in_shape = (np.array(in_shape, np.int64) + np.array(
                        [0, np.sum(paddings[0, :]), 0], np.int64)).tolist()
                    dim1 = [padded_in_shape[0],
                            padded_in_shape[1] // block_size,
                            block_size,
                            padded_in_shape[-1]]
                    dim2 = [padded_in_shape[0] * block_size,
                            padded_in_shape[1] // block_size,
                            padded_in_shape[-1]]
                    trans_perm = [2, 0, 1, 3]

                pad = get_valid_node_name(graph, s2b + '_pad')
                reshape1 = get_valid_node_name(graph, s2b + '_reshape1')
                reshape2 = get_valid_node_name(graph, s2b + '_reshape2')
                begin_name = pad if need_pad else reshape1

                graph.remove_edges_from(in_edges[1:])
                for src, _, in_attr in in_edges[:1]:
                    graph.remove_edge(src, s2b)
                    graph.add_edge(src, begin_name, **in_attr)
                graph.add_edges_from(
                    ([(pad, reshape1)] if need_pad else []) + [(reshape1, s2b), (s2b, reshape2)])
                for _, dst, out_attr in out_edges:
                    graph.remove_edge(s2b, dst)
                    graph.add_edge(reshape2, dst, **out_attr)

                pad_attr = {'name': pad,
                            'opset_version': pad_version, 'pads': full_pads}
                reshape1_attr = {'name': reshape1,
                                 'opset_version': reshape_version}
                transpose_attr = s2b_obj.copied_attr()
                transpose_attr.update(
                    {'opset_version': transpose_version, 'perm': trans_perm})
                reshape2_attr = {'name': reshape2,
                                 'opset_version': reshape_version}

                if need_pad:
                    NodeWrap(graph, pad).replace_obj('Pad', pad_attr)
                NodeWrap(graph, reshape1).replace_obj('Reshape', reshape1_attr)
                insert_constant(graph, reshape1 + '_shape', np.array(dim1,
                                                                     np.int64), reshape1, in_port=1, data_format='NHWC')
                NodeWrap(graph, s2b).replace_obj('Transpose', transpose_attr)
                NodeWrap(graph, reshape2).replace_obj('Reshape', reshape2_attr)
                insert_constant(graph, reshape2 + '_shape', np.array(dim2,
                                                                     np.int64), reshape2, in_port=1, data_format='NHWC')
                last_name = reshape2

            if s2b in graph._attr['output_names']:
                index = graph._attr['output_names'].index(s2b)
                graph._attr['output_names'][index] = last_name
        else:
            ERROR(
                '[Parser]: Meets invalid TfSpaceToBatchND Node(%s) in split_s2b!' % s2b)


def split_b2s(graph, op_type='TfBatchToSpaceND'):
    if op_type not in ('TfBatchToSpaceND', 'Tfbatch_to_space_nd'):
        ERROR('[Parser]: Meets invalid Op type (%s) in split_b2s!' % op_type)
        return
    transpose_version, d2s_version, slice_version, reshape_version = 1, 1, 1, 5
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        b2s = m['target']
        b2s_obj = NodeWrap(graph, b2s)['object']
        if b2s_obj is None:
            ERROR('[Parser]: Meets invalid TfBatchToSpaceND Node(%s) in split_b2s!' % b2s)
            continue
        in_edges = graph.sorted_in_edges(b2s, data=True)
        out_edges = graph.sorted_out_edges(b2s, data=True)
        if len(in_edges) < 3 or len(out_edges) < 1:
            continue
        input_shapes = b2s_obj.get_input_shapes()
        in_consts = b2s_obj.sorted_in_consts()
        if len(input_shapes) >= 3 \
                and input_shapes[0] is not None \
                and len(input_shapes[0]) in (3, 4) \
                and all(s is not None for s in input_shapes[0]) \
                and len(in_consts) >= 2 \
                and [c[1] for c in in_consts[:2]] == [1, 2]:
            in_shape = input_shapes[0]
            block_shape, crops = [c[2] for c in in_consts[:2]]
            is_4d = len(in_shape) == 4
            need_slice = np.any(crops != 0)
            spatial_in_shape = in_shape[1:-1]
            spatial_out_shape_before_slice = (np.array(spatial_in_shape) * block_shape).tolist()
            slice = get_valid_node_name(graph, b2s + '_slice')
            last = ''

            if is_4d and block_shape[0] == block_shape[1]:
                block_size = block_shape[0]
                trans1 = get_valid_node_name(graph, b2s + '_transpose1')
                trans2 = get_valid_node_name(graph, b2s + '_transpose2')
                last = slice if need_slice else trans2

                src, _, in_attr = in_edges[0]
                graph.remove_edges_from(in_edges)
                graph.add_edge(src, trans1, **in_attr)
                graph.add_edge(trans1, b2s)
                for _, dst, out_attr in out_edges:
                    graph.remove_edge(b2s, dst)
                    graph.add_edge(last, dst, **out_attr)
                graph.add_edge(b2s, trans2)
                if need_slice:
                    graph.add_edge(trans2, slice)

                trans1_attr = {'name': trans1, 'opset_version': transpose_version, 'perm': [3, 1, 2, 0]}
                NodeWrap(graph, trans1).replace_obj('Transpose', trans1_attr)
                d2s_attr = {'name': b2s, 'opset_version': d2s_version, 'blocksize': block_size}
                NodeWrap(graph, b2s).replace_obj('DepthToSpace', d2s_attr)
                trans2_attr = {'name': trans2, 'opset_version': transpose_version, 'perm': [3, 1, 2, 0]}
                NodeWrap(graph, trans2).replace_obj('Transpose', trans2_attr)

            else:
                dim1 = [*block_shape.tolist(), -1] + list(in_shape[1:])
                perm = [2, 3, 0, 4, 1, 5] if is_4d else [1, 2, 0, 3]
                dim2 = [-1] + spatial_out_shape_before_slice + [in_shape[-1]]

                reshape1 = get_valid_node_name(graph, b2s + '_reshape1')
                reshape2 = get_valid_node_name(graph, b2s + '_reshape2')
                last = slice if need_slice else reshape2

                src, _, in_attr = in_edges[0]
                graph.remove_edges_from(in_edges)
                graph.add_edge(src, reshape1, **in_attr)
                graph.add_edge(reshape1, b2s)
                for _, dst, out_attr in out_edges:
                    graph.remove_edge(b2s, dst)
                    graph.add_edge(last, dst, **out_attr)
                graph.add_edge(b2s, reshape2)
                if need_slice:
                    graph.add_edge(reshape2, slice)

                NodeWrap(graph, reshape1).replace_obj('Reshape',
                                                      {'name': reshape1,
                                                       'opset_version': reshape_version}
                                                      )
                insert_constant(graph,
                                reshape1 + '_shape',
                                np.array(dim1, np.int64),
                                reshape1,
                                in_port=1,
                                data_format='NHWC')

                transpose_attr = b2s_obj.copied_attr()
                transpose_attr.update({'opset_version': transpose_version, 'perm': perm})
                NodeWrap(graph, b2s).replace_obj('Transpose', transpose_attr)

                NodeWrap(graph, reshape2).replace_obj('Reshape',
                                                      {'name': reshape2,
                                                       'opset_version': reshape_version}
                                                      )
                insert_constant(graph,
                                reshape2 + '_shape',
                                np.array(dim2, np.int64),
                                reshape2,
                                in_port=1,
                                data_format='NHWC')

            if need_slice:
                starts = crops[:, 0]
                ends = -crops[:, 1]
                zero_mask = ends == 0
                ends[zero_mask] = np.array(spatial_out_shape_before_slice)[zero_mask]
                slice_attr = {'name': slice,
                              'opset_version': slice_version,
                              'axes': [1, 2] if is_4d else [1],
                              'starts': starts.tolist(),
                              'ends': ends.tolist(),
                              }
                NodeWrap(graph, slice).replace_obj('Slice', slice_attr)

            if b2s in graph._attr['output_names']:
                index = graph._attr['output_names'].index(b2s)
                graph._attr['output_names'][index] = last


def split_special_floormod(graph, op_type='TfFloorMod'):
    if op_type not in ('TfFloorMod', 'LiteFLOOR_MOD'):
        ERROR('[Parser]: Meets invalid Op type (%s) in split_special_floormod!' % op_type)
        return
    matches = single_node_matcher(graph, op_type)
    for m in matches:
        floor_mod = m['target']
        floor_mod_obj = NodeWrap(graph, floor_mod)['object']
        if floor_mod_obj is not None:
            in_edges = graph.sorted_in_edges(floor_mod, data=True)
            inputs = floor_mod_obj.get_input_tensors()
            if len(in_edges) != 2 \
                    or len(inputs) != 2 \
                    or inputs[0] is None \
                    or inputs[1] is None \
                    or str(inputs[0].dtype) != str(inputs[1].dtype):
                ERROR(
                    '[Parser]: Meets invalid inputs for Node (%s) in split_special_floormod!' % floor_mod)
                continue

            if 'float' in str(inputs[0].dtype):
                y, _, y_in_attr = in_edges[1]
                zero_value = np.zeros_like(inputs[0])
                zero = get_valid_node_name(graph, floor_mod + '_zero')
                mod_add_y = get_valid_node_name(
                    graph, floor_mod + '_mod_add_y')
                trunc_mod_equal = get_valid_node_name(
                    graph, floor_mod + '_trunc_equal')
                trunc_mod_equal_not = get_valid_node_name(
                    graph, floor_mod + '_trunc_equal_not')
                trunc_mod_less_zero = get_valid_node_name(
                    graph, floor_mod + '_trunc_less_zero')
                y_less_zero = get_valid_node_name(
                    graph, floor_mod + '_y_less_zero')
                less_zero_equal = get_valid_node_name(
                    graph, floor_mod + '_less_zero_equal')
                less_zero_equal_not = get_valid_node_name(
                    graph, floor_mod + '_less_zero_equal_not')
                logical_and = get_valid_node_name(graph, floor_mod + '_and')
                where = get_valid_node_name(graph, floor_mod + '_where')

                for _, dst, out_attr in graph.sorted_out_edges(floor_mod, data=True):
                    graph.remove_edge(floor_mod, dst)
                    graph.add_edge(where, dst, **out_attr)

                graph.add_edge(floor_mod, trunc_mod_equal)
                graph.add_edge(zero, trunc_mod_equal, **{'dst_in_port': 1})
                graph.add_edge(trunc_mod_equal, trunc_mod_equal_not)

                graph.add_edge(floor_mod, trunc_mod_less_zero)
                graph.add_edge(zero, trunc_mod_less_zero, **{'dst_in_port': 1})
                new_y_in_attr = copy.deepcopy(y_in_attr)
                new_y_in_attr['dst_in_port'] = 0
                graph.add_edge(y, y_less_zero, **new_y_in_attr)
                graph.add_edge(zero, y_less_zero, **{'dst_in_port': 1})

                graph.add_edge(trunc_mod_less_zero, less_zero_equal)
                graph.add_edge(y_less_zero, less_zero_equal,
                               **{'dst_in_port': 1})
                graph.add_edge(less_zero_equal, less_zero_equal_not)

                graph.add_edge(less_zero_equal_not, logical_and)
                graph.add_edge(trunc_mod_equal_not, logical_and,
                               **{'dst_in_port': 1})

                graph.add_edge(floor_mod, mod_add_y)
                graph.add_edge(y, mod_add_y, **y_in_attr)

                graph.add_edge(logical_and, where)
                graph.add_edge(mod_add_y, where, **{'dst_in_port': 1})
                graph.add_edge(floor_mod, where, **{'dst_in_port': 2})

                NodeWrap(graph, floor_mod).replace_obj(
                    'Mod', {'name': floor_mod, 'opset_version': 13, 'fmod': 1})
                NodeWrap(graph, zero).replace_obj('Constant', {
                    'name': zero, 'opset_version': 9, 'value': zero_value})
                NodeWrap(graph, mod_add_y).replace_obj(
                    'Add', {'name': mod_add_y, 'opset_version': 7})
                NodeWrap(graph, trunc_mod_equal).replace_obj(
                    'Equal', {'name': trunc_mod_equal, 'opset_version': 13})
                NodeWrap(graph, trunc_mod_equal_not).replace_obj(
                    'Not', {'name': trunc_mod_equal_not, 'opset_version': 1})
                NodeWrap(graph, trunc_mod_less_zero).replace_obj(
                    'Less', {'name': trunc_mod_less_zero, 'opset_version': 13})
                NodeWrap(graph, y_less_zero).replace_obj(
                    'Less', {'name': y_less_zero, 'opset_version': 13})
                NodeWrap(graph, less_zero_equal).replace_obj(
                    'Equal', {'name': less_zero_equal, 'opset_version': 13})
                NodeWrap(graph, less_zero_equal_not).replace_obj(
                    'Not', {'name': less_zero_equal_not, 'opset_version': 1})
                NodeWrap(graph, logical_and).replace_obj(
                    'And', {'name': logical_and, 'opset_version': 7})
                NodeWrap(graph, where).replace_obj(
                    'Where', {'name': where, 'opset_version': 9})

                if floor_mod in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(floor_mod)
                    graph._attr['output_names'][index] = where
        else:
            ERROR('[Parser]: Meets invalid %s Node(%s) in split_special_floormod!' % (
                op_type, floor_mod))


def merge_gru(graph):

    cell_matches = matched_patterns(graph,
                                    nodes=[('gate_weights', {'op': 'TfConst'}),
                                           ('matmul', {'op': 'TfMatMul'}),
                                           ('gate_biases', {'op': 'TfConst'}),
                                           ('biasadd', {'op': 'TfBiasAdd'}),
                                           ('sigmoid', {'op': 'TfSigmoid'}),
                                           ('split', {'op': 'TfSplit'}),
                                           ('mul', {'op': 'TfMul'}),
                                           ('sub', {'op': 'TfSub'}),
                                           ('mul_1', {'op': 'TfMul'}),
                                           ('mul_2', {'op': 'TfMul'}),
                                           ('candidate_weights',
                                            {'op': 'TfConst'}),
                                           ('matmul_1', {'op': 'TfMatMul'}),
                                           ('candidate_biases',
                                            {'op': 'TfConst'}),
                                           ('biasadd_1', {'op': 'TfBiasAdd'}),
                                           ('tanh', {'op': 'TfTanh'}),
                                           ('add', {
                                            'op': ['TfAdd', 'TfAddV2']}),
                                           ],
                                    edges=[('gate_weights', 'matmul'),
                                           ('matmul', 'biasadd'),
                                           ('gate_biases', 'biasadd',
                                            {'dst_in_port': 1}),
                                           ('biasadd', 'sigmoid'),
                                           ('sigmoid', 'split'),
                                           ('split', 'sub', {
                                            'src_out_port': 1, 'dst_in_port': 1}),
                                           ('split', 'mul'),
                                           ('split', 'mul_1', {
                                            'src_out_port': 1, 'dst_in_port': 0}),
                                           ('sub', 'mul_2'),
                                           ('candidate_weights', 'matmul_1'),
                                           ('matmul_1', 'biasadd_1'),
                                           ('candidate_biases', 'biasadd_1',
                                            {'dst_in_port': 1}),
                                           ('biasadd_1', 'tanh'),
                                           ('tanh', 'mul_2', {
                                            'src_out_port': 0, 'dst_in_port': 1}),
                                           ('mul_1', 'add'),
                                           ('mul_2', 'add', {
                                            'src_out_port': 0, 'dst_in_port': 1}),
                                           ])
    if not cell_matches:
        return

    init_state_matches = matched_patterns(graph,
                                          nodes=[
                                              ('init_state', {}),
                                              ('next', {
                                               'op': 'TfNextIteration'}),
                                              ('merge', {'op': 'TfMerge'}),
                                              ('loop_cond', {
                                               'op': 'TfLoopCond'}),
                                              ('switch', {'op': 'TfSwitch'})
                                          ],
                                          edges=[
                                              ('init_state', 'merge', {
                                               'src_out_port': 0, 'dst_in_port': 0}),
                                              ('next', 'merge', {
                                               'src_out_port': 0, 'dst_in_port': 1}),
                                              ('loop_cond', 'switch'),
                                              ('merge', 'switch')
                                          ])
    inputs_matches = matched_patterns(graph,
                                      nodes=[
                                          ('transpose', {'op': 'TfTranspose'}),
                                          ('scatter', {
                                           'op': 'TfTensorArrayScatterV3'}),
                                          ('tensor_arr', {
                                           'op': 'TfTensorArrayV3'}),
                                          ('range', {})
                                      ],
                                      edges=[
                                          ('transpose', 'scatter', {
                                           'src_out_port': 0, 'dst_in_port': 2}),
                                          ('tensor_arr', 'scatter', {
                                           'src_out_port': 0, 'dst_in_port': 0}),
                                          ('tensor_arr', 'scatter', {
                                           'src_out_port': 1, 'dst_in_port': 3}),
                                          ('range', 'scatter', {
                                           'src_out_port': 0, 'dst_in_port': 1})
                                      ])

    sequence_out_matches = matched_patterns(graph,
                                            nodes=[('tensor_arr', {'op': 'TfTensorArrayV3'}),
                                                   ('exit', {'op': 'TfExit'}),
                                                   ('gather', {
                                                    'op': 'TfTensorArrayGatherV3'}),
                                                   ('transpose', {
                                                    'op': 'TfTranspose'})
                                                   ],
                                            edges=[('tensor_arr', 'gather'),
                                                   ('exit', 'gather', {
                                                    'dst_in_port': 2}),
                                                   ('gather', 'transpose')
                                                   ]
                                            )
    state_out_matches = matched_patterns(graph,
                                         nodes=[('state', {'op': ['TfConst', 'Constant']}),
                                                ('next_iter', {
                                                    'op': 'TfNextIteration'}),
                                                ('merge', {'op': 'TfMerge'}),
                                                ('loop', {'op': 'TfLoopCond'}),
                                                ('switch', {'op': 'TfSwitch'}),
                                                ('exit_3', {'op': 'TfExit'}),
                                                ],
                                         edges=[
                                             ('state', 'merge'),
                                             ('next_iter', 'merge',
                                                 {'dst_in_port': 1}),
                                             ('merge', 'switch'),
                                             ('loop', 'switch', {
                                                 'dst_in_port': 1}),
                                             ('switch', 'exit_3'),
                                         ])

    matched = False
    if len(init_state_matches) > 0 \
            and len(inputs_matches) > 0 \
            and len(cell_matches) > 0 \
            and (len(sequence_out_matches) > 0 or len(state_out_matches) > 0):
        for cell in cell_matches:
            init_match = sorted(init_state_matches, key=lambda x: cal_path_length(
                graph, x['init_state'], cell['matmul']))[0]
            inputs_match = sorted(inputs_matches, key=lambda x: cal_path_length(
                graph, x['transpose'], cell['matmul']))[0]
            sequence_match = sorted(sequence_out_matches, key=lambda x: cal_path_length(
                graph, cell['add'], x['transpose']))[0] if sequence_out_matches else {}
            state_match = sorted(state_out_matches, key=lambda x: cal_path_length(
                graph, cell['add'], x['switch']))[0] if state_out_matches else {}

            init = init_match['init_state']
            merge = init_match['merge']
            transpose = inputs_match['transpose']
            range_name = inputs_match['range']
            scatter = inputs_match['scatter']
            sequence_out = sequence_match.get('transpose', '')
            state_out = state_match.get('exit_3', '')

            init_obj, trans_obj, range_obj, seq_out_obj, state_out_obj \
                = [NodeWrap(graph, name)['object'] if name else None for name in [init, transpose, range_name, sequence_out, state_out]]
            init_out_edges = graph.sorted_out_edges(init, data=True)
            trans_in_edges = graph.sorted_in_edges(transpose, data=True)
            trans_out_edges = graph.sorted_out_edges(transpose, keys=True)
            scatter_in_edges = graph.sorted_in_edges(scatter)

            if init_obj is not None \
                    and trans_obj is not None \
                    and range_obj is not None \
                    and (seq_out_obj is not None or state_out_obj is not None) \
                    and len(init_out_edges) >= 1 \
                    and len(trans_in_edges) == 2 \
                    and len(scatter_in_edges) >= 1 \
                    and trans_in_edges[1][2]['tensor'].value.tolist() == [1, 0, 2] \
                    and len(trans_obj.get_input_shapes()) >= 1 \
                    and trans_obj.get_input_shapes()[0] is not None \
                    and len(trans_obj.get_input_shapes()[0]) == 3 \
                    and trans_obj.get_input_shapes()[0][2] is not None:
                trans_in_shape = trans_obj.get_input_shapes()[0]
                batch_size, time_steps, input_size = trans_in_shape
                if time_steps is None:
                    range_out_edges = graph.sorted_out_edges(range_name, data=True)
                    if len(range_out_edges) < 1 \
                            or range_out_edges[0][2]['tensor'].value is None \
                            or np.ndim(range_out_edges[0][2]['tensor'].value) != 1:
                        continue
                    else:
                        time_steps = int(range_out_edges[0][2]['tensor'].value.size)

                matched = True
                gate_weights = np.transpose(
                    NodeWrap(graph, cell['gate_weights'])['object'].value)
                gate_biases = NodeWrap(graph, cell['gate_biases'])[
                    'object'].value
                candidate_weights = np.transpose(
                    NodeWrap(graph, cell['candidate_weights'])['object'].value)
                candidate_biases = NodeWrap(graph, cell['candidate_biases'])[
                    'object'].value
                cell_size = candidate_biases.size

                reset, update = np.split(gate_weights, 2, axis=0)
                hidden_w, hidden_r = np.split(
                    candidate_weights, [input_size], axis=1)
                reset_wb, update_wb = np.split(gate_biases, 2)
                reset_rb = np.zeros_like(reset_wb)
                update_rb = np.zeros_like(update_wb)
                hidden_wb = candidate_biases
                hidden_rb = np.zeros_like(hidden_wb)
                update_w, update_r = np.split(update, [input_size], axis=1)
                reset_w, reset_r = np.split(reset, [input_size], axis=1)

                W = np.stack(
                    [np.concatenate([update_w, reset_w, hidden_w], axis=0)])
                R = np.stack(
                    [np.concatenate([update_r, reset_r, hidden_r], axis=0)])
                B = np.stack([np.concatenate(
                    [update_wb, reset_wb, hidden_wb, update_rb, reset_rb, hidden_rb], axis=0)])
                if batch_size is None:
                    seq_length = np.array([], np.int64)
                else:
                    seq_length = np.array([time_steps] * batch_size, np.int64)

                inp, _, inp_out_attr = trans_in_edges[0]
                _, _, init_out_attr = init_out_edges[0]
                gru = [e[1] for e in scatter_in_edges if e[0] == transpose][0]
                gru_in_edges = graph.sorted_in_edges(gru)
                gru_out_edges = graph.sorted_out_edges(gru)

                for _, dst, k in trans_out_edges:
                    if has_path(graph, dst, gru):
                        graph.remove_edge(transpose, dst, key=k)
                graph.remove_edge(init, merge)
                graph.remove_edges_from(gru_in_edges + gru_out_edges)

                new_inp_out_attr = copy.deepcopy(inp_out_attr)
                new_inp_out_attr['dst_in_port'] = 0
                graph.add_edge(inp, gru, **new_inp_out_attr)

                insert_constant(graph, gru + '_W', W, gru,
                                in_port=1, data_format='NHWC')
                insert_constant(graph, gru + '_R', R, gru,
                                in_port=2, data_format='NHWC')
                insert_constant(graph, gru + '_B', B, gru,
                                in_port=3, data_format='NHWC')
                insert_constant(graph, gru + '_seq_length',
                                seq_length, gru, in_port=4, data_format='NHWC')

                new_init_out_attr = copy.deepcopy(init_out_attr)
                new_init_out_attr['dst_in_port'] = 5
                graph.add_edge(init, gru, **new_init_out_attr)
                insert_reshape(graph, init,
                               gru,
                               new_init_out_attr,
                               [batch_size if batch_size is not None else -
                                   1, 1, cell_size],
                               type='Reshape',
                               data_format='NHWC')

                gru_attr = NodeWrap(graph, gru)['object'].copied_attr()
                gru_attr.update({'name': gru,
                                 'opset_version': 14,
                                 'layout': True,
                                 'input_size': input_size,
                                 'time_steps': time_steps,
                                 'hidden_size': cell_size,
                                 'linear_before_reset': False,
                                 'method': 'YH' if (sequence_match and state_match) else ('Y' if sequence_match else 'H')
                                 })
                NodeWrap(graph, gru).replace_obj('GRU', gru_attr)

                if sequence_out:
                    seq_out_dim = [
                        batch_size if batch_size is not None else -1, time_steps, cell_size]
                    seq_in_edges = graph.sorted_in_edges(sequence_out)
                    graph.remove_edges_from(seq_in_edges)
                    graph.add_edge(gru, sequence_out)
                    insert_constant(graph, sequence_out + '_shape', np.array(
                        seq_out_dim, np.int64), sequence_out, in_port=1, data_format='NHWC')
                    seq_reshape_attr = NodeWrap(graph, sequence_out)[
                        'object'].copied_attr()
                    seq_reshape_attr.update({'opset_version': 5})
                    NodeWrap(graph, sequence_out).replace_obj(
                        'Reshape', seq_reshape_attr)

                if state_out:
                    state_out_dim = [
                        batch_size if batch_size is not None else -1, cell_size]
                    state_in_edges = graph.sorted_in_edges(state_out)
                    graph.remove_edges_from(state_in_edges)
                    graph.add_edge(gru, state_out, **
                                   {'src_out_port': 1, 'dst_in_port': 0})
                    insert_constant(graph, state_out + '_shape', np.array(
                        state_out_dim, np.int64), state_out, in_port=1, data_format='NHWC')
                    state_reshape_attr = NodeWrap(graph, state_out)[
                        'object'].copied_attr()
                    state_reshape_attr.update({'opset_version': 5})
                    NodeWrap(graph, state_out).replace_obj(
                        'Reshape', state_reshape_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_gru2(graph):

    cell_matches1 = matched_patterns(graph,
                                     nodes=[
                                         ('x', {}),
                                         ('state', {}),
                                         ('matmul0', {'op': 'TfMatMul'}),
                                         ('update_w', {'op': 'Constant'}),
                                         ('matmul1', {'op': 'TfMatMul'}),
                                         ('reset_w', {'op': 'Constant'}),
                                         ('matmul2', {'op': 'TfMatMul'}),
                                         ('hidden_w', {'op': 'Constant'}),
                                         ('biasadd0', {'op': 'TfBiasAdd'}),
                                         ('update_wb', {'op': 'Constant'}),
                                         ('biasadd1', {'op': 'TfBiasAdd'}),
                                         ('reset_wb', {'op': 'Constant'}),
                                         ('biasadd2', {'op': 'TfBiasAdd'}),
                                         ('hidden_wb', {'op': 'Constant'}),
                                         ('matmul3', {'op': 'TfMatMul'}),
                                         ('update_r', {'op': 'Constant'}),
                                         ('matmul4', {'op': 'TfMatMul'}),
                                         ('reset_r', {'op': 'Constant'}),
                                         ('matmul5', {'op': 'TfMatMul'}),
                                         ('hidden_r', {'op': 'Constant'}),
                                         ('add0', {'op': 'TfAddV2'}),
                                         ('add1', {'op': 'TfAddV2'}),
                                         ('add2', {'op': 'TfAddV2'}),
                                         ('z', {'op': 'TfSigmoid'}),
                                         ('r', {'op': 'TfSigmoid'}),
                                         ('ht', {'op': 'TfTanh'}),
                                         ('mul0', {'op': "TfMul"}),
                                         ('sub', {'op': 'TfSub'}),
                                         ('mul1', {'op': 'TfMul'}),
                                         ('mul2', {'op': 'TfMul'}),
                                         ('out', {'op': 'TfAddV2'}),
                                     ],
                                     edges=[
                                         ('x', 'matmul0'),
                                         ('update_w', 'matmul0'),
                                         ('x', 'matmul1'),
                                         ('reset_w', 'matmul1'),
                                         ('x', 'matmul2'),
                                         ('hidden_w', 'matmul2'),
                                         ('matmul0', 'biasadd0'),
                                         ('update_wb', 'biasadd0'),
                                         ('matmul1', 'biasadd1'),
                                         ('reset_wb', 'biasadd1'),
                                         ('matmul2', 'biasadd2'),
                                         ('hidden_wb', 'biasadd2'),
                                         ('state', 'matmul3'),
                                         ('update_r', 'matmul3'),
                                         ('state', 'matmul4'),
                                         ('reset_r', 'matmul4'),
                                         ('biasadd0', 'add0'),
                                         ('matmul3', 'add0'),
                                         ('add0', 'z'),
                                         ('biasadd1', 'add1'),
                                         ('matmul4', 'add1'),
                                         ('add1', 'r'),
                                         ('state', 'mul0'),
                                         ('r', 'mul0'),
                                         ('mul0', 'matmul5'),
                                         ('hidden_r', 'matmul5'),
                                         ('biasadd2', 'add2'),
                                         ('matmul5', 'add2'),
                                         ('add2', 'ht'),
                                         ('z', 'mul1'),
                                         ('state', 'mul1'),
                                         ('z', 'sub', {
                                             'src_out_port': 0, 'dst_in_port': 1}),
                                         ('sub', 'mul2'),
                                         ('ht', 'mul2'),
                                         ('mul1', 'out'),
                                         ('mul2', 'out'),
                                     ])
    cell_matches2 = matched_patterns(graph,
                                     nodes=[
                                         ('x', {}),
                                         ('state', {}),
                                         ('matmul3', {'op': 'TfMatMul'}),
                                         ('update_r', {'op': 'Constant'}),
                                         ('matmul4', {'op': 'TfMatMul'}),
                                         ('reset_r', {'op': 'Constant'}),
                                         ('biasadd3', {'op': 'TfBiasAdd'}),
                                         ('update_rb', {'op': 'Constant'}),
                                         ('biasadd4', {'op': 'TfBiasAdd'}),
                                         ('reset_rb', {'op': 'Constant'}),
                                         ('matmul0', {'op': 'TfMatMul'}),
                                         ('update_w', {'op': 'Constant'}),
                                         ('matmul1', {'op': 'TfMatMul'}),
                                         ('reset_w', {'op': 'Constant'}),
                                         ('biasadd0', {'op': 'TfBiasAdd'}),
                                         ('update_wb', {'op': 'Constant'}),
                                         ('biasadd1', {'op': 'TfBiasAdd'}),
                                         ('reset_wb', {'op': 'Constant'}),
                                         ('matmul5', {'op': 'TfMatMul'}),
                                         ('hidden_r', {'op': 'Constant'}),
                                         ('biasadd5', {'op': 'TfBiasAdd'}),
                                         ('hidden_rb', {'op': 'Constant'}),
                                         ('matmul2', {'op': 'TfMatMul'}),
                                         ('hidden_w', {'op': 'Constant'}),
                                         ('biasadd2', {'op': 'TfBiasAdd'}),
                                         ('hidden_wb', {'op': 'Constant'}),
                                         ('add', {'op': 'TfAddV2'}),
                                         ('add1', {'op': 'TfAddV2'}),
                                         ('z', {'op': 'TfSigmoid'}),
                                         ('r', {'op': 'TfSigmoid'}),
                                         ('mul', {'op': 'TfMul'}),
                                         ('add2', {'op': 'TfAddV2'}),
                                         ('tanh', {'op': 'TfTanh'}),
                                         ('sub', {'op': 'TfSub'}),
                                         ('mul1', {'op': 'TfMul'}),
                                         ('mul2', {'op': 'TfMul'}),
                                         ('out', {'op': 'TfAddV2'}),
                                     ],
                                     edges=[
                                         ('matmul3', 'biasadd3'),
                                         ('update_rb', 'biasadd3'),
                                         ('matmul4', 'biasadd4'),
                                         ('reset_rb', 'biasadd4'),
                                         ('matmul5', 'biasadd5'),
                                         ('hidden_rb', 'biasadd5'),
                                         ('matmul0', 'biasadd0'),
                                         ('update_wb', 'biasadd0'),
                                         ('matmul1', 'biasadd1'),
                                         ('reset_wb', 'biasadd1'),
                                         ('matmul2', 'biasadd2'),
                                         ('hidden_wb', 'biasadd2'),
                                         ('biasadd3', 'add'),
                                         ('biasadd0', 'add'),
                                         ('biasadd4', 'add1'),
                                         ('biasadd1', 'add1'),
                                         ('x', 'matmul0'),
                                         ('update_w', 'matmul0'),
                                         ('x', 'matmul1'),
                                         ('reset_w', 'matmul1'),
                                         ('x', 'matmul2'),
                                         ('hidden_w', 'matmul2'),
                                         ('state', 'matmul3'),
                                         ('update_r', 'matmul3'),
                                         ('state', 'matmul4'),
                                         ('reset_r', 'matmul4'),
                                         ('state', 'matmul5'),
                                         ('hidden_r', 'matmul5'),
                                         ('add', 'z'),
                                         ('add1', 'r'),
                                         ('z', 'sub', {
                                             'src_out_port': 0, 'dst_in_port': 1}),
                                         ('r', 'mul'),
                                         ('biasadd5', 'mul'),
                                         ('mul', 'add2'),
                                         ('biasadd2', 'add2'),
                                         ('add2', 'tanh'),
                                         ('tanh', 'mul2'),
                                         ('sub', 'mul2'),
                                         ('z', 'mul1'),
                                         ('mul1', 'out'),
                                         ('mul2', 'out'),
                                     ])
    cell_matches3 = matched_patterns(graph,
                                     nodes=[
                                         ('x', {}),
                                         ('state', {}),
                                         ('kernel_weights', {'op': 'TfConst'}),
                                         ('matmul0', {'op': 'TfMatMul'}),
                                         ('bias', {'op': 'TfConst'}),
                                         ('biasadd', {'op': 'TfBiasAdd'}),
                                         ('split', {'op': 'TfSplit'}),
                                         ('add0', {'op': 'TfAdd'}),
                                         ('sigmoid', {'op': 'TfSigmoid'}),
                                         ('mul0', {'op': 'TfMul'}),
                                         ('add2', {'op': 'TfAdd'}),
                                         ('tanh', {'op': 'TfTanh'}),
                                         ('mul2', {'op': 'TfMul'}),
                                         ('out', {'op': 'TfAdd'}),
                                         ('merge', {'op': 'TfMerge'}),
                                         ('switch', {'op': 'TfSwitch'}),
                                         ('h2h_kernel_weights',
                                          {'op': 'TfConst'}),
                                         ('matmul1', {'op': 'TfMatMul'}),
                                         ('h2h_bias', {'op': 'TfConst'}),
                                         ('biasadd1', {'op': 'TfBiasAdd'}),
                                         ('split1', {'op': 'TfSplit'}),
                                         ('add1', {'op': 'TfAdd'}),
                                         ('sigmoid1', {'op': 'TfSigmoid'}),
                                         ('mul1', {'op': 'TfMul'}),
                                         ('sub_operand', {'op': 'TfConst'}),
                                         ('sub', {'op': 'TfSub'}),
                                     ],
                                     edges=[
                                         ('x', 'matmul0'),
                                         ('kernel_weights', 'matmul0'),
                                         ('bias', 'biasadd'),
                                         ('matmul0', 'biasadd'),
                                         ('biasadd', 'split'),
                                         ('split', 'add0', {
                                             'src_out_port': 0}),
                                         ('split', 'add1', {
                                             'src_out_port': 1}),
                                         ('split', 'add2', {
                                             'src_out_port': 2}),
                                         ('add0', 'sigmoid'),
                                         ('sigmoid', 'mul0'),
                                         ('mul0', 'add2'),
                                         ('add2', 'tanh'),
                                         ('tanh', 'mul2'),
                                         ('sub', 'mul2'),
                                         ('mul2', 'out'),
                                         ('mul1', 'out'),
                                         ('state', 'merge', {
                                             'dst_in_port': 0}),
                                         ('merge', 'switch', {
                                             'dst_in_port': 0}),
                                         ('switch', 'matmul1'),
                                         ('h2h_kernel_weights', 'matmul1'),
                                         ('matmul1', 'biasadd1'),
                                         ('h2h_bias', 'biasadd1'),
                                         ('biasadd1', 'split1'),
                                         ('split1', 'add0', {
                                             'src_out_port': 0}),
                                         ('split1', 'add1', {
                                             'src_out_port': 1}),
                                         ('split1', 'mul0', {
                                             'src_out_port': 2}),
                                         ('add1', 'sigmoid1'),
                                         ('sigmoid1', 'mul1'),
                                         ('switch', 'mul1'),
                                         ('sub_operand', 'sub', {
                                             'dst_in_port': 0}),
                                         ('sigmoid1', 'sub', {
                                             'dst_in_port': 1}),
                                     ])

    cell_matches = cell_matches1 + cell_matches2 + cell_matches3
    if not cell_matches:
        return

    init_state_matches = matched_patterns(graph,
                                          nodes=[
                                              ('init_state', {}),
                                              ('next', {
                                               'op': 'TfNextIteration'}),
                                              ('merge', {'op': 'TfMerge'}),
                                              ('loop_cond', {
                                               'op': 'TfLoopCond'}),
                                              ('switch', {'op': 'TfSwitch'})
                                          ],
                                          edges=[
                                              ('init_state', 'merge', {
                                               'src_out_port': 0, 'dst_in_port': 0}),
                                              ('next', 'merge', {
                                               'src_out_port': 0, 'dst_in_port': 1}),
                                              ('loop_cond', 'switch'),
                                              ('merge', 'switch')
                                          ])
    inputs_matches = matched_patterns(graph,
                                      nodes=[
                                          ('input', {}),
                                          ('scatter', {
                                           'op': 'TfTensorArrayScatterV3'}),
                                          ('tensor_arr', {
                                           'op': 'TfTensorArrayV3'}),
                                          ('range', {})
                                      ],
                                      edges=[
                                          ('input', 'scatter', {
                                           'src_out_port': 0, 'dst_in_port': 2}),
                                          ('tensor_arr', 'scatter', {
                                           'src_out_port': 0, 'dst_in_port': 0}),
                                          ('tensor_arr', 'scatter', {
                                           'src_out_port': 1, 'dst_in_port': 3}),
                                          ('range', 'scatter', {
                                           'src_out_port': 0, 'dst_in_port': 1})
                                      ])

    sequence_out_matches = matched_patterns(graph,
                                            nodes=[('tensor_arr', {'op': 'TfTensorArrayV3'}),
                                                   ('exit', {'op': 'TfExit'}),
                                                   ('gather', {
                                                    'op': 'TfTensorArrayGatherV3'}),
                                                   ],
                                            edges=[('tensor_arr', 'gather'),
                                                   ('exit', 'gather', {
                                                    'dst_in_port': 2}),
                                                   ]
                                            )
    state_out_matches = matched_patterns(graph,
                                         nodes=[('add', {'op': ['TfAddV2', 'TfSelect']}),
                                                ('state', {}),
                                                ('next_iter', {
                                                    'op': 'TfNextIteration'}),
                                                ('merge', {'op': 'TfMerge'}),
                                                ('loop', {'op': 'TfLoopCond'}),
                                                ('switch', {'op': 'TfSwitch'}),
                                                ('out', {'op': 'TfExit'}),
                                                ],
                                         edges=[('add', 'next_iter'),
                                                ('state', 'merge'),
                                                ('next_iter', 'merge',
                                                 {'dst_in_port': 1}),
                                                ('merge', 'switch'),
                                                ('loop', 'switch', {
                                                 'dst_in_port': 1}),
                                                ('switch', 'out'),
                                                ])
    matched = False

    if len(init_state_matches) < 1 \
            or len(inputs_matches) < 1 \
            or len(cell_matches) < 1 \
            or (len(sequence_out_matches) < 1 and len(state_out_matches) < 1):
        return
    for cell in cell_matches:
        init_match = sorted(init_state_matches, key=lambda x: cal_path_length(
            graph, x['init_state'], cell['matmul0']))[0]
        inputs_match = sorted(inputs_matches, key=lambda x: cal_path_length(
            graph, x['input'], cell['matmul0']))[0]
        sequence_match = sorted(sequence_out_matches, key=lambda x: cal_path_length(
            graph, cell['out'], x['gather']))[0] if sequence_out_matches else {}
        state_match = [m for m in state_out_matches if m['add'] == cell['out']
                       or cell['out'] in graph.pred[m['add']]]
        state_match = state_match[0] if state_match else {}

        init = init_match['init_state']
        merge = init_match['merge']
        scatter = inputs_match['scatter']
        sequence_out = sequence_match.get('gather', '')
        state_outs = [state_match.get('out', '')]

        init_obj, scatter_obj \
            = [NodeWrap(graph, name)['object'] if name else None for name in [init, scatter]]
        init_out_edges = graph.sorted_out_edges(init, data=True)
        scatter_in_edges = graph.sorted_in_edges(scatter, data=True)
        scatter_out_edges = graph.sorted_out_edges(scatter)
        scatter_in_shapes = scatter_obj.get_input_shapes()
        if init_obj is None \
                or scatter_obj is None \
                or len(init_out_edges) < 1 \
                or len(scatter_in_edges) < 3 \
                or len(scatter_in_shapes) < 3 \
                or scatter_in_shapes[2] is None \
                or len(scatter_in_shapes[2]) != 3 \
                or any([s is None for s in scatter_in_shapes[2]]):
            continue
        seq_out_edges = graph.sorted_out_edges(sequence_out)
        if len(seq_out_edges) == 1 \
                and NodeWrap(graph, seq_out_edges[0][1])['object'] is not None \
                and NodeWrap(graph, seq_out_edges[0][1])['object'].type == 'TfStridedSlice' \
                and NodeWrap(graph, seq_out_edges[0][1])['object'].shrink_axis_mask == 1:
            sequence_out = ''
            sequence_match = {}
            state_outs.append(seq_out_edges[0][1])

        def get_weights_biases(targets):
            rets = []
            for target in targets:
                node_name = cell[target]
                rets.append(NodeWrap(graph, node_name)['object'].value)
            return rets
        if 'kernel_weights' in cell:
            sub_oprand_obj = NodeWrap(graph, cell['sub_operand'])['object']
            if sub_oprand_obj is None or not FLOAT_EQUAL(sub_oprand_obj.value, 1):
                continue
            kernel_weights, bias = get_weights_biases(
                ['kernel_weights', 'bias'])
            recurrent_weights, recurrent_bias = get_weights_biases(
                ['h2h_kernel_weights', 'h2h_bias'])
            if kernel_weights.shape != recurrent_weights.shape \
                    or bias.shape != recurrent_bias.shape \
                    or len(kernel_weights.shape) != 2 \
                    or len(bias.shape) != 1 \
                    or kernel_weights.shape[1] % 3 != 0 \
                    or bias.shape[0] % 3 != 0:
                continue
            reset_after = True
            # The gate order for cuDNN is [r, z, h], which is different from the canonical format [z, r, h].
            reset_w, update_w, hidden_w = np.split(kernel_weights, 3, axis=1)
            reset_wb, update_wb, hidden_wb = np.split(bias, 3)
            reset_r, update_r, hidden_r = np.split(
                recurrent_weights, 3, axis=1)
            reset_rb, update_rb, hidden_rb = np.split(recurrent_bias, 3)
        else:
            update_wb, reset_wb, hidden_wb = get_weights_biases(
                ['update_wb', 'reset_wb', 'hidden_wb'])
            update_w, reset_w, hidden_w = get_weights_biases(
                ['update_w', 'reset_w', 'hidden_w'])
            update_r, reset_r, hidden_r = get_weights_biases(
                ['update_r', 'reset_r', 'hidden_r'])
            if 'biasadd3' in cell:
                reset_after = True
                update_rb, reset_rb, hidden_rb = get_weights_biases(
                    ['update_rb', 'reset_rb', 'hidden_rb'])
            else:
                reset_after = False
                update_rb = np.zeros_like(update_wb)
                reset_rb = np.zeros_like(reset_wb)
                hidden_rb = np.zeros_like(hidden_wb)
        matched = True
        cell_size = hidden_wb.size
        W = np.stack(
            [np.transpose(np.concatenate([update_w, reset_w, hidden_w], axis=1))])
        R = np.stack(
            [np.transpose(np.concatenate([update_r, reset_r, hidden_r], axis=1))])
        B = np.stack([np.concatenate(
            [update_wb, reset_wb, hidden_wb, update_rb, reset_rb, hidden_rb], axis=0)])
        time_steps, batch_size, input_size = scatter_in_shapes[2]
        seq_length = np.array([time_steps] * batch_size, np.int64)

        inp, _, inp_out_attr = scatter_in_edges[2]
        _, _, init_out_attr = init_out_edges[0]

        graph.remove_edge(init, merge)
        graph.remove_edges_from(scatter_in_edges + scatter_out_edges)
        gru = scatter

        new_inp_out_attr = copy.deepcopy(inp_out_attr)
        new_inp_out_attr['dst_in_port'] = 0
        graph.add_edge(inp, gru, **new_inp_out_attr)

        insert_constant(graph, gru + '_W', W, gru,
                        in_port=1, data_format='NHWC')
        insert_constant(graph, gru + '_R', R, gru,
                        in_port=2, data_format='NHWC')
        insert_constant(graph, gru + '_B', B, gru,
                        in_port=3, data_format='NHWC')
        insert_constant(graph, gru + '_seq_length',
                        seq_length, gru, in_port=4, data_format='NHWC')

        new_init_out_attr = copy.deepcopy(init_out_attr)
        new_init_out_attr['dst_in_port'] = 5
        graph.add_edge(init, gru, **new_init_out_attr)
        init_shape = [
            1, (batch_size if batch_size is not None else -1), cell_size]
        insert_reshape(graph, init,
                       gru,
                       new_init_out_attr,
                       init_shape,
                       type='Reshape',
                       data_format='NHWC')

        gru_attr = NodeWrap(graph, gru)['object'].copied_attr()
        gru_attr.update({'name': gru,
                         'opset_version': 14,
                         'layout': False,
                         'input_size': input_size,
                         'time_steps': time_steps,
                         'hidden_size': cell_size,
                         'linear_before_reset': reset_after,
                         'method': 'YH' if (sequence_match and state_match) else ('Y' if sequence_match else 'H')
                         })
        NodeWrap(graph, gru).replace_obj('GRU', gru_attr)

        if sequence_out:
            seq_out_dim = [
                time_steps, batch_size if batch_size is not None else -1, cell_size]
            seq_in_edges = graph.sorted_in_edges(sequence_out)
            graph.remove_edges_from(seq_in_edges)
            graph.add_edge(gru, sequence_out)
            insert_constant(graph, sequence_out + '_shape', np.array(
                seq_out_dim, np.int64), sequence_out, in_port=1, data_format='NHWC')
            seq_reshape_attr = NodeWrap(graph, sequence_out)[
                'object'].copied_attr()
            seq_reshape_attr.update({'opset_version': 5})
            NodeWrap(graph, sequence_out).replace_obj(
                'Reshape', seq_reshape_attr)

        for state_out in state_outs:
            if not state_out:
                continue
            state_out_dim = [
                batch_size if batch_size is not None else -1, cell_size]
            state_in_edges = graph.sorted_in_edges(state_out)
            graph.remove_edges_from(state_in_edges)
            graph.add_edge(gru, state_out, **
                           {'src_out_port': 1, 'dst_in_port': 0})
            insert_constant(graph, state_out + '_shape', np.array(
                state_out_dim, np.int64), state_out, in_port=1, data_format='NHWC')
            state_reshape_attr = NodeWrap(graph, state_out)[
                'object'].copied_attr()
            state_reshape_attr.update({'opset_version': 5})
            NodeWrap(graph, state_out).replace_obj(
                'Reshape', state_reshape_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_lstm(graph):
    cell_matches = matched_patterns(graph,
                                    nodes=[
                                        ('x', {}),
                                        ('c_pre', {}),
                                        ('h_pre', {}),
                                        ('kernel_w', {'op': 'TfConst'}),
                                        ('split_w', {'op': 'TfSplit'}),
                                        ('matmul0', {'op': 'TfMatMul'}),
                                        ('matmul1', {'op': 'TfMatMul'}),
                                        ('matmul2', {'op': 'TfMatMul'}),
                                        ('matmul3', {'op': 'TfMatMul'}),
                                        ('bias', {'op': 'TfConst'}),
                                        ('split_wb', {'op': 'TfSplit'}),
                                        ('biasadd0', {'op': 'TfBiasAdd'}),
                                        ('biasadd1', {'op': 'TfBiasAdd'}),
                                        ('biasadd2', {'op': 'TfBiasAdd'}),
                                        ('biasadd3', {'op': 'TfBiasAdd'}),
                                        ('add', {'op': 'TfAddV2'}),
                                        ('add1', {'op': 'TfAddV2'}),
                                        ('add2', {'op': 'TfAddV2'}),
                                        ('add3', {'op': 'TfAddV2'}),
                                        ('matmul4', {'op': 'TfMatMul'}),
                                        ('input_r', {'op': 'Constant'}),
                                        ('matmul5', {'op': 'TfMatMul'}),
                                        ('forget_r', {'op': 'Constant'}),
                                        ('matmul6', {'op': 'TfMatMul'}),
                                        ('cell_r', {'op': 'Constant'}),
                                        ('matmul7', {'op': 'TfMatMul'}),
                                        ('output_r', {'op': 'Constant'}),
                                        ('zi', {'op': 'TfSigmoid'}),
                                        ('zf', {'op': 'TfSigmoid'}),
                                        ('z', {'op': 'TfTanh'}),
                                        ('zo', {'op': 'TfSigmoid'}),
                                        ("mul", {'op': 'TfMul'}),
                                        ("mul1", {'op': 'TfMul'}),
                                        ('ct', {'op': 'TfAddV2'}),
                                        ("tanh1", {'op': 'TfTanh'}),
                                        ('ht', {'op': 'TfMul'}),
                                    ],
                                    edges=[
                                        ('kernel_w', 'split_w', {
                                            'src_out_port': 0, 'dst_in_port': 1}),
                                        ('x', 'matmul0'),
                                        ('split_w', 'matmul0'),
                                        ('x', 'matmul1'),
                                        ('split_w', 'matmul1'),
                                        ('x', 'matmul2'),
                                        ('split_w', 'matmul2'),
                                        ('x', 'matmul3'),
                                        ('split_w', 'matmul3'),
                                        ('bias', 'split_wb', {
                                            'src_out_port': 0, 'dst_in_port': 1}),
                                        ('matmul0', 'biasadd0'),
                                        ('split_wb', 'biasadd0'),
                                        ('matmul1', 'biasadd1'),
                                        ('split_wb', 'biasadd1'),
                                        ('matmul2', 'biasadd2'),
                                        ('split_wb', 'biasadd2'),
                                        ('matmul3', 'biasadd3'),
                                        ('split_wb', 'biasadd3'),
                                        ('biasadd0', 'add'),
                                        ('biasadd1', 'add1'),
                                        ('biasadd2', 'add2'),
                                        ('biasadd3', 'add3'),
                                        ('h_pre', 'matmul4'),
                                        ('input_r', 'matmul4'),
                                        ('h_pre', 'matmul5'),
                                        ('forget_r', 'matmul5'),
                                        ('h_pre', 'matmul6'),
                                        ('cell_r', 'matmul6'),
                                        ('h_pre', 'matmul7'),
                                        ('output_r', 'matmul7'),
                                        ('matmul4', 'add'),
                                        ('matmul5', 'add1'),
                                        ('matmul6', 'add2'),
                                        ('matmul7', 'add3'),
                                        ('add', 'zi'),
                                        ('add1', 'zf'),
                                        ('add2', 'z'),
                                        ('add3', 'zo'),
                                        ("zi", "mul1"),
                                        ("z", "mul1"),
                                        ('c_pre', 'mul'),
                                        ('zf', 'mul'),
                                        ('mul', "ct"),
                                        ('mul1', "ct"),
                                        ('ct', "tanh1"),
                                        ('tanh1', "ht"),
                                        ('zo', "ht"),
                                    ])

    if not cell_matches:
        return

    Y_out_matches = matched_patterns(graph,
                                     nodes=[
                                         ('ht', {'op': 'TfMul'}),
                                         ('write', {
                                          'op': 'TfTensorArrayWriteV3'}),
                                         ('iter', {'op': 'TfNextIteration'}),
                                         ('merge', {'op': 'TfMerge'}),
                                         ('switch', {'op': 'TfSwitch'}),
                                         ('exit', {'op': 'TfExit'}),
                                         ('gather', {
                                          'op': 'TfTensorArrayGatherV3'}),
                                     ],
                                     edges=[
                                         ('ht', 'write', {
                                             'src_out_port': 0, 'dst_in_port': 2}),
                                         ('write', 'iter'),
                                         ('iter', 'merge'),
                                         ('merge', 'switch'),
                                         ('switch', 'exit'),
                                         ('exit', 'gather'),
                                     ])
    C_out_matches = matched_patterns(graph,
                                     nodes=[
                                         ('ct', {'op': 'TfAddV2'}),
                                         ('iter', {'op': 'TfNextIteration'}),
                                         ('merge', {'op': 'TfMerge'}),
                                         ('switch', {'op': 'TfSwitch'}),
                                         ('out', {'op': 'TfExit'})
                                     ],
                                     edges=[
                                         ('ct', 'iter'),
                                         ('iter', 'merge'),
                                         ('merge', 'switch'),
                                         ('switch', 'out')
                                     ])
    H_out_matches = matched_patterns(graph,
                                     nodes=[
                                         ('ht', {'op': 'TfMul'}),
                                         ('iter', {'op': 'TfNextIteration'}),
                                         ('merge', {'op': 'TfMerge'}),
                                         ('switch', {'op': 'TfSwitch'}),
                                         ('out', {'op': 'TfExit'})
                                     ],
                                     edges=[
                                         ('ht', 'iter'),
                                         ('iter', 'merge'),
                                         ('merge', 'switch'),
                                         ('switch', 'out')
                                     ])
    H_init_matches = matched_patterns(graph,
                                      nodes=[
                                          ('init', {'op': 'Constant'}),
                                          ('merge', {'op': 'TfMerge'}),
                                          ('switch', {'op': 'TfSwitch'}),
                                          ('cell_init', {'op': 'TfMatMul'})
                                      ],
                                      edges=[
                                          ('init', 'merge', {
                                              'src_out_port': 0, 'dst_in_port': 0}),
                                          ('merge', 'switch'),
                                          ('switch', 'cell_init')
                                      ])
    C_init_matches = matched_patterns(graph,
                                      nodes=[
                                          ('init', {'op': 'Constant'}),
                                          ('merge', {'op': 'TfMerge'}),
                                          ('switch', {'op': 'TfSwitch'}),
                                          ('cell_init', {'op': 'TfMul'})
                                      ],
                                      edges=[
                                          ('init', 'merge', {
                                              'src_out_port': 0, 'dst_in_port': 0}),
                                          ('merge', 'switch'),
                                          ('switch', 'cell_init')
                                      ])
    input_matches = matched_patterns(graph,
                                     nodes=[
                                         ('x', {}),
                                         ('scatter', {
                                          'op': 'TfTensorArrayScatterV3'}),
                                         ('read', {
                                          'op': 'TfTensorArrayReadV3'})
                                     ],
                                     edges=[
                                         ('x', 'scatter', {'dst_in_port': 2}),
                                         ('scatter', 'read')
                                     ])
    matched = False
    for cell in cell_matches:
        Y_out_match = [y for y in Y_out_matches if y["ht"] == cell["ht"]]
        C_out_match = [c for c in C_out_matches if c["ct"] == cell["ct"]]
        H_out_match = [h for h in H_out_matches if h["ht"] == cell["ht"]]
        H_init_match = [
            h for h in H_init_matches if h['switch'] == cell['h_pre']]
        C_init_match = [
            c for c in C_init_matches if c['switch'] == cell['c_pre']]
        input_match = [i for i in input_matches if i["read"] == cell["x"]]
        if not Y_out_match or not H_init_match or not C_init_match or not input_match:
            continue
        Y_out_match = Y_out_match[0]
        H_init_match = H_init_match[0]
        C_init_match = C_init_match[0]
        input_match = input_match[0]
        C_out_match = C_out_match[0] if C_out_match else {}
        c_out = C_out_match.get('out', '')
        H_out_match = H_out_match[0] if H_out_match else {}
        h_outs = [H_out_match.get('out', '')]

        scatter = input_match['scatter']
        scatter_obj = NodeWrap(graph, scatter)['object']
        scatter_in_edges = graph.sorted_in_edges(scatter, data=True)
        scatter_in_shapes = scatter_obj.get_input_shapes()
        scatter_out_edges = graph.sorted_out_edges(scatter)
        if scatter_obj is None \
                or len(scatter_in_edges) < 3 \
                or len(scatter_in_shapes) < 3 \
                or scatter_in_shapes[2] is None \
                or len(scatter_in_shapes[2]) != 3 \
                or any([s is None for s in scatter_in_shapes[2]]):
            continue
        time_steps, batch_size, input_size = scatter_in_shapes[2]
        h_init_obj = NodeWrap(graph, H_init_match['init'])['object']
        c_init_obj = NodeWrap(graph, C_init_match['init'])['object']
        if h_init_obj is None \
                or c_init_obj is None \
                or any([(val is None or len(val.shape) != 2 or val.shape[0] != batch_size) for val in (h_init_obj.value, c_init_obj.value)]) \
                or h_init_obj.value.shape[-1] != c_init_obj.value.shape[-1]:
            continue
        h_init = np.stack([h_init_obj.value])
        c_init = np.stack([c_init_obj.value])
        hidden_size = h_init.shape[-1]

        matched = True
        y_out = Y_out_match['gather']
        y_out_in_edges = graph.sorted_in_edges(y_out)
        y_out_out_edges = graph.sorted_out_edges(y_out, data=True)
        if len(y_out_out_edges) == 1 \
                and NodeWrap(graph, y_out_out_edges[0][1])['object'] is not None \
                and NodeWrap(graph, y_out_out_edges[0][1])['object'].type == 'TfStridedSlice' \
                and NodeWrap(graph, y_out_out_edges[0][1])['object'].shrink_axis_mask == 1:
            y_out = ''
            h_outs.append(y_out_out_edges[0][1])
        # Clear empty elements('') in h_outs
        h_outs = [h for h in h_outs if h]

        def get_weights_biases(targets):
            rets = []
            for target in targets:
                node_name = cell[target]
                rets.append(NodeWrap(graph, node_name)['object'].value)
            return rets
        kernel_w, bias = get_weights_biases(['kernel_w', 'bias'])
        input_r, output_r, forget_r, cell_r = get_weights_biases(
            ['input_r', 'output_r', 'forget_r', 'cell_r'])
        # Note that tf kernel_w and bias are in format ifco, while onnx is iofc.
        input_w, forget_w, cell_w, output_w = np.split(kernel_w, 4, axis=1)
        input_wb, forget_wb, cell_wb, output_wb = np.split(bias, 4, axis=0)
        W = np.stack([np.transpose(np.concatenate(
            [input_w, output_w, forget_w, cell_w], axis=1))])
        R = np.stack([np.transpose(np.concatenate(
            [input_r, output_r, forget_r, cell_r], axis=1))])
        bias_w = np.concatenate([input_wb, output_wb, forget_wb, cell_wb])
        bias_r = np.zeros_like(bias_w)
        B = np.stack([np.concatenate([bias_w, bias_r])])
        seq_length = np.array([time_steps] * batch_size, np.int32)

        graph.remove_edges_from(scatter_in_edges + scatter_out_edges)
        lstm = scatter
        new_in_attr = copy.deepcopy(scatter_in_edges[2][2])
        new_in_attr.update({'dst_in_port': 0})
        graph.add_edge(input_match['x'], lstm, **new_in_attr)

        insert_constant(graph, lstm + '_W', W, lstm,
                        in_port=1, data_format='NHWC')
        insert_constant(graph, lstm + '_R', R, lstm,
                        in_port=2, data_format='NHWC')
        insert_constant(graph, lstm + '_B', B, lstm,
                        in_port=3, data_format='NHWC')
        insert_constant(graph, lstm + '_seq_length', seq_length, lstm,
                        in_port=4, data_format='NHWC')
        insert_constant(graph, lstm + '_initial_h', h_init, lstm,
                        in_port=5, data_format='NHWC')
        insert_constant(graph, lstm + '_initial_c', c_init, lstm,
                        in_port=6, data_format='NHWC')
        lstm_attr = scatter_obj.copied_attr()
        method = ('Y' if y_out else '') \
            + ('H' if h_outs else '') \
            + ('C' if c_out else '')
        lstm_attr.update({'opset_version': 14,
                          'layout': False,
                          'hidden_size': hidden_size,
                          'method': method})
        NodeWrap(graph, lstm).replace_obj('LSTM', lstm_attr)

        if y_out:
            graph.remove_edges_from(y_out_in_edges)
            graph.add_edge(lstm, y_out)
            Y_out_dim = [time_steps, batch_size, hidden_size]
            insert_constant(graph, y_out + '_shape', np.array(Y_out_dim),
                            y_out, in_port=1, data_format='NHWC')
            Y_out_reshape_attr = NodeWrap(graph, y_out)['object'].copied_attr()
            Y_out_reshape_attr.update({'opset_version': 5})
            NodeWrap(graph, y_out).replace_obj('Reshape', Y_out_reshape_attr)
        for h_out in h_outs:
            h_out_in_edges = graph.sorted_in_edges(h_out)
            graph.remove_edges_from(h_out_in_edges)
            H_out_dim = [batch_size, hidden_size]
            insert_constant(graph, h_out + '_shape', np.array(H_out_dim),
                            h_out, in_port=1, data_format='NHWC')
            graph.add_edge(lstm, h_out, **
                           {'src_out_port': 1, 'dst_in_port': 0})
            reshape_attr = NodeWrap(graph, h_out)['object'].copied_attr()
            reshape_attr.update({'opset_version': 5})
            NodeWrap(graph, h_out).replace_obj('Reshape', reshape_attr)
        if c_out:
            c_out_in_edges = graph.sorted_in_edges(c_out)
            graph.remove_edges_from(c_out_in_edges)
            C_out_dim = [batch_size, hidden_size]
            insert_constant(graph, c_out + '_shape', np.array(C_out_dim),
                            c_out, in_port=1, data_format='NHWC')
            graph.add_edge(lstm, c_out, **
                           {'src_out_port': 2, 'dst_in_port': 0})
            reshape_attr = NodeWrap(graph, c_out)['object'].copied_attr()
            reshape_attr.update({'opset_version': 5})
            NodeWrap(graph, c_out).replace_obj('Reshape', reshape_attr)

    if matched:
        clear_redundant_nodes(graph)


def merge_lstm2(graph):

    cell_matches = matched_patterns(graph,
                                    nodes=[('concat', {'op': 'TfConcatV2'}),
                                           ('matmul', {'op': 'TfMatMul'}),
                                           ('weights', {'op': 'TfConst'}),
                                           ('biasadd', {'op': 'TfBiasAdd'}),
                                           ('bias', {'op': 'TfConst'}),
                                           ('split', {'op': 'TfSplit'}),
                                           ('add', {'op': 'TfAdd'}),
                                           ('const_b', {'op': 'TfConst'}),
                                           ('sig1', {'op': 'TfSigmoid'}),
                                           ('sig2', {'op': 'TfSigmoid'}),
                                           ('sig3', {'op': 'TfSigmoid'}),
                                           ('mul', {'op': 'TfMul'}),
                                           ('mul2', {'op': 'TfMul'}),
                                           ('mul3', {'op': 'TfMul'}),
                                           ('tanh', {'op': 'TfTanh'}),
                                           ],
                                    edges=[('concat', 'matmul'),
                                           ('weights', 'matmul', {
                                            'src_out_port': 0, 'dst_in_port': 1}),
                                           ('bias', 'biasadd', {
                                            'src_out_port': 0, 'dst_in_port': 1}),
                                           ('matmul', 'biasadd'),
                                           ('biasadd', 'split'),
                                           ('split', 'add'),
                                           ('split', 'sig2', {
                                            'src_out_port': 0, 'dst_in_port': 0}),
                                           ('split', 'sig3'),
                                           ('split', 'tanh'),
                                           ('const_b', 'add'),
                                           ('add', 'sig1'),
                                           ('sig1', 'mul'),
                                           ('sig2', 'mul2'),
                                           ('sig3', 'mul3'),
                                           ('tanh', 'mul2'),
                                           ])
    if not cell_matches:
        return

    begin_matches = matched_patterns(graph,
                                     nodes=[('trans', {'op': 'TfTranspose'}),
                                            ('tensor_arr_scatter', {
                                             'op': 'TfTensorArrayScatterV3'}),
                                            ('tensor_arr', {
                                             'op': 'TfTensorArrayV3'}),
                                            ('range', {}),
                                            ('tensor_read', {
                                             'op': 'TfTensorArrayReadV3'}),
                                            ('concat', {'op': 'TfConcatV2'}),
                                            ],
                                     edges=[('trans', 'tensor_arr_scatter', {'src_out_port': 0, 'dst_in_port': 2}),
                                            ('tensor_arr', 'tensor_arr_scatter'),
                                            ('tensor_arr', 'tensor_arr_scatter', {
                                             'src_out_port': 1, 'dst_in_port': 3}),
                                            ('range', 'tensor_arr_scatter', {
                                             'src_out_port': 0, 'dst_in_port': 1}),
                                            ('tensor_arr_scatter', 'tensor_read'),
                                            ('tensor_read', 'concat'),
                                            ])

    sequence_out_matches = matched_patterns(graph,
                                            nodes=[
                                                ('ht', {'op': 'TfMul'}),
                                                ('write', {
                                                 'op': 'TfTensorArrayWriteV3'}),
                                                ('iter', {
                                                 'op': 'TfNextIteration'}),
                                                ('merge', {'op': 'TfMerge'}),
                                                ('switch', {'op': 'TfSwitch'}),
                                                ('exit', {'op': 'TfExit'}),
                                                ('trans', {
                                                 'op': 'TfTranspose'}),
                                                ('concat', {}),
                                                ('gather', {'op': 'TfTensorArrayGatherV3'})],
                                            edges=[
                                                ('ht', 'write', {
                                                 'src_out_port': 0, 'dst_in_port': 2}),
                                                ('write', 'iter'),
                                                ('iter', 'merge'),
                                                ('merge', 'switch'),
                                                ('switch', 'exit'),
                                                ('exit', 'gather'),
                                                ('concat', 'trans'),
                                                ('gather', 'trans')
                                            ])

    h_init_matches = matched_patterns(graph,
                                      nodes=[
                                          ('h_init', {}),
                                          ('next_iter', {
                                           'op': 'TfNextIteration'}),
                                          ('merge', {'op': 'TfMerge'}),
                                          ('switch', {'op': 'TfSwitch'}),
                                          ('cell_concat', {
                                           'op': 'TfConcatV2'}),
                                      ],
                                      edges=[
                                          ('h_init', 'merge'),
                                          ('next_iter', 'merge'),
                                          ('merge', 'switch'),
                                          ('switch', 'cell_concat'),
                                      ])

    c_init_matches = matched_patterns(graph,
                                      nodes=[
                                          ('c_init', {}),
                                          ('next_iter', {
                                           'op': 'TfNextIteration'}),
                                          ('merge', {'op': 'TfMerge'}),
                                          ('switch', {'op': 'TfSwitch'}),
                                          ('mul', {'op': 'TfMul'}),
                                      ],
                                      edges=[
                                          ('c_init', 'merge'),
                                          ('next_iter', 'merge'),
                                          ('merge', 'switch'),
                                          ('switch', 'mul'),
                                      ])
    matched = False
    if begin_matches \
            and cell_matches \
            and h_init_matches \
            and c_init_matches \
            and sequence_out_matches \
            and len(begin_matches) >= len(cell_matches) \
            and len(sequence_out_matches) == len(cell_matches):

        for cm in cell_matches:
            begin_match = [
                b for b in begin_matches if b["concat"] == cm["concat"]]
            Y_out_match = [
                s for s in sequence_out_matches if s["ht"] == cm["mul3"]]
            h_init_match = [
                h for h in h_init_matches if h["cell_concat"] == cm["concat"]]
            c_init_match = [c for c in c_init_matches if c["mul"] == cm["mul"]]

            if not begin_match or not Y_out_match or not h_init_match or not c_init_match:
                continue

            const_b = cm['const_b']
            const_b_obj = NodeWrap(graph, const_b)['object']

            begin_name = begin_match[0]['trans']
            begin_obj = NodeWrap(graph, begin_name)['object']
            begin_in_shapes = begin_obj.get_input_shapes()

            weights_name = cm['matmul']
            weights_value_name = cm['weights']
            weights_value_name_obj = NodeWrap(
                graph, weights_value_name)['object']

            scatter = begin_match[0]['tensor_arr_scatter']
            scatter_obj = NodeWrap(graph, scatter)['object']
            scatter_in_edges = graph.sorted_in_edges(scatter, data=True)
            scatter_out_edges = graph.sorted_out_edges(scatter)

            biases_name = cm['biasadd']
            biases_value_name = cm['bias']
            biases_value_name_obj = NodeWrap(
                graph, biases_value_name)['object']

            initial_h_name = h_init_match[0]['h_init']
            initial_c_name = c_init_match[0]['c_init']
            h_init_obj = NodeWrap(graph, initial_h_name)['object']
            c_init_obj = NodeWrap(graph, initial_c_name)['object']
            h_init_out_edge = graph.sorted_out_edges(initial_h_name, data=True)
            c_init_out_edge = graph.sorted_out_edges(initial_c_name, data=True)

            if scatter_obj is None \
                    or len(scatter_in_edges) < 3 \
                    or h_init_obj is None \
                    or c_init_obj is None \
                    or begin_obj is None:
                continue

            matched = True
            # Prepare inputs for onnx lstm
            weights = weights_value_name_obj.value
            biases = biases_value_name_obj.value
            w_np = np.transpose(weights)
            b_np = biases

            batch_size, time_steps, input_size = begin_in_shapes[0]
            kernel_weights, recurrent_weights = np.split(
                w_np, [input_size], axis=1)
            input_w, cell_w, forget_w, output_w = np.split(
                kernel_weights, 4, axis=0)
            input_r, cell_r, forget_r, output_r = np.split(
                recurrent_weights, 4, axis=0)
            input_wb, cell_wb, forget_wb, output_wb = np.split(
                biases, 4, axis=0)
            forget_wb = forget_wb + \
                np.full(forget_wb.shape, float(
                    const_b_obj.value)).astype(np.float32)
            W_value = np.stack(
                [np.concatenate([input_w, output_w, forget_w, cell_w], axis=0)])
            R_value = np.stack(
                [np.concatenate([input_r, output_r, forget_r, cell_r], axis=0)])
            biases_w = np.concatenate(
                [input_wb, output_wb, forget_wb, cell_wb])
            biases_r = np.zeros_like(biases_w)
            B_value = np.stack([np.concatenate([biases_w, biases_r])])
            seq_length = np.array([time_steps] * batch_size, np.int32)
            hidden_size = begin_in_shapes[0][-1]
            Y_out_dim = [batch_size, time_steps, hidden_size]
            initial_hc_shape = [batch_size, 1, hidden_size]

            # Create a new node for LSTM
            lstm_begin = get_valid_node_name(graph, begin_name + '_lstm')

            _, _, initial_h_attr = h_init_out_edge[0]
            _, _, initial_c_attr = c_init_out_edge[0]
            new_h_init_attr = copy.deepcopy(initial_h_attr)
            new_h_init_attr['dst_in_port'] = 5
            new_c_init_attr = copy.deepcopy(initial_c_attr)
            new_c_init_attr['dst_in_port'] = 6
            trans_src, _, attr = graph.sorted_in_edges(
                begin_name, data=True)[0]

            graph.remove_edge(trans_src, begin_name)
            graph.add_edge(trans_src, lstm_begin, **attr)
            graph.add_edge(initial_h_name, lstm_begin, **new_h_init_attr)
            graph.add_edge(initial_c_name, lstm_begin, **new_c_init_attr)
            graph.remove_edges_from(scatter_in_edges + scatter_out_edges)

            seq_end_name = Y_out_match[0]['trans']
            seq_end_out_edge = graph.sorted_out_edges(
                seq_end_name, data=True)
            for _, seq_end_dst, seq_end_attr in seq_end_out_edge:
                graph.remove_edge(seq_end_name, seq_end_dst)
                graph.add_edge(lstm_begin, seq_end_dst, **seq_end_attr)
            post_reshape_name = insert_reshape_after(
                graph, lstm_begin, Y_out_dim)

            # Convert to onnx lstm
            insert_constant(graph, lstm_begin + '_W', W_value, lstm_begin,
                            in_port=1, data_format='NHWC')
            insert_constant(graph, lstm_begin + '_R', R_value, lstm_begin,
                            in_port=2, data_format='NHWC')
            insert_constant(graph, lstm_begin + '_B', B_value, lstm_begin,
                            in_port=3, data_format='NHWC')
            insert_constant(graph, lstm_begin + '_seq_length', seq_length, lstm_begin,
                            in_port=4, data_format='NHWC')
            insert_reshape(graph, initial_h_name, lstm_begin,
                           new_h_init_attr, initial_hc_shape)
            insert_reshape(graph, initial_c_name, lstm_begin,
                           new_c_init_attr, initial_hc_shape)

            lstm_attr = begin_obj.copied_attr()
            lstm_attr = {'name': lstm_begin,
                         'opset_version': 14,
                         'layout': True,
                         'hidden_size': hidden_size,
                         'method': 'Y',
                         }
            NodeWrap(graph, lstm_begin).replace_obj('LSTM', lstm_attr)

            if seq_end_name in graph._attr['output_names']:
                index = graph._attr['output_names'].index(seq_end_name)
                graph._attr['output_names'][index] = post_reshape_name

    if matched:
        clear_redundant_nodes(graph)


def merge_zero_fraction(graph):
    zf_matches = matched_patterns(graph,
                                  nodes=[
                                      ('input', {}),
                                      ('size', {'op': ['TfConst', 'Constant']}),
                                      ('le_const', {'op': 'Constant'}),
                                      ('statelessif', {'op': 'TfStatelessIf'}),
                                      ('sub', {'op': 'TfSub'}),
                                      ('cast', {'op': 'TfCast'}),
                                      ('div_operand', {'op': 'Constant'}),
                                      ('div', {'op': 'TfRealDiv'}),
                                  ],
                                  edges=[
                                      ('le_const', 'statelessif',
                                       {'dst_in_port': 0}),
                                      ('input', 'statelessif',
                                       {'dst_in_port': 1}),
                                      ('size', 'sub', {'dst_in_port': 0}),
                                      ('statelessif', 'sub',
                                       {'dst_in_port': 1}),
                                      ('sub', 'cast'),
                                      ('cast', 'div', {'dst_in_port': 0}),
                                      ('div_operand', 'div',
                                       {'dst_in_port': 1}),
                                  ]
                                  )
    matched = False
    for m in zf_matches:
        key_names = ['input', 'size', 'le_const', 'cast', 'div_operand', 'div']
        node_objs = {k: NodeWrap(graph, m[k])['object'] for k in key_names}
        if any([obj is None for obj in node_objs.values()]) or \
                len(node_objs['input'].get_output_shapes()) < 1 or \
                len(graph.sorted_in_edges(m['statelessif'])) < 2:
            ERROR('[Parser]: Meets invalid nodes in merge_zero_fraction!')
            continue
        sli_in_edges = graph.sorted_in_edges(m['statelessif'], data=True)
        input_shape = node_objs['input'].get_output_shapes()[0]
        if node_objs['size'].value != np.prod(input_shape) or \
                node_objs['le_const'].value != True or \
                node_objs['cast'].DstT != 'float32' or \
                not FLOAT_EQUAL(node_objs['div_operand'].value, node_objs['size'].value):
            continue
        matched = True
        div_in_edges = graph.sorted_in_edges(m['div'], data=True)
        graph.remove_edges_from(sli_in_edges + div_in_edges)
        _, _, in_attr = sli_in_edges[1]
        in_attr['dst_in_port'] = 0
        graph.add_edge(m['input'], m['div'], **in_attr)
        zero_fraction_attr = node_objs['div'].copied_attr()
        NodeWrap(graph, m['div']).replace_obj(
            'ZeroFraction', zero_fraction_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_fasterrcnn(graph):
    def _tile_anchor(image_size, anchor_stride, base_anchor_size, anchor_scale, anchor_aspect_ratios):
        grid_height = np.round(
            image_size[0] / anchor_stride[0]).astype(np.int32)
        grid_width = np.round(
            image_size[1] / anchor_stride[1]).astype(np.int32)
        anchor_offset = (0, 0)
        scales_grid, aspect_ratios_grid = np.meshgrid(
            anchor_scale, anchor_aspect_ratios)
        scales_grid = np.reshape(scales_grid, [-1])
        aspect_ratios_grid = np.reshape(aspect_ratios_grid, [-1])
        ratio_sqrts = np.sqrt(aspect_ratios_grid)

        heights = scales_grid / ratio_sqrts * base_anchor_size[0]
        widths = scales_grid * ratio_sqrts * base_anchor_size[1]
        y_centers = np.array(range(grid_height), dtype=np.float32)
        y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
        x_centers = np.array(range(grid_width), dtype=np.float32)
        x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
        x_centers, y_centers = np.meshgrid(x_centers, y_centers)

        widths_grid, x_centers_grid = np.meshgrid(widths, x_centers)
        heights_grid, y_centers_grid = np.meshgrid(heights, y_centers)

        bbox_centers = np.stack([y_centers_grid[:, :, np.newaxis],
                                 x_centers_grid[:, :, np.newaxis]], axis=3)
        bbox_sizes = np.stack(
            [heights_grid[:, :, np.newaxis], widths_grid[:, :, np.newaxis]], axis=3)
        bbox_centers = np.reshape(bbox_centers, [-1, 2])
        bbox_sizes = np.reshape(bbox_sizes, [-1, 2])
        bbox_corners = np.concatenate(
            [bbox_centers - .5 * bbox_sizes, bbox_centers + .5 * bbox_sizes], 1)

        y_min, x_min, y_max, x_max = np.split(bbox_corners, 4, axis=1)

        win_y_min = 0
        win_x_min = 0
        win_y_max = image_size[0]
        win_x_max = image_size[1]
        y_min_clipped = np.maximum(np.minimum(y_min, win_y_max), win_y_min)
        y_max_clipped = np.maximum(np.minimum(y_max, win_y_max), win_y_min)
        x_min_clipped = np.maximum(np.minimum(x_min, win_x_max), win_x_min)
        x_max_clipped = np.maximum(np.minimum(x_max, win_x_max), win_x_min)
        clipped = np.concatenate(
            [y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped], 1)
        areas = np.squeeze((y_max_clipped - y_min_clipped)
                           * (x_max_clipped - x_min_clipped))
        clipped = clipped[areas > 0]
        return clipped.reshape([-1, 4]).astype(np.float32)

    inp = 'Preprocessor/sub'
    roi_pooling = 'CropAndResize'
    outputs = ['Squeeze',
               'FirstStageBoxPredictor/Reshape_1',
               'SecondStageBoxPredictor/Reshape_1',
               'SecondStagePostprocessor/Reshape']
    check_names = [inp, roi_pooling] + outputs
    if any([not graph.has_node(name) for name in check_names]):
        return
    if any([name not in graph._attr['output_names'] for name in outputs]):
        ERROR('[Parser]: Please check output names in cfg file!')
        return
    obj_dict = {n: NodeWrap(graph, n)['object'] for n in check_names}
    if any([obj is None for obj in obj_dict.values()]):
        ERROR('[Parser]: Meets invalid Op in merge_fasterrcnn!')
        return
    input_shapes = obj_dict[inp].get_output_shapes()
    if len(input_shapes) < 1 \
            or input_shapes[0] is None \
            or len(input_shapes[0]) != 4:
        return

    proposal_box, proposal_prediction = outputs[:2]
    secondstage_boxpredictor, secondstage_reshape = outputs[2:]

    roi_pooling_in_edges = graph.sorted_in_edges(roi_pooling, data=True)
    proposal_box_out_edges = graph.sorted_out_edges(proposal_box, data=True)
    proposal_prediction_out_edges = graph.sorted_out_edges(
        proposal_prediction, data=True)
    secondstage_boxpredictor_out_edges = graph.sorted_out_edges(
        secondstage_boxpredictor, data=True)
    secondstage_reshape_out_edges = graph.sorted_out_edges(
        secondstage_reshape, data=True)
    if len(roi_pooling_in_edges) != 4 \
            or len(proposal_box_out_edges) < 1 \
            or len(proposal_prediction_out_edges) < 1 \
            or len(secondstage_boxpredictor_out_edges) < 1 \
            or len(secondstage_reshape_out_edges) < 1:
        return
    crop_and_resize_in_shapes = obj_dict[roi_pooling].get_input_shapes()
    if len(crop_and_resize_in_shapes) != 4:
        return

    in_shape = input_shapes[0]
    batch, img_height, img_width = in_shape[:3]
    proposal_count = crop_and_resize_in_shapes[2][0] // batch

    _, _, proposal_box_out_attr = proposal_box_out_edges[0]
    _, _, proposal_prediction_out_attr = proposal_prediction_out_edges[0]
    _, _, secondstage_boxpredictor_out_attr = secondstage_boxpredictor_out_edges[0]
    _, _, secondstage_reshape_out_attr = secondstage_reshape_out_edges[0]
    new_proposal_box_out_attr = copy.deepcopy(proposal_box_out_attr)
    new_proposal_box_out_attr['dst_in_port'] = 1
    new_secondstage_reshape_out_attr = copy.deepcopy(
        secondstage_reshape_out_attr)
    new_secondstage_reshape_out_attr['dst_in_port'] = 1

    graph.remove_edges_from(roi_pooling_in_edges[1:2])
    graph.remove_edges_from(proposal_box_out_edges +
                            proposal_prediction_out_edges)
    graph.remove_edges_from(
        secondstage_boxpredictor_out_edges + secondstage_reshape_out_edges)

    proposal = get_valid_node_name(graph, proposal_prediction + '_proposal')
    softmax1 = get_valid_node_name(graph, proposal + '_softmax')  # axis = 2
    anchor = get_valid_node_name(graph, proposal + '_anchor')
    nms = get_valid_node_name(graph, proposal + '_nms')
    nms_out_0 = get_valid_node_name(graph, nms + '_out_0')
    nms_out_2 = get_valid_node_name(graph, nms + '_out_2')
    post_nms1 = get_valid_node_name(graph, proposal + '_post_nms1')
    post_nms1_reshape = get_valid_node_name(
        graph, post_nms1 + '_reshape')  # [batch_size*100, 4]
    secondstage_box_transpose = get_valid_node_name(
        graph, secondstage_boxpredictor + '_post_transpose')  # [1, 0, 2]
    softmax2 = get_valid_node_name(
        graph, secondstage_boxpredictor + '_softmax')    # axis = 2
    detection_output = get_valid_node_name(
        graph, graph._attr['name'] + '_detection_output')
    detection_out_3 = get_valid_node_name(graph, detection_output + '_out_3')
    nms2 = get_valid_node_name(graph, detection_output + '_nms')
    nms2_out_0 = get_valid_node_name(graph, nms2 + '_out_0')
    nms2_out_1 = get_valid_node_name(graph, nms2 + '_out_1')
    nms2_out_2 = get_valid_node_name(graph, nms2 + '_out_2')
    nms2_out_3 = get_valid_node_name(graph, nms2 + '_out_2')

    graph.add_edge(proposal_prediction, softmax1,
                   **proposal_prediction_out_attr)
    graph.add_edge(softmax1, proposal)
    graph.add_edge(proposal_box, proposal, **new_proposal_box_out_attr)
    graph.add_edge(anchor, proposal, **{'dst_in_port': 2})

    graph.add_edge(proposal, nms, **{'src_out_port': 0, 'dst_in_port': 3})
    graph.add_edge(proposal, nms, **{'src_out_port': 1, 'dst_in_port': 0})
    graph.add_edge(proposal, nms, **{'src_out_port': 2, 'dst_in_port': 1})
    graph.add_edge(proposal, nms, **{'src_out_port': 3, 'dst_in_port': 2})
    graph.add_edge(proposal, post_nms1, **
                   {'src_out_port': 1, 'dst_in_port': 0})

    graph.add_edge(nms, nms_out_0, **{'src_out_port': 0})
    graph.add_edge(nms, post_nms1, **{'src_out_port': 1, 'dst_in_port': 2})
    graph.add_edge(nms, nms_out_2, **{'src_out_port': 2})
    graph.add_edge(nms, post_nms1, **{'src_out_port': 3, 'dst_in_port': 1})

    graph.add_edge(post_nms1, detection_output, **
                   {'src_out_port': 0, 'dst_in_port': 2})
    graph.add_edge(post_nms1, post_nms1_reshape, **
                   {'src_out_port': 1, 'dst_in_port': 0})
    graph.add_edge(post_nms1_reshape, roi_pooling, **
                   {'src_out_port': 0, 'dst_in_port': 1})
    graph.add_edge(secondstage_reshape, detection_output,
                   **new_secondstage_reshape_out_attr)
    graph.add_edge(secondstage_boxpredictor, secondstage_box_transpose,
                   **secondstage_boxpredictor_out_attr)
    graph.add_edge(secondstage_box_transpose, softmax2)
    graph.add_edge(softmax2, detection_output, **
                   {'src_out_port': 0, 'dst_in_port': 0})

    graph.add_edge(detection_output, nms2, **
                   {'src_out_port': 0, 'dst_in_port': 3})
    graph.add_edge(detection_output, nms2, **
                   {'src_out_port': 1, 'dst_in_port': 0})
    graph.add_edge(detection_output, nms2, **
                   {'src_out_port': 2, 'dst_in_port': 1})
    graph.add_edge(detection_output, detection_out_3, **
                   {'src_out_port': 3, 'dst_in_port': 0})
    graph.add_edge(detection_output, nms2, **
                   {'src_out_port': 4, 'dst_in_port': 2})

    graph.add_edge(nms2, nms2_out_0, **{'src_out_port': 0, 'dst_in_port': 0})
    graph.add_edge(nms2, nms2_out_1, **{'src_out_port': 1, 'dst_in_port': 0})
    graph.add_edge(nms2, nms2_out_2, **{'src_out_port': 2, 'dst_in_port': 0})
    graph.add_edge(nms2, nms2_out_3, **{'src_out_port': 3, 'dst_in_port': 0})

    graph._attr['output_names'].remove(secondstage_boxpredictor)
    graph._attr['output_names'].remove(secondstage_reshape)
    graph._attr['output_names'].extend([detection_output, nms2])

    NodeWrap(graph, proposal).replace_obj('ArmProposal',
                                          {'name': proposal,
                                           'width': img_width,
                                           'height': img_height,
                                           'score_threshold': 0.45,
                                           'class_num': 1
                                           })
    NodeWrap(graph, softmax1).replace_obj('Softmax',
                                          {'name': softmax1,
                                           'opset_version': 13,
                                           'axis': 2
                                           })

    anchor_value = _tile_anchor((img_height, img_width),
                                (16, 16),
                                (256, 256),
                                [0.25, 0.5, 1.0, 2.0],
                                [0.5, 1.0, 2.0])
    NodeWrap(graph, anchor).replace_obj('Constant',
                                        {'name': anchor,
                                         'opset_version': 1,
                                         'value': anchor_value
                                         })
    NodeWrap(graph, nms).replace_obj('ArmNMS',
                                     {'name': nms,
                                      'image_width': img_width,
                                      'image_height': img_height,
                                      'iou_threshold': 0.7,
                                      'max_box_num': 5000,
                                      'center_point_box': 0
                                      })
    NodeWrap(graph, nms_out_0).replace_obj('Out', {'name': nms_out_0})
    NodeWrap(graph, nms_out_2).replace_obj('Out', {'name': nms_out_2})
    NodeWrap(graph, post_nms1).replace_obj('ArmPostNMS1',
                                           {'name': post_nms1,
                                            'image_width': img_width,
                                            'image_height': img_height,
                                            'proposal_cnt': proposal_count
                                            })
    NodeWrap(graph, post_nms1_reshape).replace_obj('Reshape',
                                                   {'name': post_nms1_reshape,
                                                    'opset_version': 5
                                                    })
    insert_constant(graph,
                    post_nms1_reshape + '_shape',
                    np.array([-1, 4], np.int32),
                    post_nms1_reshape,
                    in_port=1,
                    data_format='NHWC')
    NodeWrap(graph, secondstage_box_transpose).replace_obj('Transpose',
                                                           {'name': secondstage_box_transpose,
                                                            'opset_version': 1,
                                                            'perm': [1, 0, 2]
                                                            })
    NodeWrap(graph, softmax2).replace_obj('Softmax',
                                          {'name': softmax2,
                                           'opset_version': 13,
                                           'axis': 2
                                           })
    NodeWrap(graph, detection_output).replace_obj('ArmDetectionOutput',
                                                  {'name': detection_output,
                                                   'image_width': img_width,
                                                   'image_height': img_height,
                                                   'class_num': 90,
                                                   'max_box_num': 9000,
                                                   'score_threshold': 0.7,
                                                   'variance': [10.0, 10.0, 5.0, 5.0]
                                                   })
    NodeWrap(graph, detection_out_3).replace_obj(
        'Out', {'name': detection_out_3})
    NodeWrap(graph, nms2).replace_obj('ArmNMS',
                                      {'name': nms2,
                                       'image_width': img_width,
                                       'image_height': img_height,
                                       'iou_threshold': 0.7,
                                       'max_box_num': 5000,
                                       'center_point_box': 0
                                       })
    NodeWrap(graph, nms2_out_0).replace_obj('Out', {'name': nms2_out_0})
    NodeWrap(graph, nms2_out_1).replace_obj('Out', {'name': nms2_out_1})
    NodeWrap(graph, nms2_out_2).replace_obj('Out', {'name': nms2_out_2})
    NodeWrap(graph, nms2_out_3).replace_obj('Out', {'name': nms2_out_3})

    clear_redundant_nodes(graph)


def merge_keras_maskrcnn(graph, params):

    def _generate_keras_maskrcnn_anchors(img_width=1024, img_height=1024):
        RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
        RPN_ANCHOR_RATIOS = [0.5, 1, 2]
        BACKBONE_STRIDES = [4, 8, 16, 32, 64]
        RPN_ANCHOR_STRIDE = 1

        backbone_shapes = np.array(
            [[int(math.ceil(img_height / stride)),
              int(math.ceil(img_width / stride))]
             for stride in BACKBONE_STRIDES])

        def _generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
            scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
            scales = scales.flatten()
            ratios = ratios.flatten()
            heights = scales / np.sqrt(ratios)
            widths = scales * np.sqrt(ratios)
            shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
            shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
            shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
            box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
            box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
            box_centers = np.stack(
                [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
            box_sizes = np.stack([box_heights, box_widths],
                                 axis=2).reshape([-1, 2])

            boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                    box_centers + 0.5 * box_sizes], axis=1)
            return boxes

        anchors = []
        for i in range(len(RPN_ANCHOR_SCALES)):
            anchors.append(_generate_anchors(RPN_ANCHOR_SCALES[i], RPN_ANCHOR_RATIOS, backbone_shapes[i],
                                             BACKBONE_STRIDES[i], RPN_ANCHOR_STRIDE))

        anchors = np.concatenate(anchors, axis=0)
        scale = np.array([img_height - 1, img_width - 1,
                          img_height - 1, img_width - 1])
        shift = np.array([0, 0, 1, 1])
        anchors = np.divide((anchors - shift), scale).astype(np.float32)
        return anchors

    # rpn_input_names = ['ROI/mul', 'ROI/strided_slice']
    rpn_out_names = ['rpn_class/concat',
                     'rpn_bbox/concat']
    # rpn_stage_out_names = ['roi_align_classifier/split']
    special_out_convs = ['fpn_p2/BiasAdd',
                         'fpn_p3/BiasAdd',
                         'fpn_p4/BiasAdd',
                         'fpn_p5/BiasAdd']
    special_in_convs = ['mrcnn_class_conv1/convolution',
                        'mrcnn_mask_conv1/convolution']

    mrcnn_box_reshape = 'mrcnn_bbox/Reshape'
    mrcnn_class_reshape = 'mrcnn_class/Reshape_1'
    mrcnn_mask_reshape = 'mrcnn_mask/Reshape_1'
    out_reshapes = [mrcnn_box_reshape, mrcnn_class_reshape]

    all_out_names = rpn_out_names + special_out_convs + out_reshapes
    all_in_names = special_in_convs
    all_names = all_out_names + all_in_names + [mrcnn_mask_reshape]

    if any([not graph.has_node(n) for n in all_names]):
        return
    obj_dict = {n: NodeWrap(graph, n)['object'] for n in all_names}
    if any([v is None for v in obj_dict.values()]):
        ERROR('[Parser]: Meets invalid node in merge_keras_maskrcnn!')
        return
    if len(obj_dict[mrcnn_class_reshape].get_input_tensors()) != 2 \
            or obj_dict[mrcnn_class_reshape].get_input_tensors()[1] is None \
            or obj_dict[mrcnn_class_reshape].get_input_tensors()[1].size != 3:
        ERROR('[Parser]: Meets Reshape node(%s) in merge_keras_maskrcnn!' % mrcnn_class_reshape)
        return

    mrcnn_class_reshape_shape = obj_dict[mrcnn_class_reshape].get_input_tensors()[1].tolist()
    N, class_num = mrcnn_class_reshape_shape[1:3]
    score_threshold = params.get('score_threshold', 0.7)
    model_name = 'mrcnn_detection'

    for out_name in all_out_names:
        out_edges = graph.sorted_out_edges(out_name)
        if out_name in special_out_convs:
            idx = special_out_convs.index(out_name)
            if idx == len(special_out_convs) - 1:
                graph.remove_edges_from(out_edges[2:])
            else:
                graph.remove_edges_from(out_edges[1:])
        else:
            graph.remove_edges_from(out_edges)
    for in_name in all_in_names:
        in_edges = graph.sorted_in_edges(in_name)
        graph.remove_edges_from(in_edges)

    rpn_probs, rpn_boxes = rpn_out_names

    rpn_probs_post_reshape = get_valid_node_name(
        graph, rpn_probs + '_reshape')  # [1,256,-1,2]
    rpn_probs_split = get_valid_node_name(
        graph, rpn_probs + '_split')  # axis = 3, splits = [1,1]
    rpn_probs_split_0_out = get_valid_node_name(
        graph, rpn_probs_split + '_0_out')  # Out node
    rpn_probs_split_1_reshape = get_valid_node_name(
        graph, rpn_probs_split + '_1_reshape')  # [1,261888]

    # axis=1, k=6000, largest=true, sorted=true
    topk = get_valid_node_name(graph, rpn_probs + '_topk')
    topk_0_out = get_valid_node_name(graph, topk + '_0_out')  # Out node
    topk_indices_reshape = get_valid_node_name(
        graph, topk + '_indices_reshape')  # [-1]

    # rpn_class/concat_proposal_nms1_gather_bbox
    box_gather = get_valid_node_name(graph, rpn_boxes + '_gather')  # axis=1
    anchor_value = _generate_keras_maskrcnn_anchors()  # Constant
    # rpn_class/concat_proposal_nms1_gather_anchor, axis=0
    anchor_gather = get_valid_node_name(graph, model_name + '_anchor_gather')

    # rpn_class/concat_proposal
    bounding_box = get_valid_node_name(graph, rpn_probs + '_proposal')
    bounding_box_1_out = get_valid_node_name(graph, bounding_box + '_1_out')
    bounding_box_2_out = get_valid_node_name(graph, bounding_box + '_2_out')
    nms1_box_num_per_class_value = np.array([[N]], dtype=np.int32)
    nms1_total_class_num_value = np.array([[1]], dtype=np.int32)
    # rpn_class/concat_proposal_nms1 ,iou_threshold=0.7, image_height=600, image_width=600
    nms1 = get_valid_node_name(graph, rpn_probs + '_nms1')
    nms1_0_out = get_valid_node_name(graph, nms1 + '_0_out')
    nms1_2_out = get_valid_node_name(graph, nms1 + '_2_out')
    post_nms1 = get_valid_node_name(graph, rpn_probs + '_post_nms1')
    post_nms1_1_out = get_valid_node_name(graph, post_nms1 + '_1_out')  # Out
    # rpn_class/concat_proposal_nms1_gather_pprob, axis=1
    class_gather = get_valid_node_name(graph, rpn_probs + '_gather')

    # begin = [0, 0, 1],  end = [1, 6000, 2], strides = [1, 1, 1]
    class_gather_slice = get_valid_node_name(graph, class_gather + '_slice')
    class_gather_reshape = get_valid_node_name(
        graph, class_gather + '_reshape')  # [1,6000]
    pyramid_roi_proposal = get_valid_node_name(
        graph, model_name + '_pyramid_roi_proposal')
    # mrcnn_detecion/reshape_deltas,   # [1000,81,4]
    deltas_reshape = get_valid_node_name(
        graph, mrcnn_box_reshape + '_deltas')
    # mrcnn_detecion/gather_delta
    deltas_gathernd = get_valid_node_name(
        graph, mrcnn_box_reshape + '_gathernd')  # batch_dims=0
    deltas_gathernd_post_reshape = get_valid_node_name(
        graph, deltas_gathernd + '_post_reshape')  # [1,1000,4]

    # mrcnn_detecion/argmax, axis=2, method=MAX, select_last_index=false
    argmax = get_valid_node_name(graph, model_name + '_argmax')
    argmax_reshape = get_valid_node_name(
        graph, argmax + '_reshape')    # [1000, 1]
    # Cast to float32
    argmax_reshape_cast = get_valid_node_name(graph, argmax_reshape + '_cast')

    # mrcnn_detecion/const_gathernd_index # Const, shape: [1000,1]
    deltas_gathernd_part_indices_value = np.reshape(
        np.array(list(range(N)), dtype=np.int32), [N, 1])

    # mrcnn_detecion/concat_gathernd_index # axis=1
    deltas_gathernd_indices_concat = get_valid_node_name(
        graph, deltas_gathernd + '_indices_concat')

    # mrcnn_detecion/decodebox
    bounding_box2 = get_valid_node_name(graph, model_name + '_decodebox')
    bounding_box2_1_out = get_valid_node_name(
        graph, bounding_box2 + '_1_out')  # Out
    bounding_box2_2_out = get_valid_node_name(
        graph, bounding_box2 + '_2_out')  # Out

    # mrcnn_detecion/reshape_probs
    mrcnn_class_reshape_post_reshape = get_valid_node_name(
        graph, mrcnn_class_reshape + '_post_reshape')  # [1000,81]

    # mrcnn_detecion/gather_score
    score_gathernd = get_valid_node_name(
        graph, mrcnn_class_reshape + '_gathernd')

    # mrcnn_detecion/threshold
    score_theshold_value = np.array([score_threshold] * N, dtype=np.float32)
    score_theshold_name = get_valid_node_name(
        graph, model_name + '_score_theshold')

    # mrcnn_detecion/compare
    greater_equal = get_valid_node_name(graph, model_name + '_compare')

    greater_equal_cast = get_valid_node_name(
        graph, greater_equal + '_cast')

    # mrcnn_detecion/score_out_to_two_dim, [1000,1]
    greater_equal_reshape = get_valid_node_name(
        graph, greater_equal + '_reshape')

    greater_equal_reshape_cast = get_valid_node_name(
        graph, greater_equal_reshape + '_cast')

    # mrcnn_detecion/score_out1
    score_mul = get_valid_node_name(graph, model_name + '_score_mul')   # Mul

    # mrcnn_detecion/class_id_used
    id_mul = get_valid_node_name(graph, model_name + '_id_mul')   # Mul

    # mrcnn_detecion/reshape_to_2DIM_mul and mrcnn_detecion/class_id_used_
    id_mul_reshape = get_valid_node_name(
        graph, id_mul + '_reshape')    # [1, 1000]

    # mrcnn_detecion/topk_sort, axis=1, k=1000, largest=true, sorted=true
    topk_sort = get_valid_node_name(graph, model_name + '_topk_sort')

    topk_sort_out_0 = get_valid_node_name(graph, topk_sort + '_out_0')  # Out

    # mrcnn_detecion/reshape_sorted_index, [1000]
    topk_sort_indices_reshape = get_valid_node_name(
        graph, topk_sort + '_indices_reshape')
    # mrcnn_detecion/reshape_sorted_index_reshape_reshape , [1000, 1]
    topk_sort_indices_reshape_post_reshape = get_valid_node_name(
        graph, topk_sort_indices_reshape + '_post_reshape')

    # mrcnn_detecion/gather_bbox, axis=1
    gather1 = get_valid_node_name(graph, model_name + 'gather_bbox')
    # mrcnn_detecion/histogram, max=80, min=1, nbins=80, discrete=true
    count = get_valid_node_name(graph, model_name + '_histogram')
    # mrcnn_detecion/reverse, batch_axis=0, time_axis=1
    reverse_seq = get_valid_node_name(graph, model_name + '_reverse')
    # mrcnn_detecion/reverse_len
    reverse_seq_len_value = np.array([class_num - 1], dtype=np.int32)
    # mrcnn_detecion/nms,  image_height = 600, image_width = 600,iou_threshold = 0.7
    nms2 = get_valid_node_name(graph, model_name + '_nms')
    # mrcnn_detecion/class_count
    nms2_class_cout_value = np.array([[class_num - 1]], dtype=np.int32)
    nms2_3_out = get_valid_node_name(graph, nms2 + '_3_out')  # Out

    # mrcnn_detecion/gather_sorted_score, axis=0
    gather2 = get_valid_node_name(graph, model_name + 'gather_sorted_score')
    # mrcnn_detecion/reshape_sorted_score_2dim, [1,1000]
    gather2_post_reshape = get_valid_node_name(
        graph, gather2 + '_post_reshape')
    # mrcnn_detecion/topk_final, axis=1, k=100, largest=true, sorted=true
    topk_final = get_valid_node_name(graph, model_name + '_topk_final')
    topk_final_0_out = get_valid_node_name(graph, topk_final + '_0_out')  # Out
    # mrcnn_detecion/reshape_final_topk_index_1dim, [100]
    topk_final_1_reshape = get_valid_node_name(
        graph, topk_final + '_1_reshape')
    # mrcnn_detecion/gather_output_bbox, axis=1
    gather3 = get_valid_node_name(graph, model_name + '_gather_output_bbox')
    # mrcnn_detecion/num_reshape, [80]
    nms2_1_reshape = get_valid_node_name(graph, nms2 + '_1_reshape')
    # PyramidRoi_mask
    pyramid_roi_mask = get_valid_node_name(
        graph, model_name + '_pyramid_roi_mask')
    # mrcnn_detecion/repeat, axis=1
    repeat = get_valid_node_name(graph, model_name + '_repeat')
    repeat_const_value = np.array(
        list(range(class_num - 1, 0, -1)), dtype=np.int32)  # [1, 80]
    repeat_const_value = np.reshape(repeat_const_value, [1, class_num - 1])
    # mrcnn_detecion/gather_output_classid, axis=1
    gather4 = get_valid_node_name(graph, model_name + '_gather_output_classid')
    gather4_out = get_valid_node_name(graph, gather4 + '_out')

    graph.add_edge(rpn_probs, rpn_probs_post_reshape)
    graph.add_edge(rpn_probs_post_reshape, rpn_probs_split)
    graph.add_edge(rpn_probs_split, rpn_probs_split_0_out)
    graph.add_edge(rpn_probs_split, rpn_probs_split_1_reshape, **{'src_out_port': 1})
    graph.add_edge(rpn_probs_split_1_reshape, topk)
    graph.add_edge(topk, topk_0_out)
    graph.add_edge(topk, topk_indices_reshape, **{'src_out_port': 1})

    graph.add_edge(rpn_boxes, box_gather)
    graph.add_edge(topk_indices_reshape, box_gather, **{'dst_in_port': 1})

    graph.add_edge(topk, anchor_gather, **{'src_out_port': 1, 'dst_in_port': 1})
    insert_constant(graph, model_name + '_anchor', anchor_value,
                    anchor_gather, data_format='NHWC')

    graph.add_edge(anchor_gather, bounding_box)
    graph.add_edge(box_gather, bounding_box, **{'dst_in_port': 1})
    graph.add_edge(bounding_box, nms1)
    graph.add_edge(bounding_box, post_nms1)
    graph.add_edge(bounding_box, bounding_box_1_out, **{'src_out_port': 1})
    graph.add_edge(bounding_box, bounding_box_2_out, **{'src_out_port': 2})

    graph.add_edge(rpn_probs, class_gather)
    graph.add_edge(topk_indices_reshape, class_gather, **{'dst_in_port': 1})
    graph.add_edge(class_gather, class_gather_slice)
    graph.add_edge(class_gather_slice, class_gather_reshape)

    insert_constant(graph,
                    nms1 + '_box_num_per_class',
                    nms1_box_num_per_class_value,
                    nms1,
                    in_port=1,
                    data_format='NHWC')
    insert_constant(graph,
                    nms1 + '_total_class_num',
                    nms1_total_class_num_value,
                    nms1,
                    in_port=2,
                    data_format='NHWC')
    graph.add_edge(class_gather_reshape, nms1, **{'dst_in_port': 3})
    graph.add_edge(nms1, nms1_0_out)
    graph.add_edge(nms1, post_nms1, **{'src_out_port': 1, 'dst_in_port': 2})
    graph.add_edge(nms1, nms1_2_out, **
                   {'src_out_port': 2})
    graph.add_edge(nms1, post_nms1, **{'src_out_port': 3, 'dst_in_port': 1})

    graph.add_edge(post_nms1, pyramid_roi_proposal)
    graph.add_edge(post_nms1, bounding_box2)
    graph.add_edge(post_nms1, post_nms1_1_out, **{'src_out_port': 1})
    for i, conv in enumerate(special_out_convs):
        graph.add_edge(conv, pyramid_roi_proposal, **
                       {'dst_in_port': i + 1})

    graph.add_edge(pyramid_roi_proposal, special_in_convs[0])
    graph.add_edge(mrcnn_box_reshape, deltas_reshape)

    graph.add_edge(mrcnn_class_reshape, argmax)
    graph.add_edge(argmax, argmax_reshape)

    graph.add_edge(argmax_reshape, deltas_gathernd_indices_concat,
                   **{'dst_in_port': 1})
    insert_constant(graph,
                    deltas_gathernd + '_part_indices',
                    deltas_gathernd_part_indices_value,
                    deltas_gathernd_indices_concat,
                    data_format='NHWC')
    graph.add_edge(argmax_reshape, argmax_reshape_cast)

    graph.add_edge(deltas_reshape, deltas_gathernd)
    graph.add_edge(deltas_gathernd_indices_concat,
                   deltas_gathernd, **{'dst_in_port': 1})
    graph.add_edge(deltas_gathernd_indices_concat,
                   score_gathernd, **{'dst_in_port': 1})
    graph.add_edge(deltas_gathernd, deltas_gathernd_post_reshape)
    graph.add_edge(deltas_gathernd_post_reshape,
                   bounding_box2, **{'dst_in_port': 1})

    graph.add_edge(mrcnn_class_reshape, mrcnn_class_reshape_post_reshape)
    graph.add_edge(mrcnn_class_reshape_post_reshape, score_gathernd)

    graph.add_edge(score_gathernd, score_mul, **{'dst_in_port': 1})
    graph.add_edge(score_gathernd, greater_equal)
    graph.add_edge(score_theshold_name, greater_equal, **{'dst_in_port': 1})
    NodeWrap(graph, score_theshold_name).replace_obj('Constant',
                                                     {'name': score_theshold_name,
                                                      'opset_version': 1,
                                                      'value': score_theshold_value,
                                                      'data_format': 'NHWC'
                                                      })
    graph.add_edge(greater_equal, greater_equal_reshape)
    graph.add_edge(greater_equal_reshape, greater_equal_reshape_cast)
    graph.add_edge(greater_equal, greater_equal_cast)
    graph.add_edge(greater_equal_cast, score_mul)
    graph.add_edge(score_mul, gather2)

    graph.add_edge(greater_equal_reshape_cast, id_mul)
    graph.add_edge(argmax_reshape_cast, id_mul, **{'dst_in_port': 1})
    graph.add_edge(id_mul, id_mul_reshape)
    graph.add_edge(id_mul_reshape, topk_sort)
    graph.add_edge(id_mul_reshape, count)
    graph.add_edge(topk_sort, topk_sort_out_0)
    graph.add_edge(topk_sort, topk_sort_indices_reshape, **{'src_out_port': 1})

    graph.add_edge(topk_sort_indices_reshape, gather1, **{'dst_in_port': 1})
    graph.add_edge(topk_sort_indices_reshape,
                   topk_sort_indices_reshape_post_reshape)
    graph.add_edge(topk_sort_indices_reshape_post_reshape,
                   gather2, **{'dst_in_port': 1})
    graph.add_edge(gather2, gather2_post_reshape)
    graph.add_edge(bounding_box2, gather1)
    graph.add_edge(bounding_box2, bounding_box2_1_out, **{'src_out_port': 1})
    graph.add_edge(bounding_box2, bounding_box2_2_out, **{'src_out_port': 2})
    graph.add_edge(count, reverse_seq)
    insert_constant(graph,
                    reverse_seq + '_len',
                    reverse_seq_len_value,
                    reverse_seq,
                    in_port=1,
                    data_format='NHWC')

    graph.add_edge(gather1, nms2)
    graph.add_edge(reverse_seq, nms2, **{'dst_in_port': 1})
    insert_constant(graph,
                    nms2 + '_class_cout',
                    nms2_class_cout_value,
                    nms2,
                    in_port=2,
                    data_format='NHWC')
    graph.add_edge(gather2_post_reshape, nms2, **{'dst_in_port': 3})
    graph.add_edge(nms2, gather3, **{'src_out_port': 0})
    graph.add_edge(nms2, nms2_1_reshape, **{'src_out_port': 1})
    graph.add_edge(nms2, topk_final, **{'src_out_port': 2})
    graph.add_edge(nms2, nms2_3_out, **{'src_out_port': 3})

    graph.add_edge(topk_final, topk_final_0_out)
    graph.add_edge(topk_final, topk_final_1_reshape, **{'src_out_port': 1})
    graph.add_edge(topk_final_1_reshape, gather3, **{'dst_in_port': 1})
    graph.add_edge(gather3, pyramid_roi_mask)
    for i, conv in enumerate(special_out_convs):
        graph.add_edge(conv, pyramid_roi_mask, **{'dst_in_port': i + 1})
    graph.add_edge(pyramid_roi_mask, special_in_convs[1])

    graph.add_edge(nms2_1_reshape, repeat, **{'dst_in_port': 1})
    insert_constant(graph, repeat + '_const',
                    repeat_const_value, repeat, data_format='NHWC')
    graph.add_edge(repeat, gather4)
    graph.add_edge(topk_final_1_reshape, gather4, **{'dst_in_port': 1})
    graph.add_edge(gather4, gather4_out)

    place_reshape(graph, rpn_probs_post_reshape, [1, 256, -1, 2])
    NodeWrap(graph, rpn_probs_split).replace_obj('Split',
                                                 {'name': rpn_probs_split,
                                                  'opset_version': 2,
                                                  'axis': 3,
                                                  'split': [1, 1]
                                                  })
    NodeWrap(graph, rpn_probs_split_0_out).replace_obj(
        'Out', {'name': rpn_probs_split_0_out})
    place_reshape(graph, rpn_probs_split_1_reshape, [1, 261888])
    NodeWrap(graph, topk).replace_obj('TopK',
                                      {'name': topk,
                                       'opset_version': 1,
                                       'axis': 1,
                                       'k': 6000,
                                       'largest': True,
                                       'sorted': True}
                                      )
    NodeWrap(graph, topk_0_out).replace_obj('Out', {'name': topk_0_out})
    place_reshape(graph, topk_indices_reshape, [-1])
    NodeWrap(graph, box_gather).replace_obj('Gather',
                                            {'name': box_gather,
                                             'opset_version': 11,
                                             'axis': 1}
                                            )
    NodeWrap(graph, anchor_gather).replace_obj('Gather',
                                               {'name': anchor_gather,
                                                'opset_version': 11,
                                                'axis': 0}
                                               )
    NodeWrap(graph, bounding_box).replace_obj(
        'ArmBoundingBox', {'name': bounding_box})
    NodeWrap(graph, bounding_box_1_out).replace_obj(
        'Out', {'name': bounding_box_1_out})
    NodeWrap(graph, bounding_box_2_out).replace_obj(
        'Out', {'name': bounding_box_2_out})
    NodeWrap(graph, nms1).replace_obj('ArmNMS',
                                      {'name': nms1,
                                       'image_width': 600,
                                       'image_height': 600,
                                       'center_point_box': 0,
                                       'iou_threshold': 0.7
                                       })
    NodeWrap(graph, nms1_0_out).replace_obj('Out', {'name': nms1_0_out})
    NodeWrap(graph, nms1_2_out).replace_obj('Out', {'name': nms1_2_out})
    NodeWrap(graph, post_nms1).replace_obj('ArmPostNMS1',
                                           {'name': post_nms1,
                                            'image_width': 1024,
                                            'image_height': 1024,
                                            'proposal_cnt': N
                                            })
    NodeWrap(graph, post_nms1_1_out).replace_obj(
        'Out', {'name': post_nms1_1_out})
    NodeWrap(graph, class_gather).replace_obj('Gather',
                                              {'name': class_gather,
                                               'opset_version': 11,
                                               'axis': 1}
                                              )
    NodeWrap(graph, class_gather_slice).replace_obj('Slice',
                                                    {'name': class_gather_slice,
                                                     'opset_version': 1,
                                                     'axes': [0, 1, 2],
                                                     'starts': [0, 0, 1],
                                                     'ends': [1, 6000, 2]
                                                     })
    place_reshape(graph, class_gather_reshape, [1, 6000])
    NodeWrap(graph, pyramid_roi_proposal).replace_obj('ArmPyramidROIAlign',
                                                      {'name': pyramid_roi_proposal,
                                                       'resize_width': 7,
                                                       'resize_height': 7
                                                       })
    place_reshape(graph, deltas_reshape, [N, class_num, 4])
    NodeWrap(graph, deltas_gathernd).replace_obj('GatherND',
                                                 {'name': deltas_gathernd,
                                                  'opset_version': 12
                                                  })
    place_reshape(graph, deltas_gathernd_post_reshape, [1, N, 4])
    NodeWrap(graph, argmax).replace_obj('ArgMax',
                                        {'name': argmax,
                                         'opset_version': 13,
                                         'axis': 2,
                                         'keepdims': True
                                         })
    place_reshape(graph, argmax_reshape, [N, 1])
    NodeWrap(graph, argmax_reshape_cast).replace_obj('Cast',
                                                     {'name': argmax_reshape_cast,
                                                      'opset_version': 1,
                                                      'to': 'float32'
                                                      })
    NodeWrap(graph, deltas_gathernd_indices_concat).replace_obj('Concat',
                                                                {'name': deltas_gathernd_indices_concat,
                                                                 'opset_version': 4,
                                                                 'axis': 1
                                                                 })
    NodeWrap(graph, bounding_box2).replace_obj(
        'ArmBoundingBox', {'name': bounding_box2})
    NodeWrap(graph, bounding_box2_1_out).replace_obj(
        'Out', {'name': bounding_box2_1_out})
    NodeWrap(graph, bounding_box2_2_out).replace_obj(
        'Out', {'name': bounding_box2_2_out})
    place_reshape(graph, mrcnn_class_reshape_post_reshape, [N, class_num])
    NodeWrap(graph, score_gathernd).replace_obj('GatherND',
                                                {'name': score_gathernd,
                                                 'opset_version': 12
                                                 })
    NodeWrap(graph, greater_equal).replace_obj('GreaterOrEqual',
                                               {'name': greater_equal,
                                                'opset_version': 12
                                                })
    NodeWrap(graph, greater_equal_cast).replace_obj('Cast',
                                                    {'name': greater_equal_cast,
                                                     'opset_version': 1,
                                                     'to': 'float32'
                                                     })
    place_reshape(graph, greater_equal_reshape, [N, 1])
    NodeWrap(graph, greater_equal_reshape_cast).replace_obj('Cast',
                                                            {'name': greater_equal_reshape_cast,
                                                             'opset_version': 1,
                                                             'to': 'float32'
                                                             })
    NodeWrap(graph, score_mul).replace_obj('Mul',
                                           {'name': score_mul,
                                            'opset_version': 7
                                            })
    NodeWrap(graph, id_mul).replace_obj('Mul',
                                        {'name': id_mul,
                                         'opset_version': 7
                                         })
    place_reshape(graph, id_mul_reshape, [1, N])
    NodeWrap(graph, topk_sort).replace_obj('TopK',
                                           {'name': topk_sort,
                                            'opset_version': 1,
                                            'axis': 1,
                                            'k': N,
                                            'largest': True,
                                            'sorted': True}
                                           )
    NodeWrap(graph, topk_sort_out_0).replace_obj(
        'Out', {'name': topk_sort_out_0})
    place_reshape(graph, topk_sort_indices_reshape, [N])
    place_reshape(graph, topk_sort_indices_reshape_post_reshape, [N, 1])
    NodeWrap(graph, gather1).replace_obj('Gather',
                                         {'name': gather1,
                                          'opset_version': 11,
                                          'axis': 1}
                                         )
    NodeWrap(graph, count).replace_obj('ArmCount',
                                       {'name': count,
                                        'discrete': True,
                                        'min': 1,
                                        'max': class_num - 1,
                                        'nbins': class_num - 1
                                        })
    NodeWrap(graph, reverse_seq).replace_obj('ReverseSequence',
                                             {'name': reverse_seq,
                                              'opset_version': 10,
                                              'batch_axis': 0,
                                              'time_axis': 1
                                              })
    NodeWrap(graph, nms2).replace_obj('ArmNMS',
                                      {'name': nms2,
                                       'image_width': 600,
                                       'image_height': 600,
                                       'center_point_box': 0,
                                       'iou_threshold': 0.7,
                                       'max_box_num': N
                                       })
    NodeWrap(graph, nms2_3_out).replace_obj('Out', {'name': nms2_3_out})
    NodeWrap(graph, gather2).replace_obj('Gather',
                                         {'name': gather2,
                                          'opset_version': 11,
                                          'axis': 0}
                                         )
    place_reshape(graph, gather2_post_reshape, [1, N])
    NodeWrap(graph, topk_final).replace_obj('TopK',
                                            {'name': topk_final,
                                             'opset_version': 1,
                                             'axis': 1,
                                             'k': 100,
                                             'largest': True,
                                             'sorted': True}
                                            )
    NodeWrap(graph, topk_final_0_out).replace_obj(
        'Out', {'name': topk_final_0_out})
    place_reshape(graph, topk_final_1_reshape, [100])
    NodeWrap(graph, gather3).replace_obj('Gather',
                                         {'name': gather3,
                                          'opset_version': 11,
                                          'axis': 1}
                                         )
    place_reshape(graph, nms2_1_reshape, [class_num - 1])
    NodeWrap(graph, pyramid_roi_mask).replace_obj('ArmPyramidROIAlign',
                                                  {'name': pyramid_roi_mask,
                                                   'resize_width': 14,
                                                   'resize_height': 14
                                                   })
    NodeWrap(graph, repeat).replace_obj('ArmRepeat',
                                        {'name': repeat,
                                         'axis': 1
                                         })
    NodeWrap(graph, gather4).replace_obj('Gather',
                                         {'name': gather4,
                                          'opset_version': 11,
                                          'axis': 1}
                                         )
    NodeWrap(graph, gather4_out).replace_obj('Out', {'name': gather4_out})

    graph._attr['output_names'] = [
        gather3, mrcnn_mask_reshape, topk_final, gather4]
    clear_redundant_nodes(graph)


def merge_embedding_lookup_sparse(graph):
    matched = False
    matches = matched_patterns(graph,
                               nodes=[
                                   ('unique', {'op': 'TfUnique'}),
                                   ('indices', {'op': ['TfConst', 'Constant']}),
                                   ('gather', {'op': 'TfGatherV2'}),
                                   ('segment_ids', {'op': ['TfConst', 'Constant']}),
                                   ('segment', {'op': ['TfSparseSegmentMean',
                                    'TfSparseSegmentSum', 'TfSparseSegmentSqrtN']}),
                               ],
                               edges=[
                                   ('unique', 'gather', {'src_out_port': 0, 'dst_in_port': 1}),
                                   ('indices', 'gather', {'dst_in_port': 2}),
                                   ('gather', 'segment', {'src_out_port': 0, 'dst_in_port': 0}),
                                   ('unique', 'segment', {'src_out_port': 1, 'dst_in_port': 1}),
                                   ('segment_ids', 'segment', {'dst_in_port': 2}),
                               ])
    for m in matches:
        names = ['unique', 'indices', 'gather', 'segment_ids', 'segment']
        objs_dict = {n: NodeWrap(graph, m[n])['object'] for n in names}
        if any(obj is None for obj in objs_dict.values()):
            ERROR('[Parser]: Meets invalid Op in merge_embedding_lookup_sparse!')
            continue
        if objs_dict['gather'].batch_dims != 0 \
                or objs_dict['gather'].axis != 0:
            continue
        unique_in_edges = graph.sorted_in_edges(m['unique'], data=True)
        gather_in_edges = graph.sorted_in_edges(m['gather'], data=True)
        segment_in_edges = graph.sorted_in_edges(m['segment'], data=True)
        unique_out_edges = graph.sorted_out_edges(m['unique'])
        gather_out_edges = graph.sorted_out_edges(m['gather'])
        if len(unique_in_edges) != 1 \
                or len(gather_in_edges) != 3 \
                or len(segment_in_edges) != 3 \
                or len(unique_out_edges) != 2 \
                or len(gather_out_edges) != 1:
            continue
        unique_inputs = objs_dict['unique'].get_input_tensors()
        if len(unique_inputs) != 1 \
                or unique_inputs[0] is None:
            continue
        matched = True
        weights = np.ones_like(unique_inputs[0], np.float32)
        src, _, in_attr = gather_in_edges[0]
        ids, _, ids_in_attr = segment_in_edges[2]
        value, _, value_in_attr = unique_in_edges[0]
        graph.remove_edges_from(segment_in_edges)
        graph.add_edge(src, m['segment'], **in_attr)
        ids_in_attr['dst_in_port'] = 1
        graph.add_edge(ids, m['segment'], **ids_in_attr)
        value_in_attr['dst_in_port'] = 2
        graph.add_edge(value, m['segment'], **value_in_attr)
        insert_constant(graph, m['segment'] + '_weights', weights,
                        m['segment'], in_port=3, data_format='NHWC')
        segment_attr = objs_dict['segment'].copied_attr()
        segment_attr.update({'combiner': re.sub('^TfSparseSegment', '', objs_dict['segment'].type).upper()})
        NodeWrap(graph, m['segment']).replace_obj('ArmEmbeddingLookupSparse', segment_attr)
    if matched:
        clear_redundant_nodes(graph)


def merge_embedding_lookup_sparse_with_weights(graph):
    matched = False
    const_types = ['TfConst', 'Constant']
    elsw_sum_matches = matched_patterns(graph,
                                        nodes=[
                                            ('unique', {'op': 'TfUnique'}),
                                            ('gather1', {'op': 'TfGatherV2'}),
                                            ('gather1_axis', {'op': const_types}),
                                            ('gather2', {'op': 'TfGatherV2'}),
                                            ('gather2_axis', {'op': const_types}),
                                            ('reshape', {'op': 'TfReshape'}),
                                            ('reshape_shape', {'op': const_types}),
                                            ('mul', {'op': 'TfMul'}),
                                            ('segment_sum', {'op': 'TfSegmentSum'}),
                                            ('segment_ids', {'op': const_types}),
                                        ],
                                        edges=[
                                            ('unique', 'gather1', {'src_out_port': 0, 'dst_in_port': 1}),
                                            ('gather1_axis', 'gather1', {'dst_in_port': 2}),
                                            ('unique', 'gather2', {'src_out_port': 1, 'dst_in_port': 1}),
                                            ('gather2_axis', 'gather2', {'dst_in_port': 2}),
                                            ('reshape_shape', 'reshape', {'dst_in_port': 1}),
                                            ('gather2', 'mul', {'dst_in_port': 0}),
                                            ('reshape', 'mul', {'dst_in_port': 1}),
                                            ('mul', 'segment_sum', {'dst_in_port': 0}),
                                            ('segment_ids', 'segment_sum', {'dst_in_port': 1})
                                        ])
    for m in elsw_sum_matches:
        names = ['unique', 'gather1', 'gather1_axis', 'gather2', 'gather2_axis',
                 'reshape', 'reshape_shape', 'mul', 'segment_sum', 'segment_ids']
        obj_dict = {k: NodeWrap(graph, m[k])['object'] for k in names}
        if any(obj is None for obj in obj_dict.values()):
            ERROR('[Parser]: Meets invalid op in merge_embedding_lookup_sparse_with_weights!')
            continue
        if obj_dict['gather1'].axis != 0 \
                or obj_dict['gather1'].batch_dims != 0 \
                or obj_dict['gather2'].axis != 0 \
                or obj_dict['gather2'].batch_dims != 0:
            continue

        gather1_in_edges = graph.sorted_in_edges(m['gather1'], data=True)
        unique_in_edges = graph.sorted_in_edges(m['unique'], data=True)
        reshape_in_edges = graph.sorted_in_edges(m['reshape'], data=True)
        segment_sum_in_edges = graph.sorted_in_edges(m['segment_sum'], data=True)

        if len(gather1_in_edges) != 3 \
                or len(unique_in_edges) != 1 \
                or len(reshape_in_edges) != 2 \
                or len(segment_sum_in_edges) != 2:
            continue

        reshape_dim = obj_dict['reshape_shape'].value
        indices = obj_dict['segment_ids'].value
        if reshape_dim is None or indices is None:
            ERROR('[Parser]: Meets invalid Constant op in merge_embedding_lookup_sparse_with_weights!')
            continue
        if reshape_dim.item(0) != indices.size or int(np.product(reshape_dim)) != indices.size:
            continue

        mean_matches = matched_patterns(graph,
                                        nodes=[
                                            ('segment_sum', {'op': 'TfSegmentSum'}),
                                            ('segment_ids', {'op': const_types}),
                                            ('reshape', {'op': 'TfReshape'}),
                                            ('segment_sum2', {'op': 'TfSegmentSum'}),
                                            ('div', {'op': 'TfDivNoNan'})
                                        ],
                                        edges=[
                                            ('segment_ids', 'segment_sum', {'dst_in_port': 1}),
                                            ('segment_ids', 'segment_sum2', {'dst_in_port': 1}),
                                            ('reshape', 'segment_sum2', {'dst_in_port': 0}),
                                            ('segment_sum', 'div', {'dst_in_port': 0}),
                                            ('segment_sum2', 'div', {'dst_in_port': 1})
                                        ])
        mean_matches = [m1 for m1 in mean_matches
                        if m1['segment_sum'] == m['segment_sum'] and m1['segment_ids'] == m['segment_ids'] and m1['reshape'] == m['reshape']]
        if len(mean_matches) > 1:
            ERROR('[Parser]: Pattern error in merge_embedding_lookup_sparse_with_weights!')
            continue
        if mean_matches:
            mean_names = ['segment_sum2', 'div']
            mean_obj_dict = {k: NodeWrap(graph, mean_matches[0][k])['object'] for k in mean_names}
            if any(obj is None for obj in mean_obj_dict.values()):
                ERROR('[Parser]: Meets invalid op in merge_embedding_lookup_sparse_with_weights!')
                continue

        sqrtn_matches = matched_patterns(graph,
                                         nodes=[
                                             ('segment_sum', {'op': 'TfSegmentSum'}),
                                             ('segment_ids', {'op': const_types}),
                                             ('reshape', {'op': 'TfReshape'}),
                                             ('pow', {'op': 'TfPow'}),
                                             ('pow_y', {'op': const_types}),
                                             ('segment_sum2', {'op': 'TfSegmentSum'}),
                                             ('sqrt', {'op': 'TfSqrt'}),
                                             ('div', {'op': 'TfDivNoNan'})
                                         ],
                                         edges=[
                                             ('segment_ids', 'segment_sum', {'dst_in_port': 1}),
                                             ('segment_ids', 'segment_sum2', {'dst_in_port': 1}),
                                             ('reshape', 'pow', {'dst_in_port': 0}),
                                             ('pow_y', 'pow', {'dst_in_port': 1}),
                                             ('pow', 'segment_sum2', {'dst_in_port': 0}),
                                             ('segment_sum', 'div', {'dst_in_port': 0}),
                                             ('segment_sum2', 'sqrt', {'dst_in_port': 0}),
                                             ('sqrt', 'div', {'dst_in_port': 1})
                                         ])
        sqrtn_matches = [m2 for m2 in sqrtn_matches
                         if m2['segment_sum'] == m['segment_sum']
                         and m2['segment_ids'] == m['segment_ids']
                         and m2['reshape'] == m['reshape']
                         and NodeWrap(graph, m2['pow_y'])['object'] is not None
                         and FLOAT_EQUAL(NodeWrap(graph, m2['pow_y'])['object'].value, 2)]
        if len(sqrtn_matches) > 1:
            ERROR('[Parser]: Pattern error in merge_embedding_lookup_sparse_with_weights!')
            continue
        if sqrtn_matches:
            sqrtn_names = ['pow', 'pow_y', 'segment_sum2', 'sqrt', 'div']
            sqrtn_obj_dict = {k: NodeWrap(graph, sqrtn_matches[0][k])['object'] for k in sqrtn_names}
            if any(obj is None for obj in sqrtn_obj_dict.values()):
                ERROR('[Parser]: Meets invalid op in merge_embedding_lookup_sparse_with_weights!')
                continue
        if len(mean_matches) == 1 and len(sqrtn_matches) == 1:
            ERROR('[Parser]: Pattern error in merge_embedding_lookup_sparse_with_weights!')
            continue

        matched = True
        second_matched = False
        src, _, in_attr = gather1_in_edges[0]
        indices_name, _, indices_in_attr = segment_sum_in_edges[1]
        ids_name, _, unique_in_attr = unique_in_edges[0]
        weights_name, _, reshape_in_attr = reshape_in_edges[0]
        if mean_matches:
            second_matched = True
            last_name = mean_matches[0]['div']
            combiner = 'MEAN'
        elif sqrtn_matches:
            second_matched = True
            last_name = sqrtn_matches[0]['div']
            combiner = 'SQRTN'
        else:
            last_name = m['segment_sum']
            combiner = 'SUM'

        last_in_edges = graph.sorted_in_edges(last_name)
        graph.remove_edges_from(last_in_edges)
        graph.add_edge(src, last_name, **in_attr)
        graph.add_edge(indices_name, last_name, **indices_in_attr)
        ids_in_attr = copy.deepcopy(unique_in_attr)
        ids_in_attr['dst_in_port'] = 2
        graph.add_edge(ids_name, last_name, **ids_in_attr)
        weights_in_attr = copy.deepcopy(reshape_in_attr)
        weights_in_attr['dst_in_port'] = 3
        graph.add_edge(weights_name, last_name, **weights_in_attr)
        embedding_lookup_attr = NodeWrap(graph, last_name)['object'].copied_attr()
        embedding_lookup_attr.update({'combiner': combiner})
        NodeWrap(graph, last_name).replace_obj('ArmEmbeddingLookupSparse', embedding_lookup_attr)
        if second_matched:
            clear_redundant_nodes(graph)

    if matched:
        clear_redundant_nodes(graph)


def convert_to_onnx(graph):
    '''Convert the model to the onnx version.'''
    tf_ops = TfOp.get_concrete_subclass_names()
    matches = extend_lists([single_node_matcher(graph, op_type)
                            for op_type in tf_ops])
    for m in matches:
        node_name = m['target']
        node_obj = NodeWrap(graph, node_name)['object']
        if node_obj is not None:
            in_edges = graph.sorted_in_edges(node_name, data=True)
            node_data_format = 'NCHW' if node_obj.data_format.startswith('NC') else 'NHWC'
            pure_type = re.sub(r'^Tf', '', node_obj.type)
            if getattr(node_obj, 'correspond_onnx_op', None) is not None:
                if isinstance(node_obj, TfHasPaddingStrides):
                    input_shapes = node_obj.get_input_shapes()
                    output_shapes = node_obj.get_output_shapes()
                    if len(input_shapes) < 1 \
                            or len(input_shapes[0]) < 3 \
                            or any(s is None for s in input_shapes[0]) \
                            or len(output_shapes) < 1 \
                            or len(output_shapes[0]) < 3 \
                            or any(s is None for s in output_shapes[0]):
                        ERROR('[Parser]: Invalid TfHasPaddingStrides Node(%s) in convert_to_onnx!' % node_name)
                        continue
                    node_obj.update_pads(input_shapes[0], output_shapes[0])
                new_node_attr = node_obj.copied_attr()

                if isinstance(node_obj, OpHasWeights):
                    if node_obj.weights is None:
                        ERROR('[Parser]: Node(%s) does not contain weights!' %
                              node_name)
                        continue
                    new_weights = node_obj.weights
                    if pure_type == 'DepthwiseConv2dNative':
                        new_weights = np.reshape(new_weights, list(
                            new_weights.shape[:2]) + [-1, 1])
                    new_weights = np.transpose(
                        new_weights, axes=type(node_obj).perm_tf_to_onnx())
                    new_node_attr.update({'weights': new_weights})

                if pure_type in ('All', 'Any', 'Max', 'Min', 'Prod'):
                    graph.remove_edges_from(in_edges[1:])
                elif pure_type in ('ArgMax', 'ArgMin'):
                    new_node_attr.update(
                        {'axis': node_obj.axis, 'keepdims': 0})
                elif pure_type == 'BiasAdd':
                    if node_data_format == 'NCHW' \
                            and len(node_obj.get_input_tensors()) > 1:
                        data1 = node_obj.get_input_tensors()[1]
                        if len(data1.shape) == 1 \
                                and node_obj.get_input_shapes()[0][1] == data1.shape[0]:
                            src, _, in_attr = in_edges[1]
                            reshape_dim = [-1] + [1] * \
                                len(node_obj.get_input_shapes()[0][2:])
                            insert_reshape(graph, src, node_name,
                                           in_attr, reshape_dim)
                elif pure_type == 'Cast':
                    new_node_attr.update({'to': new_node_attr['DstT']})
                elif pure_type == 'ComputeAccidentalHits':
                    out_edges = graph.sorted_out_edges(
                        node_name, data=True, keys=True)
                    const_name = get_valid_node_name(
                        graph, node_name + '_weights')
                    matched = False
                    const_value = None
                    for _, dst, k, out_attr in out_edges:
                        if out_attr['src_out_port'] != 2:
                            continue
                        matched = True
                        graph.remove_edge(node_name, dst, key=k)
                        const_value = out_attr['tensor'].value
                        new_out_attr = copy.deepcopy(out_attr)
                        new_out_attr.update(
                            {'src_out_port': 0, 'tensor': Tensor(value=const_value, is_const=True)})
                        graph.add_edge(const_name, dst, **new_out_attr)
                    if matched:
                        const_attr = {'name': const_name,
                                      'value': const_value,
                                      'opset_version': 9}
                        NodeWrap(graph, const_name).replace_obj(
                            'Constant', const_attr)
                elif pure_type == 'ConcatV2':
                    graph.remove_edges_from(in_edges[-1:])
                    new_node_attr.update(
                        {'axis': int(in_edges[-1][2]['tensor'].value)})
                elif pure_type == 'CropAndResize':
                    if len(in_edges) < 4 or not in_edges[3][2]['tensor'].is_const:
                        ERROR(
                            '[Parser]: Invalid TF CropAndResize Node(%s) to convert to Onnx!' % node_name)
                        continue
                    crop_size = in_edges[3][2]['tensor'].value.tolist()
                    new_node_attr.update({'crop_size': crop_size,
                                          'method': node_obj.method.upper()})
                    graph.remove_edges_from(in_edges[3:])
                elif pure_type == 'CTCGreedyDecoder':
                    if len(in_edges) == 2 \
                            and len(node_obj.get_input_shapes()) == 2 \
                            and len(node_obj.get_input_shapes()[0]) == 3:
                        src, _, in_attr = in_edges[0]
                        insert_transpose(graph, src, node_name,
                                         in_attr, [1, 0, 2])
                    else:
                        ERROR(
                            '[Parser]: Invalid TF CTCGreedyDecoder Node(%s) to convert to Onnx!' % node_name)
                        continue

                elif pure_type == 'Cumprod':
                    if len(in_edges) > 1 and in_edges[1][2]['tensor'].is_const:
                        axis_value = int(in_edges[1][2]['tensor'].value)
                        new_node_attr.update({'axis': axis_value})
                        graph.remove_edges_from(in_edges[1:])
                elif pure_type == 'ExpandDims':
                    if len(in_edges) >= 2 \
                            and len(node_obj.get_input_tensors()) >= 2 \
                            and node_obj.get_input_tensors()[0] is not None:
                        axis = node_obj.axis
                        out_tensor = np.expand_dims(
                            node_obj.get_input_tensors()[0], axis)
                        graph.remove_edges_from(in_edges[1:])
                        insert_constant(graph,
                                        node_name + '_shape',
                                        np.array(out_tensor.shape, np.int32),
                                        node_name,
                                        in_port=1,
                                        data_format='NHWC')
                    else:
                        ERROR(
                            '[Parser]: Invalid TF ExpandDims Node(%s) to convert to Onnx!' % node_name)
                        continue
                elif pure_type == 'FloorMod':
                    new_node_attr.update({'fmod': 0})
                elif pure_type == 'FloorDiv':
                    in_edges = graph.sorted_in_edges(
                        node_name, data=True, keys=True)
                    out_edges = graph.sorted_out_edges(
                        node_name, data=True, keys=True)
                    if len(out_edges) > 0 and len(node_obj.get_input_tensors()) > 1\
                            and len(in_edges) > 0:
                        floor_name = get_valid_node_name(
                            graph, node_name + '_floor')
                        dtype1 = str(node_obj.get_input_tensors()[0].dtype)
                        dtype2 = str(node_obj.get_input_tensors()[1].dtype)
                        for src_in, dst_in, k, in_attr in in_edges:
                            if 'int' in str(in_attr['tensor'].value.dtype):
                                insert_cast(graph, src_in, dst_in,
                                            'float32', in_attr=in_attr, key=k)
                        _, _, _, in_attr = in_edges[0]
                        graph.remove_edges_from(out_edges)
                        if 'int' in dtype1 \
                                and 'int' in dtype2:
                            cast_name = get_valid_node_name(
                                graph, node_name + '_post_cast')
                            graph.add_edge(floor_name, cast_name)
                            for _, dst, k, out_attr in out_edges:
                                graph.add_edge(cast_name, dst, **out_attr)
                            cast_attr = {'name': cast_name,
                                         'opset_version': 13, 'to': 'int32'}
                            NodeWrap(graph, cast_name).replace_obj(
                                'Cast', cast_attr)
                            if node_name in graph._attr['output_names']:
                                index = graph._attr['output_names'].index(
                                    node_name)
                                graph._attr['output_names'][index] = cast_name
                        else:
                            for _, dst, k, out_attr in out_edges:
                                graph.add_edge(floor_name, dst, **out_attr)
                            if node_name in graph._attr['output_names']:
                                index = graph._attr['output_names'].index(
                                    node_name)
                                graph._attr['output_names'][index] = floor_name
                        div_out_attr = copy.deepcopy(out_edges[0][3])
                        div_out_attr['tensor'].value = div_out_attr['tensor'].value.astype(
                            np.float32)
                        div_out_attr['dst_in_port'] = 0
                        graph.add_edge(node_name, floor_name, **div_out_attr)
                        floor_attr = {'name': floor_name, 'opset_version': 13}
                        NodeWrap(graph, floor_name).replace_obj(
                            'Floor', floor_attr)
                elif pure_type in ('FractionalAvgPool', 'FractionalMaxPool'):
                    method = 'AVG' if pure_type == 'FractionalAvgPool' else 'MAX'
                    new_node_attr.update({'method': method,
                                          'pseudo': node_obj.pseudo_random,
                                          'overlap': node_obj.overlapping,
                                          })
                elif pure_type in ('FusedBatchNorm', 'FusedBatchNormV3'):
                    if node_obj.is_training:
                        if not FLOAT_EQUAL(node_obj.exponential_avg_factor, 1.0):
                            WARN(
                                '[Parser]: Meets unsupported exponential_avg_factor in TF FusedBatchNorm/FusedBatchNormV3 Node(%s) to convert to Onnx!' % node_name)
                            continue
                        new_node_attr.update({'training_mode': 1})
                    else:
                        out_edges = graph.sorted_out_edges(
                            node_name, keys=True, data=True)
                        for _, dst, k, out_attr in out_edges:
                            if out_attr['src_out_port'] > 0:
                                graph.remove_edge(node_name, dst, key=k)
                elif pure_type == 'InTopKV2':
                    graph.remove_edges_from(in_edges[2:])
                elif pure_type == 'LeftShift':
                    new_node_attr.update({'direction': 'LEFT'})
                elif pure_type == 'LRN':
                    size = node_obj.depth_radius * 2 + 1
                    alpha = node_obj.alpha * size
                    new_node_attr.update({'size': size, 'alpha': alpha})
                elif pure_type == 'MaxPoolWithArgmax':
                    flatten_dim = 'NHWC' if node_obj.include_batch_in_index else 'HWC'
                    new_node_attr.update({'flatten_dim': flatten_dim})
                    graph.remove_edges_from(in_edges[1:])
                elif pure_type == 'Mean':
                    if len(in_edges) >= 2:
                        if not getattr(node_obj, 'axes'):
                            new_node_attr.update(
                                {'axes': (in_edges[1][2]['tensor'].value).tolist()})
                        graph.remove_edges_from(in_edges[1:])
                    else:
                        ERROR(
                            '[Parser]: Invalid TF Mean Node(%s) to convert to Onnx!' % node_name)
                        continue
                elif pure_type == 'Pack':
                    new_node_attr.update({'new_axis': True})
                elif pure_type in ('Pad', 'PadV2', 'MirrorPad'):
                    if pure_type == 'MirrorPad':
                        new_node_attr.update({'mode': node_obj.mode.lower()})
                    if len(in_edges) > 1:
                        if not in_edges[1][2]['tensor'].is_const:
                            WARN(
                                '[Parser]: Meets unsupported non-constant padding in Node(%s) to convert to Onnx!' % node_name)
                            continue
                        if len(node_obj.get_input_shapes()) > 1:
                            input_shape = node_obj.get_input_shapes()[1]
                            trans_shape = list(
                                range(0, len(input_shape)))[::-1]
                            if input_shape is not None and len(input_shape) == 2:
                                src1, _, in_attr1 = in_edges[1]
                                insert_transpose(graph, src1, node_name,
                                                 in_attr1, trans_shape)
                    else:
                        ERROR(
                            '[Parser]: Invalid TF Pad Node(%s) to convert to Onnx!' % node_name)
                        continue
                elif pure_type == 'Relu6':
                    new_node_attr.update({'min': 0., 'max': 6.})
                elif pure_type == 'RightShift':
                    new_node_attr.update({'direction': 'RIGHT'})
                elif pure_type == 'Roll':
                    if len(in_edges) == 3 \
                            and in_edges[1][2]['tensor'].value is not None \
                            and in_edges[2][2]['tensor'].value is not None:
                        if not in_edges[1][2]['tensor'].is_const \
                                or not in_edges[2][2]['tensor'].is_const:
                            WARN(
                                '[Parser]: Meets unsupported non-constant shift/axis in TF Roll Node(%s) to convert to Onnx!' % node_name)
                            continue
                        shift = np.atleast_1d(in_edges[1][2]['tensor'].value).tolist()
                        axes = np.atleast_1d(in_edges[2][2]['tensor'].value).tolist()
                        new_node_attr.update({'shift': shift, 'axes': axes})
                        graph.remove_edges_from(in_edges[1:])
                    else:
                        ERROR('[Parser]: Meets invalid TF Roll(%s) that cannot be converted to Onnx!' % node_name)
                        continue
                elif pure_type == 'ScatterNd':
                    # TfScatterNd should be converted in convert_scatternd.
                    # If not, then indices or shape is not constant.
                    ERROR(
                        '[Parser]: Expect indices and shape to be constant in TF ScatterNd Node(%s) to convert to Onnx!' % node_name)
                    continue
                elif pure_type == 'SegmentSum':
                    new_node_attr.update({'method': 'SUM'})
                elif pure_type == 'Silu':
                    new_node_attr.update({'alpha': node_obj.beta})
                elif pure_type == 'Slice':
                    if len(in_edges) == 3 \
                            and node_obj.get_input_tensors()[1] is not None \
                            and node_obj.get_input_tensors()[2] is not None:
                        begin, size = node_obj.get_input_tensors()[1:]
                        if np.any(size < 0):
                            ends = np.array(begin + size).astype(np.int64)
                            ends[size < 0] = INT_MAX
                        else:
                            ends = begin + size
                        graph.remove_edges_from(in_edges[2:])
                        insert_constant(graph,
                                        node_name + '_ends',
                                        np.array(ends, np.int64),
                                        node_name,
                                        in_port=2,
                                        data_format='NHWC')
                    else:
                        ERROR(
                            '[Parser]: Invalid TF Slice Node(%s) to convert to Onnx!' % node_name)
                        continue
                elif pure_type == 'SpaceToDepth':
                    new_node_attr.update(
                        {'blocksize': new_node_attr['block_size']})
                elif pure_type == 'Split':
                    if node_obj.cur_version == 1 and len(in_edges) == 2:
                        graph.remove_edges_from(in_edges)
                        src, _, in_attr = in_edges[1]
                        new_in_attr = copy.deepcopy(in_attr)
                        new_in_attr['dst_in_port'] = 0
                        graph.add_edge(src, node_name, **new_in_attr)
                        new_node_attr.update(
                            {'split': node_obj.split})
                    elif node_obj.cur_version == 2:
                        if len(in_edges) > 1:
                            graph.remove_edges_from(in_edges[1:])
                        new_node_attr.update(
                            {'split': node_obj.split})
                    else:
                        ERROR(
                            '[Parser]: Invalid TF Split Node(%s) to convert to Onnx!' % node_name)
                        continue
                elif pure_type == 'SplitV':
                    graph.remove_edges_from(in_edges[1:])
                elif pure_type == 'Sum':
                    graph.remove_edges_from(in_edges[1:])
                elif pure_type == 'TensorScatterAdd':
                    new_node_attr.update({'reduction': 'add'})
                elif pure_type == 'TopKV2':
                    if len(in_edges) != 2:
                        continue
                    k, _, in_attr = in_edges[1]
                    insert_reshape(graph, k, node_name, in_attr, [1])
                    new_node_attr.update({'axis': -1})
                elif pure_type == 'Transpose':
                    if len(in_edges) == 2:
                        new_node_attr.update(
                            {'perm': node_obj.get_input_tensors()[1].tolist()})
                    elif len(in_edges) == 1:
                        new_node_attr.update({'perm': []})
                    else:
                        ERROR(
                            '[Parser]: Invalid TF Transpose Node(%s) to convert to Onnx!' % node_name)
                        continue
                    graph.remove_edges_from(in_edges[1:])

                new_node_attr.update(
                    {'opset_version': node_obj.correspond_onnx_op['version'],
                     'data_format': node_data_format})
                NodeWrap(graph, node_name).replace_obj(
                    node_obj.correspond_onnx_op['type'], new_node_attr)
        else:
            ERROR(
                '[Parser]: Meets invalid TF op for Node(%s) in convert_to_onnx!' % node_name)
