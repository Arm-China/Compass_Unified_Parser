# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re
from ....ops.op import *
from ....graph.node_wrap import NodeWrap
from ....graph.pattern_match import single_node_matcher
from ....graph.graph_algo import get_valid_node_name, clear_redundant_nodes
from ...onnx.passes.common_passes import insert_constant
from ....common.utils import extend_lists
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL


def convert_crelu(graph):
    '''Convert Tfcrelu to neg+concat+relu as y = crelu(x, axis) = relu(concat([x, -x], axis)).
    '''
    matched = False
    matches = single_node_matcher(graph, 'Tfcrelu')
    for m in matches:
        crelu = m['target']
        crelu_obj = NodeWrap(graph, crelu)['object']
        crelu_in_edges = graph.sorted_in_edges(crelu, data=True)
        if crelu_obj is None or len(crelu_in_edges) < 1:
            WARN(
                '[Parser]: Meets invalid Tfcrelu Op (%s) in convert_crelu!' % crelu)
            continue
        matched = True
        crelu_axis = crelu_obj.axis
        neg = get_valid_node_name(graph, crelu + '_neg')
        concat = get_valid_node_name(graph, crelu + '_concat')
        graph.remove_edges_from(crelu_in_edges)
        src, _, in_attr = crelu_in_edges[0]
        graph.add_edge(src, neg, **in_attr)
        graph.add_edge(src, concat, **in_attr)
        neg_out_attr = copy.deepcopy(in_attr)
        concat_out_attr = copy.deepcopy(in_attr)
        if in_attr['tensor'] is not None and in_attr['tensor'].value is not None:
            neg_out_attr['tensor'].value = -1 * in_attr['tensor'].value
            concat_out_attr['tensor'].value = np.concatenate(
                [in_attr['tensor'].value, neg_out_attr['tensor'].value], axis=crelu_axis)
        neg_out_attr.update({'src_out_port': 0, 'dst_in_port': 1})
        graph.add_edge(neg, concat, **neg_out_attr)
        concat_out_attr.update({'src_out_port': 0, 'dst_in_port': 0})
        graph.add_edge(concat, crelu, **concat_out_attr)
        NodeWrap(graph, neg).replace_obj('Neg', {'name': neg, 'opset_version': 13})
        NodeWrap(graph, concat).replace_obj('Concat', {'name': concat, 'axis': crelu_axis, 'opset_version': 13})
        relu_attr = crelu_obj.copied_attr()
        relu_attr.update({'opset_version': 13})
        NodeWrap(graph, crelu).replace_obj('Relu', relu_attr)
    if matched:
        clear_redundant_nodes(graph)


def convert_to_onnx(graph):
    '''Convert the model to the onnx version.'''
    def _tensors_are_const(edges=list()):
        assert isinstance(edges, list), 'Expect edges to be list but got %s in tensors_are_const' % type(edges).__name__
        assert len(edges) > 0, 'Expect at least 1 edges but got 0 in tensors_are_const'
        for edge in edges:
            attr_dict = edge[-1]
            assert isinstance(
                attr_dict, dict), 'Expect items of list to contain attr dict at index -1, but got %s in tensors_are_const' % type(attr_dict).__name__
            if attr_dict.get('tensor', None) is None or not attr_dict['tensor'].is_const:
                return False
        return True

    def _remove_edges_if_const(node_name, edges=list()):
        if _tensors_are_const(edges):
            graph.remove_edges_from(edges)
        else:
            WARN('[Parser]: Meet non-const tensors of Node (%s) in remove_edges_if_const' % node_name)

    tf2_ops = Tf2Op.get_concrete_subclass_names()
    keras_ops = KerasOp.get_concrete_subclass_names()
    tf2_non_keras_ops = list(set(tf2_ops).difference(keras_ops))
    matches = extend_lists([single_node_matcher(graph, op_type)
                            for op_type in tf2_non_keras_ops])
    matched = False
    for m in matches:
        node_name = m['target']
        node_obj = NodeWrap(graph, node_name)['object']
        if node_obj is None:
            WARN(
                '[Parser]: Meets invalid TF2 op for Node(%s) in convert_to_onnx!' % node_name)
            continue
        if getattr(node_obj, 'correspond_onnx_op', None) is not None:
            in_edges = graph.sorted_in_edges(node_name, data=True)
            new_node_attr = node_obj.copied_attr()
            node_data_format = 'NCHW' if node_obj.data_format.startswith('NC') else 'NHWC'
            pure_type = re.sub(r'^Tf', '', node_obj.type)
            if isinstance(node_obj, OpHasWeights):
                if node_obj.weights is None:
                    WARN('[Parser]: Node(%s) does not contain weights!' %
                         node_name)
                    continue
                new_weights = node_obj.weights
                new_weights = np.transpose(
                    new_weights, axes=type(node_obj).perm_tf_to_onnx())
                new_node_attr.update({'weights': new_weights})
            if isinstance(node_obj, OpHasPaddingStrides):
                if hasattr(node_obj, 'strides') and len(node_obj.strides) == 4:
                    new_node_attr.update(
                        {'strides': node_obj.strides[1:3]})
                if hasattr(node_obj, 'dilations') and len(node_obj.dilations) == 4:
                    new_node_attr.update(
                        {'dilations': node_obj.dilations[1:3]})

            if pure_type in ('argmax', 'argmin'):
                _remove_edges_if_const(node_name, in_edges[1:])
                new_node_attr.update({'keepdims': 0})
            elif pure_type == 'cast':
                _remove_edges_if_const(node_name, in_edges[1:])
                new_node_attr.update({'to': node_obj.dtype})
            elif pure_type == 'concat':
                _remove_edges_if_const(node_name, in_edges[-2:])
            elif pure_type in ('conv2d', 'cumsum', 'cumprod', 'gather', 'gather_nd'):
                _remove_edges_if_const(node_name, in_edges[2:])
            elif pure_type == 'expand_dims':
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
                    WARN(
                        '[Parser]: Invalid TF2 expand_dims Node(%s) to convert to Onnx!' % node_name)
                    continue
            elif pure_type == 'floormod':
                new_node_attr.update({'fmod': 0})
                graph.remove_edges_from(in_edges[-1:])
            elif pure_type in ('fractional_avg_pool', 'fractional_max_pool'):
                if len(in_edges) < 5 \
                        or any(attr['tensor'].value is None for _, _, attr in in_edges) \
                        or any(not attr['tensor'].is_const for _, _, attr in in_edges[1:]):
                    WARN('[Parser]: Meets invalid inputs for Node(%s) in convert_to_onnx!' % node_name)
                    continue
                pooling_ratio = in_edges[1][2]['tensor'].value.tolist()
                method = 'AVG' if pure_type == 'fractional_avg_pool' else 'MAX'
                new_node_attr.update({'method': method,
                                      'pooling_ratio': pooling_ratio,
                                      'pseudo': node_obj.pseudo_random,
                                      'overlap': node_obj.overlapping,
                                      'seed': node_obj.seed,
                                      })
                graph.remove_edges_from(in_edges[1:])
            elif pure_type == 'gelu':
                approximate = 'tanh' if node_obj.approximate is True else 'none'
                new_node_attr.update({'approximate': approximate})
                graph.remove_edges_from(in_edges[-1:])
            elif pure_type == 'in_top_k':
                _remove_edges_if_const(node_name, in_edges[2:])
                if node_obj.cur_version != 1:
                    if len(in_edges) < 2:
                        WARN('[Parser]: Meets invalid in_edges for Node(%s) in convert_to_onnx!' % node_name)
                        continue
                    graph.remove_edges_from(in_edges)
                    target_src, _, target_out_attr = in_edges[0]
                    predict_src, _, predict_out_attr = in_edges[1]
                    predict_out_attr.update({'dst_in_port': 0})
                    graph.add_edge(predict_src, node_name, **predict_out_attr)
                    target_out_attr.update({'dst_in_port': 1})
                    graph.add_edge(target_src, node_name, **target_out_attr)
            elif pure_type == 'left_shift':
                new_node_attr.update({'direction': 'LEFT'})
                graph.remove_edges_from(in_edges[-1:])
            elif pure_type == 'one_hot':
                _remove_edges_if_const(node_name, in_edges[2:])
                values = np.array([node_obj.off_value, node_obj.on_value])
                insert_constant(graph, node_name + '_values', values, node_name, in_port=2)
            elif pure_type == 'right_shift':
                new_node_attr.update({'direction': 'RIGHT'})
                graph.remove_edges_from(in_edges[-1:])
            elif pure_type == 'segment_sum':
                new_node_attr.update({'method': 'SUM'})
                graph.remove_edges_from(in_edges[-1:])
            elif pure_type == 'split':
                _remove_edges_if_const(node_name, in_edges[1:])
            elif pure_type == 'stack':
                _remove_edges_if_const(node_name, in_edges[-2:])
                new_node_attr.update({'new_axis': True})
            elif len(in_edges) > 1:
                # Some ops like silu do not have 'name' input so cannot remove in_edges
                graph.remove_edges_from(in_edges[-1:])

            matched = True
            new_node_attr.update(
                {'opset_version': node_obj.correspond_onnx_op['version'],
                 'data_format': node_data_format})
            NodeWrap(graph, node_name).replace_obj(
                node_obj.correspond_onnx_op['type'], new_node_attr)
    if matched:
        clear_redundant_nodes(graph)
