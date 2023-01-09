# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re
from ....ops.op import *
from ....graph.node_wrap import NodeWrap
from ....graph.pattern_match import single_node_matcher
from ....common.utils import extend_lists
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL


def convert_to_onnx(graph):
    '''Convert the model to the onnx version.'''
    def tensors_are_const(edges=[]):
        assert isinstance(edges, list), 'Expect edges to be list but got %s in tensors_are_const' % type(edges).__name__
        assert len(edges) > 0, 'Expect at least 1 edges but got 0 in tensors_are_const'
        for edge in edges:
            attr_dict = edge[-1]
            assert isinstance(
                attr_dict, dict), 'Expect items of list to contain attr dict at index -1, but got %s in tensors_are_const' % type(attr_dict).__name__
            if attr_dict.get('tensor', None) is None or not attr_dict['tensor'].is_const:
                return False
        return True

    def remove_edges_if_const(node_name, edges=[]):
        if tensors_are_const(edges):
            graph.remove_edges_from(edges)
        else:
            WARN('[Parser]: Meet non-const tensors of Node (%s) in remove_edges_if_const' % node_name)

    tf2_ops = Tf2Op.get_concrete_subclass_names()
    keras_ops = KerasOp.get_concrete_subclass_names()
    tf2_non_keras_ops = list(set(tf2_ops).difference(keras_ops))
    matches = extend_lists([single_node_matcher(graph, op_type)
                            for op_type in tf2_non_keras_ops])
    for m in matches:
        node_name = m['target']
        node_obj = NodeWrap(graph, node_name)['object']
        if node_obj is None:
            WARN(
                '[Parser]: Meets invalid TF2 op for Node(%s) in convert_to_onnx!' % node_name)
            continue
        in_edges = graph.sorted_in_edges(node_name, data=True)
        new_node_attr = node_obj.copied_attr()
        node_data_format = 'NCHW' if node_obj.data_format.startswith('NC') else 'NHWC'
        pure_type = re.sub(r'^Tf', '', node_obj.type)
        if getattr(node_obj, 'correspond_onnx_op', None) is not None:
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

            if pure_type in ('conv2d', 'cumsum', 'cumprod'):
                remove_edges_if_const(node_name, in_edges[2:])
            elif pure_type == 'left_shift':
                new_node_attr.update({'direction': 'LEFT'})
            elif pure_type == 'right_shift':
                new_node_attr.update({'direction': 'RIGHT'})
            elif pure_type == 'split':
                remove_edges_if_const(node_name, in_edges[1:])
            elif pure_type == 'stack':
                remove_edges_if_const(node_name, in_edges[-1:])
                new_node_attr.update({'new_axis': True})

            new_node_attr.update(
                {'opset_version': node_obj.correspond_onnx_op['version'],
                 'data_format': node_data_format})
            NodeWrap(graph, node_name).replace_obj(
                node_obj.correspond_onnx_op['type'], new_node_attr)
