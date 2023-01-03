# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re
from ....ops.op import Tf2Op, KerasOp
from ....graph.node_wrap import NodeWrap
from ....graph.pattern_match import single_node_matcher
from ....common.utils import extend_lists
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL


def convert_to_onnx(graph):
    '''Convert the model to the onnx version.'''
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
            new_node_attr.update(
                {'opset_version': node_obj.correspond_onnx_op['version'],
                 'data_format': node_data_format})
            NodeWrap(graph, node_name).replace_obj(
                node_obj.correspond_onnx_op['type'], new_node_attr)
