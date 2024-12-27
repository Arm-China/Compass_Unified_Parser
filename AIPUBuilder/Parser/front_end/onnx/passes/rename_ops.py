# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from ....graph.node_wrap import NodeWrap
from ....ops.op import *


def simple_rename(graph, src_type, dst_type=None):
    if dst_type is None:
        if isinstance(src_type, str):
            dst_type = f'Arm{src_type}'
        elif isinstance(src_type, dict):
            _src_type, dst_type = list(src_type.items())[0]
            src_type = _src_type
        else:
            raise NotImplemented('Still not support this type in simple_rename!')

    assert src_type and dst_type in Op.get_concrete_subclass_names(
    ), 'dst_type is invalid or src_type_list is empty in simple_rename.'
    if isinstance(src_type, str):
        src_type_list = [src_type]
    else:
        src_type_list = src_type
    for n in graph.nodes:
        node = NodeWrap(graph, n)
        if node['object'] is not None and node['object'].type in src_type_list:
            new_attr_dict = node['object'].copied_attr()
            node.replace_obj(dst_type, new_attr_dict)
