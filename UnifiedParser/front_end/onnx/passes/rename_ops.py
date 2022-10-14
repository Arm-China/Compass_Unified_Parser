# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from ....graph.node_wrap import NodeWrap
from ....ops.op import *


def simple_rename(graph, src_type_list, dst_type):
    assert src_type_list and dst_type in Op.get_concrete_subclass_names(
    ), 'dst_type is invalid or src_type_list is empty in simple_rename.'
    if isinstance(src_type_list, str):
        src_type_list = [src_type_list]
    for n in graph.nodes:
        node = NodeWrap(graph, n)
        if node['object'] is not None and node['object'].type in src_type_list:
            new_attr_dict = node['object'].copied_attr()
            node.replace_obj(dst_type, new_attr_dict)
