"""
//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
"""

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
