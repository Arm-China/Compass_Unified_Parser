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

# cython: language_level=3
from .logger import ERROR, WARN, DEBUG
import re
from .graph.node_wrap import NodeWrap
from .graph.graph_algo import get_valid_node_name


def special_character_conversion(graph, params):
    '''Convert characters that are not allowed in IR.'''
    newname_dict = {}
    for n in graph.nodes:
        node_obj = NodeWrap(graph, n)['object']
        newname = node_obj.name
        newname = re.sub(r'[^0-9a-zA-Z\.\:\/\_\;\'\x22]', '_', newname)
        for value in newname_dict.values():
            while value == newname:
                newname = newname+'_'
        for key in newname_dict.keys():
            while key == newname:
                newname = newname+'_'
        if node_obj.name != newname:
            DEBUG('[Parser]: Duplicate layer name found! Convert layer name:(%s) to layer name: (%s)!' % (
                str(node_obj.name), str(newname)))
            newname = get_valid_node_name(graph, newname)
        newname_dict[n] = newname
    graph._attr['duplicate_name'] = newname_dict
