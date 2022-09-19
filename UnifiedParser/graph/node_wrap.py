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

from ..ops.op import Op
from ..ops.op_factory import op_factory
from ..common.errors import *


class NodeWrap(object):
    def __init__(self, graph, node_name):
        self._graph = graph
        self._name = node_name

    def __setitem__(self, key, value):
        self._graph.nodes[self._name]._attr[key] = value
        if key == 'object' and isinstance(value, Op):
            self._graph.nodes[self._name]._attr['op'] = value.type

    def __getitem__(self, key):
        if self._name not in self._graph.nodes:
            return None
        else:
            return self._graph.nodes[self._name]._attr[key] if key in self._graph.nodes[self._name]._attr else None

    def __delitem__(self, key):
        del self._graph.nodes[self._name]._attr[key]
        if key == 'object':
            self._graph.nodes[self._name]._attr['op'] = None

    def replace_obj(self, new_op_type, attr_dict):
        new_obj = op_factory(self._graph, new_op_type, attr_dict)
        if new_obj is not None:
            if type(new_obj).__name__ not in ('UndefinedOp', 'PluginOp'):
                assert type(new_obj).__name__ == new_op_type + \
                    'Op', 'The OP type to be replaced is wrong .'
            if self['object'] is not None:
                del self['object']
            self['object'] = new_obj
        else:
            ERROR('[Parser]: Meets invalid Op object of %s for Node(%s)!' %
                  (new_op_type, attr_dict.get('name', '')))
        return new_obj
