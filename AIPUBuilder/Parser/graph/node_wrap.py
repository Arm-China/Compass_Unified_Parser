# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from ..ops.op import Op
from ..ops.op_factory import op_factory
from ..logger import INFO, DEBUG, WARN, ERROR, FATAL


class NodeWrap(object):
    def __init__(self, graph, node_name):
        self._graph = graph
        self._name = node_name

    def __setitem__(self, key, value):
        self._graph.nodes[self._name][key] = value
        if key == 'object' and isinstance(value, Op):
            self._graph.nodes[self._name]['op'] = value.type

    def __getitem__(self, key):
        if self._name not in self._graph.nodes:
            return None
        else:
            return self._graph.nodes[self._name].get(key, None)

    def __delitem__(self, key):
        if key in self._graph.nodes[self._name]:
            del self._graph.nodes[self._name][key]
        if key == 'object':
            self._graph.nodes[self._name]['op'] = None

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
