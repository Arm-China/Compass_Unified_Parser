# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from collections import Mapping, Set


class NodeView(Mapping, Set):

    __slots__ = '_nodes',

    def __getstate__(self):
        return {'_nodes': self._nodes}

    def __setstate__(self, state):
        self._nodes = state['_nodes']

    def __init__(self, graph):
        self._nodes = graph._nodes_dict

    # Mapping methods
    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, n):
        return self._nodes[n]

    # Set methods
    def __contains__(self, n):
        return n in self._nodes

    @classmethod
    def _from_iterable(cls, it):
        return set(it)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, tuple(self))
