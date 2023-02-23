# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import random
import copy
import numpy as np
import sys
import itertools
from collections import defaultdict
from .node_wrap import NodeWrap
from .graph import Graph, SubGraph
from .pattern_match import single_node_matcher
from ..ops.op import InputLikeOp
from ..ops.common_ops import UndefinedOp
from ..common.defs import Tensor
from ..logger import INFO, DEBUG, WARN, ERROR, FATAL


def _shortest_path_length(g, source, target):
    ret = sys.maxsize
    if g.has_node(source) and g.has_node(target):
        if source == target:
            ret = 0
        else:

            def _bidirectional_pred_succ(G, source, target):
                """Bidirectional shortest path helper.

                   Returns (pred, succ, w) where
                   pred is a dictionary of predecessors from w to the source, and
                   succ is a dictionary of successors from w to the target.
                """
                # does BFS from both source and target and meets in the middle
                if target == source:
                    return ({target: None}, {source: None}, source)

                # handle either directed or undirected
                Gpred = G.pred
                Gsucc = G.succ

                # predecesssor and successors in search
                pred = {source: None}
                succ = {target: None}

                # initialize fringes, start with forward
                forward_fringe = [source]
                reverse_fringe = [target]

                while forward_fringe and reverse_fringe:
                    if len(forward_fringe) <= len(reverse_fringe):
                        this_level = forward_fringe
                        forward_fringe = []
                        for v in this_level:
                            for w in Gsucc[v]:
                                if w not in pred:
                                    forward_fringe.append(w)
                                    pred[w] = v
                                if w in succ:  # path found
                                    return pred, succ, w
                    else:
                        this_level = reverse_fringe
                        reverse_fringe = []
                        for v in this_level:
                            for w in Gpred[v]:
                                if w not in succ:
                                    succ[w] = v
                                    reverse_fringe.append(w)
                                if w in pred:  # found path
                                    return pred, succ, w

                raise RuntimeError(
                    "No path between %s and %s." % (source, target))

            # call helper to do the real work
            results = _bidirectional_pred_succ(g, source, target)
            pred, succ, w = results

            # build path from pred+w+succ
            path = []
            # from source to w
            while w is not None:
                path.append(w)
                w = pred[w]
            path.reverse()
            # from w to target
            w = succ[path[-1]]
            while w is not None:
                path.append(w)
                w = succ[w]

            ret = len(path) - 1
    else:
        raise RuntimeError(
            'Source %s or target %s node not in graph!' % (source, target))
    return ret


def has_path(g, source, target):
    '''Check if there is a path between two nodes.'''
    try:
        _ = _shortest_path_length(g, source, target)
    except (KeyError, RuntimeError):
        return False
    return True


def cal_path_length(g, source, target):
    try:
        sp = _shortest_path_length(g, source, target)
        return sp
    except (KeyError, RuntimeError):
        return sys.maxsize


def all_simple_paths(graph, source, target):
    '''Find all paths between the destination node and the source node.'''
    if source not in graph.nodes:
        raise Exception('source node %s not in graph' % source)
    if target not in graph.nodes:
        raise Exception('target node %s not in graph' % target)
    if source == target:
        return []
    if not has_path(graph, source, target):
        return []

    def _all_simple_paths_multigraph(G, source, target):
        queue = []
        queue.append(source)
        seen = []
        seen.append(source)
        while(len(queue) > 0):
            vertex = queue.pop(0)
            nodes = G._adj_dict[vertex].keys()
            for w in nodes:
                if w == target:
                    yield seen + [target]
                elif w not in seen:
                    queue.append(w)
                    seen.append(w)
    return _all_simple_paths_multigraph(graph, source, target)


def nodes_in_simple_paths(graph, source, target):
    all_path_nodes = set(itertools.chain(
        *list(all_simple_paths(graph, source=source, target=target))))
    return all_path_nodes


def get_valid_node_name(graph, base_name):
    max_try_times = 1000
    ret = base_name
    i = 0
    while graph.has_node(ret):
        if i == max_try_times:
            raise Exception('Cannot find valid name!')
        ret = base_name + '_' + str(i)
        i += 1
    return ret


def determined_sort(g, outputs):
    '''Get all the sorted nodes according to the outputs node of the graph.'''
    op_order = []
    stack = copy.deepcopy(outputs)
    visited = set()
    while len(stack) != 0:
        node_name = stack[0]
        stack.pop(0)
        visited.add(node_name)
        has_child = False
        in_names = [name for name in g.pred[node_name]]
        for in_node_name in in_names:
            if in_node_name not in visited:
                stack.insert(0, node_name)
                stack.insert(0, in_node_name)
                has_child = True
                break
        if not has_child and node_name not in op_order:
            op_order.append(node_name)
    return op_order


def clear_redundant_nodes(g, outputs=None):
    '''Delete redundant nodes in the graph.'''
    noop_names = [n for n in g.nodes if g.get_node(n)._attr['op'] == 'Out'
                  and g.pred[n]
                  and (any([p in g._attr.get('output_names', []) for p in g.pred[n]])
                       or len(g.succ[g.pred[n][0]]) > 1)
                  ]
    output_names = outputs if outputs else (
        noop_names if noop_names else g._attr.get('output_names', []))
    if output_names:
        valid_nodes = determined_sort(g, output_names)
        removing_nodes = set(g.nodes).difference(valid_nodes)
        g.remove_nodes_from(removing_nodes)
    else:
        ERROR('[Parser]: Can not proceed without output names in clear_redundant_nodes!')


def infer(graph, partial=False, chosen_list=None):
    if chosen_list is None:
        chosen_list = list()
    ''' Infer all nodes of the graph.   '''

    _input_cast_map = {
        'uint64': 'uint32',
        'int64': 'int32',
        'float64': 'float32',
        'bool': 'uint8'
    }

    ret = {}
    if len(graph) > 0:
        nodes_list = determined_sort(graph, graph._attr['output_names'])
        log_func = DEBUG if partial else WARN
        for node_name in nodes_list:
            if chosen_list and node_name not in chosen_list:
                continue
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is not None:
                if partial and not node_obj.is_all_inputs_const() and not isinstance(node_obj, InputLikeOp):
                    continue
                try:
                    if isinstance(node_obj, InputLikeOp):
                        if node_obj.type == 'ArmInput':
                            input_data = graph._attr['input_tensors'][node_name].value
                            if str(input_data.dtype) in _input_cast_map:
                                casted_type = _input_cast_map[str(
                                    input_data.dtype)]
                                graph._attr['input_tensors'][node_name].value = input_data.astype(
                                    np.dtype(casted_type))

                        infer_data = graph._attr['input_tensors'][node_name].value
                        node_obj.infer_shape(infer_data)
                    elif isinstance(node_obj, UndefinedOp):
                        log_func('[Parser]: Meet unsupported op type %s in Node(%s)!' % (node_obj.type, node_name))
                    else:
                        node_obj.infer_shape()
                except Exception as e:
                    log_func('[Parser]: Infer of Node(%s) meets issues: %s!' %
                             (node_name, str(e)))

                msg = ', '.join([node_obj.type,
                                 node_obj.name,
                                 node_obj.data_format,
                                 str(node_obj.get_output_shapes()),
                                 str([str(v.dtype)
                                      if v is not None else None
                                      for v in node_obj.get_output_tensors()]),
                                 str(node_obj.is_all_inputs_const())
                                 ])
                DEBUG(msg)
            else:
                ERROR('[Parser]: Meet invalid Node (%s) in infer!' % node_name)

        for out_name in graph._attr['output_names']:
            out_edges = graph.sorted_out_edges(out_name, data=True)
            for _, _, out_attr in out_edges:
                ret.update(
                    {(out_name, out_attr['src_out_port']): out_attr['tensor'].value})
    else:
        ERROR('[Parser]: Meets empty graph when inferring!')
    return ret
