# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import random
import copy
import numpy as np
import sys
import itertools
from collections import defaultdict
from .node_wrap import NodeWrap
from networkx.algorithms import has_path, all_simple_paths, shortest_path_length
from .graph import Graph, SubGraph
from .pattern_match import single_node_matcher
from ..ops.op import InputLikeOp
from ..ops.common_ops import UndefinedOp
from ..common.defs import Tensor
from ..logger import INFO, DEBUG, WARN, ERROR, FATAL


def cal_path_length(g, source, target):
    try:
        assert source in g.nodes and target in g.nodes
        sp = shortest_path_length(g, source, target)
        return int(sp)
    except:
        return sys.maxsize


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
    if outputs:
        stack = copy.deepcopy(outputs)
        pred = g.predecessor
        visited = set()
        while len(stack) != 0:
            node_name = stack[0]
            stack.pop(0)
            visited.add(node_name)
            has_child = False
            in_names = [name for name in pred[node_name]]
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
    pred = g.predecessor
    noop_names = [n for n in g.nodes if g.nodes[n]['op'] == 'Out'
                  and pred[n]
                  and any([p in g._attr.get('output_names', []) for p in pred[n]])
                  ]
    output_names = outputs if outputs else (
        noop_names if noop_names else g._attr.get('output_names', []))
    if output_names:
        valid_nodes = determined_sort(g, output_names)
        removing_nodes = set(g.nodes).difference(valid_nodes)
        valid_out_nodes = [n for n in removing_nodes if g.nodes[n]['op'] == 'Out'
                           and len(pred[n]) == 1
                           and pred[n][0] not in removing_nodes]
        removing_nodes = set(removing_nodes).difference(valid_out_nodes)
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
        'float16': 'float32',
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
                if node_obj.in_subgraph:
                    DEBUG('[Parser]: Subgraph Node(%s) is in infer, result is not guaranteed!' % node_name)

                if partial and not node_obj.is_all_inputs_const() and not isinstance(node_obj, InputLikeOp):
                    continue
                try:
                    if isinstance(node_obj, InputLikeOp):
                        if node_obj.type == 'ArmInput':
                            input_data = graph._attr['input_tensors'][node_name].value
                            input_type_str = str(input_data.dtype)
                            if input_type_str in _input_cast_map:
                                casted_type = _input_cast_map[input_type_str]
                                if casted_type != input_type_str:
                                    WARN('[Parser]: Convert unsupported type %s to type %s for Input Node (%s)!' %
                                         (input_type_str, casted_type, node_name))
                                    graph._attr['input_tensors'][node_name].value = input_data.astype(
                                        np.dtype(casted_type))

                        infer_data = graph._attr['input_tensors'][node_name].value
                        node_obj.infer_shape(infer_data)
                    elif isinstance(node_obj, UndefinedOp):
                        log_func('[Parser]: Meet unsupported op type %s in Node(%s)!' % (node_obj.type, node_name))
                    else:
                        node_obj.infer_shape()
                except Exception as e:
                    log_func('[Parser]: Infer of %s Node(%s) meets issues: %s!' %
                             (node_obj.type, node_name, str(e)))

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
                ERROR('[Parser]: Meets invalid Node (%s) in infer!' % node_name)

        for out_name in graph._attr['output_names']:
            out_edges = graph.sorted_out_edges(out_name, data=True)
            for _, _, out_attr in out_edges:
                ret.update(
                    {(out_name, out_attr['src_out_port']): out_attr['tensor'].value})
    else:
        ERROR('[Parser]: Meets empty graph when inferring!')
    return ret
