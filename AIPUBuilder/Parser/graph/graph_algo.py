# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import random
import copy
import numpy as np
import sys
import itertools
from collections import defaultdict
from datetime import datetime
from .node_wrap import NodeWrap
from networkx.algorithms import has_path, all_simple_paths, shortest_path_length
from .graph import Graph, SubGraph
from .pattern_match import single_node_matcher
from ..ops.op import InputLikeOp, SameShapeOp
from ..ops.common_ops import UndefinedOp, PluginOp
from ..common.defs import Tensor
from ..logger import INFO, DEBUG, WARN, ERROR, FATAL, WARN_EXCEPTION


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


def get_valid_node_name(graph, base_name, nodes_name=None):
    max_try_times = 1000
    ret = base_name
    i = 0
    if nodes_name is None:
        while graph.has_node(ret):
            if i == max_try_times:
                raise Exception('Cannot find valid name!')
            ret = base_name + '_' + str(i)
            i += 1
    else:
        while ret in nodes_name:
            if i == max_try_times:
                raise Exception('Cannot find valid name!')
            ret = base_name + '_' + str(i)
            i += 1
    return ret


def determined_sort(g, outputs, sort_input=False):
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
    if sort_input:
        main_input = None
        for i, n in enumerate(op_order):
            node_obj = NodeWrap(g, n)['object']
            if node_obj is None:
                ERROR('[Parser]: Meets invalid Node (%s) in determined_sort!' % n)
                continue
            if node_obj.type == 'ArmInput' and len(node_obj.get_output_shapes()[0]) > 2 and i != 0:
                main_input = n
                break
        if main_input:
            op_order.remove(main_input)
            op_order.insert(0, main_input)
    return op_order


def clear_redundant_nodes(g, outputs=None):
    '''Delete redundant nodes in the graph.'''
    pred = g.predecessor
    if g._attr.get('output_nodes', []) and None not in g._attr['output_nodes']:
        noop_names = [n for n in g.nodes if g.nodes[n]['op'] == 'Out'
                      and n in g._attr['output_nodes']]
    else:
        noop_names = [n for n in g.nodes if g.nodes[n]['op'] == 'Out'
                      and pred[n]
                      and any([p in g._attr.get('output_names', []) for p in pred[n]])
                      ]
    output_names = outputs if outputs else (
        noop_names if noop_names else g._attr.get('output_names', []))
    subgraph_map = {}
    g._attr['subgraph_depends_nodes'] = []
    if 'subgraphs' in g._attr and len(g._attr['subgraphs']) > 0:
        for k, v in g._attr['subgraphs'].items():
            for _k in v.keys():
                subgraph_map[_k] = k
    if output_names:
        valid_nodes = determined_sort(g, output_names)
        for n in g.nodes:
            node_obj = NodeWrap(g, n)['object']
            if node_obj is not None and \
                    node_obj.depend_nodes and \
                    any([a in valid_nodes for a in node_obj.depend_nodes]):
                if n not in valid_nodes:
                    valid_nodes.append(n)
                if n not in g._attr['subgraph_depends_nodes']:
                    g._attr['subgraph_depends_nodes'].append(n)
        removing_nodes = set(g.nodes).difference(valid_nodes)
        valid_out_nodes = []
        for n in removing_nodes:
            if g.nodes[n]['op'] == 'Out' and \
                    len(pred[n]) == 1 and \
                    pred[n][0] not in removing_nodes:
                if len(g.nodes[n]['object'].subgraphs) > 0:
                    if 'subgraphs' in g._attr and len(g._attr['subgraphs']) > 0:
                        for sub in g.nodes[n]['object'].subgraphs:
                            for k, v in g._attr['subgraphs'].items():
                                if sub in v:
                                    valid_out_nodes.append(n)
                                    break
                    else:
                        continue
                else:
                    valid_out_nodes.append(n)

            if isinstance(g, SubGraph) and (n in g._attr['input_tensors'] or
                                            (g.nodes[n]['op'] == 'Out' and
                                             len(pred[n]) == 1 and
                                             pred[n][0] in g._attr['input_tensors'])):
                valid_out_nodes.append(n)

        removing_nodes = set(removing_nodes).difference(valid_out_nodes)
        g.remove_nodes_from(removing_nodes)
        if 'subgraphs' in g._attr and len(g._attr['subgraphs']) > 0:
            all_valid_nodes = valid_nodes
            for k, v in list(g._attr['subgraphs'].items()):
                if k in removing_nodes:
                    g._attr['subgraphs'].pop(k)
                else:
                    for _v in v.values():
                        all_valid_nodes += set(_v.nodes)
            for k in list(g._attr['subgraphs'].keys()):
                if k not in all_valid_nodes:
                    g._attr['subgraphs'].pop(k)
        if g._attr['subgraph_depends_nodes']:
            for dep_n in g._attr['subgraph_depends_nodes']:
                out_edges = g.sorted_out_edges(dep_n, data=True)
                if not out_edges:
                    noop_node_name = get_valid_node_name(
                        g, dep_n + '_noop_0')
                    g.add_node(noop_node_name)
                    noop_node = NodeWrap(g, noop_node_name)
                    noop_node.replace_obj(
                        'Out', {'name': noop_node_name})
                    out_edge_attr = {
                        'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor(name=dep_n)}
                    g.add_edge(
                        dep_n, noop_node_name, **out_edge_attr)
    else:
        ERROR('[Parser]: Can not proceed without output names in clear_redundant_nodes!')


def infer(graph, partial=False, chosen_list=None, final=False):
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
        nodes_list = determined_sort(graph, graph._attr['subgraph_depends_nodes'] + graph._attr['output_names'])

        if 'tensor_counter' in graph._attr:
            graph._attr['tensor_counter'].clear()
            for n in nodes_list:
                for _, _, d in graph.sorted_out_edges(n, data=True):
                    edge_tensor = d['tensor']
                    if not edge_tensor.is_const:
                        graph._attr['tensor_counter'][hash(edge_tensor)] += 1

        log_func = DEBUG if partial else WARN_EXCEPTION
        for node_name in nodes_list:

            if chosen_list and node_name not in chosen_list:
                continue
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is not None:
                if node_obj.in_subgraph:
                    DEBUG('[Parser]: Subgraph Node(%s) is in infer, result is not guaranteed!' % node_name)

                if partial and not node_obj.is_all_inputs_const() and not isinstance(node_obj, InputLikeOp):
                    if isinstance(node_obj, SameShapeOp):
                        inp_tensors = node_obj.get_input_tensors()
                        if not inp_tensors or any(x is None for x in inp_tensors):
                            continue
                    else:
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
                        if node_name in graph._attr['input_tensors']:
                            infer_data = graph._attr['input_tensors'][node_name].value
                        else:
                            if node_obj.type == 'DummyInput' and isinstance(graph, SubGraph):
                                parent_node = graph._attr['parent_graph'].nodes[node_name]
                                dummy_out_edges = graph._attr['parent_graph'].sorted_out_edges(
                                    parent_node['object'].name,
                                    data=True)
                                infer_data = dummy_out_edges[0][-1]['tensor'].value
                            else:
                                log_func('[Parser]: Meet unsupported op type %s in Node(%s)!' %
                                         (node_obj.type, node_name))
                        node_obj.infer_shape(infer_data)
                    elif isinstance(node_obj, UndefinedOp):
                        log_func('[Parser]: Meet unsupported op type %s in Node(%s)!' % (node_obj.type, node_name))
                    elif isinstance(node_obj, PluginOp):
                        node_obj.infer_shape(final=final)
                    else:
                        node_obj.infer_shape()
                except Exception as e:
                    log_func('[Parser]: Infer of %s Node(%s) meets issues: %s!' %
                             (node_obj.type, node_name, str(e)))

                msg = ', '.join([
                    str(datetime.now().time()),
                    # str((psutil.Process(os.getpid()).memory_info().rss - mem1) / (1024*1024)),
                    node_obj.type,
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
