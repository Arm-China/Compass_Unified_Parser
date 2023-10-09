# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import copy
from networkx.algorithms import isomorphism
from itertools import product, permutations, combinations
from .graph import Graph


def node_feasibility(g1_node, g2_node):
    if g1_node.get('op', None) is None:
        return False
    elif g2_node.get('op', None) is None:
        return True
    else:
        if isinstance(g2_node['op'], (list, tuple)):
            return g1_node['op'] in g2_node['op']
        else:
            return g1_node['op'] == g2_node['op']


def edge_feasibility(g1_edge, g2_edge):
    def _check_edge(e1, e2):
        src_check = True if e2['src_out_port'] is None else e1['src_out_port'] == e2['src_out_port']
        dst_check = True if e2['dst_in_port'] is None else e1['dst_in_port'] == e2['dst_in_port']
        return src_check and dst_check

    g1_edge = dict(g1_edge)
    g2_edge = dict(g2_edge)
    if len(g2_edge) == 0:
        return True
    elif len(g1_edge) < len(g2_edge):
        return False
    elif len(g1_edge) == 1 and len(g2_edge) == 1:
        g1_edge = list(g1_edge.values())[0]
        g2_edge = list(g2_edge.values())[0]
        return _check_edge(g1_edge, g2_edge)
    else:
        found_match = False
        g1_edge_list = list(g1_edge.values())
        g2_edge_list = list(g2_edge.values())
        r = len(g2_edge_list)
        for part_g1_edge, part_g2_edge in product(list(combinations(g1_edge_list, r)), list(permutations(g2_edge_list, r))):
            part_found_not_match = False
            part_g1_edge = sorted(part_g1_edge, key=lambda x: (x['src_out_port'], x['dst_in_port']))
            for meta_g1_edge, meta_g2_edge in zip(part_g1_edge, part_g2_edge):
                meta_check_ret = _check_edge(meta_g1_edge, meta_g2_edge)
                if not meta_check_ret:
                    part_found_not_match = True
                    break
            if part_found_not_match:
                continue
            else:
                found_match = True
                break
        return found_match


def matched_patterns(graph, nodes, edges):
    ret = []
    if len(graph) and len(graph) >= len(nodes):
        for i, n in enumerate(nodes):
            if len(n) == 1:
                nodes[i] = tuple([n[0], {'op': None, 'unique': True}])
            elif len(n) == 2:
                if 'op' not in n[1]:
                    nodes[i][1].update({'op': None})
                if 'unique' not in n[1]:
                    nodes[i][1].update({'unique': True})
        for i, e in enumerate(edges):
            if len(e) == 2:
                edges[i] = tuple([e[0], e[1], {'src_out_port': None, 'dst_in_port': None}])
            elif len(e) == 3:
                if 'src_out_port' not in e[2]:
                    edges[i][2].update({'src_out_port': None})
                if 'dst_in_port' not in e[2]:
                    edges[i][2].update({'dst_in_port': None})
        pattern = Graph(name='pattern')
        pattern.add_nodes_from(nodes)
        pattern.add_edges_from(edges)
        matcher = isomorphism.MultiDiGraphMatcher(graph, pattern, node_feasibility, edge_feasibility)
        # matches = [{v: k for k, v in m.items()} for m in matcher.subgraph_monomorphisms_iter()]
        matches = [{v: k for k, v in m.items()} for m in matcher.subgraph_isomorphisms_iter()]
        if len(matches) > 1:
            keys = list(matches[0].keys())
            matches = sorted(matches, key=lambda x: tuple(x[k] for k in keys))
        if matches and any(n[1]['unique'] for n in nodes):
            for m in matches:
                current_found_names = set([v for meta_ret in ret for v in meta_ret.values()])
                unique_names = set([v for k, v in m.items() if pattern.nodes[k]['unique']])
                if not current_found_names.intersection(unique_names):
                    ret.append(m)
        else:
            ret = matches
    return ret


def single_node_matcher(graph, node_type):
    op_dict = {'op': node_type} if node_type else {}
    return matched_patterns(graph, nodes=[('target', op_dict)], edges=[])


def two_nodes_matcher(graph, begin_op, end_op):
    begin_dict = {'op': begin_op} if begin_op else {}
    end_dict = {'op': end_op} if end_op else {}
    return matched_patterns(graph,
                            nodes=[('begin', begin_dict), ('end', end_dict)],
                            edges=[('begin', 'end')])
