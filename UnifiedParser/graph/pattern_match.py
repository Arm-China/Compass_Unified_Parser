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

import copy
from .graph import Node, Graph


def _node_feasibility(n1, n2):
    if n1.op is None:
        return False
    elif n2.op is None:
        return True
    else:
        if isinstance(n2.op, (list, tuple)):
            return n1.op in n2.op
        else:
            return n1.op == n2.op


def _edge_feasibility(e1, e2):
    if _node_feasibility(e1.start_node, e2.start_node) and _node_feasibility(e1.end_node, e2.end_node):
        src_check = True if e2.src_out_port is None else e1.src_out_port == e2.src_out_port
        dst_check = True if e2.dst_in_port is None else e1.dst_in_port == e2.dst_in_port
        return src_check and dst_check
    else:
        return False


def _traverse(graph, u, p, unexplored_ve):
    if len(graph) > 0 and graph.has_node(u):
        u_obj = graph.get_node(u)
        p.append(u_obj)
        u_obj.explored = True
        unexplored_ve -= 1

        u_out_edges = graph.sorted_out_edges(u, keys=True, data=True)
        for u_name, v_name, k, edge_attr in u_out_edges:
            e_obj = graph.get_edge(u_name, v_name, k)
            v_obj = graph.get_node(v_name)
            if not e_obj.explored and v_obj.explored:
                p.append(e_obj)
                e_obj.explored = True
                unexplored_ve -= 1
                p.append(v_obj)
                if unexplored_ve > 0:
                    p.append(e_obj)
                    p.append(u_obj)

        u_in_edges = graph.sorted_in_edges(u, keys=True, data=True)
        for v_name, u_name, k, edge_attr in u_in_edges:
            e_obj = graph.get_edge(v_name, u_name, k)
            v_obj = graph.get_node(v_name)
            if not e_obj.explored and v_obj.explored:
                p.append(e_obj)
                e_obj.explored = True
                unexplored_ve -= 1
                p.append(v_obj)
                if unexplored_ve > 0:
                    p.append(e_obj)
                    p.append(u_obj)

        unexplored_u_out_edges = [u_out for u_out in graph.sorted_out_edges(
            u, keys=True, data=True) if not u_out[3]['explored']]
        while unexplored_u_out_edges:
            sorted_unexplored_u_out_edges = sorted(
                unexplored_u_out_edges, key=lambda x: graph.get_node(x[1]).in_degree(explored=False))
            v = sorted_unexplored_u_out_edges[0][1]
            e_obj = graph.get_edge(
                *sorted_unexplored_u_out_edges[0][0:3])
            p.append(e_obj)
            e_obj.explored = True
            unexplored_ve -= 1

            _traverse(graph, v, p, unexplored_ve)
            if unexplored_ve == 0:
                break
            p.append(e_obj)
            p.append(u_obj)
            unexplored_u_out_edges = [u_out for u_out in graph.sorted_out_edges(
                u, keys=True, data=True) if not u_out[3]['explored']]

        unexplored_u_in_edges = [u_in for u_in in graph.sorted_in_edges(
            u, keys=True, data=True) if not u_in[3]['explored']]
        while unexplored_u_in_edges:
            sorted_unexplored_u_in_edges = sorted(
                unexplored_u_in_edges, key=lambda x: graph.get_node(x[0]).out_degree(explored=False))
            v = sorted_unexplored_u_in_edges[0][0]
            e_obj = graph.get_edge(
                *sorted_unexplored_u_in_edges[0][0:3])
            p.append(e_obj)
            e_obj.explored = True
            unexplored_ve -= 1

            _traverse(graph, v, p, unexplored_ve)
            if unexplored_ve == 0:
                break
            p.append(e_obj)
            p.append(u_obj)
            unexplored_u_in_edges = [u_in for u_in in graph.sorted_in_edges(
                u, keys=True, data=True) if not u_in[3]['explored']]


def _graph_linearization(graph):
    ret = []
    if len(graph) > 0:
        graph.set_edges_explored(explored=False)
        graph.set_nodes_explored(explored=False)

        possible_starts = [
            node for node in graph.nodes.values() if node.in_degree() == 0]
        possible_starts = sorted(possible_starts, key=lambda x: (
            x.in_degree(explored=False) + x.out_degree(explored=False)))
        start_key = possible_starts[0].key if len(
            possible_starts) else list(graph.nodes)[0]
        unexplored_graph_elem = graph.num_vertices_edges
        _traverse(graph, start_key, ret, unexplored_graph_elem)

        graph.set_edges_explored(explored=False)
        graph.set_nodes_explored(explored=False)

    hash_set = set([hash(obj) for obj in ret])
    if len(hash_set) != graph.num_vertices_edges:
        ret = []
    return ret


def _extend_match(u, p, i, f, g, graph_2):
    if i + 1 == len(p):
        return True, f

    if f[p[i+2].hash_value] is None:
        u_out_edges = graph_2.sorted_out_edges(u, keys=True)
        for u_name, v_name, key in u_out_edges:
            v_node = graph_2.get_node(v_name)
            e_edge = graph_2.get_edge(u_name, v_name, key)
            v_hash, e_hash = v_node.hash_value, e_edge.hash_value
            if g[v_hash] is False \
                    and g[e_hash] is False \
                    and _node_feasibility(v_node, p[i+2]) \
                    and _edge_feasibility(e_edge, p[i+1]):
                f_tmp, g_tmp = copy.deepcopy(f), copy.deepcopy(g)
                f_tmp[p[i+1].hash_value], f_tmp[p[i+2].hash_value] = e_hash, v_hash
                g_tmp[e_hash], g_tmp[v_hash] = True, True
                ret = _extend_match(v_name, p, i+2, f_tmp,
                                    g_tmp, graph_2)
                if ret[0]:
                    return ret

        u_in_edges = graph_2.sorted_in_edges(u, keys=True)
        for v_name, u_name, key in u_in_edges:
            v_node = graph_2.get_node(v_name)
            e_edge = graph_2.get_edge(v_name, u_name, key)
            v_hash, e_hash = v_node.hash_value, e_edge.hash_value
            if g[v_hash] is False \
                    and g[e_hash] is False \
                    and _node_feasibility(v_node, p[i+2]) \
                    and _edge_feasibility(e_edge, p[i+1]):
                f_tmp, g_tmp = copy.deepcopy(f), copy.deepcopy(g)
                f_tmp[p[i+1].hash_value], f_tmp[p[i+2].hash_value] = e_hash, v_hash
                g_tmp[e_hash], g_tmp[v_hash] = True, True
                ret = _extend_match(v_name, p, i+2, f_tmp,
                                    g_tmp, graph_2)
                if ret[0]:
                    return ret
    else:
        v_hash = f[p[i+2].hash_value]
        v_node = graph_2._attr['hash_map'][v_hash]
        if f[p[i+1].hash_value] is None:
            u_out_edges = graph_2.sorted_out_edges(u, keys=True)
            u_out_edges = [
                out_edge for out_edge in u_out_edges if out_edge[1] == v_node.key]
            for u_name, v_name, key in u_out_edges:
                e_edge = graph_2.get_edge(u_name, v_name, key)
                e_hash = e_edge.hash_value
                if g[e_hash] is False and _edge_feasibility(e_edge, p[i+1]):
                    f_tmp, g_tmp = copy.deepcopy(f), copy.deepcopy(g)
                    f_tmp[p[i+1].hash_value] = e_hash
                    g_tmp[e_hash] = True
                    ret = _extend_match(v_node.key, p, i+2,
                                        f_tmp, g_tmp, graph_2)
                    if ret[0]:
                        return ret

            u_in_edges = graph_2.sorted_in_edges(u, keys=True)
            u_in_edges = [
                in_edge for in_edge in u_in_edges if in_edge[0] == v_node.key]
            for v_name, u_name, key in u_in_edges:
                e_edge = graph_2.get_edge(v_name, u_name, key)
                e_hash = e_edge.hash_value
                if g[e_hash] is False and _edge_feasibility(e_edge, p[i+1]):
                    f_tmp, g_tmp = copy.deepcopy(f), copy.deepcopy(g)
                    f_tmp[p[i+1].hash_value] = e_hash
                    g_tmp[e_hash] = True
                    ret = _extend_match(v_node.key, p, i+2,
                                        f_tmp, g_tmp, graph_2)
                    if ret[0]:
                        return ret
        else:
            ret = _extend_match(v_node.key, p, i+2, f, g, graph_2)
            if ret[0]:
                return ret

    return False, None


def _parameterized_matching(graph_1, graph_2):
    matches = []

    p = _graph_linearization(graph_1)
    if p:
        p_hash_map = Graph.element_hash_map(p)
        g1_elements_hash_map = graph_1.vertices_edges_hash_map()
        graph_2._attr['hash_map'] = graph_2.vertices_edges_hash_map()

        f = {k: None for k in p_hash_map.keys()}
        g = {k: False for k in graph_2._attr['hash_map'].keys()}

        hash_matches = []
        for u in graph_2.nodes:
            u_node = graph_2.get_node(u)
            if _node_feasibility(u_node, p[0]) \
                    and (len(p) == 1 or u_node.out_degree() >= p[0].out_degree()):
                f_tmp, g_tmp = copy.deepcopy(f), copy.deepcopy(g)
                u_hash = u_node.hash_value
                assert u_hash in g
                f_tmp[p[0].hash_value] = u_hash
                g_tmp[u_hash] = True
                ret = _extend_match(u, p, 0, f_tmp, g_tmp,
                                    graph_2)
                if ret[0]:
                    hash_matches.append(ret[1])
                    continue

        for m in hash_matches:
            node_map_dict = {}
            for k, v in m.items():
                g1_elem = g1_elements_hash_map[k]
                g2_elem = graph_2._attr['hash_map'][v]
                if isinstance(g1_elem, Node) and isinstance(g2_elem, Node):
                    node_map_dict.update({g1_elem.key: g2_elem.key})

            # check node_map
            need_break = False
            for g1_node in graph_1.nodes:
                for pred in graph_1.pred[g1_node]:
                    if node_map_dict[pred] not in graph_2.pred[node_map_dict[g1_node]]:
                        need_break = True
                        break
                if need_break:
                    break
                for succ in graph_1.succ[g1_node]:
                    if node_map_dict[succ] not in graph_2.succ[node_map_dict[g1_node]]:
                        need_break = True
                        break
                if need_break:
                    break
            if need_break:
                continue

            cur_found_names = set(
                [v for single_match in matches for v in single_match.values()])
            unique_node_map = set([v for k, v in node_map_dict.items() if graph_1.get_node(k).unique])
            if not cur_found_names.intersection(unique_node_map):
                matches.append(node_map_dict)

        del graph_2._attr['hash_map']

    return matches


def matched_patterns(graph, nodes, edges):
    if len(nodes) <= len(graph):
        for i, e in enumerate(edges):
            if len(e) == 2:
                edges[i] = tuple(
                    [e[0], e[1], {'src_out_port': None, 'dst_in_port': None}])
            elif len(e) == 3:
                if 'src_out_port' not in e[2]:
                    edges[i][2].update({'src_out_port': None})
                if 'dst_in_port' not in e[2]:
                    edges[i][2].update({'dst_in_port': None})
        sub_graph = Graph(name='pattern')
        sub_graph.add_nodes_from(nodes)
        sub_graph.add_edges_from(edges)
        matched_items = _parameterized_matching(
            sub_graph, graph)
        return matched_items
    else:
        return []


def single_node_matcher(graph, node_type):
    op_dict = {'op': node_type} if node_type else {}
    return matched_patterns(graph, nodes=[('target', op_dict)], edges=[])


def two_nodes_matcher(graph, begin_op, end_op):
    begin_dict = {'op': begin_op} if begin_op else {}
    end_dict = {'op': end_op} if end_op else {}
    return matched_patterns(graph,
                            nodes=[('begin', begin_dict), ('end', end_dict)],
                            edges=[('begin', 'end')])
