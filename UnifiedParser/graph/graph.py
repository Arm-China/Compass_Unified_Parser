# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
from collections import OrderedDict, defaultdict
from .view import NodeView
from ..common.defs import Tensor
from ..logger import INFO, DEBUG, WARN, ERROR, FATAL


class Node(object):
    '''
    The node in the calculation graph, the definition contains the detailed information of the current node, 
    as well as the input and output information (input node, output node, input edge, output edge).
    Node defines function operations and property information.
    '''
    DEFAULT_ATTR = {'op': None, 'explored': False}

    def __init__(self, graph, node_for_adding, **attr):
        assert isinstance(node_for_adding, (int, str)
                          ), 'Invalid type for a Node key!'
        self._graph = graph
        self._key = node_for_adding
        self._attr = copy.deepcopy(Node.DEFAULT_ATTR)
        self.update_attr(**attr)

    def update_attr(self, **attr):
        if attr:
            self._attr.update(attr)

    @property
    def key(self):
        return self._key

    @property
    def op(self):
        return self._attr.get('op', None)

    @property
    def hash_value(self):
        return hash(self._key)

    @property
    def explored(self):
        return self._attr.get('explored', False)

    @explored.setter
    def explored(self, value):
        self._attr['explored'] = bool(value)

    @property
    def unique(self):
        return self._attr.get('unique', True)

    @property
    def is_leaf(self):
        return len(self._graph._adj_dict[self._key]) == 0

    @property
    def is_root(self):
        for k, v in self._graph._adj_dict.items():
            for v_k in v.keys():
                if v_k == self._key:
                    return False
        return True

    def in_degree(self, explored=None):
        '''Returns the number of in-degrees for this node.'''
        assert self._graph is not None, 'The graph is empty and the in-degree of the node cannot be obtained.'
        if self._key in self._graph._nodes_dict:
            ret = 0
            for start, v in self._graph._adj_dict.items():
                for end, edges in v.items():
                    if end == self._key:
                        if explored is None:
                            ret += len(edges)
                        elif explored:
                            unexplored_edge_keys = [
                                k for k, edge in edges.items() if edge._attr['explored']]
                            ret += len(unexplored_edge_keys)
                        else:
                            explored_edge_keys = [
                                k for k, edge in edges.items() if not edge._attr['explored']]
                            ret += len(explored_edge_keys)
        else:
            ret = None
        return ret

    def out_degree(self, explored=None):
        '''Returns the number of out-degrees for this node.'''
        assert self._graph is not None, 'The graph is empty and the out-degree of the node cannot be obtained.'
        if self._key in self._graph._nodes_dict:
            ret = 0
            for start, v in self._graph._adj_dict.items():
                if start == self._key:
                    for end, edges in v.items():
                        if explored is None:
                            ret += len(edges)
                        elif explored:
                            unexplored_edge_keys = [
                                k for k, edge in edges.items() if edge._attr['explored']]
                            ret += len(unexplored_edge_keys)
                        else:
                            explored_edge_keys = [
                                k for k, edge in edges.items() if not edge._attr['explored']]
                            ret += len(explored_edge_keys)
        else:
            ret = None
        return ret


class Edge(object):
    '''
    Edges in Computational Graphs.
    Edge connects source and destination nodes.
    '''
    DEFAULT_ATTR = {'src_out_port': 0, 'dst_in_port': 0,
                    'tensor': Tensor(), 'explored': False}

    def __init__(self, u_node, v_node, **attr):
        self._start_node, self._end_node = None, None
        self._attr = copy.deepcopy(Edge.DEFAULT_ATTR)
        if isinstance(u_node, Node) and isinstance(v_node, Node):
            self._start_node = u_node
            self._end_node = v_node
            self.update_attr(**attr)
        else:
            raise RuntimeError('Meets invalid edge start node or end node!')

    def update_attr(self, **attr):
        if attr:
            self._attr.update(attr)

    @property
    def start(self):
        return self._start_node.key

    @property
    def end(self):
        return self._end_node.key

    @property
    def start_node(self):
        return self._start_node

    @property
    def end_node(self):
        return self._end_node

    @property
    def src_out_port(self):
        return self._attr.get('src_out_port', 0)

    @property
    def dst_in_port(self):
        return self._attr.get('dst_in_port', 0)

    @property
    def hash_value(self):
        return hash((self.start, self.end, self.src_out_port, self.dst_in_port))

    @property
    def explored(self):
        return self._attr.get('explored', False)

    @explored.setter
    def explored(self, value):
        self._attr['explored'] = bool(value)


class Graph(object):
    '''
    Represents a large class of computing graphs, 
    which contain the complete structure of the entire graph. 
    The definition of the graph here is the unique starting point and the only ending point, 
    as well as the available computing equipment table
    '''
    @staticmethod
    def element_hash_map(elem_list):
        ret = OrderedDict()
        for elem in elem_list:
            assert isinstance(elem, (Node, Edge)
                              ), 'The types of Node and Edge are invalid.'
            hash_value = elem.hash_value
            if hash_value not in ret:
                ret.update({hash_value: elem})
        return ret

    def __init__(self, **attr):
        self._nodes_dict = OrderedDict()
        self._adj_dict = OrderedDict()
        self._attr = defaultdict()
        self.update_attr(**attr)

    def __len__(self):
        return len(self.nodes)

    def update_attr(self, **attr):
        if attr:
            self._attr.update(attr)

    def has_node(self, node_key):
        return node_key in self._nodes_dict

    def has_edge(self, u, v, key=None):
        '''Check if there is an edge between two nodes.'''
        try:
            if key is None:
                return v in self._adj_dict[u]
            else:
                return key in self._adj_dict[u][v]
        except KeyError:
            return False

    def add_node(self, node_for_adding, **attr):
        '''Add nodes to the graph.'''
        if node_for_adding not in self.nodes:
            node_obj = Node(self, node_for_adding, **attr)
            self._nodes_dict.update({node_for_adding: node_obj})
            self._adj_dict[node_for_adding] = OrderedDict()
        else:
            if attr:
                WARN('[Parser]: Node (%s) already exits and attributes are updating ...' % str(
                    node_for_adding))
                self._nodes_dict[node_for_adding].update_attr(**attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for node in nodes_for_adding:
            try:
                if node not in self.nodes:
                    node_obj = Node(self, node, **attr)
                    self._nodes_dict.update({node: node_obj})
                    self._adj_dict[node] = OrderedDict()
                else:
                    if attr:
                        WARN(
                            '[Parser]: Node (%s) already exits and attributes are updating...' % str(node))
                        self._nodes_dict[node].update_attr(**attr)
            except TypeError:
                n, n_attr = node
                n_attr.update(attr)
                if n not in self.nodes:
                    node_obj = Node(self, n, **n_attr)
                    self._nodes_dict.update({n: node_obj})
                    self._adj_dict[n] = OrderedDict()
                else:
                    if n_attr:
                        WARN(
                            '[Parser]: Node (%s) already exits and attributes are updating...' % str(n))
                        self._nodes_dict[n].update_attr(**n_attr)

    def remove_node(self, node_for_removing):
        '''Delete a node in the graph.'''
        assert isinstance(node_for_removing, (int, str)
                          ), 'invalid type for a Node key!'
        if node_for_removing in self._attr.get('output_names', None):
            in_edges = self.sorted_in_edges(node_for_removing)
            if len(in_edges) >= 1:
                input_node_of_removing = [
                    u for u, _ in self.sorted_in_edges(node_for_removing)][0]
                found_index = self._attr['output_names'].index(
                    node_for_removing)
                self._attr['output_names'][found_index] = input_node_of_removing
                outname_dict = OrderedDict(
                    {k: i for (i, k) in enumerate(self._attr['output_names'])})
                self._attr['output_names'] = list(outname_dict.keys())
            else:
                WARN(
                    '[Parser]: Removing output node (%s) does not have preceding node in remove_node!' % node_for_removing)
                self._attr['output_names'].remove(node_for_removing)

        if node_for_removing in self._nodes_dict:
            removing_node_obj = self._nodes_dict.pop(node_for_removing)
            del removing_node_obj
            if node_for_removing in self._adj_dict:
                self._adj_dict.pop(node_for_removing)
            for k, v in self._adj_dict.items():
                if node_for_removing in v:
                    v.pop(node_for_removing)
        else:
            WARN('[Parser]: The removing node (%s) does not exist in graph!' %
                 str(node_for_removing))

    def remove_nodes_from(self, nodes_for_removing):
        for n in nodes_for_removing:
            self.remove_node(n)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        '''Add an edge between two nodes in the graph.'''
        edge_attr = copy.deepcopy(
            attr) if attr else copy.deepcopy(Edge.DEFAULT_ATTR)
        if 'src_out_port' not in edge_attr:
            edge_attr.update({'src_out_port': 0})
        if 'dst_in_port' not in edge_attr:
            edge_attr.update({'dst_in_port': 0})
        self.add_nodes_from([u_of_edge, v_of_edge])
        node_pair = (self._nodes_dict[u_of_edge], self._nodes_dict[v_of_edge])
        edge_obj = Edge(*node_pair, **attr)
        if u_of_edge not in self._adj_dict or v_of_edge not in self._adj_dict[u_of_edge]:
            self._adj_dict[u_of_edge][v_of_edge] = {0: edge_obj}
        else:
            updated = False
            for k, v in self._adj_dict[u_of_edge][v_of_edge].items():
                if v.src_out_port == edge_attr['src_out_port'] and v.dst_in_port == edge_attr['dst_in_port']:
                    WARN('[Parser]: Meets the same out/in port between two nodes (%s,%s)! updating attributes...'
                         % (str(u_of_edge), str(v_of_edge)))
                    v.update_attr(**edge_attr)
                    updated = True
                    break
            if not updated:
                cur_key = None
                cur_edge_num = len(self._adj_dict[u_of_edge][v_of_edge])
                for k in range(cur_edge_num):
                    if k not in self._adj_dict[u_of_edge][v_of_edge]:
                        cur_key = k
                        break
                if cur_key is None:
                    cur_key = max(
                        list(self._adj_dict[u_of_edge][v_of_edge].keys())) + 1
                self._adj_dict[u_of_edge][v_of_edge].update(
                    {cur_key: edge_obj})

    def add_edges_from(self, ebunch_to_add):
        for e in ebunch_to_add:
            if len(e) == 3:
                u, v, d = e
            elif len(e) == 2:
                u, v = e
                d = {}
            else:
                raise RuntimeError(
                    '[Parser]: Invalid edge to be added in add_edges_from!')
            self.add_edge(u, v, **d)

    def remove_edge(self, u_of_edge, v_of_edge, key=None):
        '''Remove an edge between two nodes in the graph.'''
        assert u_of_edge in self.nodes and v_of_edge in self.nodes, 'The edge to be deleted is not in the graph.'
        if v_of_edge in self._adj_dict[u_of_edge]:
            if len(self._adj_dict[u_of_edge][v_of_edge]):
                if key is None or isinstance(key, dict):
                    self._adj_dict[u_of_edge].pop(v_of_edge)
                elif key in self._adj_dict[u_of_edge][v_of_edge]:
                    self._adj_dict[u_of_edge][v_of_edge].pop(key)
                    if len(self._adj_dict[u_of_edge][v_of_edge]) == 0:
                        self._adj_dict[u_of_edge].pop(v_of_edge)

    def remove_edges_from(self, ebunch):
        for e in ebunch:
            try:
                self.remove_edge(*e[:3])
            except Exception as e:
                WARN('[Parser]: Meets error (%s) in remove_edges_from!' % str(e))

    def sorted_in_edges(self, n, keys=False, data=False):
        '''Arrange in_edges in the order of dst_in_port.'''
        assert n in self.nodes, ('Node(%s) does not exist in the graph!' % n)
        input_edges = []
        for start, v in self._adj_dict.items():
            for end, edges in v.items():
                if end == n:
                    for edge_key, edge in edges.items():
                        input_edges.append((start, end, edge_key, edge._attr))
        input_edges = sorted(
            input_edges, key=lambda x: (x[3]['dst_in_port'] if x[3]['dst_in_port'] is not None else 0, x[2]))
        if keys and data:
            ret = [(u, v, k, d) for u, v, k, d in input_edges]
        elif keys:
            ret = [(u, v, k) for u, v, k, _ in input_edges]
        elif data:
            ret = [(u, v, d) for u, v, _, d in input_edges]
        else:
            ret = [(u, v) for u, v, _, _ in input_edges]
        return ret

    def sorted_out_edges(self, n, keys=False, data=False):
        '''Arrange out_edges in the order of dst_in_port.'''
        assert n in self.nodes, ('Node(%s) does not exist in the graph!' % n)
        output_edges = []
        for start, v in self._adj_dict.items():
            if start == n:
                for end, edges in v.items():
                    for edge_key, edge in edges.items():
                        output_edges.append((start, end, edge_key, edge._attr))
        output_edges = sorted(
            output_edges, key=lambda x: (x[3]['src_out_port'] if x[3]['src_out_port'] is not None else 0, x[2]))
        if keys and data:
            ret = [(u, v, k, d) for u, v, k, d in output_edges]
        elif keys:
            ret = [(u, v, k) for u, v, k, _ in output_edges]
        elif data:
            ret = [(u, v, d) for u, v, _, d in output_edges]
        else:
            ret = [(u, v) for u, v, _, _ in output_edges]
        return ret

    def set_nodes_explored(self, explored=True):
        for node in self._nodes_dict.values():
            node.explored = explored

    def set_edges_explored(self, explored=True):
        for start, v in self._adj_dict.items():
            for end, edges in v.items():
                for _, edge in edges.items():
                    edge.explored = explored

    def sorted_in_consts(self, u):
        ret = []
        for u_pred, _ in self.sorted_in_edges(u):
            if self.nodes[u_pred]._attr['op'] == 'Constant' and self.nodes[u_pred]._attr.get('object', None) is not None:
                ret.append(
                    (u_pred, self.nodes[u_pred]._attr['object']._attr['value']))
        return ret

    def get_node(self, node_key):
        assert node_key in self.nodes, 'The node key is not in the graph.'
        return self._nodes_dict[node_key]

    def get_edge(self, u, v, key=0):
        assert u in self._adj_dict and v in self._adj_dict[u] and key in self._adj_dict[
            u][v], 'The edge to be obtained is not in the adjacency list.'
        return self._adj_dict[u][v][key]

    def vertices_edges_hash_map(self):
        elem_list = []
        for node in self._nodes_dict.values():
            elem_list.append(node)
        for start, v in self._adj_dict.items():
            for end, edges in v.items():
                for k, edge in edges.items():
                    elem_list.append(edge)
        return Graph.element_hash_map(elem_list)

    def clear(self):
        self._nodes_dict.clear()
        self._adj_dict.clear()
        self._attr.clear()

    @property
    def num_vertices(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        ret = 0
        for start, v in self._adj_dict.items():
            for end, edges in v.items():
                ret += len(edges)
        return ret

    @property
    def num_vertices_edges(self):
        return len(self) + self.num_edges

    @property
    def nodes(self):
        nodes = NodeView(self)
        self.__dict__['nodes'] = nodes
        return nodes

    @property
    def pred(self):
        ret = OrderedDict()
        for start, v in self._adj_dict.items():
            if start not in ret:
                ret[start] = []
            for end, edges in v.items():
                if end not in ret:
                    ret[end] = [start]
                else:
                    ret[end].append(start)
        return ret

    @property
    def succ(self):
        ret = OrderedDict()
        for start, v in self._adj_dict.items():
            if start not in ret:
                ret[start] = []
            for end, edges in v.items():
                if start not in ret:
                    ret[start] = [end]
                else:
                    ret[start].append(end)
        return ret

    def dot(self):
        keys = list(self.nodes._nodes.keys())
        ret = "digraph \"%s\" {\nnode [shape=record];\n" % self._attr["name"]

        available_colors = ["chartreuse", "beige",
                            "gold", "greenyellow", "purple"]

        for i, (n_name, n) in enumerate(self.nodes._nodes.items()):
            id_s = str(i)
            color = "white"
            label = "node_%d [style=\"filled\",color=\"black\",fillcolor=\"%s\",label=\"{%s %s|%s" % (
                keys.index(n_name), color, id_s, n_name, n.op)

            in_tensor = [str(a["tensor"].shape)
                         for _, _, a in self.sorted_in_edges(n_name, data=True)]
            out_tensor = [str(a["tensor"].shape)
                          for _, _, a in self.sorted_out_edges(n_name, data=True)]
            label += "|"
            if len(in_tensor):
                label += "%s \-\> " % (",".join(in_tensor))
            label += ",".join(out_tensor)
            label += '|'

            for name, successors in self._adj_dict.items():
                for s in successors:
                    if s == n_name:
                        in_port = successors[s][0]._attr['dst_in_port']
                        out_port = successors[s][0]._attr['src_out_port']
                        label += "dst_in: %s " % (out_port)
                        label += "src_out: %s, " % (in_port)

            label_end = "}\" ];\n"
            label += "%s"+label_end
            comment = ""
            ret += label % (comment)

        for name, successors in self._adj_dict.items():
            p_id = keys.index(name)
            for s in successors:
                if s in keys:
                    ret += "node_%d -> {node_%d};\n" % (
                        p_id, keys.index(s))
        ret += "};"
        return ret

    def to_png(self, file="graph.png"):
        try:
            import pydot
            ret = self.dot()
            g = pydot.graph_from_dot_data(self.dot())[0]
            g.write_png(file)

        except Exception as e:
            WARN(
                "Dependence pydot and graphviz! please install the dependencies.\n%s", str(e))

    def to_svg(self, file="graph.svg"):
        try:
            import pydot
            g = pydot.graph_from_dot_data(self.dot())[0]
            g.write_svg(file)

        except Exception as e:
            WARN(
                "Dependence pydot and graphviz! please install the dependencies.\n%s", str(e))


class ReadOnlyGraph(object):
    def not_allowed(self, *args, **kwds):
        msg = "SubGraph is readonly. Mutations not allowed"
        FATAL('[Parser]: ' + msg)

    update_attr = not_allowed
    add_node = not_allowed
    remove_node = not_allowed
    add_nodes_from = not_allowed
    remove_nodes_from = not_allowed
    add_edge = not_allowed
    remove_edge = not_allowed
    add_edges_from = not_allowed
    remove_edges_from = not_allowed
    clear = not_allowed


class SubGraph(ReadOnlyGraph, Graph):
    def __init__(self, graph, filter_node=None, filter_edge=None):
        if filter_node is None:
            filter_node = list()
        if filter_edge is None:
            filter_edge = list()
        self._root = graph
        self._filter_node = filter_node
        self._filter_edge = filter_edge
        self._attr = defaultdict()
        self._attr['input_tensors'] = {}
        self._attr['output_names'] = []
        self._attr['root_in_ports'] = []

    @property
    def _nodes_dict(self):
        ret = OrderedDict()
        for n in self._filter_node:
            if n in self._root.nodes:
                ret.update({n: self._root.nodes[n]})
        return ret

    @property
    def _adj_dict(self):
        ret = OrderedDict()
        for n in self._filter_node:
            if n in self._root.nodes and n not in ret:
                ret[n] = OrderedDict()

        for filter_edge_info in self._filter_edge:
            if len(filter_edge_info) == 2:
                u, v = filter_edge_info
                src_out_port, dst_in_port = 0, 0
            elif len(filter_edge_info) >= 3:
                u, v, d = filter_edge_info[:3]
                src_out_port = d.get('src_out_port', 0)
                dst_in_port = d.get('dst_in_port', 0)
            else:
                WARN('[Parser]: Meets invalid Subgraph edge!')
                continue

            if u in self._filter_node \
                    and u in self._root.nodes \
                    and v in self._filter_node \
                    and v in self._root.nodes:
                if u in self._root._adj_dict and v in self._root._adj_dict[u]:
                    if v not in ret[u]:
                        ret[u][v] = OrderedDict()
                    for k, edge_obj in self._root._adj_dict[u][v].items():
                        if edge_obj.src_out_port == src_out_port and edge_obj.dst_in_port == dst_in_port:
                            sub_k = len(ret[u][v])
                            ret[u][v][sub_k] = edge_obj
        return ret
