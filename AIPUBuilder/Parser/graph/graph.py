# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from collections import OrderedDict, defaultdict
import networkx as nx
from ..common.defs import Tensor
from ..logger import INFO, DEBUG, WARN, ERROR, FATAL


class Graph(nx.MultiDiGraph):
    def __init__(self, **attr):
        super(Graph, self).__init__(incoming_graph_data=None, multigraph_input=None, **attr)
        self._attr = self.graph

    @property
    def successor(self):
        ret = OrderedDict()
        for start, v in self.adj.items():
            if start not in ret:
                ret[start] = []
            for end, _ in v.items():
                if start not in ret:
                    ret[start] = [end]
                else:
                    ret[start].append(end)
        return ret

    @property
    def predecessor(self):
        ret = OrderedDict()
        for start, v in self.adj.items():
            if start not in ret:
                ret[start] = []
            for end, _ in v.items():
                if end not in ret:
                    ret[end] = [start]
                else:
                    ret[end].append(start)
        return ret

    def children(self, node):
        return list(self.successors(node))

    def parents(self, node):
        return list(self.predecessors(node))

    def sorted_in_edges(self, n, keys=False, data=False):
        if n not in self.nodes:
            raise Exception('[Parser]: Node(%s) dose not exist in graph!' % n)
        in_edges = []
        for p in self.predecessors(n):
            edges = self.adj[p][n]
            for edge_k, edge_attr in edges.items():
                in_edges.append((p, n, edge_k, edge_attr))
        in_edges = sorted(in_edges, key=lambda x: (x[3]['dst_in_port'] if x[3]['dst_in_port'] is not None else 0, x[2]))
        if keys and data:
            ret = [(u, v, k, d) for u, v, k, d in in_edges]
        elif keys:
            ret = [(u, v, k) for u, v, k, _ in in_edges]
        elif data:
            ret = [(u, v, d) for u, v, _, d in in_edges]
        else:
            ret = [(u, v) for u, v, _, _ in in_edges]
        return ret

    def sorted_out_edges(self, n, keys=False, data=False):
        if n not in self.nodes:
            raise Exception('[Parser]: Node(%s) dose not exist in graph!' % n)
        out_edges = []
        for s in self.successors(n):
            edges = self.adj[n][s]
            for edge_k, edge_attr in edges.items():
                out_edges.append((n, s, edge_k, edge_attr))
        out_edges = sorted(out_edges, key=lambda x: (x[3]['src_out_port']
                                                     if x[3]['src_out_port'] is not None else 0, x[2]))
        if keys and data:
            ret = [(u, v, k, d) for u, v, k, d in out_edges]
        elif keys:
            ret = [(u, v, k) for u, v, k, _ in out_edges]
        elif data:
            ret = [(u, v, d) for u, v, _, d in out_edges]
        else:
            ret = [(u, v) for u, v, _, _ in out_edges]
        return ret

    def add_node(self, node_for_adding, **attr):
        if not isinstance(node_for_adding, str):
            node_for_adding = str(node_for_adding)
        if not attr:
            attr = {'op': None, 'unique': True}
        else:
            if 'op' not in attr:
                attr['op'] = None
            if 'unique' not in attr:
                attr['unique'] = True
        super(Graph, self).add_node(node_for_adding, **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for node in nodes_for_adding:
            try:
                if node not in self.nodes:
                    self.add_node(node, **attr)
                else:
                    if attr:
                        self.nodes[node].update(**attr)
            except TypeError:
                n, n_attr = node
                n_attr.update(attr)
                if n not in self.nodes:
                    self.add_node(n, **n_attr)
                else:
                    if n_attr:
                        self.nodes[node].update(**n_attr)

    def remove_node(self, n):
        if isinstance(n, (list, tuple)):
            if len(n) != 2:
                raise Exception(
                    '[Parser]: Length of node(%s) should be 2 when it is composed of name and attributes!' % n)
        if n in self.graph.get('output_names', []):
            in_edges = self.sorted_in_edges(n)
            if len(in_edges) >= 1:
                index = self.graph['output_names'].index(n)
                self.graph['output_names'].pop(index)
                for u, _ in in_edges:
                    if u not in self.graph['output_names']:
                        self.graph['output_names'].insert(index, u)
                        index += 1
            else:
                self.graph['output_names'].remove(n)
        super(Graph, self).remove_node(n)

    def remove_nodes_from(self, nodes):
        for n in nodes:
            self.remove_node(n)

    def add_edge(self, u_for_edge, v_for_edge, key=None, **attr):
        if not attr:
            attr = {'src_out_port': 0, 'dst_in_port': 0, 'tensor': Tensor()}
        else:
            if 'src_out_port' not in attr:
                attr['src_out_port'] = 0
            if 'dst_in_port' not in attr:
                attr['dst_in_port'] = 0
            if 'tensor' not in attr:
                attr['tensor'] = Tensor()
        self.add_nodes_from([u_for_edge, v_for_edge])
        key = super(Graph, self).add_edge(u_for_edge, v_for_edge, key=key, **attr)
        return key

    # def add_edges_from(self, ebunch_to_add, **attr):
    #     ret = super(Graph, self).add_nodes_from(ebunch_to_add, attr=attr)
    #     return ret

    # def remove_edge(self, u, v, key=None):
    #     super(Graph, self).remove_edge(u, v, key=key)

    def remove_edges_from(self, ebunch):
        for e in ebunch:
            try:
                if (len(e) == 3 and isinstance(e[2], int)) or len(e) == 4:
                    self.remove_edge(*e[:3])
                else:
                    self.remove_edge(*e[:2])
            except:
                pass

    def dot(self):
        keys = list(self.nodes._nodes.keys())
        ret = "digraph \"%s\" {\nnode [shape=record];\n" % self._attr["name"]

        for i, (n_name, n) in enumerate(self.nodes._nodes.items()):
            # n is a dictionary whose keys include 'op', 'unique' and 'object'.
            id_s = str(i)
            label = "node_%d [style=\"filled\",color=\"black\",fillcolor=\"%s\",label=\"{%s %s|%s" % (
                keys.index(n_name), "white", id_s, n_name, n.get('op', ''))
            in_tensor = [str(a["tensor"].shape) + str(a["tensor"].dtype)
                         for _, _, a in self.sorted_in_edges(n_name, data=True)]
            out_tensor = [str(a["tensor"].shape) + str(a["tensor"].dtype)
                          for _, _, a in self.sorted_out_edges(n_name, data=True)]
            label += "|"
            if len(in_tensor):
                label += "%s \-\> " % (",".join(in_tensor))
            label += ",".join(out_tensor)
            label_end = "}\" ];\n"
            label += "%s" + label_end
            comment = ""
            ret += label % (comment)

        for name, successors in self.adj.items():
            p_id = keys.index(name)
            for s in successors:
                if s in keys:
                    for _, edge in successors[s].items():
                        label = "node_%d -> {node_%d} [style=\"filled\",color=\"black\",label=\"" % (
                            p_id, keys.index(s))
                        in_port = edge['dst_in_port']
                        scale_zp = edge['tensor'].scale_zp if edge['tensor'] is not None else ''
                        out_port = edge['src_out_port']
                        label += "%s %s \n\n" % (out_port, str(scale_zp))
                        label += "%s " % (in_port)
                        label_end = "\"];"
                        label += label_end
                        ret += label
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


class SubGraph(Graph):
    def __init__(self, **attr):
        super(SubGraph, self).__init__(**attr)
        self._root = None
        self._attr = defaultdict(int)
        self._attr['framework'] = None
        self._attr['parent_graph'] = None
        self._attr['input_tensors'] = OrderedDict()
        self._attr['output_names'] = []
        self._attr['output_tensor_names'] = []
        self._attr['output_ports'] = OrderedDict()
        self._attr['root_in_ports'] = []
