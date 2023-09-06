# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from ..op import *


class IfOp(OpHasSubGraph, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'else_branch': {'type': AttrType.GRAPH, 'required': True},
                    'then_branch': {'type': AttrType.GRAPH, 'required': True},
                    },
                11: {'else_branch': {'type': AttrType.GRAPH, 'required': True},
                     'then_branch': {'type': AttrType.GRAPH, 'required': True},
                     },
                13: {'else_branch': {'type': AttrType.GRAPH, 'required': True},
                     'then_branch': {'type': AttrType.GRAPH, 'required': True},
                     }
                }

    def __init__(self, graph, attr_dict=None):
        super(IfOp, self).__init__(graph, attr_dict)
        self.update_attributes(IfOp, attr_dict)
        assert self.check_required(), 'IfOp is missing a required parameter.'

    def infer_shape(self):
        super(IfOp, self).infer_shape()
        inputs = self.get_input_tensors()
        in_ports = self.then_branch._attr['root_in_ports'] if inputs[0] else self.else_branch._attr['root_in_ports']
        self.set_out_tensor(inputs[in_ports[0]: in_ports[-1] + 1])


class LoopOp(OpHasSubGraph, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'body': {'type': AttrType.GRAPH, 'required': True}},
                11: {'body': {'type': AttrType.GRAPH, 'required': True}},
                13: {'body': {'type': AttrType.GRAPH, 'required': True}},
                }

    def __init__(self, graph, attr_dict=None):
        super(LoopOp, self).__init__(graph, attr_dict)
        self.update_attributes(LoopOp, attr_dict)
        assert self.check_required(), 'LoopOp is missing a required parameter.'

    def infer_shape(self):
        super(LoopOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 2
        in_edges = self._graph.sorted_in_edges(self.name, data=True)
        appendix_in_edges_num = len(self.body._attr['root_in_ports'])
        if len(in_edges[:-appendix_in_edges_num]) == 2 \
                and in_edges[0][2]['tensor'].is_const \
                and in_edges[0][2]['tensor'].value is not None \
                and in_edges[1][2]['tensor'].is_const \
                and in_edges[1][2]['tensor'].value is not None:
            from ...graph.graph_algo import determined_sort
            sub_nodes = determined_sort(self.body, self.body._attr['output_names'])
            count, cond = int(inputs[0]), bool(inputs[1])
            output_list = []
            for i in range(count):
                if cond:
                    for n in sub_nodes:
                        sub_node_obj = self._graph.nodes[n]['object']
                        sub_in_edges = self._graph.sorted_in_edges(n, data=True)
                        for sub_src, _, in_attr in sub_in_edges:
                            if self._graph.nodes[sub_src]['op'] == 'Dummy' \
                                    and (in_attr['tensor'].name, in_attr['src_out_port']) in self.body._attr['input_tensors']:
                                in_attr['tensor'].value = np.array(i, np.dtype(in_attr['tensor'].dtype))
                        sub_node_obj.infer_shape()
                    meta_out = self._graph.nodes[self.body._attr['output_names'][-1]]['object'].get_output_tensors()[0]
                    output_list.append(meta_out)
                else:
                    break
            out_tensor = np.stack(output_list)
            self.set_out_tensor([out_tensor])
        else:
            WARN('[Parser]: Loop(%s) with more than 2 inputs or non-const trip-count/condition is not supported!' % self.name)
