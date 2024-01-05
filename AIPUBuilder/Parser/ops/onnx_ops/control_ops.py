# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


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
        out_edges = self._graph.sorted_out_edges(self.name)
        appendix_in_edges_num = len(self.body._attr['root_in_ports'])
        if (len(in_edges[:-appendix_in_edges_num]) == 2
                or len(in_edges[:-appendix_in_edges_num]) == 3
                or len(in_edges[:-appendix_in_edges_num]) == 0) \
                and in_edges[0][2]['tensor'].is_const \
                and in_edges[0][2]['tensor'].value is not None \
                and in_edges[1][2]['tensor'].is_const \
                and in_edges[1][2]['tensor'].value is not None:
            from ...graph.graph_algo import determined_sort
            sub_nodes = determined_sort(
                self.body, self.body._attr['output_names'])
            count, cond = int(inputs[0]), bool(inputs[1])
            output_list = []
            for i in range(count):
                if cond:
                    meta_out = None
                    for n in sub_nodes:
                        try:
                            sub_node_obj = self._graph.nodes[n]['object']
                        except:
                            # if sub_node is not in the graph, skip it.
                            continue
                        sub_in_edges = self._graph.sorted_in_edges(
                            n, data=True)
                        for sub_src, sub_dst, in_attr in sub_in_edges:
                            # reset iter_count in each loop.
                            # TODO: need to support more forms.
                            if self._graph.nodes[sub_src]['op'] == 'Dummy' \
                                    and (in_attr['tensor'].name, in_attr['src_out_port']) in self.body._attr['input_tensors']:
                                in_attr['tensor'].value = np.array(
                                    i, np.dtype(in_attr['tensor'].dtype))
                            if self._graph.nodes[sub_src]['op'] == 'Constant' \
                                    and sub_src == in_edges[0][0] \
                                    and (self._graph.nodes[sub_dst]['op'] == 'Unsqueeze'
                                         or self._graph.nodes[sub_dst]['op'] == 'Add'):
                                in_attr['tensor'].value = np.array(
                                    i, in_attr['tensor'].dtype)

                            # reset last loop body y_out as next y_in if necessary.
                            if i > 0 \
                                    and len(self.body._attr['output_names']) == 3 \
                                    and self.body._attr['output_names'][1] == n \
                                    and meta_out is not None:
                                in_attr['tensor'].value = np.array(
                                    meta_out, in_attr['tensor'].dtype)

                        sub_node_obj.infer_shape()

                    try:
                        if self.body._attr['output_names'][-1] in self._graph.nodes:
                            meta_out = self._graph.nodes[self.body._attr['output_names'][-1]]['object'].get_output_tensors()[
                                0]
                        else:
                            meta_out = self._graph.nodes[self.body._attr['output_names'][-2]]['object'].get_output_tensors()[
                                0]
                    except:
                        WARN(
                            'unsupported loop: loop body output name is not in the graph.')

                    output_list.append(meta_out)
                else:
                    break

            if len(output_list) > 0:
                out_tensor_list = np.stack(output_list)
                out_tensor = out_tensor_list[-1]
            else:
                out_tensor_list = np.array(output_list)
                if len(in_edges) == 3:
                    out_tensor = in_edges[-1][2]['tensor'].value
                else:
                    out_tensor = None
                    # TODO: need to support condition is False and v_initial is None
                    WARN('need to support more forms.')

            if len(out_edges) == 2:
                self.set_out_tensor([out_tensor, out_tensor_list])
            else:
                self.set_out_tensor([out_tensor_list])
        else:
            WARN('[Parser]: Loop(%s) with more than 2 inputs or non-const trip-count/condition is not supported!' % self.name)
