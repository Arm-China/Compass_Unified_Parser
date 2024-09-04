# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.
import numpy as np

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
        self.real_loop_cnt = None
        self.update_attributes(LoopOp, attr_dict)
        assert self.check_required(), 'LoopOp is missing a required parameter.'

    def infer_shape(self):
        super(LoopOp, self).infer_shape()
        # 2 + N inputs: max_count, cond_in, ...
        inputs = self.get_input_tensors()
        assert len(inputs) >= 2
        in_edges = self._graph.sorted_in_edges(self.name, data=True)
        out_edges = self._graph.sorted_out_edges(self.name)
        from ...graph.graph_algo import determined_sort
        if in_edges[0][2]['tensor'].is_const and \
                in_edges[0][2]['tensor'].value is not None and \
                in_edges[1][2]['tensor'].is_const and \
                in_edges[1][2]['tensor'].value is not None:
            max_count, ori_cond_in = int(inputs[0]), bool(inputs[1])
            count_cond_is_const = True
        else:
            WARN(f'[Parser]: Loop({self.name}) max_count/cond_in is non-const, the infer shape is unreliable.')
            max_count, ori_cond_in = 5, True
            count_cond_is_const = False

        cond_in = ori_cond_in
        body_inputs_num = len(self.body._attr['input_tensors'])  # 2+N
        body_outputs_num = len(self.body._attr['output_names'])  # 1+K+N
        N = body_inputs_num - 2
        K = body_outputs_num - 1 - N
        k_carried_away = [[] for i in range(K)]

        sub_nodes = determined_sort(self.body, self.body._attr['output_names'])
        output_list = []
        loop_cnt = 0
        cond_out_root_inputs = self.body.nodes[self.body._attr['output_names'][0]]['object'].get_root_inputs()
        cond_out_root_input_const = {}
        for inp in cond_out_root_inputs:
            cond_out_root_input_const[inp] = False
        for i in range(max_count):
            if cond_in:
                for n in sub_nodes:
                    sub_node_obj = self.body.nodes[n]['object']
                    if sub_node_obj is None:
                        ERROR(f'[Parser]: Get Node Obj Failed in Loop Infer of Node: {n}.')
                    if sub_node_obj.type == 'Input':
                        inp_idx = list(self.body._attr['input_tensors'].keys()).index(n)
                        if inp_idx == 0:
                            out_tensor = np.array(loop_cnt, np.int64)
                        elif inp_idx == 1:
                            out_tensor = np.array(cond_in, bool)
                        else:
                            if loop_cnt == 0:
                                out_tensor = copy.deepcopy(inputs[inp_idx])
                            else:
                                out_tensor = copy.deepcopy(output_list[inp_idx - 1])
                        if n in cond_out_root_input_const:
                            cond_out_root_input_const[n] = in_edges[inp_idx][-1]['tensor'].is_const
                        sub_node_obj.infer_shape(out_tensor)
                    elif sub_node_obj.type == 'Dummy':
                        # input data from parent graph
                        parent_node = self.body._attr['parent_graph'].nodes[n]
                        dummy_out_edges = self.body._attr['parent_graph'].sorted_out_edges(parent_node['object'].name,
                                                                                           data=True)
                        out_tensor = dummy_out_edges[0][-1]['tensor'].value
                        if n in cond_out_root_input_const:
                            cond_out_root_input_const[n] = dummy_out_edges[0][-1]['tensor'].is_const
                        sub_node_obj.infer_shape(out_tensor)
                    else:
                        if sub_node_obj.type == 'Constant' and n in cond_out_root_input_const:
                            cond_out_root_input_const[n] = True
                        sub_node_obj.infer_shape()
                loop_cnt += 1
                # Loop body outputs: 1 + N + K
                for out in self.body._attr['output_names']:
                    out_tensor = self.body.nodes[out]['object'].get_output_tensors()[0]
                    output_list.append(out_tensor)
                cond_in = output_list[0]
                if K > 0:
                    for k in range(K):
                        k_carried_away[k].append(output_list[k - K])
            else:
                break
        if count_cond_is_const and all(list(cond_out_root_input_const.values())):
            self.real_loop_cnt = loop_cnt
        # Loop outputs: N + K
        if ori_cond_in:
            output_list = output_list[1: 1 + N]
            output_list.extend([np.vstack(x) for x in k_carried_away])
        else:
            output_list = inputs[2:]
        while len(output_list) < len(out_edges):
            output_list.append(np.array([]))
        self.set_out_tensor(output_list)
