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
        in_edges = self._graph.sorted_in_edges(self.name, data=True)
        out_edges = self._graph.sorted_out_edges(self.name)
        if in_edges[0][2]['tensor'].is_const and \
                in_edges[0][2]['tensor'].value is not None:
            if_cond = bool(inputs[0])
            if_cond_is_const = True
        else:
            WARN(f'[Parser]: If({self.name}) condition is non-const, the infer shape is unreliable.')
            if_cond = True
            if_cond_is_const = False
        # True: then branch, False: else branch
        from ...graph.graph_algo import determined_sort
        cur_sub_graph = self.then_branch if if_cond else self.else_branch
        sub_nodes = determined_sort(cur_sub_graph, cur_sub_graph._attr['output_names'])
        output_list = []
        output_const_list = []
        for n in sub_nodes:
            sub_node_obj = cur_sub_graph.nodes[n]['object']
            if sub_node_obj is None:
                ERROR(f'[Parser]: Get Node Obj Failed in If Infer of Node: {n}.')
            if sub_node_obj.type == 'DummyInput':
                # input data from parent graph
                parent_node = cur_sub_graph._attr['parent_graph'].nodes[n]
                dummy_out_edges = cur_sub_graph._attr['parent_graph'].sorted_out_edges(parent_node['object'].name,
                                                                                       data=True)
                out_tensor = dummy_out_edges[0][-1]['tensor'].value
                sub_node_obj.infer_shape(out_tensor)
            else:
                sub_node_obj.infer_shape()

        for out in cur_sub_graph._attr['output_names']:
            is_const = cur_sub_graph.nodes[out]['object'].is_all_inputs_const()
            for _, dst, out_attr in cur_sub_graph.sorted_out_edges(out, data=True):
                if cur_sub_graph.nodes[dst]['object'].type == 'Out':
                    out_port = out_attr['src_out_port']
                    out_tensor = cur_sub_graph.nodes[out]['object'].get_output_tensors()[out_port]
                    output_list.append(out_tensor)
                    output_const_list.append(is_const)
        self.set_out_tensor(output_list, all(output_const_list))


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

        from ...graph.graph_algo import determined_sort
        sub_nodes = determined_sort(self.body, self.body._attr['output_names'])
        loop_cnt = 0
        last_output_list = []
        last_output_const_list = []
        cond_out_root_inputs = self.body.nodes[self.body._attr['output_names'][0]]['object'].get_root_inputs()
        cond_out_root_input_const = {}
        for inp in cond_out_root_inputs:
            cond_out_root_input_const[inp] = False
        for i in range(max_count):
            if cond_in:
                output_list = []
                output_const_list = []
                for n in sub_nodes:
                    sub_node_obj = self.body.nodes[n]['object']
                    if sub_node_obj is None:
                        ERROR(f'[Parser]: Get Node Obj Failed in Loop Infer of Node: {n}.')
                    try:
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
                                    out_tensor = copy.deepcopy(last_output_list[inp_idx - 1])
                            if n in cond_out_root_input_const:
                                cond_out_root_input_const[n] = in_edges[inp_idx][-1]['tensor'].is_const
                            sub_node_obj.infer_shape(out_tensor)
                        elif sub_node_obj.type == 'DummyInput':
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
                    except Exception as e:
                        WARN_EXCEPTION(
                            f'[Parser]: Infer of {sub_node_obj.type} Node({n}) in {self.name} meets issues: {str(e)}!')
                loop_cnt += 1
                # Loop body outputs: 1 + N + K
                for out in self.body._attr['output_names']:
                    is_const = self.body.nodes[out]['object'].is_all_inputs_const()
                    for _, dst, out_attr in self.body.sorted_out_edges(out, data=True):
                        if self.body.nodes[dst]['object'].type == 'Out':
                            out_port = out_attr['src_out_port']
                            out_tensor = self.body.nodes[out]['object'].get_output_tensors()[out_port]
                            output_list.append(out_tensor)
                            output_const_list.append(is_const)
                last_output_list = output_list.copy()
                last_output_const_list = output_const_list.copy()
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
            loop_output_list = last_output_list[1: 1 + N]
            loop_output_list.extend([np.vstack(x) for x in k_carried_away])
            loop_output_const_list = last_output_const_list[1: 1 + N]
            for i in range(K):
                loop_output_const_list.append(last_output_const_list[1 + N + i])
        else:
            loop_output_list = inputs[2:]
            loop_output_const_list = [attr['tensor'].is_const for _, _, attr in in_edges[2:]]
        while len(loop_output_list) < len(out_edges):
            loop_output_list.append(np.array([]))
            loop_output_const_list.append(False)
        self.set_out_tensor(loop_output_list, all(loop_output_const_list))
