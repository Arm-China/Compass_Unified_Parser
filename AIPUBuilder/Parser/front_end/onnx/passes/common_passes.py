# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import copy
import itertools
from collections import OrderedDict
from ....plugin_loader import PARSER_OP_DICT

from ....common.defs import Tensor, Framework, FLOAT_EQUAL
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL
from ....common.utils import extend_lists, get_converted_dtype
from ....ops.op import Op, OpHasWeights, OpHasBiases, OpHasOneOutPort, ConstLikeOp, OnnxReduceOp, \
    OpHasVariableOutPorts, OpHasMultipleOutPorts
from ....ops.onnx_ops.array_ops import CastOp
from ....graph.graph import SubGraph
from ....ops.release_ops import ArmCastOp, ArmTransposeOp
from ....graph.node_wrap import NodeWrap
from ....graph.graph_algo import has_path, get_valid_node_name, all_simple_paths, clear_redundant_nodes
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher


def fuse_const(graph):
    matches = single_node_matcher(graph, '')
    for m in matches:
        node_name = m['target']
        if node_name in graph.nodes:
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is None:
                ERROR(
                    '[Parser]: Meets invalid Node (%s) in fuse_const!' % node_name)
                continue
            if not isinstance(node_obj, ConstLikeOp) \
                    and isinstance(node_obj, OpHasOneOutPort) \
                    and node_obj.is_all_inputs_const():
                out_edge = graph.sorted_out_edges(node_name, data=True)
                if len(out_edge) >= 1 and out_edge[0][2]['tensor'] is not None and out_edge[0][2]['tensor'].value is not None:
                    const_value = out_edge[0][2]['tensor'].value
                    const_attr = {'name': node_name,
                                  'value': const_value,
                                  'data_format': node_obj.data_format,
                                  'opset_version': 9}
                    NodeWrap(graph, node_name).replace_obj(
                        'Constant', const_attr)
                    in_edges = graph.sorted_in_edges(node_name)
                    graph.remove_edges_from(in_edges)
    clear_redundant_nodes(graph)


def convert_64bit_const(graph):
    matches = single_node_matcher(graph, ['Constant', 'ArmConstant'])
    for m in matches:
        node_obj = NodeWrap(graph, m['target'])['object']
        if node_obj is None:
            ERROR(
                '[Parser]: Meets invalid Node (%s) in convert_64bit_const!' % m['target'])
            continue
        value = getattr(node_obj, 'value') if node_obj.type == 'Constant' else getattr(node_obj, 'weights')
        value_dtype = str(value.dtype)
        if value_dtype in ['int64', 'uint64', 'float64']:
            if value_dtype == 'int64':
                value = value.astype(np.int32)
            elif value_dtype == 'uint64':
                value = value.astype(np.uint32)
            elif value_dtype == 'float64':
                value = value.astype(np.float32)
            setattr(node_obj, 'value' if node_obj.type == 'Constant' else 'weights', value)


def convert_to_const(graph, op_type_name_list):
    if len(graph) and op_type_name_list:
        for node_name in graph.nodes:
            node = NodeWrap(graph, node_name)
            node_obj = node['object']
            if node_obj is None:
                ERROR('[Parser]: Meets invalid Node(%s) in convert_to_const!' % node_name)
                continue
            if isinstance(node_obj, OpHasOneOutPort) and node_obj.type in op_type_name_list:
                out_tensors = node_obj.get_output_tensors()
                if len(out_tensors) >= 1 and out_tensors[0] is not None and node_obj.is_all_outputs_const():
                    new_attr = node_obj.copied_attr()
                    new_attr.update({'value': out_tensors[0].copy()})
                    node.replace_obj('Constant', new_attr)
                    const_in_edges = graph.sorted_in_edges(node_name)
                    graph.remove_edges_from(const_in_edges)
        clear_redundant_nodes(graph)
    else:
        WARN('[Parser]: Invalid params for convert_to_const!')


def convert_dummyinput_to_input(graph):
    matches = single_node_matcher(graph, 'DummyInput')
    for m in matches:
        node_name = m['target']
        if node_name in graph.nodes:
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is None:
                ERROR(
                    '[Parser]: Meets invalid Node (%s) in convert_dummyinput_to_input!' % node_name)
                continue
            out_edges = graph.sorted_out_edges(node_name, data=True)
            assert len(out_edges) == 1, 'DummyInput should only have 1 output.'
            out_tensor = out_edges[0][-1]['tensor']
            new_attr = node_obj.copied_attr()
            NodeWrap(graph, node_name).replace_obj(
                'ArmInput', new_attr)
            graph._attr['input_tensors'].update({out_tensor.name: out_tensor})


def convert_multi_outputs_to_const(graph, op_type_name_list):
    matches = single_node_matcher(graph, '')
    for m in matches:
        node_name = m['target']
        if node_name in graph.nodes:
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is None:
                ERROR(
                    '[Parser]: Meets invalid Node (%s) in convert_multi_outputs_to_const!' % node_name)
                continue
            if isinstance(node_obj, (OpHasVariableOutPorts, OpHasMultipleOutPorts)) and \
                    node_obj.type in op_type_name_list and \
                    node_obj.is_all_outputs_const():
                out_edges = graph.sorted_out_edges(node_name, data=True)
                in_edges = graph.sorted_in_edges(node_name)
                graph.remove_edges_from(in_edges)
                graph.remove_edges_from(out_edges)
                if node_name in graph._attr['output_names']:
                    idx = graph._attr['output_names'].index(node_name)
                    graph._attr['output_names'].pop(idx)
                else:
                    idx = None
                for i, o_edge in enumerate(out_edges):
                    if o_edge[2]['tensor'] is not None and o_edge[2]['tensor'].value is not None:
                        const_value = o_edge[2]['tensor'].value
                        const_node_name = get_valid_node_name(graph, node_name)
                        graph.add_node(const_node_name)
                        const_attr = {'name': const_node_name,
                                      'value': const_value,
                                      'data_format': node_obj.data_format,
                                      'opset_version': 9}
                        NodeWrap(graph, const_node_name).replace_obj(
                            'Constant', const_attr)
                        _, dst, out_attr = o_edge
                        out_attr['src_out_port'] = 0
                        graph.add_edge(const_node_name, dst, **out_attr)
                        if idx is not None:
                            graph._attr['output_names'].insert(idx + i, const_node_name)
    clear_redundant_nodes(graph)


def remove_node_safely(graph, n):
    assert graph.has_node(
        n), 'The node %s does not exist, cannot remove_node_safely.' % (n)
    in_edges = graph.sorted_in_edges(n, data=True)
    out_edges = graph.sorted_out_edges(n, data=True)
    if len(in_edges) >= 1:
        src_name, _, in_attr = in_edges[0]
        new_attr = copy.deepcopy(in_attr)
        for _, dst_name, out_attr in out_edges:
            if NodeWrap(graph, src_name)['object'].type == 'Constant':
                const = NodeWrap(graph, src_name)['object'].value
                if new_attr['tensor'] is None:
                    new_attr['tensor'] = Tensor(value=const)
                else:
                    new_attr['tensor'].value = const
            if new_attr['tensor'] is None:
                new_attr['tensor'] = Tensor(min_max=out_attr['tensor'].min_max)
            else:
                new_attr['tensor'].min_max = out_attr['tensor'].min_max if out_attr['tensor'].min_max else in_attr['tensor'].min_max
            new_attr.update({'dst_in_port': out_attr['dst_in_port']})
            graph.add_edge(src_name, dst_name, **new_attr)
        graph.remove_node(n)
    elif len(in_edges) == 0 and len(out_edges) == 0:
        graph.remove_node(n)


def remove_useless_op(graph, op_type_list):
    '''Remove some redundant unwanted OP.'''
    for op_type in op_type_list:
        removing_nodes = []
        matched = single_node_matcher(graph, op_type)
        for m in matched:
            node_name = m['target']
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is None:
                continue
            if op_type == 'Cast':
                in_edges = graph.sorted_in_edges(node_name, data=True)
                if len(in_edges) > 0 \
                        and in_edges[0][2].get('tensor', None) is not None \
                        and not node_obj.quantize \
                        and in_edges[0][2]['tensor'].get_dtype() is not None \
                        and in_edges[0][2]['tensor'].get_dtype() == node_obj.to:
                    removing_nodes.append(node_name)
                else:
                    continue
            elif op_type == 'ArmCast':
                in_edges = graph.sorted_in_edges(node_name, data=True)
                out_edges = graph.sorted_out_edges(node_name, data=True)
                if not node_obj.quantize \
                        and len(in_edges) > 0 \
                        and in_edges[0][2]['tensor'].get_dtype() is not None \
                        and str(in_edges[0][2]['tensor'].get_dtype()) == node_obj.to_dtype:
                    removing_nodes.append(node_name)
                elif len(in_edges) > 0 \
                        and len(out_edges) > 0 \
                        and in_edges[0][2]['tensor'] is not None \
                        and in_edges[0][2]['tensor'].scale_zp \
                        and out_edges[0][2]['tensor'] is not None \
                        and out_edges[0][2]['tensor'].scale_zp \
                        and str(in_edges[0][2]['tensor'].dtype) == node_obj.to_dtype\
                        and FLOAT_EQUAL(out_edges[0][2]['tensor'].scale_zp[0], in_edges[0][2]['tensor'].scale_zp[0])\
                        and FLOAT_EQUAL(out_edges[0][2]['tensor'].scale_zp[1], in_edges[0][2]['tensor'].scale_zp[1]):
                    removing_nodes.append(node_name)
                else:
                    continue
            elif op_type == 'Blank':
                removing_nodes.append(node_name)
            elif op_type == 'ChannelShuffle':
                input_shape = node_obj.get_input_shapes()[0]
                if node_obj.splits == 1:
                    if node_obj.group == 1:
                        removing_nodes.append(node_name)
                    elif input_shape and None not in input_shape:
                        ic = input_shape[-1] if node_obj.data_format == 'NHWC' else input_shape[1]
                        if node_obj.group == ic:
                            removing_nodes.append(node_name)
            elif op_type in ('Concat', 'Sum'):
                in_edges = graph.sorted_in_edges(node_name)
                if len(in_edges) <= 1:
                    removing_nodes.append(node_name)
            elif op_type in ('Identity', 'DummyInput'):
                if op_type == 'DummyInput' and isinstance(graph, SubGraph):
                    out_edges = graph.sorted_out_edges(node_name, data=True)
                    graph.remove_edges_from(out_edges)
                removing_nodes.append(node_name)
            elif op_type == 'Dropout':
                in_edges = graph.sorted_in_edges(node_name, data=True)
                if len(in_edges) != 3:
                    continue
                if not in_edges[1][2]['tensor'].is_const or not in_edges[2][2]['tensor'].is_const:
                    continue
                if not node_obj.training_mode or FLOAT_EQUAL(node_obj.ratio, 0.0):
                    out_ports = node_obj.get_out_ports()
                    if len(out_ports) == 2:
                        in_shapes = node_obj.get_input_shapes()
                        if len(in_shapes) < 1 or in_shapes[0] is None:
                            continue
                        mask_value = np.tile(True, in_shapes[0])
                        mask_name = get_valid_node_name(graph, node_name + '_out_mask')
                        out_edges = graph.sorted_out_edges(node_name, keys=True, data=True)
                        for _, dst, k, out_attr in out_edges:
                            if out_attr['src_out_port'] == 1:
                                graph.remove_edge(node_name, dst, key=k)
                                graph.add_edge(mask_name,
                                               dst,
                                               **{'src_out_port': 0,
                                                  'dst_in_port': out_attr['dst_in_port'],
                                                  'tensor': Tensor(value=mask_value, name=mask_name, is_const=True)
                                                  })
                        NodeWrap(graph, mask_name).replace_obj('Constant', {
                            'name': mask_name, 'value': mask_value, 'opset_version': 9})
                    removing_nodes.append(node_name)
                    graph.remove_edges_from(in_edges[1:])
                else:
                    continue
            elif op_type == 'Expand':
                in_shapes = node_obj.get_input_shapes()
                if len(in_shapes) < 1 or in_shapes[0] is None:
                    continue
                in_edges = graph.sorted_in_edges(node_name, data=True)
                if len(in_edges) == 2 \
                        and in_edges[1][2].get('tensor', None) is not None \
                        and in_edges[1][2]['tensor'].is_const:
                    if FLOAT_EQUAL(in_edges[1][2]['tensor'].value, 1) \
                            and in_edges[1][2]['tensor'].value.size <= len(in_shapes[0]):
                        graph.remove_edges_from(in_edges[1:])
                        removing_nodes.append(node_name)
                    else:
                        out_shapes = node_obj.get_output_shapes()
                        if len(out_shapes) >= 1 and out_shapes[0] is not None \
                                and in_shapes[0] == out_shapes[0]:
                            graph.remove_edges_from(in_edges[1:])
                            removing_nodes.append(node_name)
            elif op_type in ('AveragePool', 'MaxPool'):
                in_shapes = node_obj.get_input_shapes()
                out_shapes = node_obj.get_output_shapes()
                if in_shapes[0] is None or out_shapes[0] is None:
                    ERROR(
                        '[Parser]: Meets invalid input/output shape for node (%s) in remove_useless_op.' % node_name)
                    continue
                if (node_obj.data_format == 'NCHW' and in_shapes[0][2:] == [1, 1] and out_shapes[0][2:] == [1, 1]) \
                        or (node_obj.data_format == 'NHWC' and in_shapes[0][1:3] == [1, 1] and out_shapes[0][1:3] == [1, 1]):
                    removing_nodes.append(node_name)
            elif op_type == 'Pad':
                if np.all(np.array(node_obj.pads, np.int64) == 0):
                    removing_nodes.append(node_name)
            elif op_type == 'Pow':
                in_shapes = node_obj.get_input_shapes()
                out_shapes = node_obj.get_output_shapes()
                if len(in_shapes) < 1 or len(out_shapes) < 1 or in_shapes[0] != out_shapes[0]:
                    continue
                in_edges = graph.sorted_in_edges(node_name, data=True)
                if len(in_edges) < 2 or in_edges[1][2]['tensor'] is None \
                        or not in_edges[1][2]['tensor'].is_const \
                        or in_edges[1][2]['tensor'].value is None \
                        or not FLOAT_EQUAL(in_edges[1][2]['tensor'].value, 1):
                    continue
                removing_nodes.append(node_name)
            elif op_type == 'ArmTile':
                reps = node_obj.reps
                if all(rep == 1 for rep in reps):
                    removing_nodes.append(node_name)
            elif op_type in OnnxReduceOp.get_concrete_subclass_names():
                in_edges = graph.sorted_in_edges(node_name)
                if len(in_edges) < 1:
                    ERROR('[Parser]: Meets invalid Reduce(%s) to remove in remove_useless_op!' % node_name)
                    continue
                if node_obj.axes is None and node_obj.noop_with_empty_axes:
                    removing_nodes.append(node_name)
                    if node_name in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(node_name)
                        graph._attr['output_names'][index] = in_edges[0][0]
            elif op_type == 'ArmReduce':
                input_shapes = node_obj.get_input_shapes()
                if len(input_shapes) > 0 \
                        and input_shapes[0] is not None \
                        and None not in input_shapes[0] \
                        and np.product(np.array(input_shapes[0])[np.array(node_obj.axes, np.int32)]) == 1:
                    in_edges = graph.sorted_in_edges(node_name)
                    removing_nodes.append(node_name)
                    if node_name in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(node_name)
                        if in_edges[0][0] not in graph._attr['output_ names']:
                            graph._attr['output_names'][index] = in_edges[0][0]
                        else:
                            graph._attr['output_names'].pop(index)
            elif op_type in ('Reshape', 'ArmReshape', 'LiteRESHAPE'):
                reshape_in_edges = graph.sorted_in_edges(node_name, data=True)
                src_name = reshape_in_edges[0][0]
                src_node_obj = NodeWrap(graph, src_name)['object']
                if src_node_obj is None:
                    continue
                if len(reshape_in_edges) > 1 \
                        and (reshape_in_edges[1][2].get('tensor', None) is None
                             or not reshape_in_edges[1][2]['tensor'].is_const):
                    continue
                in_shapes = node_obj.get_input_shapes()
                out_shapes = node_obj.get_output_shapes()
                if src_node_obj.type == 'Constant' \
                        and src_node_obj.value is not None \
                        and len(graph.sorted_out_edges(src_name)) == 1:
                    new_shape = node_obj.shape if op_type == 'Reshape' else node_obj.dim
                    src_node_obj.value = np.reshape(
                        src_node_obj.value, newshape=new_shape)
                    removing_nodes.append(node_name)
                elif len(in_shapes) >= 1 \
                        and in_shapes[0] is not None \
                        and len(out_shapes) >= 1 \
                        and in_shapes[0] == out_shapes[0]:
                    removing_nodes.append(node_name)
            elif op_type == 'Resize':
                in_shapes = node_obj.get_input_shapes()
                in_tensors = node_obj.get_input_tensors()
                if node_obj.cur_version == 10:
                    if len(in_tensors) == 2 and node_obj.cur_version == 10 and in_tensors[1].size > 0 and np.all(in_tensors[1] == 1):
                        removing_nodes.append(node_name)
                else:
                    if len(in_tensors) >= 3 and in_tensors[2].size > 0 and np.all(in_tensors[2] == 1):
                        removing_nodes.append(node_name)
                    elif len(in_tensors) == 4 and in_tensors[3].size > 0 and in_shapes[0] == in_tensors[3].tolist():
                        removing_nodes.append(node_name)
            elif op_type == 'Roll':
                axis_value = node_obj.axes[0] if len(
                    node_obj.axes) == 1 else node_obj.axes
                roll_shift = node_obj.shift
                roll_shape = node_obj.get_input_shapes()[0]
                if len(roll_shift) == 1:
                    roll_shift = roll_shift[0]
                    from ....ops.common_ops import RollOp
                    roll_shift, start1, end1, steps1, axes1, start2, end2, steps2, axes2 \
                        = RollOp.cal_roll_parm(axis_value, roll_shift, roll_shape)
                    if roll_shift == 0 \
                            or np.any(np.abs(start1) >= np.abs(end1)) \
                            or np.any(np.abs(start2) >= np.abs(end2)):
                        removing_nodes.append(node_name)
                else:
                    if np.all(np.array(roll_shift) == 0):
                        removing_nodes.append(node_name)
                    else:
                        WARN(
                            '[Parser]: Meets unsupported Roll(%s) to remove in remove_useless_op!' % node_name)
                        continue
            elif op_type == 'Slice':
                in_edges = graph.sorted_in_edges(node_name, data=True)
                if len(in_edges) > 1 and any(not in_attr['tensor'].is_const for _, _, in_attr in in_edges[1:]):
                    continue
                input_shapes = node_obj.get_input_shapes()
                output_shapes = node_obj.get_output_shapes()
                if len(input_shapes) >= 1 \
                        and len(output_shapes) >= 1 \
                        and (input_shapes[0] is not None and None not in input_shapes[0]) \
                        and (output_shapes[0] is not None and None not in output_shapes[0]) \
                        and input_shapes[0] == output_shapes[0] \
                        and all(d == 0 for d in node_obj.starts) \
                        and all(input_shapes[0][axis] == node_obj.ends[idx] for idx, axis in enumerate(node_obj.axes)):
                    removing_nodes.append(node_name)
            elif op_type == 'Split':
                input_shapes = node_obj.get_input_shapes()
                if len(input_shapes) >= 1 and len(node_obj.split) == 1 and \
                        input_shapes[0] is not None and None not in input_shapes[0] and \
                        input_shapes[0][node_obj.axis] == node_obj.split[0]:
                    removing_nodes.append(node_name)
            elif op_type in ('Transpose', 'ArmTranspose'):
                trans_in_edges = graph.sorted_in_edges(node_name, data=True)
                if len(trans_in_edges) != 1:
                    ERROR(
                        '[Parser]: Meets invalid Transpose(%s) to remove in remove_useless_op!' % node_name)
                    continue
                trans_in_tensor_shape = trans_in_edges[0][2]['tensor'].shape
                perm = node_obj.perm
                src_name = trans_in_edges[0][0]
                src_node_obj = NodeWrap(graph, src_name)['object']
                if src_node_obj is None:
                    continue
                if src_node_obj.type == 'Constant' \
                        and src_node_obj.value is not None \
                        and len(graph.sorted_out_edges(src_name)) == 1:
                    src_node_obj.value = np.transpose(
                        src_node_obj.value, axes=perm)
                    removing_nodes.append(node_name)
                elif trans_in_tensor_shape is not None:
                    no_change_perm = list(range(len(trans_in_tensor_shape)))
                    if (no_change_perm == perm or
                            int(np.prod(trans_in_tensor_shape)) == 1):
                        removing_nodes.append(node_name)
                    else:
                        diff_values = [v1 for i, (v1, v2) in enumerate(zip(perm, no_change_perm)) if v1 != v2]
                        if all([trans_in_tensor_shape[axis] == 1 for axis in diff_values]):
                            removing_nodes.append(node_name)
                elif perm and list(perm) == list(range(max(perm) + 1)):
                    removing_nodes.append(node_name)
            elif op_type == 'Upsample':
                if node_obj.scales and np.all(np.array(node_obj.scales, np.float32) == 1):
                    upsample_in_edges = graph.sorted_in_edges(node_name)
                    if len(upsample_in_edges) >= 1:
                        graph.remove_edges_from(upsample_in_edges[1:])
                    removing_nodes.append(node_name)
            else:
                removing_nodes.append(node_name)

        for rn in removing_nodes:
            if rn in graph._attr['output_names']:
                index = graph._attr['output_names'].index(rn)
                pred_node_name = graph.sorted_in_edges(rn)[0][0]
                if pred_node_name not in graph._attr['output_names']:
                    graph._attr['output_names'][index] = pred_node_name
                else:
                    graph._attr['output_names'].remove(rn)
            remove_node_safely(graph, rn)


def remove_redundant_bn(graph, max_branches=6):
    for b in range(max_branches, 0, -1):
        nodes = [('pre_bn', {'op': 'ArmBatchNorm'})] \
            + [('post_bn_%s' % str(i + 1),
                {'op': 'ArmBatchNorm'}) for i in range(b)]
        edges = [('pre_bn', 'post_bn_%s' % str(i + 1),
                  {'dst_in_port': 0}) for i in range(b)]
        matches = matched_patterns(graph, nodes, edges)
        for m in matches:
            pre_bn = m['pre_bn']
            post_bns = [m['post_bn_%s' % str(i + 1)] for i in range(b)]
            obj_dict = {k: NodeWrap(graph, k)['object']
                        for k in [pre_bn] + post_bns}
            if all([obj is not None for obj in obj_dict.values()]):
                pre_bn_out_edges = graph.sorted_out_edges(pre_bn)
                if len(pre_bn_out_edges) != b:
                    continue
                if any([obj_dict[name].axis != obj_dict[pre_bn].axis for name in post_bns]):
                    continue
                pre_bn_in_edges = graph.sorted_in_edges(pre_bn, data=True)
                if len(pre_bn_in_edges) < 1:
                    ERROR(
                        '[Parser]: Meets invalid BatchNorm Op (%s) in remove_redundant_bn!' % pre_bn)
                    continue
                src, _, in_attr = pre_bn_in_edges[0]
                graph.remove_edge(src, pre_bn)
                for name in post_bns:
                    graph.remove_edge(pre_bn, name)
                    graph.add_edge(src, name, **in_attr)
                    new_weights = obj_dict[pre_bn].weights * \
                        obj_dict[name].weights
                    new_biases = obj_dict[pre_bn].biases * \
                        obj_dict[name].weights + obj_dict[name].biases
                    obj_dict[name].weights = new_weights
                    obj_dict[name].biases = new_biases
                remove_node_safely(graph, pre_bn)
            else:
                ERROR('[Parser]: Meets invalid BatchNorm Op in remove_redundant_bn!')


def remove_redundant_reshape(graph, type='Reshape'):
    matches = matched_patterns(graph,
                               nodes=[
                                   ('reshape_1', {'op': type}), ('reshape_2', {'op': type})],
                               edges=[('reshape_1', 'reshape_2')]
                               )
    for m in matches:
        reshape_1, reshape_2 = m['reshape_1'], m['reshape_2']
        reshape_1_obj = NodeWrap(graph, reshape_1)['object']
        reshape_2_obj = NodeWrap(graph, reshape_2)['object']
        reshape_1_in_shapes = reshape_1_obj.get_input_shapes()
        reshape_2_out_shapes = reshape_2_obj.get_output_shapes()
        reshape_1_edges = graph.sorted_in_edges(reshape_1, data=True)
        reshape_2_edges = graph.sorted_in_edges(reshape_2, data=True)
        reshape_1_out_edges = graph.sorted_out_edges(reshape_1, data=True)
        reshape_2_out_edges = graph.sorted_out_edges(reshape_2, data=True)
        if len(reshape_1_in_shapes) >= 1 \
                and len(reshape_2_out_shapes) >= 1 \
                and reshape_1 not in graph._attr['output_names']:
            if len(reshape_1_out_edges) == 1:
                remove_node_safely(graph, reshape_1)
                if type == 'Reshape' and len(reshape_2_edges) == 2:
                    graph.remove_edges_from(reshape_2_edges[1:])
                    reshape_2_obj.cur_version = 1
                    reshape_2_obj.shape = reshape_2_out_shapes[0]
                if reshape_1_in_shapes[0] == reshape_2_out_shapes[0]:
                    src, _, in_attr = reshape_1_edges[0]
                    graph.remove_edges_from(reshape_1_edges)
                    graph.remove_edges_from(reshape_2_out_edges)
                    for _, dst, out_attr in reshape_2_out_edges:
                        new_out_attr = copy.deepcopy(out_attr)
                        new_out_attr['src_out_port'] = in_attr['src_out_port']
                        graph.add_edge(src, dst, **new_out_attr)
                    if reshape_2 in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(reshape_2)
                        graph._attr['output_names'][index] = src
                clear_redundant_nodes(graph)
            elif len(reshape_1_out_edges) > 1:
                reshape_1_out_node_objs = []
                reshape_2_nodes = []
                for src, dst, out_attr in reshape_1_out_edges:
                    reshape_2_nodes.append(dst)
                    reshape_1_out_node_objs.append(NodeWrap(graph, dst)['object'])
                if all([rs2 in graph._attr['output_names'] for rs2 in reshape_2_nodes]):
                    continue
                if all([n_obj.type == type for n_obj in reshape_1_out_node_objs]):
                    all_reshape_2_out_shapes = []
                    for n_obj in reshape_1_out_node_objs:
                        all_reshape_2_out_shapes.append(n_obj.get_output_shapes())
                    if all([reshape_1_in_shapes[0] == out_shapes[0] for out_shapes in all_reshape_2_out_shapes]):
                        src, _, in_attr = reshape_1_edges[0]
                        graph.remove_edges_from(reshape_1_edges)
                        for _, dst1, out_attr1 in reshape_1_out_edges:
                            reshape_2_out_edges = graph.sorted_out_edges(dst1, data=True)
                            graph.remove_edges_from(reshape_2_out_edges)
                            for _, dst2, out_attr2 in reshape_2_out_edges:
                                new_out_attr = copy.deepcopy(out_attr2)
                                new_out_attr['src_out_port'] = in_attr['src_out_port']
                                graph.add_edge(src, dst2, **new_out_attr)
                            if dst1 in graph._attr['output_names']:
                                index = graph._attr['output_names'].index(dst1)
                                graph._attr['output_names'][index] = src
                        clear_redundant_nodes(graph)


def remove_redundant_transpose(graph):
    matched = False
    transpose_types = ['Transpose', 'ArmTranspose']
    matches = [matched_patterns(graph,
                                nodes=[('transpose1', {'op': tt, 'unique': False}),
                                       ('transpose2', {'op': tt})],
                                edges=[('transpose1', 'transpose2')]
                                ) for tt in transpose_types]
    matches = extend_lists(matches)
    for m in matches:
        trans1, trans2 = m['transpose1'], m['transpose2']
        trans1_obj, trans2_obj = NodeWrap(
            graph, trans1)['object'], NodeWrap(graph, trans2)['object']
        if trans1_obj is not None \
                and trans2_obj is not None \
                and len(trans1_obj.perm) == len(trans2_obj.perm):
            trans1_in_edges = graph.sorted_in_edges(trans1, data=True)
            if len(trans1_in_edges) < 1:
                continue
            matched = True
            src, _, in_attr = trans1_in_edges[0]
            trans2_in_edges = graph.sorted_in_edges(trans2, data=True)
            graph.remove_edges_from(trans2_in_edges)
            graph.add_edge(src, trans2, **in_attr)
            trans2_obj.perm = ArmTransposeOp.cal_merged_perm(
                trans1_obj.perm, trans2_obj.perm)
        else:
            ERROR('[Parser]: Meets invalid Transpose (%s or %s) in remove_redundant_transpose!'
                  % (trans1, trans2))
    if matched:
        clear_redundant_nodes(graph)


def remove_redundant_transpose2(graph):
    '''Remove redundant transpose and change dim of Reshape if the following patterns are matched:
    Transpose(NHWC to NCHW) + Reshape(NCHW to NCH'W') + Transpose(NCH'W' to NH'W'C) [=> Reshape(NHWC to NH'W'C)]
    or
    Transpose(NCHW to NHWC) + Reshape(NHWC to NH'W'C) + Transpose(NH'W'C to NCH'W') [=> Reshape(NCHW to NCH'W')]
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('transpose1', {'op': ['Transpose', 'ArmTranspose'], 'unique': False}),
                                      ('reshape', {'op': ['Reshape', 'ArmReshape']}),
                                      ('transpose2', {'op': ['Transpose', 'ArmTranspose']})],
                               edges=[('transpose1', 'reshape'),
                                      ('reshape', 'transpose2')]
                               )
    for m in matches:
        trans1, reshape, trans2 = m['transpose1'], m['reshape'], m['transpose2']
        trans1_obj, reshape_obj, trans2_obj = [NodeWrap(
            graph, node)['object'] for node in [trans1, reshape, trans2]]
        trans1_in_edges = graph.sorted_in_edges(trans1, data=True)
        if trans1_obj is None or reshape_obj is None or trans2_obj is None or len(trans1_in_edges) < 1:
            ERROR('[Parser]: Meets invalid Transpose/Reshape Node in remove_redundant_transpose2!')
            continue
        reshape_in_shapes = reshape_obj.get_input_shapes()
        if len(reshape_in_shapes) < 1 or reshape_in_shapes[0] is None or None in reshape_in_shapes[0]:
            continue
        reshape_out_shapes = reshape_obj.get_output_shapes()
        if len(reshape_out_shapes) < 1 or reshape_out_shapes[0] is None or None in reshape_out_shapes[0]:
            continue
        reshape_in_shape = reshape_in_shapes[0]
        reshape_dim = reshape_out_shapes[0]
        if len(reshape_in_shape) != len(reshape_dim):
            continue
        trans1_perm = trans1_obj.perm
        trans2_perm = trans2_obj.perm
        if len(trans1_perm) != len(trans2_perm) or len(trans1_perm) != len(reshape_dim):
            continue
        rank = len(trans1_perm)
        nhwc_to_nchw = [0, rank - 1] + list(range(1, rank - 1))
        nchw_to_nhwc = [0] + list(range(2, rank)) + [1]
        input_data_format = None
        if trans1_perm == nhwc_to_nchw and trans2_perm == nchw_to_nhwc:
            input_data_format = 'NHWC'
        elif trans1_perm == nchw_to_nhwc and trans2_perm == nhwc_to_nchw:
            input_data_format = 'NCHW'
        else:
            continue
        if input_data_format == 'NHWC':
            # If input data format is NHWC, then data format of reshape is NCHW.
            if reshape_in_shape[:2] != reshape_dim[:2]:
                continue
            new_dim = [reshape_dim[0]] + reshape_dim[2:] + [reshape_dim[1]]
        else:  # input_data_format == 'NCHW'
            # If input data format is NCHW, then data format of reshape is NHWC.
            if reshape_in_shape[0] != reshape_dim[0] or reshape_in_shape[-1] != reshape_dim[-1]:
                continue
            new_dim = [reshape_dim[0]] + [reshape_dim[-1]] + reshape_dim[1:-1]
        matched = True
        src, _, in_attr = trans1_in_edges[0]
        trans2_out_edges = graph.sorted_out_edges(trans2, data=True)
        graph.remove_edges_from(trans2_out_edges)
        new_reshape = get_valid_node_name(graph, trans2 + '_reshape')
        graph.add_edge(src, new_reshape, **in_attr)
        for _, dst, out_attr in trans2_out_edges:
            graph.add_edge(new_reshape, dst, **out_attr)
        new_reshape_attr = reshape_obj.copied_attr()
        new_reshape_attr.update({'name': new_reshape, 'dim': new_dim})
        NodeWrap(graph, new_reshape).replace_obj('ArmReshape', new_reshape_attr)
        if trans2 in graph._attr['output_names']:
            index = graph._attr['output_names'].index(trans2)
            graph._attr['output_names'][index] = new_reshape
    if matched:
        clear_redundant_nodes(graph)


def remove_redundant_transpose_unaware(graph):
    '''Remove Transpose nodes from the following patterns if their perm are inversed.
    sink_single_transpose won't work for this case because the first Transpose has more than 1 children.
        x -> Transpose -> Quantize(or others) -> Transpose
                     \--> (other children)
    Merge to:
        x -> Quantize(or others)
         \-> Transpose --> (other children)
    '''
    matched = False
    matches = matched_patterns(graph,
                               nodes=[('trans1', {'op': 'ArmTranspose', 'unique': False}),
                                      ('unaware', {'op': ['ArmQuantize', 'ArmDeQuantize']}),  # or other ops
                                      ('trans2', {'op': 'ArmTranspose'})],
                               edges=[('trans1', 'unaware'),
                                      ('unaware', 'trans2')]
                               )
    for m in matches:
        trans1, trans2, unaware = m['trans1'], m['trans2'], m['unaware']
        trans1_obj, trans2_obj, unaware_obj = [NodeWrap(
            graph, name)['object'] for name in [trans1, trans2, unaware]]
        trans1_in_edges = graph.sorted_in_edges(trans1, data=True)
        if trans1_obj is None or trans2_obj is None or unaware_obj is None \
                or len(trans1_obj.perm) != len(trans2_obj.perm) \
                or len(trans1_in_edges) < 1:
            ERROR('[Parser]: Meets invalid nodes in remove_redundant_transpose3!')
            continue
        merged_perm = ArmTransposeOp.cal_merged_perm(trans1_obj.perm, trans2_obj.perm)
        if merged_perm != list(range(len(merged_perm))):
            continue
        unaware_out_edges = graph.sorted_out_edges(unaware)
        if len(unaware_out_edges) != 1:
            continue
        matched = True
        unaware_in_edges = graph.sorted_in_edges(unaware)
        trans2_out_edges = graph.sorted_out_edges(trans2, data=True)
        graph.remove_edges_from(unaware_in_edges + unaware_out_edges + trans2_out_edges)
        src, _, in_attr = trans1_in_edges[0]
        graph.add_edge(src, unaware, **in_attr)
        for _, dst, out_attr in trans2_out_edges:
            graph.add_edge(unaware, dst, **out_attr)
        if trans2 in graph._attr['output_names']:
            index = graph._attr['output_names'].index(trans2)
            graph._attr['output_names'][index] = unaware
    if matched:
        clear_redundant_nodes(graph)


def remove_redundant_cast(graph):
    cast_types = ['Cast', 'ArmCast']
    cast_combinations = itertools.product(cast_types, cast_types)
    for cast1_type, cast2_type in cast_combinations:
        matches = two_nodes_matcher(graph, cast1_type, cast2_type)
        for m in matches:
            cast1, cast2 = m['begin'], m['end']
            cast1_obj = NodeWrap(graph, cast1)['object']
            cast2_obj = NodeWrap(graph, cast2)['object']
            if cast1_obj is not None and cast2_obj is not None:
                cast1_dst_type = cast1_obj.to if cast1_type == 'Cast' else cast1_obj.to_dtype
                cast2_dst_type = cast2_obj.to if cast2_type == 'Cast' else cast2_obj.to_dtype
                if 'int' in cast1_dst_type and 'float' in cast2_dst_type:
                    # int+float is not same as float. For example, if input is 1.5, int+float
                    # will get 1.0, while float will get 1.5.
                    continue
                cast1_out_edges = graph.sorted_out_edges(cast1)
                if len(cast1_out_edges) == 1:
                    remove_node_safely(graph, cast1)
            else:
                ERROR('[Parser]: Meets invalid Cast Node (%s or %s) in remove_redundant_cast!' % (
                    cast1, cast2))


def insert_cast(graph, src, dst, dst_type, in_attr=None, key=None, type='Cast'):
    ret = None
    allowed_dtypes = ArmCastOp.attributes()['to_dtype']['options']
    if type == 'Cast':
        allowed_dtypes += ['int64', 'float64']
    if graph is not None \
            and len(graph) >= 2 \
            and dst_type in allowed_dtypes \
            and graph.has_node(src) \
            and graph.has_node(dst) \
            and type in ('Cast', 'ArmCast'):
        if not in_attr:
            in_attr = {'src_out_port': 0, 'dst_in_port': 0}
        if has_path(graph, src, dst):
            graph.remove_edge(src, dst, key=key)
        cast = get_valid_node_name(
            graph, dst + '_pre_cast_' + str(in_attr.get('dst_in_port', 0)))
        cast_in_attr = copy.deepcopy(in_attr)
        cast_in_attr['dst_in_port'] = 0
        cast_out_tensor = Tensor()
        if in_attr.get('tensor', None) is not None and in_attr['tensor'].value is not None:
            cast_out_tensor.value = in_attr['tensor'].value.astype(np.dtype(dst_type))
        else:
            cast_out_tensor.dtype = dst_type
        cast_out_attr = {'src_out_port': 0, 'dst_in_port': in_attr.get(
            'dst_in_port', 0), 'tensor': cast_out_tensor}
        graph.add_edge(src, cast, **cast_in_attr)
        graph.add_edge(cast, dst, **cast_out_attr)
        if type == 'Cast':
            cast_attr = {'name': cast, 'opset_version': 1, 'to': dst_type}
        else:
            cast_attr = {'name': cast, 'to_dtype': dst_type}
        NodeWrap(graph, cast).replace_obj(type, cast_attr)
        ret = cast
    else:
        ERROR('[Parser]: Invalid params for insert_cast!')
    return ret


def insert_cast_after(graph, src, from_dtype, to_dtype, out_port=0, type='Cast'):
    ret = None
    allowed_dtypes = ArmCastOp.attributes()['to_dtype']['options']
    if type == 'Cast':
        allowed_dtypes += ['int64', 'float64']
    if graph.has_node(src) \
            and to_dtype in allowed_dtypes \
            and type in ('Cast', 'ArmCast') \
            and NodeWrap(graph, src)['object'] is not None \
            and out_port in NodeWrap(graph, src)['object'].get_out_ports():
        if out_port == 0:
            cast = src + '_post_cast'
        else:
            cast = src + '_post_cast_' + str(out_port)
        cast = get_valid_node_name(graph, cast)
        cast_in_tensor = Tensor()
        for _, dst, k, out_attr in graph.sorted_out_edges(src, keys=True, data=True):
            if out_attr['src_out_port'] == out_port:
                new_out_attr = copy.deepcopy(out_attr)
                new_out_attr['src_out_port'] = 0
                if new_out_attr['tensor'] is not None:
                    if new_out_attr['tensor'].value is not None:
                        new_out_attr['tensor'].value = new_out_attr['tensor'].value.astype(
                            np.dtype(to_dtype))
                        cast_in_tensor.value = new_out_attr['tensor'].value.astype(
                            np.dtype(from_dtype))
                    else:
                        new_out_attr['tensor'].dtype = to_dtype
                        cast_in_tensor.dtype = from_dtype
                graph.remove_edge(src, dst, k)
                graph.add_edge(cast, dst, **new_out_attr)
        graph.add_edge(src, cast, **{'src_out_port': out_port,
                                     'dst_in_port': 0, 'tensor': cast_in_tensor})
        if type == 'Cast':
            cast_attr = {'name': cast, 'opset_version': 1, 'to': to_dtype}
        else:
            cast_attr = {'name': cast, 'to_dtype': to_dtype}
        NodeWrap(graph, cast).replace_obj(type, cast_attr)
        ret = cast
    else:
        ERROR('[Parser]: Invalid params for insert_cast_after!')
    return ret


def insert_cast_sub_mul_for_quant(graph, src, dst, scale, zero_point, in_attr=None, key=None, data_format='NCHW'):
    ret = None
    if graph is None \
            or len(graph) < 2 \
            or not graph.has_node(src) \
            or not graph.has_node(dst) \
            or not has_path(graph, src, dst):
        ERROR('[Parser]: Meets invalid params for insert_cast_sub_mul_for_quant!')
        return ret

    scale = np.array(scale).astype(np.float32)
    zero_point = np.array(zero_point).astype(np.int32)

    cast_out_attr = copy.deepcopy(in_attr)
    cast_out_attr.update({'src_out_port': 0, 'dst_in_port': 0})
    sub_out_attr = copy.deepcopy(cast_out_attr)
    sub_cast_out_attr = copy.deepcopy(cast_out_attr)
    mul_out_attr = copy.deepcopy(cast_out_attr)
    if not in_attr:
        in_attr = {'src_out_port': 0, 'dst_in_port': 0}
    else:
        mul_out_attr.update({'dst_in_port': in_attr.get('dst_in_port', 0)})
        in_attr.update({'dst_in_port': 0})
        if in_attr['tensor'] is not None and in_attr['tensor'].value is not None:
            cast_out_attr['tensor'].value = np.array(in_attr['tensor'].value).astype(np.int32)
            sub_out_attr['tensor'].value = cast_out_attr['tensor'].value - zero_point
            sub_cast_out_attr['tensor'].value = np.array(sub_out_attr['tensor'].value).astype(np.float32)
            mul_out_attr['tensor'].value = sub_cast_out_attr['tensor'].value * scale

    graph.remove_edge(src, dst, key=key)

    cast = get_valid_node_name(graph, src + '_cast')
    sub = get_valid_node_name(graph, src + '_sub')
    sub_cast = get_valid_node_name(graph, sub + '_cast')
    mul = get_valid_node_name(graph, src + '_mul')

    graph.add_edge(src, cast, **in_attr)
    graph.add_edge(cast, sub, **cast_out_attr)
    graph.add_edge(sub, sub_cast, **sub_out_attr)
    graph.add_edge(sub_cast, mul, **sub_cast_out_attr)
    graph.add_edge(mul, dst, **mul_out_attr)

    insert_constant(graph, src + '_zp', zero_point, sub, in_port=1, data_format=data_format)
    insert_constant(graph, src + '_scale', scale, mul, in_port=1, data_format=data_format)
    NodeWrap(graph, cast).replace_obj('Cast', {'name': cast, 'opset_version': 1, 'to': 'int32'})
    NodeWrap(graph, sub).replace_obj('Sub', {'name': sub, 'opset_version': 7})
    NodeWrap(graph, sub_cast).replace_obj(
        'Cast', {'name': sub_cast, 'opset_version': 1, 'to': 'float32'})
    NodeWrap(graph, mul).replace_obj('Mul', {'name': mul, 'opset_version': 7})
    ret = cast
    return ret


def insert_mul_add_cast_after_for_dequant(graph, src, to_dtype, scale, zero_point, data_format='NCHW'):
    '''Insert nodes for the output of quantized nodes according to formula:
    y = saturate(round(x / y_scale) + y_zero_point)
    '''
    ret = None
    if not graph.has_node(src) \
            or NodeWrap(graph, src)['object'] is None \
            or to_dtype not in ArmCastOp.attributes()['to_dtype']['options']:
        ERROR('[Parser]: Meets invalid params for insert_mul_add_cast_after_for_dequant!')
        return ret
    out_edges = graph.sorted_out_edges(src, data=True)
    if len(out_edges) < 1:
        ERROR('[Parser]: Meets invalid outputs for Node(%s) in insert_mul_add_cast_after_for_dequant!' % src)
        return ret

    mul_scale = np.array(1 / scale).astype(np.float32)
    zero_point = np.array(zero_point).astype(np.float32)
    clip_min = float(np.iinfo(to_dtype).min)
    clip_max = float(np.iinfo(to_dtype).max)

    out_attr = out_edges[0][2]
    src_out_attr = copy.deepcopy(out_attr)
    src_out_attr.update({'dst_in_port': 0})
    mul_out_attr = copy.deepcopy(src_out_attr)
    mul_out_attr.update({'src_out_port': 0})
    round_out_attr = copy.deepcopy(mul_out_attr)
    add_out_attr = copy.deepcopy(mul_out_attr)
    clip_out_attr = copy.deepcopy(mul_out_attr)
    if out_attr['tensor'] is not None and out_attr['tensor'].value is not None:
        src_out_attr['tensor'].value = np.array(out_attr['tensor'].value).astype(np.float32)
        mul_out_attr['tensor'].value = src_out_attr['tensor'].value * mul_scale
        round_out_attr['tensor'].value = np.around(mul_out_attr['tensor'].value)
        add_out_attr['tensor'].value = round_out_attr['tensor'].value + zero_point
        clip_out_attr['tensor'].value = np.clip(add_out_attr['tensor'].value, clip_min, clip_max)

    out_mul = get_valid_node_name(graph, src + '_out_mul')
    out_round = get_valid_node_name(graph, src + '_out_round')
    out_add = get_valid_node_name(graph, src + '_out_add')
    out_clip = get_valid_node_name(graph, src + '_out_clip')
    out_cast = get_valid_node_name(graph, src + '_out_cast')

    graph.remove_edges_from(out_edges)
    graph.add_edge(src, out_mul, **src_out_attr)
    graph.add_edge(out_mul, out_round, **mul_out_attr)
    graph.add_edge(out_round, out_add, **round_out_attr)
    graph.add_edge(out_add, out_clip, **add_out_attr)
    graph.add_edge(out_clip, out_cast, **clip_out_attr)

    insert_constant(graph, out_mul + '_scale', mul_scale,
                    out_mul, in_port=1, data_format=data_format)
    insert_constant(graph, out_add + '_zp', zero_point,
                    out_add, in_port=1, data_format=data_format)

    for _, dst, out_attr in out_edges:
        graph.add_edge(out_cast, dst, **out_attr)

    NodeWrap(graph, out_mul).replace_obj('Mul', {'name': out_mul, 'opset_version': 7})
    NodeWrap(graph, out_round).replace_obj('Round', {'name': out_round, 'opset_version': 11})
    NodeWrap(graph, out_add).replace_obj('Add', {'name': out_add, 'opset_version': 7})
    NodeWrap(graph, out_clip).replace_obj('Clip', {'name': out_clip, 'opset_version': 6,
                                                   'min': clip_min, 'max': clip_max})
    NodeWrap(graph, out_cast).replace_obj('Cast', {'name': out_cast, 'opset_version': 1, 'to': str(to_dtype)})
    ret = out_cast
    return ret


def insert_constant(graph, name, value, dst, in_port=0, data_format='NCHW', const_ver=9, scale_zp=None, quantize=False):
    if graph.has_node(dst) and value is not None and isinstance(value, np.ndarray):
        const_name = get_valid_node_name(graph, name)
        graph.add_node(const_name)
        const_attr = {'name': const_name,
                      'value': value,
                      'data_format': data_format,
                      'opset_version': const_ver,
                      'quantize': quantize}
        NodeWrap(graph, const_name).replace_obj('Constant', const_attr)
        edge_attr = {'src_out_port': 0, 'dst_in_port': in_port,
                     'tensor': Tensor(value=value, is_const=True)}
        if isinstance(scale_zp, (tuple, list)) \
                and len(scale_zp) == 2:
            edge_attr['tensor'].scale_zp = tuple(scale_zp)
            edge_attr['tensor'].dtype = str(value.dtype)
        graph.add_edge(const_name, dst, **edge_attr)
    else:
        ERROR('[Parser]: Invalid params for insert_constant (%s)!' % name)


def insert_dequant_quant(graph, src, dst, in_attr, op_type, key=None, data_format='NHWC'):
    ret = None
    if graph.has_node(src) and graph.has_node(dst):
        if has_path(graph, src, dst):
            graph.remove_edge(src, dst, key=key)
        op_postfix = 'dequantize' if op_type == 'DequantizeLinear' else 'quantize'
        new_op = get_valid_node_name(graph, src + f'_post_{op_postfix}')
        graph.add_node(new_op)
        new_op_attr = {'name': new_op, 'opset_version': 13, 'quantize': True}

        if in_attr['tensor'].scale_zp is None:
            ERROR('[Parser]: No scale_zp info for insert_dequant_quant!')

        scale, zp = in_attr['tensor'].scale_zp

        scale = np.array(scale[0], dtype=np.float32)
        zp = np.array(zp[0], dtype=in_attr['tensor'].dtype)

        scale_const = get_valid_node_name(graph, new_op + '_scale')
        graph.add_node(scale_const)
        scale_attr = {'name': scale_const, 'value': scale,
                      'data_format': data_format, 'opset_version': 9}
        NodeWrap(graph, scale_const).replace_obj('Constant', scale_attr)
        scale_const_edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                                 'tensor': Tensor(value=scale, is_const=True)}
        graph.add_edge(scale_const, new_op, **scale_const_edge_attr)

        zp_const = get_valid_node_name(graph, new_op + '_zp')
        graph.add_node(zp_const)
        zp_attr = {'name': zp_const, 'value': zp,
                   'data_format': data_format, 'opset_version': 9}
        NodeWrap(graph, zp_const).replace_obj('Constant', zp_attr)
        zp_const_edge_attr = {'src_out_port': 0, 'dst_in_port': 2,
                              'tensor': Tensor(value=zp, is_const=True)}
        graph.add_edge(zp_const, new_op, **zp_const_edge_attr)

        NodeWrap(graph, new_op).replace_obj(op_type, new_op_attr)

        new_op_in_attr = copy.deepcopy(in_attr)
        new_op_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, new_op, **new_op_in_attr)

        new_op_out_attr = copy.deepcopy(in_attr)
        out_tensor = Tensor()
        if in_attr.get('tensor', None) is not None:
            out_tensor = copy.deepcopy(in_attr['tensor'])
            if in_attr['tensor'].value is not None:
                if op_type == 'DequantizeLinear':
                    out_tensor.value = (in_attr['tensor'].value - zp) * scale.astype(np.float32)
                    out_tensor.min_max = ()
                    out_tensor.scale_zp = ()
                else:
                    out_tensor.value = np.round(in_attr['tensor'].value / scale + zp).astype(zp.dtype)
                out_tensor.shape = out_tensor.value.shape
                out_tensor.dtype = str(out_tensor.value.dtype)
        new_op_out_attr.update({'src_out_port': 0, 'tensor': out_tensor})
        graph.add_edge(new_op, dst, **new_op_out_attr)
        ret = new_op
    else:
        ERROR('[Parser]: Invalid params for insert_dequant_quant!')
    return ret


def insert_gather(graph, src, dst, indices, axis=0, edge_attr=None, key=None, type='Gather'):
    ret = None
    if edge_attr is None:
        edge_attr = dict()
    assert type in (
        'Gather', 'ArmGather'), 'The type of node is invalid in insert_gather.'
    if graph.has_node(src) and graph.has_node(dst) and indices is not None:
        if has_path(graph, src, dst):
            graph.remove_edge(src, dst, key=key)
        if not edge_attr:
            edge_attr = {'src_out_port': 0, 'dst_in_port': 0}
        gather = get_valid_node_name(graph, dst + '_pre_gather')
        gather_in_attr = copy.deepcopy(edge_attr)
        gather_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, gather, **gather_in_attr)
        if isinstance(indices, str):
            # FIXME, indices's output port != 0
            assert graph.has_node(indices), f'{indices} not in graph.'
            graph.add_edge(indices, gather, **{'src_out_port': 0, 'dst_in_port': 1})
        else:
            insert_constant(graph, gather + '_indices', indices,
                            gather, in_port=1, data_format='NHWC')
        gather_out_attr = {'src_out_port': 0,
                           'dst_in_port': edge_attr['dst_in_port']}
        if edge_attr and \
                edge_attr.get('tensor', None) is not None and \
                getattr(edge_attr['tensor'], 'value', None) is not None:
            if isinstance(indices, str):
                idx_shape = NodeWrap(graph, indices)['object'].get_output_shapes()[0]
                out_tensor = np.take(edge_attr['tensor'].value, np.zeros(idx_shape, dtype=np.int64), axis=axis)
                gather_out_attr.update({'tensor': Tensor(shape=out_tensor.shape)})
            else:
                out_tensor = np.take(edge_attr['tensor'].value, indices, axis=axis)
                gather_out_attr.update({'tensor': Tensor(value=out_tensor)})
        graph.add_edge(gather, dst, **gather_out_attr)
        gather_attr = {'name': gather, 'axis': axis}
        if type == 'Gather':
            gather_attr.update({'opset_version': 11})
        NodeWrap(graph, gather).replace_obj(type, gather_attr)
        ret = gather
    else:
        ERROR('[Parser]: Invalid params for insert_gather!')
    return ret


def insert_repeat(graph, src, dst, in_attr, reps, axis=None, key=None, type='Repeat', data_format='NHWC'):
    ret = None
    if graph.has_node(src) \
            and graph.has_node(dst) \
            and reps is not None \
            and ((isinstance(reps, (list, tuple)) and len(reps) > 0) or (isinstance(reps, np.ndarray) and reps.size > 0)) \
            and type in ('Repeat', 'ArmRepeat'):
        if isinstance(reps, (list, tuple)):
            reps = np.array(reps, np.int32)
        if has_path(graph, src, dst):
            graph.remove_edge(src, dst, key=key)
        repeat = get_valid_node_name(graph, dst + '_pre_repeat')
        graph.add_node(repeat)

        repeat_in_attr = copy.deepcopy(in_attr)
        repeat_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, repeat, **repeat_in_attr)
        dst_in_attr = copy.deepcopy(in_attr)
        if dst_in_attr['tensor'] is not None:
            if dst_in_attr['tensor'].value is not None:
                tensor = Tensor(min_max=in_attr['tensor'].min_max,
                                value=np.repeat(dst_in_attr['tensor'].value, reps.tolist()))
            if dst_in_attr['tensor'].dtype is not None:
                tensor.dtype = dst_in_attr['tensor'].dtype
                tensor.scale_zp = dst_in_attr['tensor'].scale_zp
        else:
            tensor = Tensor(min_max=in_attr['tensor'].min_max)
        dst_in_attr.update({'src_out_port': 0, 'tensor': tensor})
        graph.add_edge(repeat, dst, **dst_in_attr)
        insert_constant(graph, repeat + '_reps', reps, repeat, in_port=1, data_format=data_format)
        repeat_attr = {'name': repeat, 'axis': axis}
        if type == 'Repeat':
            repeat_attr.update({'opset_version': 1})
        NodeWrap(graph, repeat).replace_obj(type, repeat_attr)
        ret = repeat
    else:
        ERROR('[Parser]: Invalid params for insert_repeat!')
    return ret


def insert_reshape(graph, src, dst, in_attr, dim, key=None, type='Reshape', data_format='NHWC', quantize=False):
    ret = None
    if graph.has_node(src) and graph.has_node(dst) and dim:
        if has_path(graph, src, dst):
            graph.remove_edge(src, dst, key=key)
        reshape = get_valid_node_name(graph, src + '_post_reshape')
        graph.add_node(reshape)
        reshape_attr = {'name': reshape, 'opset_version': 5, 'quantize': quantize}
        if type == 'Reshape':
            reshape_dim = np.array(dim, np.int32)
            const = get_valid_node_name(graph, reshape + '_shape')
            graph.add_node(const)
            const_attr = {'name': const, 'value': reshape_dim,
                          'data_format': data_format, 'opset_version': 9}
            NodeWrap(graph, const).replace_obj('Constant', const_attr)
            const_edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                               'tensor': Tensor(value=reshape_dim, is_const=True)}
            graph.add_edge(const, reshape, **const_edge_attr)
        else:
            reshape_attr.update({'dim': dim})
        NodeWrap(graph, reshape).replace_obj(type, reshape_attr)

        reshape_in_attr = copy.deepcopy(in_attr)
        reshape_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, reshape, **reshape_in_attr)

        reshape_out_attr = copy.deepcopy(in_attr)
        out_tensor = Tensor()
        if in_attr.get('tensor', None) is not None:
            out_tensor = copy.deepcopy(in_attr['tensor'])
            if in_attr['tensor'].value is not None:
                out_tensor.value = np.reshape(
                    in_attr['tensor'].value, newshape=dim)
                out_tensor.shape = out_tensor.value.shape
            else:
                out_tensor.shape = tuple(dim)
        reshape_out_attr.update({'src_out_port': 0, 'tensor': out_tensor})
        graph.add_edge(reshape, dst, **reshape_out_attr)
        ret = reshape
    else:
        ERROR('[Parser]: Invalid params for insert_reshape!')
    return ret


def insert_reshape_after(graph, src, new_dim, old_dim=None, out_port=0, type='Reshape', quantize=False):
    ret = None
    if old_dim is None:
        old_dim = list()
    if graph.has_node(src) and type in ('Reshape', 'ArmReshape'):
        if out_port == 0:
            reshape_name = src + '_post_reshape'
        else:
            reshape_name = src + '_post_reshape_' + str(out_port)
        reshape = get_valid_node_name(graph, reshape_name)
        graph.add_node(reshape)
        reshape_attr = {'name': reshape, 'opset_version': 5, 'quantize': quantize}
        if type == 'Reshape':
            reshape_dim = np.array(new_dim, np.int64)
            const = get_valid_node_name(graph, reshape + '_shape')
            graph.add_node(const)
            const_attr = {'name': const, 'value': reshape_dim,
                          'data_format': 'NHWC', 'opset_version': 9}
            NodeWrap(graph, const).replace_obj('Constant', const_attr)
            const_edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                               'tensor': Tensor(value=reshape_dim, is_const=True)}
            graph.add_edge(const, reshape, **const_edge_attr)
        else:
            reshape_attr.update({'dim': new_dim})
        NodeWrap(graph, reshape).replace_obj(type, reshape_attr)

        src_out_attr = {'src_out_port': out_port,
                        'dst_in_port': 0, 'tensor': Tensor()}
        out_edges = graph.sorted_out_edges(src, keys=True, data=True)
        for _, dst, key, out_attr in out_edges:
            if out_attr.get('src_out_port', 0) == out_port:
                graph.remove_edge(src, dst, key)
                new_out_attr = copy.deepcopy(out_attr)
                new_out_attr['src_out_port'] = 0
                graph.add_edge(reshape, dst, **new_out_attr)
                if new_out_attr.get('tensor', None) is not None:
                    new_out_tensor_shape = new_out_attr['tensor'].get_shape()
                    new_out_tensor_value = new_out_attr['tensor'].value
                    if old_dim:
                        if new_out_tensor_value is not None:
                            new_src_out_tensor = np.reshape(
                                new_out_attr['tensor'].value, newshape=old_dim)
                            src_out_attr.update(
                                {'tensor': Tensor(value=new_src_out_tensor)})
                        else:
                            src_out_attr.update({'tensor': Tensor(shape=tuple(old_dim))})
                    elif new_dim and new_out_tensor_shape is not None and new_dim != list(new_out_tensor_shape):
                        if new_out_tensor_value is not None:
                            new_src_out_tensor = np.reshape(
                                new_out_attr['tensor'].value, newshape=new_dim)
                            src_out_attr.update(
                                {'tensor': Tensor(value=new_src_out_tensor)})
                        else:
                            src_out_attr.update({'tensor': Tensor(shape=tuple(new_dim))})

                    if new_out_attr['tensor'].dtype is not None \
                            or len(new_out_attr['tensor'].scale_zp) > 0:
                        if src_out_attr.get('tensor', None) is None:
                            src_out_attr.update({'tensor': Tensor()})
                        src_out_attr['tensor'].dtype = new_out_attr['tensor'].dtype
                        src_out_attr['tensor'].scale_zp = new_out_attr['tensor'].scale_zp
        graph.add_edge(src, reshape, **src_out_attr)
        ret = reshape
    else:
        ERROR('[Parser]: Invalid params for insert_reshape_after!')
    return ret


def place_reshape(graph, reshape, dim, data_format='NHWC'):
    if not isinstance(dim, np.ndarray):
        dim = np.array(dim, np.int32)
    NodeWrap(graph, reshape).replace_obj('Reshape',
                                         {'name': reshape,
                                          'opset_version': 5}
                                         )
    insert_constant(graph,
                    reshape + '_shape',
                    dim,
                    reshape,
                    in_port=1,
                    data_format=data_format)


def insert_slice(graph, src, dst, in_attr, begin, size, key=None, type='Slice', data_format='NHWC', quantize=False):
    ret = None
    if graph.has_node(src) and graph.has_node(dst) and begin and size and type in ('Slice', 'ArmSlice'):
        if has_path(graph, src, dst):
            graph.remove_edge(src, dst, key=key)
        slice = get_valid_node_name(graph, src + '_post_slice')
        graph.add_node(slice)
        slice_attr = {'name': slice, 'quantize': quantize}

        starts = np.array(begin, np.int32)
        size = np.array(size, np.int32)
        ends = starts + size

        if type == 'Slice':
            slice_attr.update({'opset_version': 10})
            insert_constant(graph, slice + '_starts', starts,
                            slice, in_port=1, data_format=data_format)
            insert_constant(graph, slice + '_ends', ends, slice,
                            in_port=2, data_format=data_format)
        else:
            slice_attr.update({'starts': starts.tolist(),
                               'ends': ends.tolist(),
                               'steps': [1] * starts.size
                               })
        NodeWrap(graph, slice).replace_obj(type, slice_attr)

        slice_in_attr = copy.deepcopy(in_attr)
        slice_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, slice, **slice_in_attr)

        slice_out_attr = copy.deepcopy(in_attr)
        slice_out_attr.update({'src_out_port': 0})
        graph.add_edge(slice, dst, **slice_out_attr)
        ret = slice
    else:
        ERROR('[Parser]: Invalid params for insert_slice!')
    return ret


def insert_slice_after(graph, src, begin, size, slice_before_shape, out_port=0, type='Slice', data_format='NHWC'):
    ret = None
    if not (graph.has_node(src) and begin and size and type in ('Slice', 'ArmSlice')):
        ERROR('[Parser]: Invalid params for insert_slice_after!')
        return ret
    if out_port == 0:
        name = src + '_post_slice'
    else:
        name = src + '_post_slice_' + str(out_port)
    slice_name = get_valid_node_name(graph, name)
    graph.add_node(slice_name)
    slice_attr = {'name': slice_name}

    starts = np.array(begin, np.int32)
    size = np.array(size, np.int32)
    ends = starts + size

    if type == 'Slice':
        slice_attr.update({'opset_version': 10})
        insert_constant(graph, slice_name + '_starts', starts,
                        slice_name, in_port=1, data_format=data_format)
        insert_constant(graph, slice_name + '_ends', ends,
                        slice_name, in_port=2, data_format=data_format)
    else:
        slice_attr.update({'starts': starts.tolist(),
                           'ends': ends.tolist(),
                           'steps': [1] * starts.size
                           })
    NodeWrap(graph, slice_name).replace_obj(type, slice_attr)

    src_out_attr = {'src_out_port': out_port, 'dst_in_port': 0, 'tensor': Tensor(shape=slice_before_shape)}
    out_edges = graph.sorted_out_edges(src, data=True)
    for _, dst, out_attr in out_edges:
        if out_attr.get('src_out_port', 0) != out_port:
            continue
        graph.remove_edge(src, dst)
        new_out_attr = copy.deepcopy(out_attr)
        new_out_attr['src_out_port'] = 0
        graph.add_edge(slice_name, dst, **new_out_attr)
    graph.add_edge(src, slice_name, **src_out_attr)
    ret = slice_name
    return ret


def insert_tile(graph, src, dst, in_attr, reps, key=None, type='Tile', data_format='NHWC', quantize=False):
    ret = None
    if graph.has_node(src) \
            and graph.has_node(dst) \
            and reps is not None \
            and ((isinstance(reps, (list, tuple)) and len(reps) > 0) or (isinstance(reps, np.ndarray) and reps.size > 0)) \
            and type in ('Tile', 'ArmTile'):
        if isinstance(reps, (list, tuple)):
            reps = np.array(reps, np.int32)
        if has_path(graph, src, dst):
            graph.remove_edge(src, dst, key=key)
        tile = get_valid_node_name(graph, dst + '_pre_tile')
        graph.add_node(tile)

        tile_in_attr = copy.deepcopy(in_attr)
        tile_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, tile, **tile_in_attr)
        dst_in_attr = copy.deepcopy(in_attr)
        tensor = dst_in_attr['tensor']
        if tensor.value is not None:
            tensor.value = np.tile(dst_in_attr['tensor'].value, reps.tolist())
            tensor.shape = tensor.value.shape
        else:
            tensor_shape = tensor.get_shape()
            if tensor_shape is not None and None not in tensor_shape:
                tensor.shape = tuple([int(shape * rep) for shape, rep in zip(tensor_shape, reps)])
        dst_in_attr.update({'src_out_port': 0, 'tensor': tensor})
        graph.add_edge(tile, dst, **dst_in_attr)

        if type == 'Tile':
            const = get_valid_node_name(graph, tile + '_reps')
            graph.add_node(const)
            const_edge_attr = {'src_out_port': 0, 'dst_in_port': 1,
                               'tensor': Tensor(value=reps, is_const=True)}
            graph.add_edge(const, tile, **const_edge_attr)
            NodeWrap(graph, const).replace_obj('Constant', {'name': const,
                                                            'value': reps,
                                                            'data_format': data_format,
                                                            'opset_version': 9
                                                            }
                                               )
            tile_attr = {'name': tile, 'opset_version': 6, 'quantize': quantize}
            NodeWrap(graph, tile).replace_obj('Tile', tile_attr)
        else:
            tile_attr = {'name': tile, 'reps': reps.tolist(), 'quantize': quantize}
            NodeWrap(graph, tile).replace_obj('ArmTile', tile_attr)

        ret = tile
    else:
        ERROR('[Parser]: Invalid params for insert_tile!')
    return ret


def insert_transpose(graph, src, dst, in_attr, perm, key=None, type='Transpose', quantize=False):
    ret = None
    if graph.has_node(src) \
            and graph.has_node(dst) \
            and perm is not None \
            and isinstance(perm, (list, np.ndarray)) \
            and type in ('Transpose', 'ArmTranspose'):
        if isinstance(perm, np.ndarray):
            perm = perm.tolist()
        if has_path(graph, src, dst):
            graph.remove_edge(src, dst, key=key)
        transpose = get_valid_node_name(graph, src + '_post_transpose')
        graph.add_node(transpose)
        transpose_attr = {'name': transpose, 'perm': perm, 'quantize': quantize}
        if type == 'Transpose':
            transpose_attr.update({'opset_version': 1})
        NodeWrap(graph, transpose).replace_obj(type, transpose_attr)

        transpose_in_attr = copy.deepcopy(in_attr)
        transpose_in_attr.update({'dst_in_port': 0})
        graph.add_edge(src, transpose, **transpose_in_attr)

        transpose_out_attr = copy.deepcopy(in_attr)
        out_tensor = Tensor()
        if in_attr.get('tensor', None) is not None:
            out_tensor = copy.deepcopy(in_attr['tensor'])
            if in_attr['tensor'].value is not None:
                out_tensor.value = np.transpose(
                    in_attr['tensor'].value, axes=perm)
                out_tensor.shape = out_tensor.value.shape
            elif out_tensor.shape is not None and len(out_tensor.shape) == len(perm):
                out_tensor.shape = tuple(out_tensor.shape[idx] for idx in perm)
        transpose_out_attr.update({'src_out_port': 0, 'tensor': out_tensor})
        graph.add_edge(transpose, dst, **transpose_out_attr)
        ret = transpose
    else:
        ERROR('[Parser]: Invalid params for insert_transpose!')
    return ret


def insert_transpose_after(graph, src, perm, port=0, type='Transpose', quantize=False):
    ret = None
    if graph.has_node(src) \
            and perm is not None \
            and isinstance(perm, (list, np.ndarray)) \
            and type in ('Transpose', 'ArmTranspose'):
        if isinstance(perm, np.ndarray):
            perm = perm.tolist()
        if port == 0:
            candidate_name = '_post_transpose'
        else:
            candidate_name = '_post_transpose_%s' % str(port)
        transpose = get_valid_node_name(graph, src + candidate_name)
        found_port = False
        out_tensor = None
        for _, dst, k, out_attr in graph.sorted_out_edges(src, keys=True, data=True):
            if out_attr['src_out_port'] == port:
                found_port = True
                new_out_attr = copy.deepcopy(out_attr)
                new_out_attr['src_out_port'] = 0
                graph.remove_edge(src, dst, key=k)
                graph.add_edge(transpose, dst, **new_out_attr)
                if out_tensor is None:
                    out_tensor = copy.deepcopy(new_out_attr['tensor'])

        if found_port:
            out_edge_attr = {'src_out_port': port, 'dst_in_port': 0}
            if out_tensor is not None:
                inverse_perm = Op.cal_inverse_perm(perm)
                if out_tensor.value is not None:
                    out_tensor.value = np.transpose(out_tensor.value, inverse_perm)
                    out_tensor.shape = out_tensor.value.shape
                elif out_tensor.shape is not None and len(out_tensor.shape) == len(inverse_perm):
                    out_tensor.shape = tuple(out_tensor.shape[idx] for idx in inverse_perm)
                out_edge_attr.update({'tensor': out_tensor})
            graph.add_edge(src, transpose, **out_edge_attr)
            node_attr = {'name': transpose, 'perm': perm, 'quantize': quantize}
            if type == 'Transpose':
                node_attr.update({'opset_version': 1})
            NodeWrap(graph, transpose).replace_obj(type, node_attr)
            ret = transpose
        else:
            ERROR('[Parser]: Cannot find port=%d in insert_transpose_after!' % port)
    else:
        ERROR('[Parser]: Invalid params for insert_transpose_after!')
    return ret


def apply_subgraph_plugin(graph):
    try:
        apply_named_subgraph_plugin(graph)
        apply_pattern_subgraph_plugin(graph)
        apply_preprocess_plugin(graph)
    except Exception as e:
        import traceback
        ERROR('Applying Subgraph plugin Failed. %s' % (str(e)))
        print(traceback.format_exc())


def merge_pattern_to_plugin(graph, plugin_node_optype, innodes, outnodes, non_unique_in_edge_dict={}, match=None):
    assert len(
        outnodes) > 0, '[Parser]: Meet invalid outnodes in merge_pattern_to_plugin!'
    plugin_node = get_valid_node_name(graph, outnodes[0] + '_plugin')

    def add_plugin_in_edge(src, in_attr, in_port):
        new_in_attr = copy.deepcopy(in_attr)
        new_in_attr.update({'dst_in_port': in_port})
        graph.add_edge(src, plugin_node, **new_in_attr)

    cutoff = None if match is None else len(match)

    all_nodes = set()
    for src in innodes:
        for dst in outnodes:
            if src == dst:
                all_nodes.add(src)
                continue
            for path in all_simple_paths(graph, src, dst, cutoff=cutoff):
                all_nodes.update(path)

    inp_tensor_names = []
    input_index_map = []
    for innode in innodes:
        input_map = []
        innode_in_edges = graph.sorted_in_edges(innode, data=True)
        graph.remove_edges_from(innode_in_edges)
        if not innode_in_edges and non_unique_in_edge_dict and innode in non_unique_in_edge_dict:
            innode_in_edges = non_unique_in_edge_dict[innode]
        for src, _, in_attr in innode_in_edges:
            if match is not None and src in list(match.values()):
                continue
            if (src, in_attr['src_out_port']) not in inp_tensor_names:
                inp_tensor_names.append((src, in_attr['src_out_port']))
            else:
                continue
            in_port = len(inp_tensor_names) - 1
            add_plugin_in_edge(src, in_attr, in_port)
            input_map.append(in_port)
        input_index_map.append(input_map)

    out_tensor_names = []
    for outnode in outnodes:
        input_map = []
        outnode_in_edges = graph.sorted_in_edges(outnode, data=True)
        graph.remove_edges_from(outnode_in_edges)
        for src, _, in_attr in outnode_in_edges:
            if any((node == src or has_path(graph, node, src)) for node in innodes):
                continue
            if match is not None and src in list(match.values()):
                continue
            if (src, in_attr['src_out_port']) not in inp_tensor_names:
                inp_tensor_names.append((src, in_attr['src_out_port']))
            else:
                continue
            in_port = len(inp_tensor_names) - 1
            add_plugin_in_edge(src, in_attr, in_port)
            input_map.append(in_port)
        input_index_map.append(input_map)
        outnode_out_edges = graph.sorted_out_edges(outnode, data=True)
        graph.remove_edges_from(outnode_out_edges)
        for out, dst, out_attr in outnode_out_edges:
            if (out, out_attr['dst_in_port']) not in out_tensor_names:
                out_tensor_names.append((out, out_attr['dst_in_port']))
            new_out_attr = copy.deepcopy(out_attr)
            new_out_attr.update({'src_out_port': len(out_tensor_names) - 1})
            graph.add_edge(plugin_node, dst, **new_out_attr)

    # get attributes before remove it
    attr_names_map = {}
    if graph._attr['framework'].name == 'TENSORFLOW':
        from ...tf.load import tf_attr_names_map
        attr_names_map.update({v: k for k, v in tf_attr_names_map.items()})

    attrs = {}
    for name in all_nodes:
        attrs[name] = {}
        for k, v in getattr(NodeWrap(graph, name)['object'],
                            '_attr',
                            {}).items():
            # TODO: do not wrap attr values, e.g.  keepdims = false --> keepdims = 0
            try:
                v_value = getattr(v, 'value')
            except AttributeError:
                continue

            attrs[name].update({k: v_value})
            if k in attr_names_map:
                attrs[name].update({attr_names_map[k]: v_value})

    if match:
        inverse_match = {v: k for k, v in match.items()}
        attrs = {inverse_match[k]: v for k, v in attrs.items()}

    new_attrs = NodeWrap(graph, outnodes[0])['object'].copied_attr()
    new_attrs.update(attrs)
    new_attrs.update({'name': plugin_node, '_nest_inputs': input_index_map})
    NodeWrap(graph, plugin_node).replace_obj(plugin_node_optype, new_attrs)

    for node in all_nodes:
        if node in graph._attr['output_names']:
            index = graph._attr['output_names'].index(node)
            if plugin_node not in graph._attr['output_names']:
                graph._attr['output_names'][index] = plugin_node
            else:
                graph._attr['output_names'].remove(index)
        remove_node_safely(graph, node)


def apply_pattern_subgraph_plugin(graph):
    pattern_subgraph = set()
    for name, plugin in PARSER_OP_DICT.items():
        if hasattr(plugin, '_subgraph_type') and plugin._subgraph_type == 'pattern_subgraph':
            pattern_subgraph.add(plugin)
    if not pattern_subgraph:
        return

    pattern_subgraph = list(pattern_subgraph)
    pattern_subgraph.sort(key=lambda x: x.priority, reverse=True)

    plugin_ops = []

    def get_op_name(optype):
        if optype in plugin_ops:
            return f'Plugin{optype}'
        optype_prefix = {
            Framework.TFLITE: lambda x: 'Lite' + x,
            Framework.CAFFE: lambda x: 'Caffe' + x.upper(),
            Framework.TENSORFLOW: lambda x: 'Tf' + x,
        }

        framework = graph._attr.get('framework', Framework.NONE)
        op_name = optype_prefix[framework](optype) if \
            framework in optype_prefix else optype
        return op_name

    def get_io_nodes(nodes, edges, graph, match):
        def get_edges_num(n_name, _edges, edge_type):
            idx = 0 if edge_type == 'out' else 1
            edg_num = 0
            for e in _edges:
                if e[idx] == n_name:
                    edg_num += 1
            return edg_num

        _in_nodes = []
        _out_nodes = []
        _non_unique_in_edges = {}
        for n in nodes:
            node_in_edges = graph.sorted_in_edges(match[n[0]], data=True)
            if node_in_edges:
                pattern_in_edge_num = get_edges_num(n[0], edges, 'in')
                if len(node_in_edges) > pattern_in_edge_num:
                    _in_nodes.append(match[n[0]])

            node_out_edges = graph.sorted_out_edges(match[n[0]], data=True)
            _unique = n[1]['unique'] if 'unique' in n[1] else True
            if not _unique and match[n[0]] in _in_nodes:
                _non_unique_in_edges[match[n[0]]] = node_in_edges.copy()
            if node_out_edges:
                pattern_out_edge_num = get_edges_num(n[0], edges, 'out')
                if len(node_out_edges) > pattern_out_edge_num and _unique:
                    _out_nodes.append(match[n[0]])

        return _in_nodes, _out_nodes, _non_unique_in_edges

    for plugin in pattern_subgraph:
        plugin_ops.append(plugin.op_type)
        nodes = []
        for name, optype in plugin.pattern_nodes:
            if isinstance(optype, dict):
                assert 'op' in optype, 'key: op MUST be set in node'
                if 'unique' in optype:
                    nodes.append((name, {'op': get_op_name(optype['op']),
                                         'unique': bool(optype['unique'])}))
                else:
                    nodes.append((name, {'op': get_op_name(optype['op'])}))
            else:
                nodes.append((name, {'op': get_op_name(optype)}))

        matches = matched_patterns(graph,
                                   nodes=nodes,
                                   edges=plugin.pattern_edges)
        in_out_nodes_list = []
        non_unique_in_edges = []
        for m in matches:
            in_nodes, out_nodes, non_unique_in_edges_dict = get_io_nodes(
                plugin.pattern_nodes, plugin.pattern_edges, graph, m)
            in_out_nodes_list.append((in_nodes, out_nodes))
            non_unique_in_edges.append(non_unique_in_edges_dict)
        for i, m in enumerate(matches):
            in_nodes, out_nodes = in_out_nodes_list[i]
            merge_pattern_to_plugin(
                graph, plugin.op_type, in_nodes, out_nodes, non_unique_in_edges[i], m)
            DEBUG('[Parser]: pattern based subgraph plugin applied: {[%s]->[%s]} merged to %s' %
                  (','.join([str(n) for n in in_nodes]), ','.join([str(n) for n in out_nodes]), plugin.op_type))


def apply_named_subgraph_plugin(graph):
    named_subgraph = set()
    for _, plugin in PARSER_OP_DICT.items():
        if hasattr(plugin, '_subgraph_type') and plugin._subgraph_type == 'named_subgraph':
            named_subgraph.add(plugin)
    named_subgraph = list(named_subgraph)
    named_subgraph.sort(key=lambda x: x.priority, reverse=True)
    for plugin in named_subgraph:
        if not all(graph.has_node(n) for n in plugin.start_nodes + plugin.end_nodes):
            continue

        merge_pattern_to_plugin(
            graph, '.named_subgraph.' + plugin.op_type, plugin.start_nodes, plugin.end_nodes)
        DEBUG('[Parser]: name_based subgraph plugin applied: {[%s]->[%s]} merged to %s' % (
            ','.join(plugin.start_nodes), ','.join(plugin.end_nodes), plugin.op_type))


def insert_preprocess_plugin(graph, plugin_op_type, input_nodes, input_shapes, use_default_output):
    '''Return a list of input names, in which preprocess plugin is applied.
    '''
    ret = []
    graph_inputs = graph._attr['input_tensors']
    if any(n not in graph_inputs.keys() for n in input_nodes):
        return ret
    nodes_cnt = len(input_nodes)
    shapes_cnt = len(input_shapes)
    if nodes_cnt != shapes_cnt:
        ERROR('[Parser]: The length of input nodes should be same as input shapes in insert_preprocess_plugin, but got (%d, %d)!' % (
            nodes_cnt, shapes_cnt))
        return ret
    DEBUG('[Parser]: preprocess subgraph plugin applied: preprocess %s is added before [%s](shape: %s)' % (
        plugin_op_type, ','.join(input_nodes), ','.join([str(shape) for shape in input_shapes])))
    for input_name, shape in zip(input_nodes, input_shapes):
        # update graph input
        original_input = graph_inputs[input_name]
        original_input_value = original_input.value
        dtype = original_input_value.dtype
        new_input = np.zeros(shape, dtype=dtype) if 'int' in str(dtype) else np.random.ranf(shape).astype(dtype)
        input_tensor = graph._attr['input_tensors'][input_name]
        input_tensor.value = new_input
        input_tensor.shape = tuple(shape)

        # set in_attr for preprocess
        input_out_edges = graph.sorted_out_edges(input_name, data=True)
        if len(input_out_edges) < 1:
            continue
        input_out_attr = input_out_edges[0][2]
        preprocess_in_attr = copy.deepcopy(input_out_attr)
        preprocess_in_attr.update({'src_out_port': 0, 'dst_in_port': 0})
        if preprocess_in_attr['tensor'] is not None:
            preprocess_in_attr['tensor'].value = new_input
        else:
            preprocess_in_attr['tensor'] = Tensor(value=new_input, shape=tuple(shape))

        # insert preprocess node after input node
        graph.remove_edges_from(input_out_edges)
        preprocess_node = get_valid_node_name(graph, plugin_op_type)
        graph.add_edge(input_name, preprocess_node, **preprocess_in_attr)
        if use_default_output:
            preprocess_out_value = original_input_value
            preprocess_out_shape = tuple(original_input_value.shape)
        else:
            preprocess_out_value, preprocess_out_shape = None, None
        for _, dst, out_attr in input_out_edges:
            dst_in_attr = copy.deepcopy(out_attr)
            if dst_in_attr['tensor'] is not None:
                dst_in_attr['tensor'].value = preprocess_out_value
                dst_in_attr['tensor'].shape = preprocess_out_shape
            else:
                dst_in_attr['tensor'] = Tensor(value=preprocess_out_value, shape=preprocess_out_shape, dtype=dtype)
            graph.add_edge(preprocess_node, dst, **dst_in_attr)
        NodeWrap(graph, preprocess_node).replace_obj('.preprocess.' + plugin_op_type,
                                                     {'name': preprocess_node,
                                                      'out_tensors': [preprocess_out_value] if use_default_output else []})
        ret.append(input_name)
    return ret


def apply_preprocess_plugin(graph):
    preprocess_subgraph = set()
    for _, plugin in PARSER_OP_DICT.items():
        if plugin.input_nodes is not None and plugin.input_shapes is not None:
            preprocess_subgraph.add(plugin)
    preprocess_subgraph = list(preprocess_subgraph)
    preprocess_subgraph.sort(key=lambda x: x.priority, reverse=True)
    applied_input_names = []
    for plugin in preprocess_subgraph:
        if all(name in applied_input_names for name in plugin.input_nodes):
            continue
        plugin_op_type = plugin.op_type
        use_default_output = 'infer_shape' not in plugin.__dict__
        applied_input_names.extend(insert_preprocess_plugin(
            graph, plugin_op_type, plugin.input_nodes, plugin.input_shapes, use_default_output))


def merge_same_op_at_out_port(graph, op_types=['ArmTranspose', 'ArmReshape']):
    '''Merge ANY->Transpose/Reshape/others(*n) to ANY->Transpose/Reshape/others(*1) if other
    inputs and attrs of n Transpose/Reshape/others nodes are same.
    '''
    matches = single_node_matcher(graph, {})
    for m in matches:
        node = m['target']
        if not graph.has_node(node):
            continue
        node_obj = NodeWrap(graph, node)['object']
        if node_obj is None or node_obj.type == 'Out':
            continue
        out_ports = node_obj.get_out_ports()
        if len(out_ports) < 1:
            continue
        out_edges = graph.sorted_out_edges(node, keys=True, data=True)
        if len(out_edges) < 2:
            continue
        for p in out_ports:
            cur_p_edges = [e for e in out_edges if (e[3]['src_out_port'] == p and graph.has_node(
                e[1]) and NodeWrap(graph, e[1])['object'] is not None)]
            if len(cur_p_edges) < 2:
                continue
            if any(e[3]['dst_in_port'] != 0 for e in cur_p_edges):
                continue
            for op in op_types:
                cur_type_edges = [e for e in cur_p_edges if (graph.has_node(e[1]) and NodeWrap(
                    graph, e[1])['object'] is not None and NodeWrap(graph, e[1])['object'].type == op)]
                if len(cur_type_edges) < 2:
                    continue
                cur_objs = [NodeWrap(graph, e[1])['object'] for e in cur_type_edges]
                if op == 'ArmReshape':
                    if any(cur_objs[0].dim != obj.dim for obj in cur_objs[1:]):
                        continue
                elif op == 'ArmTranspose':
                    if any(cur_objs[0].perm != obj.perm for obj in cur_objs[1:]):
                        continue
                elif op == 'Cast':
                    if any((cur_objs[0].to != obj.to or cur_objs[0].saturate != obj.saturate) for obj in cur_objs[1:]):
                        continue
                elif op in ['QuantizeLinear', 'DequantizeLinear']:
                    if op == 'QuantizeLinear':
                        scale_attr = 'y_scale'
                        zp_attr = 'y_zero_point'
                    else:
                        scale_attr = 'x_scale'
                        zp_attr = 'x_zero_point'
                    quant_nodes = [e[1] for e in cur_p_edges]
                    for quant_node in quant_nodes:
                        quant_in_edges = graph.sorted_in_edges(quant_node, data=True)
                        if any(e[2]['tensor'].value is None for e in quant_in_edges[1:]) \
                                or any(not e[2]['tensor'].is_const for e in quant_in_edges[1:]):
                            continue
                    if any((cur_objs[0].axis != obj.axis or not FLOAT_EQUAL(getattr(cur_objs[0], scale_attr), getattr(obj, scale_attr))
                            or getattr(cur_objs[0], zp_attr) != getattr(obj, zp_attr)) for obj in cur_objs[1:]):
                        continue
                else:
                    # not supported yet
                    continue
                keep_out_node = cur_type_edges[0][1]
                for _, removing_out, k, _ in cur_type_edges[1:]:
                    graph.remove_edge(node, removing_out, key=k)
                    for _, dst, meta_k, out_attr in graph.sorted_out_edges(removing_out, keys=True, data=True):
                        graph.remove_edge(removing_out, dst, key=meta_k)
                        graph.add_edge(keep_out_node, dst, **out_attr)
                    graph.remove_node(removing_out)
                    if removing_out in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(removing_out)
                        if keep_out_node not in graph._attr['output_names']:
                            graph._attr['output_names'][index] = keep_out_node
                        else:
                            graph._attr['output_names'].pop(index)


def record_output_tensors(graph, params={}):
    # using Out Op to record the output tensors order
    out_tensors = graph._attr['output_tensor_names']

    matches = single_node_matcher(graph, 'Out')
    out_nodes = [None] * len(out_tensors)
    for m in matches:
        node_name = m['target']
        if node_name in graph.nodes:
            try:
                _, _, _, info = graph.sorted_in_edges(
                    node_name, keys=True, data=True)[0]
                t = info.get('tensor', None)
                if t is not None and t.name in out_tensors:
                    idx = out_tensors.index(t.name)
                    out_nodes[idx] = node_name
            except Exception:
                pass
    graph._attr['output_nodes'] = out_nodes
    if params.get('model_type', '') == 'torch' and None not in out_nodes:
        original_outputs = params.get('output_tensor_map', {}).keys()
        if len(out_nodes) == len(original_outputs):
            for out_name_from_cfg, node_name in zip(original_outputs, out_nodes):
                params['output_tensor_map'][out_name_from_cfg] = [node_name]
