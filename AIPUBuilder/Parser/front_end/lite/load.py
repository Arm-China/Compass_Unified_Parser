# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import onnx
import numpy as np
import copy
import re
from collections import defaultdict, OrderedDict
from functools import partial
from ...graph.graph import Graph
from ...graph.node_wrap import NodeWrap
from ...graph.graph_algo import get_valid_node_name, clear_redundant_nodes
from ...common.defs import Framework, get_opset_version, Tensor
from ...common.utils import get_version, extend_lists
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL
from .buffer import read_tflite_model, parse_operator, parse_tensor, get_act_info_from_tensor


def convert_attr_to_onnx(attr_dict):
    new_attr = copy.deepcopy(attr_dict)
    for k, v in attr_dict.items():
        if k == 'AlignCorners':
            new_attr.update({'align_corners': int(v)})
            new_attr.pop(k)
        elif k == 'Alpha':
            new_attr.update({'alpha': v})
            new_attr.pop(k)
        elif k == 'AsymmetricQuantizeInputs':
            new_attr.update({'asymmetric_quantize_inputs': int(v)})
            new_attr.pop(k)
        elif k == 'Axis':
            new_attr.update({'axis': v})
            new_attr.pop(k)
        elif k == 'AdjX':
            new_attr.update({'adj_x': v})
            new_attr.pop(k)
        elif k == 'AdjY':
            new_attr.update({'adj_y': v})
            new_attr.pop(k)
        elif k == 'BeginMask':
            new_attr.update({'begin_mask': v})
            new_attr.pop(k)
        elif k == 'Beta':
            new_attr.update({'beta': v})
            new_attr.pop(k)
        elif k == 'Bias':
            new_attr.update({'bias': v})
            new_attr.pop(k)
        elif k == 'BlockSize':
            new_attr.update({'blocksize': v})
            new_attr.pop(k)
        elif k == 'CellClip':
            new_attr.update({'cell_clip': v})
            new_attr.pop(k)
        elif k == 'DepthMultiplier':
            new_attr.update({'multiplier': int(v)})
            new_attr.pop(k)
        elif k == 'DilationDFactor':
            new_attr.update(
                {'dilations': [v, *new_attr.get('dilations', [1, 1, 1])[1:3]]})
            new_attr.pop(k)
        elif k == 'DilationHFactor':
            if 'DilationDFactor' in attr_dict:
                new_attr.update(
                    {'dilations': [new_attr.get('dilations', [1, 1, 1])[0], v, new_attr.get('dilations', [1, 1, 1])[2]]})
            else:
                new_attr.update(
                    {'dilations': [v, new_attr.get('dilations', [1, 1])[1]]})
            new_attr.pop(k)
        elif k == 'DilationWFactor':
            if 'DilationDFactor' in attr_dict:
                new_attr.update(
                    {'dilations': [*new_attr.get('dilations', [1, 1, 1])[0:2], v]})
            else:
                new_attr.update(
                    {'dilations': [new_attr.get('dilations', [1, 1])[0], v]})
            new_attr.pop(k)
        elif k == 'EllipsisMask':
            new_attr.update({'ellipsis_mask': v})
            new_attr.pop(k)
        elif k == 'EndMask':
            new_attr.update({'end_mask': v})
            new_attr.pop(k)
        elif k == 'Exclusive':
            new_attr.update({'exclusive': int(v)})
            new_attr.pop(k)
        elif k == 'FilterHeight':
            new_attr.update(
                {'kernel_shape': [v, new_attr.get('kernel_shape', [1, 1])[1]]})
            new_attr.pop(k)
        elif k == 'FilterWidth':
            new_attr.update(
                {'kernel_shape': [new_attr.get('kernel_shape', [1, 1])[0], v]})
            new_attr.pop(k)
        elif k == 'FusedActivationFunction':
            if v not in ('RELU', 'RELU_N1_TO_1', 'RELU6', 'TANH', 'SIGN_BIT'):
                v = 'NONE'
            new_attr.update({'activations': v})
            new_attr.pop(k)
        elif k == 'HalfPixelCenters':
            new_attr.update({'half_pixel': int(v)})
            new_attr.pop(k)
        elif k in ('KeepDims', 'KeepNumDims'):
            new_attr.update({'keepdims': int(v)})
            new_attr.pop(k)
        elif k == 'Mode':
            new_attr.update({'mode': v})
            new_attr.pop(k)
        elif k == 'NewAxisMask':
            new_attr.update({'new_axis_mask': v})
            new_attr.pop(k)
        elif k == 'NewShapeAsNumpy':
            new_attr.update({'shape': v.tolist()})
            new_attr.pop(k)
        elif k == 'Num':
            new_attr.update({'num': int(v)})
            new_attr.pop(k)
        elif k in ('OutDataType', 'OutputType'):
            new_attr.update({'to': v})
            new_attr.pop(k)
        elif k == 'Padding':
            if v == 'SAME':
                v = 'SAME_UPPER'
            new_attr.update({'auto_pad': v})
            new_attr.pop(k)
        elif k == 'ProjClip':
            new_attr.update({'proj_clip': v})
            new_attr.pop(k)
        elif k == 'NumSplits':
            new_attr.update({'num_splits': v})
            new_attr.pop(k)
        elif k == 'Radius':
            new_attr.update({'radius': v})
            new_attr.pop(k)
        elif k == 'Reverse':
            new_attr.update({'reverse': int(v)})
            new_attr.pop(k)
        elif k == 'ShrinkAxisMask':
            new_attr.update({'shrink_axis_mask': v})
            new_attr.pop(k)
        elif k == 'SqueezeDims':
            new_attr.update({'axes': v})
            new_attr.pop(k)
        elif k == 'SqueezeDimsAsNumpy':
            new_attr.update({'axes': v.tolist()})
            new_attr.pop(k)
        elif k == 'StrideD':
            new_attr.update(
                {'strides': [v, *new_attr.get('strides', [1, 1, 1])[1:3]]})
            new_attr.pop(k)
        elif k == 'StrideH':
            if 'StrideD' in attr_dict:
                new_attr.update(
                    {'strides': [new_attr.get('strides', [1, 1, 1])[0], v, new_attr.get('strides', [1, 1, 1])[2]]})
            else:
                new_attr.update(
                    {'strides': [v, new_attr.get('strides', [1, 1])[1]]})
            new_attr.pop(k)
        elif k == 'StrideW':
            if 'StrideD' in attr_dict:
                new_attr.update(
                    {'strides': [*new_attr.get('strides', [1, 1, 1])[0:2], v]})
            else:
                new_attr.update(
                    {'strides': [new_attr.get('strides', [1, 1])[0], v]})
            new_attr.pop(k)
        elif k == 'TimeMajor':
            new_attr.update({'time_major': int(v)})
            new_attr.pop(k)
        elif k == 'WeightsFormat':
            new_attr.update({'weights_format': v})
            new_attr.pop(k)
        elif k == 'BatchDim':
            new_attr.update({'batch_dim': v})
            new_attr.pop(k)
        elif k == 'BatchDims':
            new_attr.update({'batch_dims': v})
            new_attr.pop(k)
        elif k == 'SeqDim':
            new_attr.update({'seq_dim': v})
            new_attr.pop(k)
    return new_attr


def convert_tflite_to_graph(model_path, params):
    '''Parse the tflite model into a graph structure.'''

    from ...plugin_loader import PARSER_OP_DICT
    graph = Graph(name=params.get('model_name', ''))
    graph._attr['framework'] = Framework.TFLITE
    graph._attr['output_tensor_names'] = params.get('output_tensor_names', [])
    quantize = True \
        if params.get('compat_quantized_model', 'false').lower() == 'true' \
        else False
    graph._attr['quantize'] = quantize

    try:
        ret, model, buffer = read_tflite_model(model_path)
        if ret:
            consumer_ver = get_version(onnx)
            if consumer_ver >= 1.04:
                opset_version = get_opset_version(consumer_ver)
                graph._attr['opset_version'] = opset_version
                sub_graph = model.Subgraphs(0)
                net_inputs = sub_graph.InputsAsNumpy().tolist()
                net_outputs = sub_graph.OutputsAsNumpy().tolist()
                tensors_table = [sub_graph.Tensors(
                    i) for i in range(sub_graph.TensorsLength())]
                operators_table = [sub_graph.Operators(
                    i) for i in range(sub_graph.OperatorsLength())]

                parsed_operators_table = list(
                    map(partial(parse_operator, tflite_model=model, buffer=buffer), operators_table))
                for single_input in net_inputs:
                    parsed_operators_table.insert(
                        0, {'type': 'Input', 'attr': {}, 'inputs': [], 'outputs': [single_input]})
                linear_weights_tensor_id = {
                    k: v for po in parsed_operators_table for k, v in po.get('linear_weights', {}).items()}

                # identify whether tensor is non-Const
                non_const_tensor_ids = set(
                    net_outputs + extend_lists([op_info['outputs'] for op_info in parsed_operators_table]))
                tensors_table = [(tensor, ti not in non_const_tensor_ids, linear_weights_tensor_id.get(
                    ti, '')) for ti, tensor in enumerate(tensors_table)]
                parsed_tensors_table = list(
                    map(partial(parse_tensor, tflite_model=model, quantize=quantize), tensors_table))

                tensor_name_count = defaultdict(int)
                for in_tensor_id in non_const_tensor_ids:
                    in_tensor = parsed_tensors_table[in_tensor_id]
                    if in_tensor['data'].dtype == 'bool':
                        in_tensor['data'] = in_tensor['data'].astype(np.uint8)
                    tensor_name = in_tensor['name']
                    tensor_name_count[tensor_name] += 1
                    if tensor_name_count[tensor_name] > 1:
                        parsed_tensors_table[in_tensor_id]['name'] = tensor_name + \
                            '_' + str(tensor_name_count[tensor_name] - 1)

                out_tensor_operator_map = {k: oi for oi, op_info in enumerate(
                    parsed_operators_table) for k in op_info['outputs']}
                out_tensor_op_name_map = OrderedDict()

                # const sharing exits in tflite
                const_tensor_count = defaultdict(int)
                for op_id, op_info in enumerate(parsed_operators_table):
                    node_name = parsed_tensors_table[op_info['outputs'][0]]['name']
                    node_name = get_valid_node_name(graph, node_name)
                    graph.add_node(node_name)
                    for o in op_info.get('outputs', []):
                        out_tensor_op_name_map.update(
                            {parsed_tensors_table[o]['name']: node_name})
                    if op_info['type'] == 'Input':
                        op_type = op_info['type']
                        for out in op_info['outputs']:
                            if out not in net_inputs:
                                net_inputs.append(out)
                    else:
                        op_type = ('Tf' if op_info['is_tf_op'] else 'Lite') + op_info['type']
                    attr_dict = copy.deepcopy(op_info['attr'])
                    attr_dict.update(
                        {'name': node_name, 'data_format': 'NHWC', 'quantize': quantize})
                    if re.search(r'Lite', op_type):
                        attr_dict.update(
                            {'opcode_version': op_info['opcode_version']})
                    activation = get_act_info_from_tensor(
                        parsed_tensors_table[op_info['outputs'][0]])
                    if activation and attr_dict.get('FusedActivationFunction', 'NONE') == 'NONE':
                        attr_dict['FusedActivationFunction'] = activation['act_type']
                    if op_type not in PARSER_OP_DICT:
                        attr_dict = convert_attr_to_onnx(attr_dict)
                    node = NodeWrap(graph, node_name)
                    node.replace_obj(op_type, attr_dict)

                    for in_port, in_tensor_id in enumerate(op_info['inputs']):
                        if in_tensor_id < 0:
                            in_tensor_name = get_valid_node_name(
                                graph, node_name + '_in_tensor_' + str(in_port))
                            graph.add_node(in_tensor_name)
                            dummy_node = NodeWrap(graph, in_tensor_name)
                            dummy_attr = {'name': in_tensor_name}
                            dummy_node.replace_obj('Dummy', dummy_attr)
                            edge_attr = {'src_out_port': 0, 'dst_in_port': in_port, 'tensor': Tensor(
                                name=in_tensor_name)}
                            graph.add_edge(
                                in_tensor_name, node_name, **edge_attr)
                        elif in_tensor_id in non_const_tensor_ids:
                            in_tensor = parsed_tensors_table[in_tensor_id]
                            pre_op_id = out_tensor_operator_map[in_tensor_id]
                            pre_op_info = parsed_operators_table[pre_op_id]
                            pre_node_name = parsed_tensors_table[pre_op_info['outputs'][0]]['name']
                            src_out_port = pre_op_info['outputs'].index(
                                in_tensor_id)
                            edge_tensor = Tensor(shape=in_tensor['data'].shape)
                            edge_tensor.value = in_tensor['data']
                            if 'quant_info' in in_tensor:
                                if 'Max' in in_tensor['quant_info'] \
                                        and 'Min' in in_tensor['quant_info']:
                                    edge_tensor.min_max = (
                                        in_tensor['quant_info']['Min'], in_tensor['quant_info']['Max'])
                                if 'Scale' in in_tensor['quant_info'] \
                                        and 'ZeroPoint' in in_tensor['quant_info']:
                                    edge_tensor.scale_zp = (in_tensor['quant_info']['Scale'],
                                                            in_tensor['quant_info']['ZeroPoint'])
                            if 'dtype' in in_tensor:
                                edge_tensor.dtype = in_tensor['dtype']
                            edge_attr = {'src_out_port': src_out_port,
                                         'dst_in_port': in_port, 'tensor': edge_tensor}
                            assert graph.has_node(
                                pre_node_name), 'The node does not exist, cannot add edge into graph in convert_tflite_to_graph.'
                            graph.add_edge(
                                pre_node_name, node_name, **edge_attr)
                        else:
                            const_name = parsed_tensors_table[in_tensor_id]['name']
                            const_tensor_count[const_name] += 1
                            if const_tensor_count[const_name] > 1:
                                const_name = const_name + '_' + str(in_tensor_id) + '_' + \
                                    str(const_tensor_count[const_name] - 1)
                            const_name = get_valid_node_name(graph, const_name)
                            assert not graph.has_node(
                                const_name), ('The const node(%s) already exist, cannot add node into graph in convert_tflite_to_graph.' % const_name)
                            const_tensor = parsed_tensors_table[in_tensor_id]
                            edge_attr = {'src_out_port': 0,
                                         'dst_in_port': in_port,
                                         'tensor': Tensor(value=const_tensor['data'], is_const=True)
                                         }
                            if 'quant_info' in const_tensor:
                                if 'Max' in const_tensor['quant_info'] \
                                        and 'Min' in const_tensor['quant_info']:
                                    edge_attr['tensor'].min_max = (const_tensor['quant_info']['Min'],
                                                                   const_tensor['quant_info']['Max'])
                                if 'Scale' in const_tensor['quant_info'] \
                                        and 'ZeroPoint' in const_tensor['quant_info']:
                                    edge_attr['tensor'].scale_zp = (const_tensor['quant_info']['Scale'],
                                                                    const_tensor['quant_info']['ZeroPoint'])
                            if 'dtype' in const_tensor:
                                edge_attr['tensor'].dtype = const_tensor['dtype']
                            graph.add_edge(const_name, node_name, **edge_attr)
                            const_node = NodeWrap(graph, const_name)
                            const_node.replace_obj('Constant',
                                                   {'name': const_name,
                                                    'value': const_tensor['data'],
                                                    'data_format': 'NHWC',
                                                    'opset_version': opset_version,
                                                    'quantize': quantize
                                                    }
                                                   )

                if ret:
                    input_tensors = OrderedDict()
                    for net_in in net_inputs:
                        net_in_name = parsed_tensors_table[net_in]['name']
                        net_in_tensor = parsed_tensors_table[net_in]['data']
                        shape = list(
                            net_in_tensor.shape) if net_in_tensor.shape else []
                        if net_in_name in params['input_shapes'] \
                                and shape != params['input_shapes'][net_in_name]:
                            new_shape = params['input_shapes'][net_in_name]
                            WARN('Original model expects input shape of input %s to be %s. '
                                 'Now reset it to %s basing on config file!' % (net_in_name, str(shape), str(new_shape)))
                            shape = new_shape
                        if str(net_in_tensor.dtype).find('int') != -1:
                            net_in_tensor = np.zeros(
                                shape, net_in_tensor.dtype)
                        else:
                            net_in_tensor = (np.random.ranf(
                                size=shape) * 100).astype(net_in_tensor.dtype)

                        input_tensors.update(
                            {net_in_name: Tensor(value=net_in_tensor)})
                    graph._attr['input_tensors'] = copy.deepcopy(input_tensors)

                    # add output node into graph
                    output_names = []
                    for output_index, output_tensor_id in enumerate(net_outputs):
                        out_tensor = parsed_tensors_table[output_tensor_id]
                        out_op_id = out_tensor_operator_map[output_tensor_id]
                        out_op_info = parsed_operators_table[out_op_id]
                        out_node_name = parsed_tensors_table[out_op_info['outputs'][0]]['name']
                        noop_node_name = parsed_tensors_table[output_tensor_id]['name'] + '_noop_' + str(
                            output_index)
                        assert graph.has_node(out_node_name) and not graph.has_node(
                            noop_node_name), 'The output node does not exist, cannot add output node into graph in convert_tflite_to_graph.'
                        graph.add_node(noop_node_name)
                        noop_node = NodeWrap(graph, noop_node_name)
                        noop_node.replace_obj('Out', {'name': noop_node_name})

                        src_out_port = out_op_info['outputs'].index(
                            output_tensor_id)
                        out_edge_tensor = Tensor(
                            name=out_tensor.get('name', ''))
                        if 'quant_info' in out_tensor:
                            if 'Max' in out_tensor['quant_info'] \
                                    and 'Min' in out_tensor['quant_info']:
                                out_edge_tensor.min_max = (
                                    out_tensor['quant_info']['Min'], out_tensor['quant_info']['Max'])
                            if 'Scale' in out_tensor['quant_info'] \
                                    and 'ZeroPoint' in out_tensor['quant_info']:
                                out_edge_tensor.scale_zp = (
                                    out_tensor['quant_info']['Scale'], out_tensor['quant_info']['ZeroPoint'])
                        if 'dtype' in out_tensor:
                            out_edge_tensor.dtype = out_tensor['dtype']
                        if out_tensor.get('data', None) is not None:
                            out_edge_tensor.value = out_tensor['data']
                            out_edge_tensor.shape = out_tensor['data'].shape
                        out_edge_attr = {'src_out_port': src_out_port,
                                         'dst_in_port': 0, 'tensor': out_edge_tensor}
                        graph.add_edge(
                            out_node_name, noop_node_name, **out_edge_attr)

                        if out_node_name not in output_names:
                            output_names.append(out_node_name)

                    if params.get('output_names', []):
                        names = list()
                        for n in params['output_names']:
                            if graph.has_node(n):
                                if n not in names:
                                    names.append(n)
                            elif n in out_tensor_op_name_map:
                                if out_tensor_op_name_map[n] not in names:
                                    names.append(out_tensor_op_name_map[n])
                            else:
                                WARN(
                                    '[Parser]: Output name (%s) is neither a node name nor a tensor name! Please check config file!' % n)
                        if not names:
                            WARN(
                                ['Parser: Cannot get valid output names from config file! Will use parsed node names instead!'])
                            graph._attr['output_names'] = copy.deepcopy(
                                output_names)
                        else:
                            graph._attr['output_names'] = copy.deepcopy(names)
                            # convert node name to tensor name
                            if any([t not in out_tensor_op_name_map for t in graph._attr['output_tensor_names']]) \
                                    and all([graph.has_node(n) for n in graph._attr['output_tensor_names']]):
                                out_tensors = []
                                for n in graph._attr['output_tensor_names']:
                                    out_edges = graph.sorted_out_edges(
                                        n, data=True)
                                    for u, v, d in out_edges:
                                        t = d.get('tensor', None)
                                        if t is not None:
                                            out_tensors.append(t.name)
                                        else:
                                            pass
                    else:
                        graph._attr['output_names'] = copy.deepcopy(
                            output_names)

                    for name in graph._attr['output_names']:
                        if graph.has_node(name):
                            if not graph.succ[name] \
                                    or all([graph.get_node(succ)._attr['op'] != 'Out' for succ in graph.succ[name]]):
                                out_edges = graph.sorted_out_edges(
                                    name, data=True)
                                try:
                                    out_ports = graph.get_node(
                                        name)._attr['object'].get_out_ports()
                                except:
                                    out_ports = []
                                for p in out_ports:
                                    tensor = None
                                    for _, _, out_attr in out_edges:
                                        if out_attr['src_out_port'] == p:
                                            tensor = out_attr['tensor']
                                            break
                                    out_name = get_valid_node_name(
                                        graph, name + '_out_' + str(p))
                                    graph.add_edge(
                                        name, out_name, **{'src_out_port': p, 'dst_in_port': 0, 'tensor': tensor})
                                    NodeWrap(graph, out_name).replace_obj(
                                        'Out', {'name': out_name})

                    clear_redundant_nodes(graph)
            else:
                ERROR('Onnx version is too low, needs updating!')
    except Exception as e:
        ret = False
        WARN('[Parser]: Read Tflite %s meets error in convert_tflite_to_graph! %s' % (
            model_path, str(e)))
    if not ret:
        graph.clear()
    return graph
