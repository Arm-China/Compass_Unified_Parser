# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import re
import numpy as np
import onnx
import yaml
from collections import Iterable, defaultdict, OrderedDict
from ...graph.graph import Graph
from ...graph.node_wrap import NodeWrap
from ...graph.graph_algo import get_valid_node_name
from .buffer import read_caffe_model
from ...ops.op import OpHasWeights, OpHasBiases
from ..onnx.passes.common_passes import insert_constant
from ...common.defs import Attribute, Framework, Tensor, get_opset_version
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL
from ...common.utils import get_version, string_list_to_string


def num_to_multiple_elem_list(value, multiplier=2):
    if isinstance(value, np.ndarray):
        if value.size == multiplier:
            ret = value.tolist()
        else:
            ret = [int(value)] * multiplier
    else:
        ret = [int(value)] * multiplier
    return ret


def convert_attr_to_onnx(attr_dict):
    new_attr = attr_dict.copy()
    for k, v in attr_dict.items():
        if k.endswith('_param') and isinstance(v, dict):
            for vk, vv in v.items():
                if vk == 'aspect_ratio':
                    new_attr.update({'aspect_ratio': np.array(vv).tolist()})
                elif vk == 'num_classes':
                    new_attr.update({'num_classes': int(vv)})
                elif vk == 'coeff':
                    new_attr.update({'coeff': np.array(vv).tolist()})
                elif vk == 'dilation':
                    if len(attr_dict.get('blobs', [])) >= 1 and len(attr_dict['blobs'][0].shape) == 5:
                        multiplier = 3
                    else:
                        multiplier = 2
                    new_attr.update(
                        {'dilations': num_to_multiple_elem_list(vv, multiplier=multiplier)})
                elif vk == 'eps':
                    new_attr.update({'epsilon': float(vv)})
                elif vk == 'kernel_size':
                    if len(attr_dict.get('blobs', [])) >= 1 and len(attr_dict['blobs'][0].shape) == 5:
                        multiplier = 3
                    else:
                        multiplier = 2
                    new_attr.update(
                        {'kernel_shape': num_to_multiple_elem_list(vv, multiplier=multiplier)})
                elif vk == 'local_size':
                    new_attr.update({'size': int(vv)})
                elif vk == 'min_size':
                    new_attr.update({'min_size': np.array(vv).tolist()})
                elif vk == 'max_size':
                    new_attr.update({'max_size': np.array(vv).tolist()})
                elif vk == 'negative_slope':
                    new_attr.update({'alpha': float(vv)})
                elif vk == 'offset':
                    new_attr.update({'offset': vv.tolist()})
                elif vk == 'operation':
                    new_attr.update({'method': str(vv)})
                elif vk == 'order':
                    if isinstance(vv, np.ndarray):
                        new_attr.update({'order': vv.tolist()})
                elif vk == 'pad':
                    if len(attr_dict.get('blobs', [])) >= 1 and len(attr_dict['blobs'][0].shape) == 5:
                        multiplier = 3
                    else:
                        multiplier = 2
                    new_attr.update(
                        {'pads': num_to_multiple_elem_list(vv, multiplier=multiplier) * 2})
                elif vk == 'pool':
                    new_attr.update({'method': str(vv)})
                elif vk == 'scales':
                    new_attr.update({'scales': list(vv)})
                elif vk == 'shape':
                    if isinstance(vv, list):
                        if 'dim' in vv[0]:
                            new_attr.update({'shape': vv[0]['dim'].tolist()})
                        else:
                            new_attr.update({'shape': []})
                    else:
                        if 'dim' in vv:
                            new_attr.update({'shape': vv['dim'].tolist()})
                        else:
                            new_attr.update({'shape': []})
                elif vk == 'stride':
                    if len(attr_dict.get('blobs', [])) >= 1 and len(attr_dict['blobs'][0].shape) == 5:
                        multiplier = 3
                    else:
                        multiplier = 2
                    new_attr.update(
                        {'strides': num_to_multiple_elem_list(vv, multiplier=multiplier)})
                elif vk == 'top_k':
                    new_attr.update({'k': int(vv)})
                elif vk == 'variance':
                    new_attr.update({'variance': np.array(vv).tolist()})
                else:
                    new_attr.update({vk: vv})
            new_attr.pop(k)
    return new_attr


def convert_caffe_to_graph(model_path, params):
    '''Parse the caffe model into a graph structure.'''
    if not params.get('model_name', ''):
        params['model_name'] = 'caffe_model'
    graph = Graph(name=params['model_name'])
    graph._attr['framework'] = Framework.CAFFE
    graph._attr['output_tensor_names'] = params.get('output_tensor_names', [])

    consumer_ver = get_version(onnx)
    if consumer_ver >= 1.04:
        opset_version = get_opset_version(consumer_ver)
        graph._attr['opset_version'] = opset_version

        prototxt_path = params.get('caffe_prototxt', '')
        if params.get('input_names', []) \
                and params.get('input_shapes', {}) \
                and len(params['input_names']) != len(params['input_shapes']):
            FATAL(
                '[Parser]: The length of input_names should be equal to the length of input_shapes!')

        ret, model = read_caffe_model(model_path, prototxt_path, params)
        if ret and model.get('layers', []):
            graph._attr['input_tensors'] = OrderedDict()
            for inp in model['inputs'][::-1]:
                model['layers'].insert(
                    0, {'name': inp, 'type': 'DATA', 'top': [inp]})
            for op_info in model['layers']:
                if op_info['type'] in ('INPUT', 'HDF5DATA', 'IMAGEDATA', 'ANNOTATEDDATA'):
                    op_info['type'] = 'DATA'
                elif op_info['type'] == 'PYTHON':
                    if op_info['python_param'].get('layer', '') \
                            and re.search(r'Layer$', op_info['python_param']['layer']):
                        op_info['type'] = op_info['python_param']['layer'].rstrip(
                            'Layer').upper()
                    else:
                        op_info['type'] = 'DATA'
                    extra_params = yaml.load(op_info['python_param'].get(
                        'param_str', ''), Loader=yaml.FullLoader)
                    if extra_params:
                        op_info['python_param'].update(extra_params)
                        op_info['python_param'].pop('param_str')
                elif op_info['type'] == 'INNERPRODUCT':
                    op_info['type'] = 'INNER_PRODUCT'

                if op_info['type'] == 'DATA':
                    if op_info['name'] not in params['input_names']:
                        try:
                            if op_info['name'] in model['inputs'] and len(model['inputs']) == len(model['shapes']):
                                index = model['inputs'].index(op_info['name'])
                                input_shape = model['shapes'][index]
                            elif 'input_param' in op_info:
                                input_shape = list(
                                    op_info['input_param']['shape'][0].get('dim', []))
                            else:
                                input_shape = list(model['shapes'])
                        except:
                            FATAL(
                                '[Parser]: Node (%s) does not have valid shape! Please check cfg or prototxt!' % op_info['name'])
                            input_shape = []
                        # replace tensor name with node name
                        if op_info['top'][0] in params['input_names']:
                            params['input_names'].remove(op_info['top'][0])
                        params['input_names'].append(op_info['name'])
                        params['input_shapes'].update(
                            {op_info['name']: input_shape})
                    else:
                        input_shape = params['input_shapes'][op_info['name']]
                    input_tensor = (np.random.ranf(
                        size=input_shape) * 2 - 1).astype(np.float32)

                    graph._attr['input_tensors'].update(
                        {op_info['name']: Tensor(name=op_info['name'], value=input_tensor)})

            if not params.get('input_names', []) or not params.get('input_shapes', {}):
                FATAL('[Parser]: Got invalid input_names or input_shapes for model %s!' %
                      params['model_name'])

            inplace_ops = defaultdict(list)
            for oi, op_info in enumerate(model['layers']):
                tops = op_info['top'] if 'top' in op_info else []
                name = op_info['name']
                if len(tops) == 1 and tops == op_info.get('bottom', []):
                    for i in range(oi):
                        if 'top' in model['layers'][i] and tops[0] in model['layers'][i]['top']:
                            inplace_ops[(model['layers'][i]['name'], i)].extend(
                                [(name, oi)])
                            break

            for k, v in inplace_ops.items():
                layer_name, layer_id = k
                cur_top = layer_name + '_inner'
                model['layers'][layer_id]['top'][0] = cur_top
                for j, (v_name, v_id) in enumerate(v):
                    model['layers'][v_id]['bottom'][0] = cur_top
                    if j < len(v) - 1:
                        cur_top = v_name + '_inner'
                        model['layers'][v_id]['top'][0] = cur_top

            out_tensor_operator_map = {}
            for oi, op_info in enumerate(model['layers']):
                tops = op_info['top'] if 'top' in op_info else []
                out_tensor_operator_map.update({k[0]: (oi, k[1]) for k in [(
                    top_name, out_port) for out_port, top_name in enumerate(tops)]})

            layer_map = OrderedDict()
            for l in model['layers']:
                op_name = get_valid_node_name(graph, l['name'])
                graph.add_node(op_name)
                layer_map.update({op_name: l})
                attr_dict = convert_attr_to_onnx(l)
                attr_dict.update(
                    {'data_format': params.get('input_data_format', 'NCHW')})
                NodeWrap(graph, op_name).replace_obj(
                    'Caffe'+l['type'], attr_dict)

                for in_port, in_tensor_name in enumerate(l.get('bottom', [])):
                    pred_op_index, out_port = out_tensor_operator_map[in_tensor_name]
                    pre_op_name = model['layers'][pred_op_index]['name']
                    edge_attr = {'src_out_port': out_port,
                                 'dst_in_port': in_port}
                    graph.add_edge(pre_op_name, op_name, **edge_attr)

                node_obj = NodeWrap(graph, op_name)['object']
                cur_in_ports = len(l.get('bottom', []))
                node_obj._attr.update(
                    {'blobs': Attribute('blobs', {'value': attr_dict.get('blobs', [])})})
                if isinstance(node_obj, OpHasWeights) and isinstance(node_obj, OpHasBiases) and len(attr_dict.get('blobs', [])) >= 2:
                    w, b = attr_dict['blobs'][0:2]
                    if b.size > 1:
                        b = np.squeeze(b)
                    if node_obj.type == 'CaffeINNER_PRODUCT':
                        if len(w.shape) == 4:
                            w = np.squeeze(w, axis=(0, 1))
                        if getattr(node_obj, 'transpose', False):
                            w = np.transpose(w)
                            node_obj.transpose = False
                    insert_constant(graph, op_name + '_weights',
                                    w, op_name, in_port=cur_in_ports)
                    insert_constant(graph, op_name + '_biases',
                                    b, op_name, in_port=cur_in_ports+1)
                elif isinstance(node_obj, OpHasWeights) and len(attr_dict.get('blobs', [])) >= 1:
                    w = attr_dict['blobs'][0]
                    if node_obj.type == 'CaffeINNER_PRODUCT':
                        if len(w.shape) == 4:
                            w = np.squeeze(w, axis=(0, 1))
                        if getattr(node_obj, 'transpose', False):
                            w = np.transpose(w)
                            node_obj.transpose = False
                    insert_constant(graph, op_name + '_weights',
                                    w, op_name, in_port=cur_in_ports)
                elif isinstance(node_obj, OpHasBiases) and len(attr_dict.get('blobs', [])) >= 1:
                    b = attr_dict['blobs'][0]
                    if b.size > 1:
                        b = np.squeeze(b)
                    insert_constant(graph, op_name + '_biases',
                                    b, op_name, in_port=cur_in_ports)
                if node_obj.type == 'CaffeBATCHNORM' and len(attr_dict.get('blobs', [])) == 3:
                    node_obj.scale_factor = attr_dict['blobs'][2]
                elif node_obj.type == 'CaffeCONVOLUTIONDEPTHWISE' and len(attr_dict.get('blobs', [])) >= 1:
                    node_obj.group = attr_dict['blobs'][0].shape[0]
                elif node_obj.type in ('CaffeBIAS', 'CaffeSCALE') and len(attr_dict.get('blobs', [])) >= 1:
                    for b in attr_dict['blobs']:
                        insert_constant(
                            graph, op_name + '_blob' + str(cur_in_ports), b, op_name, in_port=cur_in_ports)
                        cur_in_ports += 1
                elif node_obj.type == 'CaffeLSTM' and len(attr_dict.get('blobs', [])) >= 1:
                    node_obj.weights_list = attr_dict['blobs']

            if any([name not in layer_map and name not in model['inputs'] for name in params['input_names']]):
                FATAL('[Parser]: Got invalid input_names (%s) for model %s! Please check cfg file!' % (
                    string_list_to_string(params['input_names']), params['model_name']))

            graph._attr['output_names'] = copy.deepcopy(
                params.get('output_names', []))
            if graph._attr['output_names']:
                for i, out_name in enumerate(graph._attr['output_names']):
                    if not graph.has_node(out_name) or out_name not in layer_map:
                        if out_name not in out_tensor_operator_map:
                            FATAL(
                                '[Parser]: Output node or tensor (%s) does not exists in graph! Please check config file!' % out_name)
                        else:
                            out_node = model['layers'][out_tensor_operator_map[out_name][0]]['name']
                            WARN('[Parser]: the output name (%s) is not a node but a tensor. However, we will use the node (%s) as output node.' % (
                                out_name, out_node))
                            if out_node not in graph._attr['output_names']:
                                graph._attr['output_names'][i] = out_node
                            else:
                                graph._attr['output_names'][i] = None
                graph._attr['output_names'] = [
                    i for i in graph._attr['output_names'] if i is not None]
            else:
                node_names = [n for n in graph.nodes]
                succ = graph.succ
                for n in node_names:
                    if len(succ[n]) == 0:
                        graph._attr['output_names'].append(n)

            if not graph._attr['output_names']:
                FATAL('[Parser]: Got no output names for graph, cannot proceed!')

            for out_name in graph._attr['output_names']:
                out_layer = layer_map[out_name]
                out_edges = graph.sorted_out_edges(out_name)
                graph.remove_edges_from(out_edges)
                for i, top in enumerate(out_layer.get('top', [])):
                    noop_node_name = get_valid_node_name(
                        graph, out_name + '_noop_' + str(i))
                    out_edge_attr = {'src_out_port': i,
                                     'dst_in_port': 0, 'tensor': Tensor(name=top)}
                    graph.add_edge(out_name, noop_node_name, **out_edge_attr)
                    NodeWrap(graph, noop_node_name).replace_obj(
                        'Out', {'name': noop_node_name})

            # set out tensor names
            for out_name in graph._attr['output_tensor_names'][:]:
                if out_name not in out_tensor_operator_map \
                        and out_name not in layer_map \
                        and not graph.has_node(out_name):
                    WARN('[Parser]: Output tensor name (%s) does not exist in graph and will be removed, '
                         'please check config file!' % out_name)
                    graph._attr['output_tensor_names'].remove(out_name)
            for out_node in graph._attr['output_names']:
                out_edges = graph.sorted_out_edges(out_node, data=True)
                for _, _, out_attr in out_edges:
                    out_tensor_name = getattr(out_attr['tensor'], 'name', '')
                    if out_tensor_name and out_tensor_name not in graph._attr['output_tensor_names']:
                        graph._attr['output_tensor_names'].append(
                            out_tensor_name)

        else:
            FATAL(
                '[Parser]: Reads caffe model error or meets empty model or invalid params!')
    else:
        FATAL('[Parser]: Onnx version is too low, needs updating!')
    return graph
