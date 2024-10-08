# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import os
import copy
from collections import defaultdict, OrderedDict
import itertools
import numpy as np
import onnx
from ...graph.graph import Graph, SubGraph
from .buffer import get_model_content, parse_proto, get_tensor_content, get_value_content, get_node_content, get_graph_content
from ...graph.node_wrap import NodeWrap
from ...graph.graph_algo import get_valid_node_name, clear_redundant_nodes, has_path
from ...common.defs import Framework, Tensor, get_opset_version
from ...common.utils import is_file, get_version, multi_string_to_list
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


onnx_source_map = {
    '': Framework.NONE,
    'keras2onnx': Framework.TENSORFLOW,
    'onnxmltools': Framework.COREML,
    'pytorch': Framework.TORCH,
    'tf2onnx': Framework.TENSORFLOW

}


def build_subgraph(params, root_graph_info, opset_ver):

    nodes = params.get('nodes', [])

    for ni, node in enumerate(nodes):
        if not node['name']:
            if node['output'][0]['name']:
                node.update({'name': node['output'][0]['name']})
                nodes[ni].update(
                    {'name': node['output'][0]['name']})
            else:
                name = get_valid_node_name(root_graph_info['graph'], node['type'])
                node.update({'name': name})
                nodes[ni].update({'name': name})

    inputs = params.get('inputs', [])
    outputs = params.get('outputs', [])
    consts = params.get('consts', [])
    const_names = params.get('const_names', [])
    nodes_names = [n['name'] for n in nodes]

    root_graph = root_graph_info['graph']
    root_nodes = root_graph_info['nodes']
    root_inputs = root_graph_info['inputs']
    root_outputs = root_graph_info['outputs']
    root_const_names = root_graph_info['const_names']

    inputs_dict = {(info.get('name', ''), info.get('out_port', 0)): info.get('type', {}) for info in inputs}
    root_inputs_dict = {(info.get('name', ''), info.get('out_port', 0)): info.get('type', {}) for info in root_inputs}

    filter_nodes, filter_edges = [], []

    all_nodes = nodes + root_nodes
    out_tensor_operator_map = {}
    newly_added_consts = []
    for oi, op_info in enumerate(all_nodes):
        out_tensor_operator_map.update({k: oi for k in [(
            out_info['name'], out_info['out_port']) for out_info in op_info['output']]})
        node_name = op_info['name']
        if node_name in nodes_names \
                and node_name not in (const_names + root_const_names) \
                and op_info['type'] in ('Constant', 'ConstantOfShape') \
                and isinstance(op_info.get('value'), dict) \
                and op_info['value'].get('tensor', None) is not None:
            op_info.update({'tensor': op_info['value']['tensor']})
            newly_added_consts.append(op_info)
            const_names.append(node_name)

    all_consts = itertools.chain(consts, newly_added_consts)

    const_tensor_operator_map = {}
    all_const_idx = 0
    for const_node_name in root_const_names:
        const_tensor_operator_map.update({(const_node_name, 0): all_const_idx})
        all_const_idx += 1

    for op_info in all_consts:
        const_node_name = op_info['name']
        const_node_out_port = op_info['out_port']
        const_tensor_operator_map.update({(const_node_name, const_node_out_port): all_const_idx})
        if not root_graph.has_node(const_node_name):
            root_graph.add_node(const_node_name)
            const_node = NodeWrap(root_graph, const_node_name)
            const_attr = {'name': const_node_name,
                          'value': op_info['tensor'],
                          'data_format': params.get('input_data_format', 'NCHW'),
                          'opset_version': opset_ver}
            const_node.replace_obj('Constant', const_attr)
            filter_nodes.append(const_node_name)
        all_const_idx += 1

    for n in nodes:
        name = n.get('name', '')
        if name in const_names:
            continue
        n.update({'opset_version': opset_ver, 'in_subgraph': True})
        root_graph.add_node(name)
        op_type = n['type']
        if n.get('domain', '') == 'com.microsoft':
            op_type = op_type + 'Ms'
        NodeWrap(root_graph, name).replace_obj(op_type, n)
        filter_nodes.append(name)

        for in_port, in_tensor_info in enumerate(n['input']):
            in_tensor_name, in_tensor_out_port = in_tensor_info['name'], in_tensor_info['out_port']
            if (in_tensor_name, in_tensor_out_port) in root_inputs_dict:
                edge_attr = {'src_out_port': in_tensor_out_port,
                             'dst_in_port': in_port,
                             'tensor': Tensor(name=in_tensor_name)
                             }
                if in_tensor_name in root_graph._attr['input_tensors'] \
                        and root_graph._attr['input_tensors'][in_tensor_name].is_const:
                    edge_attr['tensor'].value = root_graph._attr['input_tensors'][in_tensor_name].value
                    edge_attr['tensor'].is_const = True
                root_graph.add_edge(in_tensor_name, name, **edge_attr)
            elif (in_tensor_name, in_tensor_out_port) in inputs_dict:
                assert in_tensor_name not in out_tensor_operator_map and in_tensor_out_port == 0
                if not root_graph.has_node(in_tensor_name):
                    root_graph.add_node(in_tensor_name)
                    NodeWrap(root_graph, in_tensor_name).replace_obj('Dummy', {'name': in_tensor_name})
                tensor_dtype = inputs_dict[(in_tensor_name, in_tensor_out_port)]['tensor_type'].get('elem_type', None)
                tensor_shape = inputs_dict[(in_tensor_name, in_tensor_out_port)]['tensor_type'].get('shape', None)
                edge_attr = {'src_out_port': in_tensor_out_port,
                             'dst_in_port': in_port,
                             'tensor': Tensor(name=in_tensor_name,
                                              shape=tensor_shape,
                                              dtype=np.dtype(tensor_dtype))
                             }
                root_graph.add_edge(in_tensor_name, name, **edge_attr)
            elif (in_tensor_name, in_tensor_out_port) in out_tensor_operator_map:
                pre_op_id = out_tensor_operator_map[(
                    in_tensor_name, in_tensor_out_port)]
                pre_op = all_nodes[pre_op_id]

                pre_op_name = pre_op['name'] if pre_op['name'] else pre_op['output'][0]['name']
                if in_tensor_out_port == 0:
                    in_tensor_out_port = [
                        out_info['name'] for out_info in pre_op['output']].index(in_tensor_name)
                edge_attr = {'src_out_port': in_tensor_out_port,
                             'dst_in_port': in_port}
                root_graph.add_edge(pre_op_name, name, **edge_attr)
                if pre_op_name in nodes_names and name in nodes_names:
                    filter_edges.append((pre_op_name, name, {
                                        'src_out_port': edge_attr['src_out_port'], 'dst_in_port': edge_attr['dst_in_port']}))
            elif (in_tensor_name, in_tensor_out_port) in const_tensor_operator_map:
                edge_attr = {'src_out_port': in_tensor_out_port,
                             'dst_in_port': in_port,
                             'tensor': Tensor(name=in_tensor_name,
                                              value=NodeWrap(root_graph, in_tensor_name)['object'].value,
                                              is_const=True)
                             }
                root_graph.add_edge(in_tensor_name, name, **edge_attr)

                if in_tensor_name in nodes_names and name in nodes_names:
                    filter_edges.append((in_tensor_name, name, {
                                        'src_out_port': edge_attr['src_out_port'], 'dst_in_port': edge_attr['dst_in_port']}))
            else:
                pass

    ret = SubGraph(root_graph, filter_nodes, filter_edges)
    for k in inputs_dict.keys():
        ret._attr['input_tensors'].update({k: None})

    for out_index, output in enumerate(outputs):
        output_info = (output['name'], output['out_port'])
        if output_info in out_tensor_operator_map:
            op_index_has_output = out_tensor_operator_map[output_info]
            assert op_index_has_output < len(
                all_nodes), 'Meet invalid op_index_has_output (%d) in build_subgraph' % op_index_has_output
            out_node_name = all_nodes[op_index_has_output]['name']
            if not out_node_name:
                out_node_name = output['name']
        elif output_info in const_tensor_operator_map:
            out_node_name = output['name']
        else:
            ERROR('[Parser]: Meets error in build_subgraph: Key %s is not in tensor_operator_map!' % str(output_info))
            continue
        assert ret.has_node(out_node_name), ('Node(%s) does not exist in build_subgraph!' % out_node_name)
        ret._attr['output_names'].append(out_node_name)
        ret._attr['output_ports'].update({out_node_name: output_info[1]})

    return ret


def attr_value_converter(attr_dict, source, root_graph_info=None):
    for key in attr_dict:
        if key == 'activations':
            acts = [act.upper() for act in attr_dict[key]]
            attr_dict.update({key: acts})
        elif key == 'pads':
            if len(attr_dict[key]) == 2:
                pads = attr_dict[key]
            elif len(attr_dict[key]) == 4:
                if source == Framework.MXNET:
                    pad_t, pad_b, pad_l, pad_r = attr_dict[key]
                else:
                    pad_t, pad_l, pad_b, pad_r = attr_dict[key]
                pads = [pad_t, pad_l, pad_b, pad_r]
            elif len(attr_dict[key]) in (6, 8):
                pads = attr_dict[key]
            else:
                ERROR('[Parser]: Invalid length of paddings %d in attr_value_converter!' %
                      len(attr_dict[key]))
                pads = attr_dict[key]
            attr_dict.update({key: pads})
        elif key == 'value':
            if attr_dict.get(key, None) is not None \
                    and isinstance(attr_dict[key], dict) \
                    and attr_dict[key].get('tensor', None) is not None:
                attr_dict.update({'value': attr_dict[key]['tensor']})
        elif key in ('then_branch', 'else_branch', 'body'):
            sub_graph = build_subgraph(
                attr_dict[key], root_graph_info, attr_dict.get('opset_version', 1))
            attr_dict.update({key: sub_graph})


def convert_onnx_to_graph(graph, model_path, params):
    '''Parse the onnx model into a graph structure.'''
    graph._attr['framework'] = Framework.ONNX
    graph._attr['output_tensor_names'] = params.get('output_tensor_names', [])
    graph._attr['output_names'] = copy.deepcopy(params.get('output_names', []))
    graph._attr['tensor_counter'] = defaultdict(int)
    force_not_quantize = True \
        if ((params.get('force_float_ir', 'false').lower() == 'true')
            or (params.get('compat_quantized_model', 'true').lower() == 'false')) \
        else False

    graph._attr['subgraph_output_names'] = OrderedDict()

    meta_ret = True
    consumer_ver = get_version(onnx)
    if consumer_ver >= 1.04:
        model = None
        try:
            model = onnx.load(model_path, load_external_data=False)
            try:
                onnx.checker.check_model(model)
            except Exception as e:
                WARN(
                    '[Parser]: Meets issue(%s) in onnx.check_model, but still proceeds!' % str(e))
        except IOError:
            ERROR('[Parser]: Reading onnx file %s meets error in convert_onnx_to_graph!' % (
                model_path))
            meta_ret = False
        except Exception as e:
            ERROR('[Parser]: Reading onnx file %s meets exception %s in convert_onnx_to_graph.' %
                  (model_path, str(e)))
            meta_ret = False

        if meta_ret:
            model_dir = os.path.dirname(model_path)
            model_attr = get_model_content(model)
            params['source'] = onnx_source_map.get(
                model_attr['producer_name'].lower(), Framework.NONE)
            opset_ver = None
            opset_import_map = dict()
            for version_dict in model_attr['opset_import']:
                domain = version_dict.get('domain', '')
                version = version_dict.get('version', None)
                assert version is not None and version > 0, \
                    'Meets invalid version for domain %s of opset_import in convert_onnx_to_graph!' % domain
                if not domain or domain == 'ai.onnx':
                    opset_ver = version
                else:
                    opset_import_map[domain] = version
            if opset_ver is None:
                WARN('[Parser]: Meets empty opset_import in convert_onnx_to_graph! Please check model file.')
                opset_ver = get_opset_version(consumer_ver)
                WARN('[Parser]: Try to use suggested opset version(%s) for current Onnx!' % opset_ver)
            graph._attr['opset_version'] = max([opset_ver] + list(opset_import_map.values()))
            g = model.graph
            if len(g.node) > 0:
                g_content = get_graph_content(g, model_dir, params=params)
                const_values = g_content['consts']
                const_names = g_content['const_names']
                const_names = {c_info['name']: None for c_info in const_names}
                inputs = g_content['inputs']
                outputs = g_content['outputs']
                nodes = g_content['nodes']

                root_info = OrderedDict()
                root_info.update({'graph': graph,
                                  'inputs': copy.deepcopy(inputs),
                                  'outputs': copy.deepcopy(outputs),
                                  'nodes': nodes,
                                  })
                graph._attr['input_tensors'] = OrderedDict()

                for single_input in inputs:
                    if single_input['name'] not in const_names:
                        nodes.insert(0, {'name': single_input['name'],
                                         'out_port': 0,
                                         'type': 'Input',
                                         'input': [],
                                         'output': [{'name': single_input['name'], 'out_port': 0}]}
                                     )
                        input_shape = single_input['type']['tensor_type']['shape'].tolist()
                        if 'input_shapes' in params and len(input_shape) >= 1:
                            name = single_input['name']
                            if name not in params['input_shapes']:
                                # for support use node name
                                for node in nodes:
                                    for inp in node.get('input', []):
                                        if name == inp.get('name', ''):
                                            name = node.get('name', name)
                            elif params['input_shapes'][name] is not None \
                                    and len(input_shape) == len(params['input_shapes'][name]):
                                input_shape[:] = params['input_shapes'][name][:]
                        if 0 in input_shape and single_input['name'] not in params.get('input_shapes', {}):
                            WARN('[Parser]: Shape 0 found in Input node(%s), please check config file!' %
                                 single_input['name'])

                        if single_input['name'] in params['input_npy']:
                            input_tensor = params['input_npy'][single_input['name']]
                            is_const = True
                        else:
                            input_type = np.dtype(
                                single_input['type']['tensor_type']['elem_type'])
                            if input_type.name in ('int32', 'int64'):
                                input_tensor = np.zeros(shape=input_shape).astype(input_type)
                            elif input_type.name in ('float32', 'float64',):
                                input_tensor = np.random.ranf(size=input_shape).astype(input_type)
                            elif input_type.name == 'bool':
                                input_tensor = np.random.randint(
                                    0, 2, size=input_shape).astype(bool)
                            else:
                                input_tensor = np.random.ranf(
                                    size=input_shape).astype(input_type)
                            is_const = False

                        graph._attr['input_tensors'].update({
                            single_input['name']: Tensor(
                                name=single_input['name'], value=input_tensor, is_const=is_const)
                        })

                out_tensor_operator_map = {}
                for oi, op_info in enumerate(nodes):
                    out_tensor_operator_map.update({k: oi for k in [(
                        out_info['name'], out_info['out_port']) for out_info in op_info['output']]})
                tensor_name_map = {t_name: (port, op_id) for (
                    t_name, port), op_id in out_tensor_operator_map.items()}

                if not inputs:
                    WARN('[Parser]: The model does not have any inputs! Ignore input names from config file.')
                    params['input_names'] = []
                else:
                    input_names = copy.deepcopy(params.get('input_names', []))
                    for name in input_names:
                        if name not in tensor_name_map and name not in [n.get('name', '') for n in nodes]:
                            WARN(
                                '[Parser]: Input name (%s) does not exit in node names or tensor names! Will ignore it!' % name)
                            params['input_names'].remove(name)

                anchor_tensors = params.get('anchor_tensor_name', [])
                anchor_tensors_value = []
                if anchor_tensors:
                    anchor_tensors = multi_string_to_list(anchor_tensors)
                    anchor_tensors_value = [None] * len(anchor_tensors)
                graph._attr['anchors'] = None

                graph_is_quantized = False

                nodes_name = [n['name'] for n in nodes]

                for c in const_values:
                    c_name = get_valid_node_name(graph, c['name'], nodes_name)  # const tensor name maybe same with node
                    const_names[c['name']] = c_name
                    graph.add_node(c_name)
                    NodeWrap(graph, c_name).replace_obj('Constant', {'name': c_name,
                                                                     'value': c['tensor'],
                                                                     'data_format': params.get('input_data_format', 'NCHW'),
                                                                     'opset_version': opset_ver})
                root_info.update({'const_names': list(const_names.values())})

                in_tensor_names = set()
                for ni, node in enumerate(nodes):
                    op_attr = {k: v for k, v in node.items()}
                    node_type = node['type']
                    domain = op_attr.get('domain', '')
                    if not domain or domain == 'ai.onnx':
                        opset_version = opset_ver
                    else:
                        assert domain in opset_import_map, \
                            'Meets domain(%s) with unknown opset version in convert_onnx_to_graph!' % domain
                        opset_version = opset_import_map[domain]
                        if domain == 'com.microsoft':
                            node_type = node_type + 'Ms'
                    quantize = False
                    if not force_not_quantize:
                        if node['type'] in ('QuantizeLinear', 'QGemm') or node['type'].startswith('QLinear'):
                            quantize = True
                            graph_is_quantized = True
                        elif node['type'] == 'DequantizeLinear':
                            graph_is_quantized = True
                    op_attr.update({'data_format': params.get('input_data_format', 'NCHW'),
                                    'opset_version': opset_version,
                                    'quantize': quantize
                                    })
                    attr_value_converter(
                        op_attr, params['source'], root_graph_info=root_info)
                    if not op_attr['name']:
                        if node['output'][0]['name']:
                            op_attr.update({'name': node['output'][0]['name']})
                            nodes[ni].update(
                                {'name': node['output'][0]['name']})
                        else:
                            name = get_valid_node_name(graph, op_attr['type'])
                            op_attr.update({'name': name})
                            nodes[ni].update({'name': name})
                    op_name = op_attr['name']
                    if not graph.has_node(op_name):
                        graph.add_node(op_name)
                    NodeWrap(graph, op_name).replace_obj(node_type, op_attr)
                    for in_port, in_tensor_info in enumerate(node['input']):
                        edge_attr = {}
                        in_tensor_name, in_tensor_out_port = in_tensor_info[
                            'name'], in_tensor_info['out_port']
                        in_tensor_names.add(in_tensor_name)
                        if not in_tensor_name:
                            in_tensor_name = get_valid_node_name(
                                graph, 'no_name_const')
                            assert not graph.has_node(
                                in_tensor_name), 'Node %s does not exist in convert_onnx_to_graph.' % (in_tensor_name)
                            graph.add_node(in_tensor_name)
                            dummy_node = NodeWrap(graph, in_tensor_name)
                            dummy_attr = {'name': in_tensor_name}
                            dummy_node.replace_obj('Dummy', dummy_attr)
                            edge_attr = {'src_out_port': in_tensor_out_port, 'dst_in_port': in_port, 'tensor': Tensor(
                                name=in_tensor_name, is_const=True)}
                            graph.add_edge(
                                in_tensor_name, op_name, **edge_attr)
                        elif in_tensor_name in const_names:
                            real_const_name = const_names[in_tensor_name]
                            edge_attr = {'src_out_port': 0,
                                         'dst_in_port': in_port,
                                         'tensor': Tensor(name=real_const_name,
                                                          value=NodeWrap(graph, real_const_name)['object'].value,
                                                          is_const=True)
                                         }
                            graph.add_edge(real_const_name, op_name, **edge_attr)
                        else:
                            if in_tensor_name in tensor_name_map:
                                in_tensor_out_port = tensor_name_map[in_tensor_name][0]
                            pre_op_id = out_tensor_operator_map[(
                                in_tensor_name, in_tensor_out_port)]
                            pre_op = nodes[pre_op_id]
                            pre_op_name = pre_op['name'] if pre_op['name'] else pre_op['output'][0]['name']
                            if in_tensor_out_port == 0:
                                in_tensor_out_port = [
                                    out_info['name'] for out_info in pre_op['output']].index(in_tensor_name)
                            edge_attr = {
                                'src_out_port': in_tensor_out_port, 'dst_in_port': in_port}
                            if pre_op.get('type', '') == 'Constant' and pre_op.get('value', None) is not None:
                                edge_attr.update({'tensor': Tensor(name=in_tensor_name,
                                                                   value=pre_op['value']['tensor'],
                                                                   is_const=True)}
                                                 )
                            elif pre_op_name in graph._attr['input_tensors'] \
                                    and graph._attr['input_tensors'][pre_op_name].is_const:
                                edge_attr.update({'tensor': Tensor(name=in_tensor_name,
                                                                   value=graph._attr['input_tensors'][pre_op_name].value,
                                                                   is_const=True)}
                                                 )
                            else:
                                edge_attr.update({'tensor': Tensor(name=in_tensor_name)})
                            graph.add_edge(pre_op_name, op_name, **edge_attr)
                        if anchor_tensors \
                                and any(val is None for val in anchor_tensors_value) \
                                and in_tensor_name in anchor_tensors \
                                and edge_attr.get('tensor', None) is not None:
                            if not edge_attr['tensor'].is_const:
                                WARN('[Parser]: Meet non-const anchor (%s) in convert_onnx_to_graph!' % in_tensor_name)
                            else:
                                index = anchor_tensors.index(in_tensor_name)
                                anchor_tensors_value[index] = edge_attr['tensor'].value

                    if node['type'] == 'If':
                        if_in_port = 1
                        for b in ['then_branch', 'else_branch']:
                            branch = getattr(
                                NodeWrap(graph, op_name)['object'], b)
                            for branch_out in branch._attr['output_names']:
                                branch._attr['root_in_ports'].append(
                                    if_in_port)
                                edge_attr = {'src_out_port': 0,
                                             'dst_in_port': if_in_port}
                                graph.add_edge(
                                    branch_out, op_name, **edge_attr)
                                if_in_port += 1
                                graph._attr['subgraph_output_names'].update(
                                    branch._attr['output_ports'])
                    elif node['type'] == 'Loop':
                        loop_obj = NodeWrap(graph, op_name)['object']
                        if loop_obj is not None:
                            loop_in_shape = loop_obj.get_input_shapes()
                            loop_in_tensors = loop_obj.get_input_tensors()
                            loop_in_port = len(loop_obj.get_in_ports())

                            if len(loop_in_shape) == 3 \
                                    and len(loop_in_tensors) == 3 \
                                    and len(loop_obj.body._attr['input_tensors']) == 3 \
                                    and len(loop_obj.body._attr['output_names']) == 3:
                                if not loop_in_tensors[1]:
                                    continue
                                loop_in_edges = graph.sorted_in_edges(
                                    op_name, data=True)
                                for index, (body_inp, in_port) in enumerate(loop_obj.body._attr['input_tensors']):
                                    body_in_obj = NodeWrap(
                                        graph, body_inp)['object']
                                    if body_in_obj is None:  # could be a standalone node
                                        continue
                                    body_inp_port = body_in_obj.get_in_ports()
                                    if len(body_inp_port) != 0 or len(loop_in_edges) != 3:
                                        continue
                                    src, _, in_attr = loop_in_edges[index]
                                    new_in_attr = copy.deepcopy(in_attr)
                                    new_in_attr['dst_in_port'] = 0
                                    graph.add_edge(
                                        src, body_inp, **new_in_attr)

                                # TODO: need to support complex condition loop.
                                for body_out in loop_obj.body._attr['output_names'][1:-1]:
                                    loop_obj.body._attr['root_in_ports'].append(
                                        loop_in_port)
                                    edge_attr = {'src_out_port': 0,
                                                 'dst_in_port': loop_in_port}
                                    graph.add_edge(
                                        body_out, op_name, **edge_attr)
                                    loop_in_port += 1
                            else:
                                for body_out in loop_obj.body._attr['output_names']:
                                    loop_obj.body._attr['root_in_ports'].append(
                                        loop_in_port)
                                    edge_attr = {'src_out_port': 0,
                                                 'dst_in_port': loop_in_port}
                                    graph.add_edge(
                                        body_out, op_name, **edge_attr)
                                    loop_in_port += 1

                            graph._attr['subgraph_output_names'].update(
                                loop_obj.body._attr['output_ports'])
                        else:
                            ERROR(
                                '[Parser]: Meets invalid Loop Op(%s) in convert_onnx_to_graph!' % op_name)

                graph._attr['quantize'] = graph_is_quantized

                if anchor_tensors and anchor_tensors_value:
                    try:
                        if all(len(value.shape) > 1 and value.shape[-1] == 4 for value in anchor_tensors_value):
                            concat_axis = -2
                        else:
                            concat_axis = -1
                        graph._attr['anchors'] = np.reshape(np.concatenate(
                            anchor_tensors_value, axis=concat_axis), [-1, 4])
                    except:
                        WARN('[Parser]: Cannot get anchor tensors (%s) in convert_onnx_to_graph!' % str(anchor_tensors))

                if graph._attr['output_names']:
                    removing_names = []
                    for i, out_name in enumerate(graph._attr['output_names']):
                        if graph.has_node(out_name) or out_name in tensor_name_map.keys():
                            out_port, out_node_name = 0, copy.deepcopy(
                                out_name)
                            if not graph.has_node(out_name):
                                out_port, op_id = tensor_name_map[out_name]
                                out_node_name = nodes[op_id].get('name', '')
                                if out_node_name:
                                    if out_node_name not in graph._attr['output_names']:
                                        if out_node_name not in graph._attr['output_names']:
                                            graph._attr['output_names'][i] = out_node_name
                                        else:
                                            removing_names.append(
                                                graph._attr['output_names'][i])
                                        WARN('[Parser]: The output name %s is not a node but a tensor. However, we will use the node %s as output node.'
                                             % (out_name, out_node_name))
                                    else:
                                        removing_names.append(out_name)
                                else:
                                    FATAL(
                                        '[Parser]: Meets invalid node name:%s!' % out_node_name)
                            noop_node_name = get_valid_node_name(
                                graph, out_node_name + '_noop_' + str(out_port))
                            graph.add_node(noop_node_name)
                            noop_node = NodeWrap(graph, noop_node_name)
                            noop_node.replace_obj(
                                'Out', {'name': noop_node_name})
                            out_edge_attr = {
                                'src_out_port': out_port, 'dst_in_port': 0, 'tensor': Tensor(name=out_name)}
                            graph.add_edge(
                                out_node_name, noop_node_name, **out_edge_attr)
                            if out_name in params.get('output_tensor_map', {}):
                                params['output_tensor_map'][out_name] = [noop_node_name]
                            elif out_node_name in params.get('output_tensor_map', {}):
                                params['output_tensor_map'][out_node_name] = [noop_node_name]
                        else:
                            FATAL(
                                '[Parser]: Graph does not contain such a node/tensor name:%s' % out_name)
                    for rm in removing_names:
                        graph._attr['output_names'].remove(rm)

                    removing_sg_out = []
                    for sub_out in graph._attr['subgraph_output_names']:
                        if graph.has_node(sub_out):
                            try:
                                if all(not has_path(graph, sub_out, go) for go in graph._attr['output_names']):
                                    removing_sg_out.append(sub_out)
                            except:
                                pass
                        else:
                            removing_sg_out.append(sub_out)
                    for rn in removing_sg_out:
                        graph._attr['subgraph_output_names'].pop(rn)

                    for subgraph_out in graph._attr['subgraph_output_names']:
                        out_op_name = get_valid_node_name(graph, subgraph_out + '_out')
                        out_edge_attr = {
                            'src_out_port': graph._attr['subgraph_output_names'][subgraph_out],
                            'dst_in_port': 0,
                            'tensor': Tensor(name=subgraph_out)}
                        graph.add_edge(subgraph_out, out_op_name, **out_edge_attr)

                    graph._attr['output_names'] += list(graph._attr['subgraph_output_names'])

                    # convert node name to tensor name
                    if any([t not in tensor_name_map for t in graph._attr['output_tensor_names']]) \
                            and all([graph.has_node(n) for n in graph._attr['output_tensor_names']]):
                        out_tensors = []
                        for n in graph._attr['output_tensor_names']:
                            out_edges = graph.sorted_out_edges(n, data=True)
                            for u, v, d in out_edges:
                                t = d.get('tensor', None)
                                if t is not None:
                                    out_tensors.append(t.name)
                                else:
                                    pass
                        graph._attr['output_tensor_names'] = out_tensors
                else:
                    for out_index, output in enumerate(outputs):
                        if (output['name'], output['out_port']) not in out_tensor_operator_map \
                                and output['name'] in const_names:
                            # Ignore the const outputs because they will make the graph not connected
                            continue
                        op_index_has_output = out_tensor_operator_map[(
                            output['name'], output['out_port'])]
                        out_node_name = nodes[op_index_has_output]['name']
                        if not out_node_name:
                            out_node_name = output['name']
                        noop_node_name = out_node_name + \
                            '_noop_' + str(out_index)
                        assert graph.has_node(out_node_name) and not graph.has_node(
                            noop_node_name), 'Node(%s) does not exist in convert_onnx_to_graph.' % (out_node_name)
                        if out_node_name not in graph._attr['output_names']:
                            graph._attr['output_names'].append(out_node_name)

                        graph.add_node(noop_node_name)
                        noop_node = NodeWrap(graph, noop_node_name)
                        noop_node.replace_obj('Out', {'name': noop_node_name})
                        pending_out_port = output['out_port']
                        current_out_ports = NodeWrap(graph, out_node_name)[
                            'object'].get_out_ports()
                        if pending_out_port in current_out_ports:
                            found_non_out_node = False
                            cur_out_edges = graph.sorted_out_edges(
                                out_node_name, data=True)
                            for _, dst, out_attr in cur_out_edges:
                                if out_attr.get('src_out_port', 0) == pending_out_port \
                                        and NodeWrap(graph, dst)['object'] is not None \
                                        and NodeWrap(graph, dst)['object'].type != 'Out':
                                    found_non_out_node = True
                                    break
                            if not found_non_out_node:
                                pending_out_port = max(current_out_ports) + 1
                        out_edge_attr = {
                            'src_out_port': pending_out_port, 'dst_in_port': 0, 'tensor': Tensor(name=output['name'])}
                        graph.add_edge(
                            out_node_name, noop_node_name, **out_edge_attr)

                    for subgraph_out in graph._attr['subgraph_output_names']:
                        if subgraph_out not in graph._attr['output_names']:
                            graph._attr['output_names'].append(subgraph_out)
                        out_op_name = get_valid_node_name(graph, subgraph_out + '_out')
                        out_edge_attr = {
                            'src_out_port': graph._attr['subgraph_output_names'][subgraph_out],
                            'dst_in_port': 0,
                            'tensor': Tensor(name=subgraph_out)}
                        graph.add_edge(subgraph_out, out_op_name, **out_edge_attr)

                # Add Out node after the nodes with output tensors but without successors.
                for (out_tensor_name, out_port), node_idx in out_tensor_operator_map.items():
                    node_name = nodes[node_idx]['name']
                    if out_tensor_name in in_tensor_names or node_name in graph._attr['output_names']:
                        continue
                    out_node_name = out_tensor_name + '_out_' + str(out_port)
                    graph.add_edge(node_name, out_node_name, **
                                   {'src_out_port': out_port, 'dst_in_port': 0, 'tensor': Tensor(name=out_tensor_name)})
                    NodeWrap(graph, out_node_name).replace_obj('Out', {'name': out_node_name})

                if not graph._attr['output_names']:
                    ERROR('[Parser]: Got no output names for graph, cannot proceed!')
                else:
                    if any(in_name not in graph._attr['input_tensors'] for in_name in params['input_names']):
                        for in_name in params['input_names']:
                            if graph.has_node(in_name) \
                                    and in_name not in graph._attr['input_tensors']:
                                out_edges = graph.sorted_out_edges(in_name, data=True)
                                if len(out_edges) > 0:
                                    in_edges = graph.sorted_in_edges(in_name)
                                    graph.remove_edges_from(in_edges)
                                    obj = NodeWrap(graph, in_name)['object']
                                    if obj is not None:
                                        NodeWrap(graph, in_name).replace_obj('Input', obj.copied_attr())
                                    else:
                                        NodeWrap(graph, in_name).replace_obj('Input', {'name': in_name})
                                    if out_edges[0][2]['tensor'] is not None and out_edges[0][2]['tensor'].value is not None:
                                        graph._attr['input_tensors'].update({in_name: out_edges[0][2]['tensor']})
                                    else:
                                        cur_shape = params['input_shapes'][in_name]
                                        if cur_shape is None:
                                            ERROR(
                                                '[Parser]: Shape of Input(%s) is unknown! Please provide input_shape in cfg file!' % in_name)
                                            continue
                                        graph._attr['input_tensors'].update(
                                            {in_name: Tensor(name=in_name, value=np.random.ranf(cur_shape).astype(np.float32))})
                        graph._attr['input_tensors'] = OrderedDict(
                            {k: v for k, v in graph._attr['input_tensors'].items() if graph.has_node(k)})
                    clear_redundant_nodes(graph)
            else:
                WARN('[Parser]: Meets empty graph in convert_onnx_to_graph!')
    else:
        ERROR('[Parser]: Onnx version is too low, needs updating!')

    return graph
