# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import os
import copy
from collections import defaultdict, OrderedDict
import itertools
import numpy as np
import onnx
from ...graph.graph import Graph, SubGraph
from .buffer import get_model_content, parse_proto, get_tensor_content, get_value_content, get_node_content, \
    get_graph_content
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


def build_subgraph(name, g_content, root_graph, parent_graph_info, a_nodes, opset_ver):
    sub_graph = SubGraph(name=name)
    sub_graph._attr['framework'] = root_graph._attr['framework']

    nodes = g_content.get('nodes', [])

    for ni, node in enumerate(nodes):
        if not node['name']:
            if node['output'][0]['name']:
                node.update({'name': node['output'][0]['name']})
                nodes[ni].update(
                    {'name': node['output'][0]['name']})
            else:
                name = get_valid_node_name(sub_graph, node['type'])
                node.update({'name': name})
                nodes[ni].update({'name': name})

    inputs = g_content.get('inputs', [])
    outputs = g_content.get('outputs', [])
    consts = g_content.get('consts', [])
    const_names = g_content.get('const_names', [])
    const_names = {c_info['name']: None for c_info in const_names}
    nodes_names = [n['name'] for n in nodes]

    parent_graph = parent_graph_info['graph']
    parent_nodes = parent_graph_info['nodes']
    # parent_inputs = parent_graph_info['inputs']
    # parent_outputs = parent_graph_info['outputs']
    parent_const_names = parent_graph_info['const_names']

    sub_graph_info = OrderedDict()
    sub_graph_info.update({'graph': sub_graph,
                           'inputs': copy.deepcopy(inputs),
                           'outputs': copy.deepcopy(outputs),
                           'nodes': nodes,
                           })

    sub_graph._root = root_graph
    sub_graph._attr['parent_graph'] = parent_graph

    for single_input in inputs:
        if single_input['name'] not in const_names:
            nodes.insert(0, {'name': single_input['name'],
                             'out_port': 0,
                             'type': 'Input',
                             'input': [],
                             'output': [{'name': single_input['name'], 'out_port': 0}]}
                         )
            input_shape = single_input['type']['tensor_type']['shape'].tolist()
            if 0 in input_shape:
                WARN('[Parser]: Shape 0 found in Input node(%s), please check subgraph!' %
                     single_input['name'])

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

            sub_graph._attr['input_tensors'].update({
                single_input['name']: Tensor(
                    name=single_input['name'], value=input_tensor, is_const=is_const)
            })

    a_nodes += nodes
    out_tensor_operator_map = {}
    for oi, op_info in enumerate(a_nodes):
        out_tensor_operator_map.update({k: oi for k in [(
            out_info['name'], out_info['out_port']) for out_info in op_info['output']]})
        node_name = op_info['name']
        if node_name in nodes_names \
                and node_name not in (list(const_names.keys()) + parent_const_names) \
                and op_info['type'] in ('Constant', 'ConstantOfShape') \
                and isinstance(op_info.get('value'), dict) \
                and op_info['value'].get('tensor', None) is not None:
            op_info.update({'tensor': op_info['value']['tensor']})
            const_names[node_name] = node_name
    tensor_name_map = {t_name: (port, op_id) for (
        t_name, port), op_id in out_tensor_operator_map.items()}

    for c in consts:
        c_name = get_valid_node_name(sub_graph, c['name'], nodes_names)  # const tensor name maybe same with node
        const_names[c['name']] = c_name
        sub_graph.add_node(c_name)
        NodeWrap(sub_graph, c_name).replace_obj('Constant', {'name': c_name,
                                                             'value': c['tensor'],
                                                             'data_format': root_graph._attr['data_format'],
                                                             'opset_version': opset_ver})
    sub_graph_info.update({'const_names': list(const_names.values())})

    all_consts = list(const_names.values()) + parent_const_names

    if isinstance(parent_graph, SubGraph):
        all_consts += root_graph._attr['const_names']

    in_tensor_names = set()
    graph_is_quantized = False
    for ni, node in enumerate(nodes):
        op_attr = {k: v for k, v in node.items()}
        node_type = node['type']
        domain = op_attr.get('domain', '')
        if domain == 'com.microsoft':
            node_type = node_type + 'Ms'
        opset_version = opset_ver

        quantize = False
        if not root_graph._attr['force_not_quantize']:
            if node['type'] in ('QuantizeLinear', 'QGemm') or node['type'].startswith('QLinear'):
                quantize = True
                graph_is_quantized = True
            elif node['type'] == 'DequantizeLinear':
                graph_is_quantized = True
        op_attr.update({'data_format': root_graph._attr['data_format'],
                        'opset_version': opset_version,
                        'quantize': quantize
                        })
        if not op_attr['name']:
            if node['output'][0]['name']:
                op_attr.update({'name': node['output'][0]['name']})
                nodes[ni].update(
                    {'name': node['output'][0]['name']})
            else:
                name = get_valid_node_name(sub_graph, op_attr['type'])
                op_attr.update({'name': name})
                nodes[ni].update({'name': name})
        attr_value_converter(
            op_attr, root_graph._attr['source'], root_graph,
            a_nodes=a_nodes,
            parent_graph_info=sub_graph_info)
        op_name = op_attr['name']
        if not sub_graph.has_node(op_name):
            sub_graph.add_node(op_name)
        NodeWrap(sub_graph, op_name).replace_obj(node_type, op_attr)
        if op_name in root_graph._attr['node_in_subgraphs']:
            op_obj = sub_graph.nodes[op_name]['object']
            op_obj.in_subgraph = True
            op_obj.subgraphs.extend(root_graph._attr['node_in_subgraphs'][op_name])
        for in_port, in_tensor_info in enumerate(node['input']):
            in_tensor_name, in_tensor_out_port = in_tensor_info[
                'name'], in_tensor_info['out_port']
            in_tensor_names.add(in_tensor_name)
            if not in_tensor_name:
                in_tensor_name = get_valid_node_name(
                    sub_graph, 'no_name_const')
                assert not sub_graph.has_node(
                    in_tensor_name), 'Node %s does not exist in build_subgraph.' % (in_tensor_name)
                sub_graph.add_node(in_tensor_name)
                dummy_node = NodeWrap(sub_graph, in_tensor_name)
                dummy_attr = {'name': in_tensor_name}
                dummy_node.replace_obj('DummyInput', dummy_attr)
                edge_attr = {'src_out_port': in_tensor_out_port, 'dst_in_port': in_port, 'tensor': Tensor(
                    name=in_tensor_name, is_const=True)}
                sub_graph.add_edge(
                    in_tensor_name, op_name, **edge_attr)
            elif in_tensor_name in all_consts:
                if in_tensor_name in const_names:
                    real_const_name = const_names[in_tensor_name]
                    edge_attr = {'src_out_port': 0,
                                 'dst_in_port': in_port,
                                 'tensor': Tensor(name=real_const_name,
                                                  value=NodeWrap(sub_graph, real_const_name)['object'].value,
                                                  is_const=True)
                                 }
                    sub_graph.add_edge(real_const_name, op_name, **edge_attr)
                else:
                    # from root const
                    if sub_graph.has_node(in_tensor_name):
                        edge_attr = {'src_out_port': in_tensor_out_port, 'dst_in_port': in_port, 'tensor': Tensor(
                            name=in_tensor_name, is_const=True)}
                        sub_graph.add_edge(
                            in_tensor_name, op_name, **edge_attr)
                    else:
                        n_name = get_valid_node_name(sub_graph, in_tensor_name)
                        sub_graph.add_node(n_name)
                        if in_tensor_name in parent_graph.nodes:
                            op_obj = parent_graph.nodes[in_tensor_name]['object']
                        else:
                            # from root graph
                            op_obj = root_graph.nodes[in_tensor_name]['object']
                        cons_value = op_obj.value
                        op_obj.in_subgraph = True
                        if sub_graph.name not in op_obj.subgraphs:
                            op_obj.subgraphs.append(sub_graph.name)

                        NodeWrap(sub_graph, n_name).replace_obj('DummyInput', {'name': n_name})
                        edge_attr = {'src_out_port': in_tensor_out_port, 'dst_in_port': in_port, 'tensor': Tensor(
                            name=in_tensor_name, shape=cons_value.shape, is_const=True)}
                        sub_graph.add_edge(
                            n_name, op_name, **edge_attr)
            else:
                if in_tensor_name in tensor_name_map:
                    in_tensor_out_port = tensor_name_map[in_tensor_name][0]
                pre_op_id = out_tensor_operator_map[(
                    in_tensor_name, in_tensor_out_port)]
                pre_op = a_nodes[pre_op_id]
                pre_op_name = pre_op['name'] if pre_op['name'] else pre_op['output'][0]['name']
                if not sub_graph.has_node(pre_op_name):
                    # root graph op
                    if pre_op_name in parent_graph.nodes:
                        op_obj = parent_graph.nodes[pre_op_name]['object']
                        op_obj.in_subgraph = True
                        if sub_graph.name not in op_obj.subgraphs:
                            op_obj.subgraphs.append(sub_graph.name)
                    elif pre_op_name in root_graph.nodes:
                        op_obj = root_graph.nodes[pre_op_name]['object']
                        op_obj.in_subgraph = True
                        if sub_graph.name not in op_obj.subgraphs:
                            op_obj.subgraphs.append(sub_graph.name)
                    else:
                        parent_graph_raw_node_names = [n['name'] for n in parent_nodes]
                        assert pre_op_name in parent_graph_raw_node_names
                        if pre_op_name not in root_graph._attr['node_in_subgraphs']:
                            root_graph._attr['node_in_subgraphs'][pre_op_name] = [sub_graph.name]
                        else:
                            if sub_graph.name not in root_graph._attr['node_in_subgraphs'][pre_op_name]:
                                root_graph._attr['node_in_subgraphs'][pre_op_name].append(sub_graph.name)
                    n_name = get_valid_node_name(
                        sub_graph, pre_op_name)
                    sub_graph.add_node(n_name)
                    dummy_node = NodeWrap(sub_graph, n_name)
                    dummy_attr = {'name': n_name}
                    dummy_node.replace_obj('DummyInput', dummy_attr)
                    edge_attr = {'src_out_port': in_tensor_out_port, 'dst_in_port': in_port, 'tensor': Tensor(
                        name=in_tensor_name, is_const=False)}
                    sub_graph.add_edge(
                        n_name, op_name, **edge_attr)
                else:
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
                    elif pre_op_name in sub_graph._attr['input_tensors'] \
                            and sub_graph._attr['input_tensors'][pre_op_name].is_const:
                        edge_attr.update({'tensor': Tensor(name=in_tensor_name,
                                                           value=sub_graph._attr['input_tensors'][pre_op_name].value,
                                                           is_const=True)}
                                         )
                    else:
                        edge_attr.update({'tensor': Tensor(name=in_tensor_name)})
                    sub_graph.add_edge(pre_op_name, op_name, **edge_attr)
    sub_graph._attr['quantize'] = graph_is_quantized

    for out_index, output in enumerate(outputs):
        if (output['name'], output['out_port']) not in out_tensor_operator_map \
                and output['name'] in const_names:
            # In sub graph, not connected graph is allowed
            out_node_name = output['name']
        else:
            op_index_has_output = out_tensor_operator_map[(
                output['name'], output['out_port'])]
            out_node_name = a_nodes[op_index_has_output]['name']
            if not out_node_name:
                out_node_name = output['name']
        noop_node_name = out_node_name + '_noop_' + str(out_index)
        assert sub_graph.has_node(out_node_name) and not sub_graph.has_node(
            noop_node_name), 'Node(%s) does not exist in build_subgraph.' % (out_node_name)
        if out_node_name not in sub_graph._attr['output_names']:
            sub_graph._attr['output_names'].append(out_node_name)
        if output['name'] not in sub_graph._attr['output_tensor_names']:
            sub_graph._attr['output_tensor_names'].append(output['name'])

        sub_graph.add_node(noop_node_name)
        noop_node = NodeWrap(sub_graph, noop_node_name)
        noop_node.replace_obj('Out', {'name': noop_node_name})
        pending_out_port = output['out_port']
        current_out_ports = NodeWrap(sub_graph, out_node_name)[
            'object'].get_out_ports()
        if pending_out_port in current_out_ports:
            found_non_out_node = False
            cur_out_edges = sub_graph.sorted_out_edges(
                out_node_name, data=True)
            for _, dst, out_attr in cur_out_edges:
                if out_attr.get('src_out_port', 0) == pending_out_port \
                        and NodeWrap(sub_graph, dst)['object'] is not None \
                        and NodeWrap(sub_graph, dst)['object'].type != 'Out':
                    found_non_out_node = True
                    break
            if not found_non_out_node:
                pending_out_port = max(current_out_ports) + 1
        out_edge_attr = {
            'src_out_port': pending_out_port, 'dst_in_port': 0, 'tensor': Tensor(name=output['name'])}
        sub_graph.add_edge(
            out_node_name, noop_node_name, **out_edge_attr)

    for inp_name, inp_tensor in sub_graph._attr['input_tensors'].items():
        out_port = 0
        out_edges = sub_graph.sorted_out_edges(inp_name)
        if len(out_edges) == 0:
            noop_node_name = inp_name + '_noop_' + str(out_port)
            assert sub_graph.has_node(inp_name) and not sub_graph.has_node(
                noop_node_name), 'Node(%s) does not exist in build_subgraph.' % (inp_name)
            sub_graph.add_node(noop_node_name)
            noop_node = NodeWrap(sub_graph, noop_node_name)
            noop_node.replace_obj('Out', {'name': noop_node_name})
            pending_out_port = out_port
            current_out_ports = NodeWrap(sub_graph, inp_name)[
                'object'].get_out_ports()
            if pending_out_port in current_out_ports:
                found_non_out_node = False
                cur_out_edges = sub_graph.sorted_out_edges(
                    inp_name, data=True)
                for _, dst, out_attr in cur_out_edges:
                    if out_attr.get('src_out_port', 0) == pending_out_port \
                            and NodeWrap(sub_graph, dst)['object'] is not None \
                            and NodeWrap(sub_graph, dst)['object'].type != 'Out':
                        found_non_out_node = True
                        break
                if not found_non_out_node:
                    pending_out_port = max(current_out_ports) + 1
            out_edge_attr = {
                'src_out_port': pending_out_port, 'dst_in_port': 0, 'tensor': inp_tensor}
            sub_graph.add_edge(
                inp_name, noop_node_name, **out_edge_attr)

    clear_redundant_nodes(sub_graph)
    return sub_graph


def attr_value_converter(attr_dict, source, root_graph, a_nodes, parent_graph_info=None):
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
            sub_graph_name = attr_dict['name'] + '_' + key + '_subgraph'
            sub_graph = build_subgraph(sub_graph_name,
                                       attr_dict[key],
                                       root_graph,
                                       parent_graph_info,
                                       a_nodes,
                                       attr_dict.get('opset_version', 1))
            if attr_dict['name'] in root_graph._attr['subgraphs']:
                root_graph._attr['subgraphs'][attr_dict['name']][sub_graph_name] = sub_graph
            else:
                root_graph._attr['subgraphs'][attr_dict['name']] = {sub_graph_name: sub_graph}
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

    graph._attr['force_not_quantize'] = force_not_quantize
    graph._attr['subgraphs'] = OrderedDict()
    graph._attr['node_in_subgraphs'] = OrderedDict()
    graph._attr['subgraph_depends_nodes'] = []
    graph._attr['subgraph_output_names'] = []

    meta_ret = True
    all_nodes = []
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
            graph._attr['source'] = params['source']
            graph._attr['data_format'] = params.get('input_data_format', 'NCHW')
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

                all_nodes += nodes
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
                                                                     'data_format': params.get('input_data_format',
                                                                                               'NCHW'),
                                                                     'opset_version': opset_ver})
                root_info.update({'const_names': list(const_names.values())})
                graph._attr['const_names'] = list(const_names.values())

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
                    if not op_attr['name']:
                        if node['output'][0]['name']:
                            op_attr.update({'name': node['output'][0]['name']})
                            nodes[ni].update(
                                {'name': node['output'][0]['name']})
                        else:
                            name = get_valid_node_name(graph, op_attr['type'])
                            op_attr.update({'name': name})
                            nodes[ni].update({'name': name})
                    attr_value_converter(
                        op_attr, params['source'], root_graph=graph,
                        a_nodes=all_nodes, parent_graph_info=root_info)
                    op_name = op_attr['name']
                    if not graph.has_node(op_name):
                        graph.add_node(op_name)
                    NodeWrap(graph, op_name).replace_obj(node_type, op_attr)
                    if op_name in graph._attr['node_in_subgraphs']:
                        op_obj = graph.nodes[op_name]['object']
                        op_obj.in_subgraph = True
                        op_obj.subgraphs.extend(graph._attr['node_in_subgraphs'][op_name])
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
                                                                   value=graph._attr['input_tensors'][
                                                                       pre_op_name].value,
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
                                        WARN(
                                            '[Parser]: The output name %s is not a node but a tensor. However, we will use the node %s as output node.'
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

                # Set output tensors if not set in params
                if not graph._attr['output_tensor_names']:
                    for out_index, output in enumerate(outputs):
                        if (output['name'], output['out_port']) not in out_tensor_operator_map \
                                and output['name'] in const_names:
                            # Ignore the const outputs because they will make the graph not connected
                            continue
                        if output['name'] not in graph._attr['output_tensor_names']:
                            graph._attr['output_tensor_names'].append(output['name'])

                # Add Out node after the nodes with output tensors but without successors.
                for (out_tensor_name, out_port), node_idx in out_tensor_operator_map.items():
                    node_name = nodes[node_idx]['name']
                    if out_tensor_name in in_tensor_names or node_name in graph._attr['output_names']:
                        continue
                    out_node_name = out_tensor_name + '_out_' + str(out_port)
                    if len(NodeWrap(graph, node_name)['object'].subgraphs) > 0:
                        att_dict = {
                            'name': out_node_name,
                            'subgraphs': NodeWrap(graph, node_name)['object'].subgraphs
                        }
                    else:
                        att_dict = {
                            'name': out_node_name
                        }
                    graph.add_edge(node_name, out_node_name, **
                                   {'src_out_port': out_port, 'dst_in_port': 0, 'tensor': Tensor(name=out_tensor_name)})
                    NodeWrap(graph, out_node_name).replace_obj('Out', att_dict)

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
                                    if out_edges[0][2]['tensor'] is not None and out_edges[0][2][
                                            'tensor'].value is not None:
                                        graph._attr['input_tensors'].update({in_name: out_edges[0][2]['tensor']})
                                    else:
                                        cur_shape = params['input_shapes'][in_name]
                                        if cur_shape is None:
                                            ERROR(
                                                '[Parser]: Shape of Input(%s) is unknown! Please provide input_shape in cfg file!' % in_name)
                                            continue
                                        graph._attr['input_tensors'].update(
                                            {in_name: Tensor(name=in_name,
                                                             value=np.random.ranf(cur_shape).astype(np.float32))})
                        graph._attr['input_tensors'] = OrderedDict(
                            {k: v for k, v in graph._attr['input_tensors'].items() if graph.has_node(k)})
                    clear_redundant_nodes(graph)
            else:
                WARN('[Parser]: Meets empty graph in convert_onnx_to_graph!')
    else:
        ERROR('[Parser]: Onnx version is too low, needs updating!')

    return graph
