# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
from collections import OrderedDict
from .graph.graph_algo import determined_sort
from .graph.node_wrap import NodeWrap
from .ops.op import ArmOp, OpHasWeights, OpHasBiases, BaseActivationOp, BaseQuantizeDequantizeOp, BaseRnnOp, OpHasAnchors
from .ops.release_ops import ArmActivationOp
from .ops.common_ops import PluginOp
from .common.utils import is_dir, list_list_to_string, string_list_to_string
from .logger import INFO, DEBUG, WARN, ERROR, FATAL


def write_net_attrs(txt_file, attr):
    ret = True
    if not txt_file.closed and txt_file.mode == 'a':
        for k, v in attr.items():
            if k in ['input_tensors', 'output_tensors']:
                txt_file.write('%s=[%s]\n' % (k, v))
            else:
                txt_file.write('%s=%s\n' % (k, v))
    else:
        ERROR('[Parser]: Can not write net attr!')
        ret = False
    return ret


def write_nodes_attrs(txt_path, net_attr, nodes_list, graph):
    writing_node = ''
    arm_op_types = ArmOp.get_concrete_subclass_names()
    ret = True
    try:
        with open(txt_path, 'a') as txt_file:
            if write_net_attrs(txt_file, net_attr):
                for i, n in enumerate(nodes_list):
                    writing_node = n
                    op_obj = NodeWrap(graph, n)['object']
                    if op_obj is not None:
                        if op_obj.type not in arm_op_types and not isinstance(op_obj, PluginOp):
                            ERROR('[Parser]: Writing %s op(%s) that is not supported in serialize!' % (
                                op_obj.type, op_obj.name))
                        txt_file.write('\nlayer_id=%d\n' % i)
                        op_obj.write_attrs(txt_file)
                    else:
                        ERROR(
                            '[Parser]: Meets invalid op for Node (%s) in serialize!' % n)
                txt_file.write('\n\n')
    except IOError as e:
        ERROR('[Parser]: Meets IOError (%s) when writing attributes of node (%s) in serialize!' % (
            str(e), writing_node))
        ret = False
    except Exception as e:
        ERROR('[Parser]: Meets Exception (%s) when writing attributes of node (%s) in serialize!' % (
            str(e), writing_node))
        ret = False
    return ret


def write_nodes_weights(bin_path, nodes_list, graph):
    writing_node = ''
    ret = True
    try:
        with open(bin_path, 'ab') as bin_file:
            for n in nodes_list:
                writing_node = n
                op_obj = NodeWrap(graph, n)['object']
                if op_obj is not None:
                    op_obj.write_top_range(bin_file)
                    op_obj.write_top_scale_zp(bin_file)

                    if isinstance(op_obj, OpHasWeights):
                        if not op_obj.write_weights(bin_file):
                            ret = False
                            break
                    if isinstance(op_obj, OpHasBiases):
                        if not op_obj.write_biases(bin_file):
                            ret = False
                            break
                    if isinstance(op_obj, OpHasAnchors):
                        if not op_obj.write_anchors(bin_file):
                            ret = False
                            break
                    if isinstance(op_obj, (BaseActivationOp, ArmActivationOp)):
                        if not op_obj.write_negative_slope(bin_file):
                            ret = False
                            break
                    if isinstance(op_obj, BaseQuantizeDequantizeOp):
                        if not op_obj.write_scale_zp(bin_file):
                            ret = False
                            break
                    if isinstance(op_obj, BaseRnnOp):
                        if not op_obj.write_scale_zp(bin_file):
                            ret = False
                            break
                    if isinstance(op_obj, PluginOp):
                        if not op_obj.write_constants(bin_file):
                            ret = False
                            break
                else:
                    ERROR(
                        '[Parser]: Meets invalid op for Node (%s) in serialize!' % n)
    except IOError as e:
        ERROR('[Parser]: Meets IOError (%s) when writing binary data of node (%s) in serialize!' % (
            str(e), writing_node))
        ret = False
    except Exception as e:
        ERROR('[Parser]: Meets Exception (%s) when writing binary data of node (%s) in serialize!' % (
            str(e), writing_node))
        ret = False
    return ret


def write_net(txt_path, bin_path, net_attr, graph):
    sorted_list = determined_sort(graph,
                                  graph._attr['subgraph_depends_nodes'] + graph._attr['output_names'],
                                  sort_input=True)
    ret = write_nodes_attrs(txt_path, net_attr, sorted_list, graph)

    if ret:
        ret = write_nodes_weights(bin_path, sorted_list, graph)
    return ret


def get_output_tensor_names(graph):
    output_tops_names = []
    out_nodes = graph._attr.get('output_nodes', [])
    if (len(out_nodes)
            and all([i is not None for i in out_nodes])
            and all([NodeWrap(graph, name)['object'] for name in
                     out_nodes])):  # due to some post process adding,the out node may be removed.
        for name in out_nodes:
            obj = NodeWrap(graph, name)['object']
            out_tops = obj.get_inputs_info()
            output_tops_names.extend(out_tops[0])
    else:
        for name in graph._attr['output_names']:
            obj = NodeWrap(graph, name)['object']
            out_tops = obj.get_outputs_info()
            if len(out_tops) > 0:
                output_tops_names.extend(out_tops[0])
    return output_tops_names


def serialize(graph, params):
    '''Serialize graph and write to IR txt and IR bin.
    Return True/False for serializing status and also txt path and bin path.'''
    ret = True
    txt_path, bin_path = '', ''
    model_name = params['model_name'] \
        if params.get('model_name') \
        else os.path.splitext(os.path.basename(params['tflite_file']))[0]
    output_dir = params.get('output_dir', './')
    model_domain = params.get('model_domain', 'image_classification')

    if is_dir(output_dir):
        txt_path = os.path.join(output_dir, model_name + '.txt')
        bin_path = os.path.join(output_dir, model_name + '.bin')

        if os.path.exists(txt_path):
            os.remove(txt_path)
        if os.path.exists(bin_path):
            os.remove(bin_path)

        sorted_list = determined_sort(graph,
                                      graph._attr['subgraph_depends_nodes'] + graph._attr['output_names'],
                                      sort_input=True)

        net_attr = OrderedDict()
        net_attr['model_name'] = model_name
        net_attr['model_domain'] = model_domain
        net_attr['layer_number'] = str(len(sorted_list))
        net_attr['precision'] = 'int8' if graph._attr.get('quantize', False) else 'float32'
        net_attr['compat_quantized_model'] = 'true' if graph._attr.get('quantize', False) else 'false'
        net_attr['model_bin'] = './' + os.path.basename(bin_path)

        input_names = params['input_names']
        if any([i not in graph.nodes or graph.nodes[i]['op'] != 'ArmInput' for i in input_names]) or len(input_names) == 0:
            WARN('[Parser]: the input node(s) has changed or not set, please check the IR to confirm the input tensors order.')
            input_names = [n for n in graph.nodes if graph.nodes[n]['op'] == 'ArmInput']
        input_objs = [NodeWrap(graph, name)['object'] for name in input_names]
        input_tops = [obj.get_outputs_info() for obj in input_objs]
        input_tops_names = [t[0][0] for t in input_tops]
        net_attr['input_tensors'] = string_list_to_string(input_tops_names)
        INFO('[Parser]: The input tensor(s) is/are: %s' %
             (net_attr['input_tensors']))

        output_tensor_names = get_output_tensor_names(graph)

        net_attr['output_tensors'] = string_list_to_string(output_tensor_names)

        ret = write_net(txt_path, bin_path, net_attr, graph)

        if 'subgraphs' in graph._attr and graph._attr['subgraphs']:
            for n_name, v in graph._attr['subgraphs'].items():
                for subgraph_name, subgraph in v.items():
                    sorted_list = determined_sort(subgraph, subgraph._attr['output_names'], sort_input=True)
                    sub_net_attr = OrderedDict()
                    sub_net_attr['subgraph_name'] = subgraph_name
                    sub_net_attr['layer_number'] = str(len(sorted_list))
                    sub_net_attr['precision'] = 'int8' if subgraph._attr.get('quantize', False) else 'float32'
                    sub_net_attr['compat_quantized_model'] = 'true' if subgraph._attr.get(
                        'quantize', False) else 'false'
                    sub_input_names = list(subgraph._attr['input_tensors'].keys())
                    sub_input_objs = [NodeWrap(subgraph, name)['object'] for name in sub_input_names]
                    sub_input_tops = [obj.get_outputs_info() for obj in sub_input_objs]
                    sub_input_tops_names = [t[0][0] for t in sub_input_tops]
                    sub_net_attr['input_tensors'] = string_list_to_string(sub_input_tops_names)
                    sub_output_tensor_names = get_output_tensor_names(subgraph)
                    sub_net_attr['output_tensors'] = string_list_to_string(sub_output_tensor_names)

                    ret = write_net(txt_path, bin_path, sub_net_attr, subgraph)

    else:
        ERROR('[Parser]: Meets invalid output dir in serialize!')
        ret = False
    return ret, txt_path, bin_path


def show_in_out_map(graph, params):
    '''Show the mapping of inputs from cfg and input tensors from IR, and also
    the mappings of outputs from cfg and output tensors from IR.
    '''
    ret = True
    input_tensor_map = params.get('input_tensor_map', {})
    for input_name_from_cfg, input_node_name in input_tensor_map.items():
        if input_node_name is None:
            input_node_name = input_name_from_cfg
        if input_node_name not in graph.nodes:
            INFO('[Parser]: Input %s from cfg is removed!' % input_name_from_cfg)
            continue
        output_info = NodeWrap(graph, input_node_name)['object'].get_outputs_info()
        if len(output_info) > 0 and len(output_info[0]) > 0:
            input_tensor_name = output_info[0][0]
            INFO('[Parser]: Input %s from cfg is shown as tensor %s in IR!' %
                 (input_name_from_cfg, input_tensor_name))
        else:
            ERROR('[Parser]: Meets invalid input node (%s) in serialize!' % input_node_name)
            ret = False

    output_tensor_map = params.get('output_tensor_map', {})
    for output_name_from_cfg, out_node_names in output_tensor_map.items():
        output_name_from_ir = []
        for out_node_name in out_node_names:
            if out_node_name not in graph.nodes:
                continue
            input_info = NodeWrap(graph, out_node_name)['object'].get_inputs_info()
            if len(input_info) > 0 and len(input_info[0]) > 0:
                output_name_from_ir.append(input_info[0][0])
            else:
                ERROR('[Parser]: Meets invalid Out node (%s) in serialize!' % out_node_name)
                ret = False
        if len(output_name_from_ir) > 0:
            output_name_str = ', '.join(output_name_from_ir)
            INFO('[Parser]: Output %s from cfg is shown as tensor %s in IR!' %
                 (output_name_from_cfg, output_name_str))
        else:
            INFO('[Parser]: Output %s from cfg is removed/replaced by other tensors!' % output_name_from_cfg)
    return ret
