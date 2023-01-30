# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
from collections import OrderedDict
from .graph.graph_algo import determined_sort
from .graph.node_wrap import NodeWrap
from .ops.op import ArmOp, OpHasWeights, OpHasBiases, BaseActivationOp
from .ops.release_ops import ArmActivationOp
from .ops.common_ops import PluginOp
from .common.utils import is_dir, list_list_to_string, string_list_to_string
from .logger import INFO, DEBUG, WARN, ERROR, FATAL


def write_net_attrs(txt_file, attr):
    ret = True
    if not txt_file.closed and txt_file.mode == 'w':
        for k, v in attr.items():
            if k in ['input_tensors', 'output_tensors']:
                txt_file.write('%s=[%s]\n' % (k, v))
            else:
                txt_file.write('%s=%s\n' % (k, v))
    else:
        WARN('[Parser]: Can not write net attr!')
        ret = False
    return ret


def serialize(graph, params):
    '''Serialize graph and write to IR txt and IR bin.'''
    ret = True
    model_name = params['model_name'] \
        if params.get('model_name') \
        else os.path.splitext(os.path.basename(params['tflite_file']))[0]
    output_dir = params.get('output_dir', './')
    model_domain = params.get('model_domain', 'image_classification')

    if is_dir(output_dir):
        txt_path = os.path.join(output_dir, model_name + '.txt')
        bin_path = os.path.join(output_dir, model_name + '.bin')

        arm_op_types = ArmOp.get_concrete_subclass_names()
        sorted_list = determined_sort(graph, graph._attr['output_names'])
        main_input = None
        for i, n in enumerate(sorted_list):
            node_obj = NodeWrap(graph, n)['object']
            if node_obj is None:
                WARN(
                    '[Parser]: Node (%s) has invalid op that cannot be written to IR in serialize!' % n)
                continue
            if node_obj.type == 'ArmInput' and len(node_obj.get_output_shapes()[0]) > 2 and i != 0:
                main_input = n
                break
        if main_input:
            sorted_list.remove(main_input)
            sorted_list.insert(0, main_input)

        net_attr = OrderedDict()
        net_attr['model_name'] = model_name
        net_attr['model_domain'] = model_domain
        net_attr['layer_number'] = str(len(sorted_list))
        net_attr['precision'] = 'int8' if graph._attr.get('quantize', False) else 'float32'
        net_attr['model_bin'] = './' + os.path.basename(bin_path)

        input_names = params['input_names']
        if any([i not in graph.nodes or graph.nodes[i].op != 'ArmInput' for i in input_names]) or len(input_names) == 0:
            WARN('[Parser]: the input node(s) has changed or not set, please check the IR to confirm the input tensors order.')
            input_names = [
                graph.nodes[n].key for n in graph.nodes if graph.nodes[n].op == 'ArmInput']
        input_objs = [NodeWrap(graph, name)['object'] for name in input_names]
        input_tops = [obj.get_outputs_info() for obj in input_objs]
        input_tops_names = [t[0][0] for t in input_tops]
        net_attr['input_tensors'] = string_list_to_string(input_tops_names)
        INFO('[Parser]: The input tensor(s) is/are: %s' %
             (net_attr['input_tensors']))

        output_tops_names = []
        out_nodes = graph._attr.get('output_nodes', [])
        if (len(out_nodes)
            and all([i is not None for i in out_nodes])
                and all([NodeWrap(graph, name)['object'] for name in out_nodes])):   # due to some post process adding,the out node may be removed.
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

        net_attr['output_tensors'] = string_list_to_string(output_tops_names)

        writing_node = ''
        try:
            with open(txt_path, 'w') as txt_file:
                if write_net_attrs(txt_file, net_attr):
                    for i, n in enumerate(sorted_list):
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
        except IOError as e:
            ERROR('[Parser]: Meets IOError (%s) when writing attributes of node (%s) in serialize!' % (
                str(e), writing_node))
            ret = False
        except Exception as e:
            ERROR('[Parser]: Meets Exception (%s) when writing attributes of node (%s) in serialize!' % (
                str(e), writing_node))
            ret = False

        if ret:
            writing_node = ''
            try:
                with open(bin_path, 'wb') as bin_file:
                    for n in sorted_list:
                        writing_node = n
                        op_obj = NodeWrap(graph, n)['object']
                        if op_obj is not None:
                            if isinstance(op_obj, OpHasWeights):
                                if not op_obj.write_weights(bin_file):
                                    ret = False
                                    break
                            if isinstance(op_obj, OpHasBiases):
                                if not op_obj.write_biases(bin_file):
                                    ret = False
                                    break
                            if isinstance(op_obj, (BaseActivationOp, ArmActivationOp)):
                                if not op_obj.write_negative_slope(bin_file):
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

    else:
        ERROR('[Parser]: Meets invalid output dir in serialize!')
        ret = False
    return ret
