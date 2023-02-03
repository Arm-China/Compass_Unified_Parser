# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from collections import OrderedDict
from ...common.defs import FLOAT_EQUAL
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


def get_nodes_input_and_attr(configs):
    '''Get the inputs of attr of all the nodes from model config.
    For constants, it's not saved as a layer by tf, so record all the nodes' name
    to create Const node for them later.
    '''
    def flatten(mixed_nested_list):
        if isinstance(mixed_nested_list, list):
            for item in mixed_nested_list:
                yield from flatten(item)
        else:
            yield mixed_nested_list

    input_nodes_dict = {}
    if not configs or 'layers' not in configs.keys():
        return input_nodes_dict

    for layer_configs in configs['layers']:
        inbound_nodes = layer_configs.get('inbound_nodes', [])
        node_name = layer_configs.get('name', '')
        if not inbound_nodes or not node_name:
            continue
        # input_info_dict is a dict, whose key is kwarg name or index, and value is a tuple with 5 items:
        # input node name, dst out port, whether is control edge, whether is constant, value(None if not constant)
        input_info_dict = {}
        flatten_inbound_nodes = list(flatten(inbound_nodes))
        nodes_iter = iter(flatten_inbound_nodes)
        for node in nodes_iter:
            if isinstance(node, str):
                next(nodes_iter)
                if node == '_CONSTANT_VALUE':
                    const_value = next(nodes_iter)
                    const_node_name = node_name + '/' + node
                    input_node_info = (const_node_name, 0, False, True, const_value)
                    input_info_dict.update({node: input_node_info})
                    continue
                tensor_index = next(nodes_iter)
                # TODO: Consider control edge(the third arg means is_control)
                input_node_info = (node, tensor_index, False, False, None)
                input_info_dict.update({node: input_node_info})
            elif isinstance(node, dict):
                for key, value in node.items():
                    if key == 'name':
                        continue
                    if isinstance(value, list) and len(value) == 3 and isinstance(value[0], str):
                        # node name and its parent's out port
                        input_node_info = (value[0], value[2], False, False, None)
                        input_info_dict.update({key: input_node_info})
                    else:
                        input_node_info = (node_name + '/' + key, 0, False, True, value)
                        input_info_dict.update({key: input_node_info})
        input_nodes_dict.update({node_name: input_info_dict})
    return input_nodes_dict


def get_node_attr(layer):
    ret = {}
    try:
        layer_config = layer.get_config()
        if isinstance(layer_config, dict):
            ret.update(layer_config)
    except Exception as e:
        DEBUG('[Parser]: Fail to get config for layer (%s) because %s' % (layer.name, str(e)))

    for key in layer.__dir__():
        if key.startswith('_'):
            # Ignore internal attributes
            continue
        if key in ('activity_regularizer', 'build', 'built', 'call', 'compute_dtype', 'count_params',
                   'dynamic', 'dtype_policy', 'finalize_state', 'get_config',
                   'inbound_nodes', 'input', 'input_mask', 'input_spec',
                   'losses', 'metric', 'metrics', 'name_scope', 'non_trainable_variables', 'non_trainable_weights',
                   'outbound_nodes', 'output', 'output_mask', 'OVERLOADABLE_OPERATORS',
                   'symbol', 'stateful', 'states', 'state_spec', 'submodules', 'supports_masking',
                   'trainable_weights', 'trainable_variables',
                   'updates', 'variables', 'variable_dtype', 'with_name_scope'):
            # Ignore inputs/outputs and other attributes that are not used
            continue
        try:
            value = eval('layer.' + key)
            if eval('callable(layer.' + key + ')'):
                if any(key.startswith(func) for func in ('add_', 'apply', 'compute_', 'from_', 'get_', 'reset_', 'set_')) \
                        or any(key.endswith(func) for func in ('_initializer', '_constraint')) \
                        or '__name__' not in dir(value):
                    # Ignore functions that are not used
                    continue
                func_name = value.__name__
                ret.update({key: func_name})
            elif key == 'weights' and isinstance(value, list):
                weights_list = []
                for variable in value:
                    weights_list.append(variable.numpy())
                key = 'weights_list'
                ret.update({key: weights_list})
                if len(value) == 2:
                    try:
                        biases = layer.bias.numpy()
                    except:
                        biases = None
                    if biases is not None and FLOAT_EQUAL(weights_list[1], biases):
                        ret.update({'weights': weights_list[0]})
                elif len(value) == 1:
                    ret.update({'weights': weights_list[0]})
            elif 'numpy' in dir(value):
                ret.update({key: value.numpy()})
            else:
                ret.update({key: copy.deepcopy(value)})
        except Exception as e:
            DEBUG('[Parser]: Fail to get key (%s) for layer (%s) because %s' % (key, layer.name, str(e)))
            continue
        DEBUG('[Parser]: layer: %s, key: %s, value: %s' % (layer.name, key, str(ret[key])))
    return ret


def get_node_type(layer):
    layer_type = type(layer).__name__
    if layer_type == 'TFOpLambda':
        node_type = layer.get_config().get('function', layer_type)
    else:
        node_type = layer_type

    opcode_version = 1 if 'compat.v1' in node_type else 2

    # Add prefix 'Keras' for keras op type to distiguish from raw ops
    if node_type in dir(tf.keras.layers):
        node_type = 'Keras' + node_type
        return (node_type, opcode_version)

    # Remove prefix like 'math.', 'nn.' and etc
    node_type = node_type.split('.')[-1]

    return (node_type, opcode_version)


def get_node_input(layer, input_info_dict):
    assert isinstance(input_info_dict, dict), 'Expect input_info_dict to be a dict!'
    arg_pos_dict = layer._call_fn_arg_positions
    arg_defaults_dict = layer._call_fn_arg_defaults

    if not input_info_dict:
        return []

    if not arg_pos_dict:
        return [value for _, value in input_info_dict.items()]

    node_input_info = []
    inbound_nodes_cnt = len(layer.inbound_nodes)

    for arg_name, arg_pos in arg_pos_dict.items():
        if arg_pos < inbound_nodes_cnt:
            input_tensors = layer.get_input_at(arg_pos)
            if isinstance(input_tensors, (list, tuple)):
                if len(input_tensors) == 0:
                    continue
            else:
                if not tf.is_tensor(input_tensors):
                    continue
                elif 'numpy' in dir(input_tensors) and '_CONSTANT_VALUE' in input_info_dict:
                    input_info = input_info_dict['_CONSTANT_VALUE']
                    node_input_info.append(input_info)
                    continue
            inbound_layers = layer.inbound_nodes[arg_pos].inbound_layers
            inbound_layers = inbound_layers if isinstance(inbound_layers, (list, tuple)) else [inbound_layers]
            inbound_nodes = [node.name for node in inbound_layers]
            for node_name in inbound_nodes:
                if node_name in input_info_dict:
                    input_info = input_info_dict[node_name]
                    node_input_info.append(input_info)
                else:
                    WARN('[Parser]: Meet invalid node (%s) in get_node_input!' % node_name)
        elif arg_name in input_info_dict \
                and len(input_info_dict[arg_name]) == 5:
            input_info = input_info_dict[arg_name]
            node_input_info.append(input_info)
        elif arg_name in arg_defaults_dict:
            value = arg_defaults_dict[arg_name]
            input_info = (layer.name + '/' + arg_name, 0, False, True, value)
            node_input_info.append(input_info)
        else:
            WARN('[Parser]: Missing node (%s) in get_node_input!' % arg_name)
    return node_input_info


def get_const_node_content(node_name, const_value):
    '''Create Const node for inputs that are not shown as model layer.
    '''
    ret = {}
    if const_value is not None:
        const_value = np.array(const_value)
    else:
        const_value = np.array(None)
    out_tensor_name = node_name + ':0'
    ret = {'name': node_name,
           'type': 'constant',
           'input': [],
           'output': [(out_tensor_name, list(const_value.shape))],
           'attr': {'value': const_value, 'dtype': const_value.dtype.name},
           'opcode_version': 2
           }
    return ret


def get_node_content(layer):
    layer_outputs = layer.output if isinstance(layer.output, (list, tuple)) else [layer.output]
    output_name_shape = [(out.name, out.shape.as_list() if out.shape is not None else [])
                         for out in layer_outputs]
    DEBUG('layer: %s, output: %s' % (layer.name, str(output_name_shape)))
    node_type, opcode_version = get_node_type(layer)
    ret = {'name': layer.name,
           'type': node_type,
           'output': output_name_shape,
           'attr': get_node_attr(layer),
           'opcode_version': opcode_version
           }
    return ret


def get_nodes_content(layers, model_configs):
    layers = layers if isinstance(layers, list) else [layers]
    inputs_info_dict = get_nodes_input_and_attr(model_configs)
    nodes_content = []
    for layer in layers:
        # if hasattr(layer, 'layers'):
        #     nodes_content.extend(get_nodes_content(layer.layers, exp_input_names))
        #     continue
        node_content = get_node_content(layer)
        input_info_dict = inputs_info_dict.get(layer.name, {})
        node_input_info = get_node_input(layer, input_info_dict)
        node_content.update({
            'input': [(name, src_out_port, control_edge) for name, src_out_port, control_edge, _, _ in node_input_info]})
        nodes_content.append(node_content)
        const_nodes = [(name, value) for name, _, _, is_const, value in node_input_info if is_const]
        for name, value in const_nodes:
            nodes_content.append(get_const_node_content(name, value))
    return nodes_content


def parse_keras(model_path, params):
    nodes = list()
    nodes_dict, tensors, np_tensors = OrderedDict(), OrderedDict(), OrderedDict()
    input_shapes = params['input_shapes'].copy()
    try:
        load_options = tf.saved_model.LoadOptions(allow_partial_checkpoint=True)
        model = tf.keras.models.load_model(model_path, compile=False, options=load_options)
    except Exception as e:
        WARN('[Parser]: Reading saved model/h5 file (%s) meets error (%s)!' %
             (model_path, str(e)))
        return nodes, nodes_dict, tensors, np_tensors, input_shapes

    nodes = get_nodes_content(model.layers, model.get_config())
    model_inputs = model.inputs
    model_inputs_names = model.input_names

    for n in nodes:
        if n['name'] in model_inputs_names and n['name'] not in input_shapes:
            tensor_shape = n['output'][0][1]
            input_shapes.update({n['name']: tensor_shape})
        if n['type'] == 'constant' and n['attr'].get('value', None) is not None:
            const_tensor_name = n['output'][0][0]
            const_tensor_value = n['attr']['value']
            np_tensors.update({const_tensor_name: const_tensor_value})
        nodes_dict.update({n['name']: n})

    feed_model_inputs = []
    for model_input_name, model_input in zip(model_inputs_names, model_inputs):
        model_input_shape = input_shapes[model_input_name]
        if any([d is None for d in model_input_shape]):
            WARN(
                '[Parser]: Found None in the shape of Input (%s): %s!' %
                (model_input_name, str(model_input_shape)))
            feed_model_inputs.append([])
            continue
        try:
            type_str = model_input.dtype.name
        except Exception as e:
            WARN('[Parser]: Meets error when getting dtype of input tensor (%s): %s!' %
                 (model_input, str(e)))
            type_str = 'float32'
        np_tensor = np.random.randint(0, 1, size=model_input_shape).astype(type_str) \
            if 'int' in type_str \
            else np.random.ranf(model_input_shape).astype(type_str)
        feed_model_inputs.append(np_tensor)
        np_tensors.update({model_input.name: np_tensor})
    outputs = []
    for layer in model.layers:
        if layer.name in input_shapes.keys():
            continue
        if isinstance(layer.output, (list, tuple)):
            outputs.extend(layer.output)
        else:
            outputs.append(layer.output)
    try:
        functors = K.function([model.input], outputs)
        outputs_value = functors(feed_model_inputs)
    except Exception as e:
        outputs_value = [None] * len(outputs)
        DEBUG('Fail to get outputs of tensors: %s' % str(e))
    for out, out_value in zip(outputs, outputs_value):
        tensors.update({out.name: out})
        if out_value is None:
            out_value = np.random.ranf(out.shape.as_list()).astype(out.dtype.name)
        np_tensors.update({out.name: np.array(out_value)})

    return nodes, nodes_dict, tensors, np_tensors, input_shapes
