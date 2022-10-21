# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from collections import OrderedDict
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
    const_inputs_dict = {}
    nodes_attr_dict = {}
    if not configs or 'layers' not in configs.keys():
        return input_nodes_dict, attr_dict
    for layer_configs in configs['layers']:
        inbound_nodes = layer_configs.get('inbound_nodes', [])
        node_name = layer_configs.get('name', '')
        if not inbound_nodes or not node_name:
            continue
        class_name = layer_configs.get('class_name', '')
        function = layer_configs.get('config', {}).get('function', '')
        input_nodes_info = []
        node_attr_dict = {}
        const_inputs_name = []
        flatten_inbound_nodes = list(flatten(inbound_nodes))
        nodes_iter = iter(flatten_inbound_nodes)
        for node in nodes_iter:
            if isinstance(node, str):
                node_index = next(nodes_iter)
                if node == '_CONSTANT_VALUE':
                    const_value = next(nodes_iter)
                    const_node_name = node_name + '/' + node
                    input_node_info = (const_node_name, 0, False)
                    input_nodes_info.append(input_node_info)
                    const_inputs_name.append(const_node_name)
                    continue
                tensor_index = next(nodes_iter)
                # TODO: Consider control edge(the third arg means is_control)
                input_node_info = (node, tensor_index, False)
                input_nodes_info.append(input_node_info)
            elif isinstance(node, dict):
                for key, value in node.items():
                    if key == 'name':
                        continue
                    if key == 'dtype' and class_name == 'TFOpLambda' and function == 'cast':
                        node_attr_dict.update({'DstT': value})
                    elif isinstance(value, list) and len(value) == 3 and isinstance(value[0], str):
                        input_node_info = (value[0], value[2], False)  # node name and its parent's out port
                        input_nodes_info.append(input_node_info)
                    elif value is not None:
                        # Some inputs in tf2 are attributes for raw ops, so save them to both
                        # inputs and attributes.
                        const_node_name = node_name + '/' + key
                        input_node_info = (const_node_name, 0, False)
                        input_nodes_info.append(input_node_info)
                        node_attr_dict.update({key: value})
                        const_inputs_name.append(const_node_name)
        input_nodes_dict.update({node_name: input_nodes_info})
        const_inputs_dict.update({node_name: const_inputs_name})
        nodes_attr_dict.update({node_name: node_attr_dict})
    return input_nodes_dict, const_inputs_dict, nodes_attr_dict


def get_node_attr(layer):
    ret = {}
    for key in layer.__dir__():
        if key.startswith('_'):
            # Ignore internal attributes
            continue
        if key in ('activity_regularizer', 'built', 'compute_dtype',
                   'dynamic', 'dtype_policy',
                   'inbound_nodes', 'input', 'input_mask', 'input_spec',
                   'losses', 'metric', 'metrics', 'non_trainable_variables', 'non_trainable_weights',
                   'outbound_nodes', 'output', 'output_mask', 'OVERLOADABLE_OPERATORS',
                   'symbol', 'stateful', 'states', 'state_spec', 'submodules', 'supports_masking',
                   'trainable_weights', 'trainable_variables',
                   'updates', 'variables', 'variable_dtype'):
            # Ignore inputs/outputs and other attributes that are not used
            continue
        try:
            if eval('callable(layer.' + key + ')'):
                continue
            value = eval('layer.' + key)
            if key == 'weights':
                weights_list = []
                for variable in value:
                    weights_list.append(variable.numpy())
                key = 'weights_list'
                ret.update({key: weights_list})
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

    # Convert type like 'math.add' to 'Add', 'cast' to 'Cast'(similar to type of raw ops)
    node_type = node_type.split('.')[-1]
    node_type = node_type[0].upper() + node_type[1:]

    # Convert type like 'Conv2d' to 'Conv2D' to match with type in raw ops
    if node_type.endswith('2d') or node_type.endswith('3d'):
        node_type = node_type[:-1] + 'D'
    return node_type


def get_const_node_content(layer, const_input_name):
    '''Create Const node for inputs that are not shown as model layer. Note that some inputs are also
    attributes because we cannot recognize whether they are taken as input or attribute.
    '''
    ret = {}
    arg_name = const_input_name
    prefix = layer.name + '/'
    if const_input_name.startswith(prefix):
        arg_name = const_input_name.replace(prefix, '', 1)
    const_value = None
    for node in layer.inbound_nodes:
        if not node.call_kwargs or arg_name not in node.call_kwargs.keys():
            continue
        try:
            kwargs_value = node.call_kwargs[arg_name]
            const_value = np.array(kwargs_value)
        except:
            break
    if const_value is not None:
        out_tensor_name = const_input_name + ':0'
        ret = {'name': const_input_name,
               'type': 'Const',
               'input': [],
               'output': [(out_tensor_name, list(const_value.shape))],
               'attr': {'value': const_value, 'dtype': const_value.dtype.name},
               'opcode_version': 1
               }
    else:
        WARN('[Parser]: Fail to get const value for (%s) in layer %s!' % (const_input_name, layer.name))
    return ret


def get_node_content(layer):
    layer_outputs = layer.output if isinstance(layer.output, list) else [layer.output]
    output_name_shape = [(out.name, out.shape.as_list() if out.shape is not None else [])
                         for out in layer_outputs]
    ret = {'name': layer.name,
           'type': get_node_type(layer),
           'output': output_name_shape,
           'attr': get_node_attr(layer),
           'opcode_version': 2
           }
    return ret


def get_nodes_content(layers, model_configs):
    layers = layers if isinstance(layers, list) else [layers]
    input_nodes_dict, const_inputs_dict, extra_attrs_dict = get_nodes_input_and_attr(model_configs)
    nodes_content = []
    for layer in layers:
        node_input_info = input_nodes_dict.get(layer.name, [])
        extra_attr_dict = extra_attrs_dict.get(layer.name, {})
        const_input_names = const_inputs_dict.get(layer.name, [])
        # if hasattr(layer, 'layers'):
        #     nodes_content.extend(get_nodes_content(layer.layers, exp_input_names))
        #     continue
        node_content = get_node_content(layer)
        if extra_attr_dict:
            node_content['attr'].update(extra_attr_dict)
        node_content.update({'input': node_input_info})
        nodes_content.append(node_content)
        for const_input in const_input_names:
            nodes_content.append(get_const_node_content(layer, const_input))
    return nodes_content


def parse_keras(model_path, params):
    nodes = list()
    nodes_dict, tensors, np_tensors = OrderedDict(), OrderedDict(), OrderedDict()
    input_shapes = params['input_shapes'].copy()
    try:
        tf.compat.v1.enable_eager_execution()
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
        if n['type'] == 'Const' and n['attr'].get('value', None) is not None:
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
        if isinstance(layer.output, list):
            outputs.extend(layer.output)
        else:
            outputs.append(layer.output)
    functors = K.function([model.input], outputs)
    outputs_value = functors(feed_model_inputs)
    for out, out_value in zip(outputs, outputs_value):
        tensors.update({out.name: out})
        np_tensors.update({out.name: out_value})

    tf.compat.v1.disable_eager_execution()

    return nodes, nodes_dict, tensors, np_tensors, input_shapes
