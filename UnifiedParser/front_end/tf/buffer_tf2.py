# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from collections import OrderedDict
from .utils import trim_tensor_name
from ...common.errors import *


def get_nodes_name_and_attr(configs):
    '''Get all the nodes from model config. For constants, it may not be saved as a layer by tf.
    Therefore we need to record all the nodes' name to create Const node for them later.
    '''
    input_nodes_dict = {}
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
        input_nodes_name = []
        node_attr_dict = {}
        for nodes in inbound_nodes:
            if not isinstance(nodes, list):
                continue
            for node in nodes:
                if isinstance(node, str):
                    input_nodes_name.append(node)
                elif isinstance(node, dict):
                    for key, value in node.items():
                        if key == 'name':
                            continue
                        if key == 'dtype' and class_name == 'TFOpLambda' and function == 'cast':
                            node_attr_dict.update({'DstT': value})
                        elif isinstance(value, list) and len(value) == 3 and isinstance(value[0], str):
                            input_nodes_name.append(value[0])
                        elif value is not None:
                            # Some inputs in tf2 are attributes for raw ops, so save them to both
                            # inputs and attributes.
                            const_node_name = node_name + '/' + key
                            input_nodes_name.append(const_node_name)
                            node_attr_dict.update({key: value})
        input_nodes_dict.update({node_name: input_nodes_name})
        nodes_attr_dict.update({node_name: node_attr_dict})
    return input_nodes_dict, nodes_attr_dict


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
                   'symbol', 'stateful', 'submodules', 'supports_masking',
                   'trainable_weights', 'trainable_variables',
                   'updates', 'variable_dtype'):
            # Ignore inputs/outputs and other attributes that are not used
            continue
        try:
            if eval('callable(layer.' + key + ')'):
                continue
            value = eval('layer.' + key)
            ret.update({key: copy.deepcopy(value)})
        except Exception as e:
            DEBUG('[Parser]: Fail to get key (%s) for layer (%s) because %s' % (key, layer.name, str(e)))
            continue
        ret.update({key: value})
        DEBUG('[Parser]: layer: %s, key: %s, value: %s' % (layer.name, key, str(ret[key])))
    return ret


def get_node_input(layer, expect_input_names):
    input_infos, input_nodes, cur_input_names = [], [], []
    for node in layer.inbound_nodes:
        input_nodes.extend(node.parent_nodes)
    for node in input_nodes:
        node_name = node.layer.name
        tensor_name = node.outputs.name
        is_control = tensor_name.startswith('^')
        name_input = tensor_name.split(':')
        if len(name_input) == 1:
            ret = (node_name, 0, is_control)
        else:
            ret = (node_name, int(name_input[-1]), is_control)
        input_infos.append(ret)
        cur_input_names.append(node_name)
    const_input_names = []
    for idx, exp_inp_name in enumerate(expect_input_names):
        if exp_inp_name not in cur_input_names:
            const_input_info = (exp_inp_name, 0, False)
            input_infos.insert(idx, const_input_info)
            const_input_names.insert(idx, exp_inp_name)
    return input_infos, const_input_names


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
               'output': [out_tensor_name, list(const_value.shape)],
               'attr': {'value': const_value, 'dtype': const_value.dtype},
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
    input_nodes_dict, extra_attrs_dict = get_nodes_name_and_attr(model_configs)
    nodes_content = []
    for layer in layers:
        exp_input_names = input_nodes_dict.get(layer.name, [])
        extra_attr_dict = extra_attrs_dict.get(layer.name, {})
        # if hasattr(layer, 'layers'):
        #     nodes_content.extend(get_nodes_content(layer.layers, exp_input_names))
        #     continue
        node_content = get_node_content(layer)
        if extra_attr_dict:
            node_content['attr'].update(extra_attr_dict)
        node_input_info, const_input_names = get_node_input(layer, exp_input_names)
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
        load_options = tf.saved_model.LoadOptions(allow_partial_checkpoint=True)
        model = tf.keras.models.load_model(model_path, compile=False, options=load_options)
    except Exception as e:
        WARN('[Parser]: Reading saved model/h5 file (%s) meets error (%s)!' %
             (model_path, str(e)))
        return nodes, nodes_dict, tensors, np_tensors, input_shapes

    nodes = get_nodes_content(model.layers, model.get_config())
    model_inputs = model.inputs
    model_inputs_names = [trim_tensor_name(inp_name) for inp_name in model.input_names]

    for n in nodes:
        if n['name'] in model_inputs_names and n['name'] not in input_shapes:
            tensor_shape = n['output'][0][1]
            input_shapes.update({n['name']: tensor_shape})
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
    outputs = [layer.output for layer in model.layers if layer.name not in input_shapes.keys()]
    functors = K.function([model.input], outputs)
    outputs_value = functors(feed_model_inputs)
    for out, out_value in zip(outputs, outputs_value):
        tensors.update({out.name: out})
        np_tensors.update({out.name: out_value})

    return nodes, nodes_dict, tensors, np_tensors, input_shapes
