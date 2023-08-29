# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import re
import copy
import tensorflow as tf
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


tf_types_convert_mapping = [
    ('DT_INVALID', None, lambda pb: pb),
    ('DT_FLOAT', np.float32, lambda pb: np.float32(pb.float_val)),
    ('DT_DOUBLE', np.float64, lambda pb: np.float64(pb.double_val)),
    ('DT_INT32', np.int32, lambda pb: np.int32(pb.int_val)),
    ('DT_UINT8', np.uint8, lambda pb: np.uint8(pb.int_val)),
    ('DT_INT16', np.int16, lambda pb: np.int16(pb.int_val)),
    ('DT_INT8', np.int8, lambda pb: np.int8(pb.int_val)),
    ('DT_STRING', str, lambda pb: str(pb.string_val)),
    ('DT_COMPLEX64', np.complex64, lambda pb: np.complex64(pb.scomplex_val)),
    ('DT_INT64', np.int64, lambda pb: np.int64(pb.int64_val)),
    ('DT_BOOL', bool, lambda pb: pb.bool_val),
    ('DT_QINT8', 'qint8', lambda pb: pb.int_val),
    ('DT_QUINT8', 'quint8', lambda pb: pb.int_val),
    ('DT_QINT32', 'qint32', lambda pb: pb.int_val),
    ('DT_BFLOAT16', 'bfloat16', lambda pb: pb.half_val),
    ('DT_QINT16', 'qint16', lambda pb: pb.int_val),
    ('DT_QUINT16', 'quint16', lambda pb: pb.int_val),
    ('DT_UINT16', np.uint16, lambda pb: np.uint16(pb.int_val)),
    ('DT_COMPLEX128', np.complex128, lambda pb: np.complex128(pb.dcomplex_val)),
    ('DT_HALF', np.float16, lambda pb: np.float16(pb.half_val)),
    ('DT_RESOURCE', 'resouce', lambda pb: pb.resource_handle_val),
    ('DT_VARIANT', 'variant', lambda pb: pb.variant_val),
    ('DT_UINT32', np.uint32, lambda pb: np.uint32(pb.uint32_val)),
    ('DT_UINT64', np.uint64, lambda pb: np.uint64(pb.uint64_val)),
]


tf_attr_value_types_map = {
    's': lambda x: x.decode('utf-8'),
    'i': lambda x: np.int64(x),
    'f': lambda x: float(x),
    'b': lambda x: bool(x),
    'type': lambda x: str(np.dtype(tf_types_convert_mapping[x][1]))
    if isinstance(tf_types_convert_mapping[x][1], type)
    else tf_types_convert_mapping[x][1],
    'shape': lambda x: parse_shape(x),
    'tensor': lambda x: parse_tensor(x),
    'list': lambda x: parse_list(x),
    'func': lambda x: x,
}


def get_attr_value_type(attr_value):
    type_str = ''
    for k in tf_attr_value_types_map.keys():
        if attr_value.HasField(k):
            type_str = k
            break
    if not type_str:
        ERROR('[Parser]: Meets unsupported attr value type!')
    return type_str


def parse_shape(tensor_shape_proto):
    unknown_rank = tensor_shape_proto.unknown_rank
    if unknown_rank:
        dim = np.array([], np.int64)
    else:
        dim = np.array(
            [dim.size for dim in tensor_shape_proto.dim], dtype=np.int64)
    return {'unkwown_rank': unknown_rank, 'dim': dim}


def parse_list(list_attr):
    ret = []
    if len(list_attr.ListFields()):
        type_name = list_attr.ListFields()[0][0].name
        ret = list(
            map(tf_attr_value_types_map[type_name], list_attr.ListFields()[0][1]))
    return ret


def parse_tensor(tensor_pb):
    shape = np.array(
        [dim.size for dim in tensor_pb.tensor_shape.dim], dtype=np.int64)
    dtype = tensor_pb.dtype
    if len(shape) == 0:
        value = tf_types_convert_mapping[dtype][2](tensor_pb)
        if tf_types_convert_mapping[dtype][0] == 'DT_STRING':
            return np.array(value, dtype='<U0')
        else:
            value = np.atleast_1d(value).copy()
            if len(value) == 1:
                return np.array(value[0], dtype=tf_types_convert_mapping[dtype][1])
            elif not tensor_pb.tensor_content:
                WARN('[Parser]: Meet invalid tensor_pb in parse_tensor!')
                return np.array(value, dtype=tf_types_convert_mapping[dtype][1])

    if tensor_pb.tensor_content:
        flat = np.array(np.frombuffer(
            tensor_pb.tensor_content, tf_types_convert_mapping[dtype][1]))
        if len(flat) == shape.prod():
            return flat.reshape(shape)
        else:
            return flat
    else:
        value = np.array(tf_types_convert_mapping[dtype][2](
            tensor_pb), dtype=tf_types_convert_mapping[dtype][1])
        if len(value) == 0:
            value = np.zeros(shape, value.dtype)
        else:
            try:
                value = np.broadcast_to(value, shape=shape).copy()
            except:
                value = np.reshape(value, shape)
        return value


def parse_proto(container, func):
    for elem in container:
        value = func(elem)
        yield value


def parse_node_input(node_input):
    splitter = ':'
    is_control = node_input.startswith('^')
    name_input = node_input.split(splitter)
    if len(name_input) == 1:
        return name_input[0], 0, is_control
    else:
        return splitter.join(name_input[:-1]), int(name_input[-1]), is_control


def parse_node_attr(node_attr):
    ret = {}
    for k, v in node_attr.items():
        if k.startswith('_'):
            continue
        try:
            value_type = get_attr_value_type(v)
            value = tf_attr_value_types_map[value_type](getattr(v, value_type))
            ret.update({k: value})
        except Exception as e:
            WARN('[Parser]: Reading TF attr (%s) meets error (%s) in parse_node_attr!' % (
                k, str(e)))
        # DEBUG('[Parser]: Reading TF node info: key: (%s) ,value: (%s)!' %
        #       (str(k), str(value)))
    return ret


def get_node_content(node_proto):
    ret = {'name': node_proto.name,
           'type': node_proto.op,
           'input': list(map(parse_node_input, node_proto.input)),
           'attr': parse_node_attr(node_proto.attr),
           'opcode_version': 1
           }
    return ret


def get_op_content(operation):
    ret = get_node_content(operation.node_def)
    output = [(out.name, out.shape.as_list() if out.shape.dims is not None else None)
              for out in operation.outputs]
    ret.update({'output': output})
    return ret


def get_out_tensors(nodes):
    return [(out[0], tf.get_tensor_by_name(out[0])) for out in nodes['output']]


def get_function_node_content(function, nodes):
    '''Update nodes by replacing the original function node(for example swish_f32) with actual nodes.
    There are 4 main operations:
    1) Use function_def_to_graph to get graph of the function;
    2) Replace 'Placeholder' in function graph with 'Identity' and its input is the original input of
       function node;
    3) Change op type of function node to Identity and change its input to the output of function graph;
    4) Rename node name and tensor name in function graph because the function can be called multiple
       times.
    '''
    from tensorflow.python.framework.function_def_to_graph import function_def_to_graph
    func_name = function.signature.name
    func_inputs = [inp.name for inp in function.signature.input_arg]
    func_graph = function_def_to_graph(function)
    func_graph_outputs = list(map(parse_node_input, [out.name for out in func_graph.outputs]))
    if len(func_inputs) != 1 or len(func_graph_outputs) != 1:
        WARN('[Parser]: Meet tf function (%s) with more than 1 input/output, which is not yet supported!' % func_name)
    func_nodes = list(parse_proto(func_graph.get_operations(), get_op_content))
    nodes_to_add = []
    for idx, original_node in enumerate(nodes):
        if original_node['type'] != func_name:
            continue
        # Update op type of original node to 'Identity'
        nodes[idx]['type'] = 'Identity'
        # Update 'from_function' to True to note that its tensor may not exist in original model
        nodes[idx]['from_function'] = True
        original_node_outputs = original_node['output']
        original_node_inputs = original_node['input']
        name_prefix = original_node['name'] + '_'
        for fn in func_nodes:
            func_node = copy.deepcopy(fn)
            func_node['name'] = name_prefix + fn['name']
            if func_node['type'] == 'Placeholder':
                func_node['type'] = 'Identity'
                # Update input of Placeholder to the input of original node
                func_node['input'] = original_node_inputs
            else:
                func_node['input'] = [(name_prefix + name, in_port, is_control)
                                      for name, in_port, is_control in fn['input']]
            # rename node name and tensor name to avoid duplicate when the function is called multiple times
            if fn['name'] == func_graph_outputs[0][0]:
                # original function node becomes the last node in function graph
                nodes[idx]['input'] = [(func_node['name'], func_graph_outputs[0][1], func_graph_outputs[0][2])]
            else:
                func_node['output'] = [(name_prefix + name, shape) for name, shape in fn['output']]
            func_node['from_function'] = True
            nodes_to_add.append(func_node)
    nodes.extend(nodes_to_add)
    return nodes
