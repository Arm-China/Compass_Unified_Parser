# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import numpy as np
import re
import copy


# TenserProto
# UNDEFINED = 0;
# FLOAT = 1; // float
# UINT8 = 2; // uint8_t
# INT8 = 3; // int8_t
# UINT16 = 4; // uint16_t
# INT16 = 5; // int16_t
# INT32 = 6; // int32_t
# INT64 = 7; // int64_t
# STRING = 8; // string
# BOOL = 9; // bool
# FLOAT16 = 10;
# DOUBLE = 11;
# UINT32 = 12;
# UINT64 = 13;
# COMPLEX64 = 14; // complex
# COMPLEX128 = 15; // complex
# BFLOAT16 = 16;
onnx_tensor_np_mapping = [
    ('UNDEFINED', None, lambda pb: pb),
    ('FLOAT', np.float32, lambda pb: pb.float_data if len(pb.float_data)
     > 0 else np.frombuffer(pb.raw_data, dtype=np.float32)),
    ('UINT8', np.uint8, lambda pb: pb.int32_data if len(pb.int32_data) > 0 else np.frombuffer(pb.raw_data, dtype=np.uint8)),
    ('INT8', np.int8, lambda pb: pb.int32_data if len(pb.int32_data) > 0 else np.frombuffer(pb.raw_data, dtype=np.int8)),
    ('UINT16', np.uint16, lambda pb: pb.int32_data if len(pb.int32_data) > 0 else np.frombuffer(pb.raw_data, dtype=np.uint16)),
    ('INT16', np.int16, lambda pb: pb.int32_data if len(pb.int32_data) > 0 else np.frombuffer(pb.raw_data, dtype=np.int16)),
    ('INT32', np.int32, lambda pb: pb.int32_data if len(pb.int32_data) > 0 else np.frombuffer(pb.raw_data, dtype=np.int32)),
    ('INT64', np.int64, lambda pb: pb.int64_data if len(pb.int64_data) > 0 else np.frombuffer(pb.raw_data, dtype=np.int64)),
    ('STRING', str, lambda pb: pb.string_data),
    ('BOOL', bool, lambda pb: pb.int32_data if len(pb.int32_data) > 0 else np.frombuffer(pb.raw_data, dtype=bool)),
    ('FLOAT16', np.float16, lambda pb: pb.int32_data if len(pb.int32_data)
     > 0 else np.frombuffer(pb.raw_data, dtype=np.float16)),
    ('DOUBLE', np.double, lambda pb: pb.double_data if len(pb.double_data)
     > 0 else np.frombuffer(pb.raw_data, dtype=np.float64)),
    ('UINT32', np.uint32, lambda pb: pb.uint64_data if len(pb.uint64_data)
     > 0 else np.frombuffer(pb.raw_data, dtype=np.uint32)),
    ('UINT64', np.uint64, lambda pb: pb.uint64_data if len(pb.uint64_data)
     > 0 else np.frombuffer(pb.raw_data, dtype=np.uint64)),
    ('COMPLEX64', np.complex64, lambda pb: pb.float_data),
    ('COMPLEX128', np.complex128, lambda pb: pb.double_data),
]


onnx_tensor_type_decode = {index: map_info for index,
                           map_info in enumerate(onnx_tensor_np_mapping)}


def parse_proto_name(proto_name):
    name_splits = proto_name.split(':')
    try:
        if len(name_splits) > 1 and len(name_splits[-1]) > 0:
            port = int(name_splits[-1][0])
        else:
            port = 0
    except:
        port = 0
    return {'name': proto_name, 'out_port': port}


def get_tensor_shape_content(tensor_shape_proto):
    shape_list = []
    for d in tensor_shape_proto.dim:
        dim_value = d.dim_value
        if dim_value == 0:
            try:
                if 'batch' in getattr(d, 'dim_param', '') \
                        or 'N' in getattr(d, 'dim_param', ''):
                    dim_value = 1
                else:
                    dim_value = int(getattr(d, 'dim_param', ''))
            except:
                pass
        shape_list.append(dim_value)
    return np.array([dim for dim in shape_list], dtype=np.int64)


def get_tensor_message(tensor_message):
    ret = {}
    for field in tensor_message.ListFields():
        field_name = field[0].name
        field_value = field[1]
        if field_name == 'elem_type':
            field_value = onnx_tensor_np_mapping[field_value][1].__name__
        elif field_name == 'shape':
            field_value = get_tensor_shape_content(field[1])
        else:
            continue
        ret.update({field_name: field_value})
    return ret


def get_type_content(type_proto):
    ret = {}
    for attr in ['tensor_type', 'sequence_type', 'map_type']:
        if type_proto.HasField(attr):
            ret.update({attr: get_tensor_message(
                getattr(type_proto, 'tensor_type'))})
            break
    return ret


def get_tensor_content(tensor_proto):
    ret = {}
    tensor_shape = np.array([dim for dim in tensor_proto.dims], dtype=np.int64)
    tensor_type = tensor_proto.data_type
    tensor_type_name, np_type, tensor_field = onnx_tensor_type_decode[tensor_type]
    if tensor_type_name != 'UNDEFINED':
        name_info = parse_proto_name(tensor_proto.name)
        ret.update(name_info)
        if len(tensor_shape) == 0:
            value = tensor_field(tensor_proto)
            value = np.array(value).copy()
            assert len(
                value) == 1, 'The length of Tensor is invalid in get_tensor_content.'
            if np.ndim(value[0]) == 0:
                ret.update({'tensor': value[0].astype(np_type) if hasattr(
                    value[0], 'astype') else value[0]})
            else:
                ret.update({'tensor': np.array(value[0], dtype=np_type)})
        else:
            flat = np.array(np.array(tensor_field(tensor_proto), np_type))
            if len(flat) == tensor_shape.prod():
                ret.update({'tensor': flat.reshape(tensor_shape)})
            else:
                ret.update({'tensor': flat})
        if ret.get('tensor', None) is not None \
                and ret['tensor'].dtype == 'float16':
            ret['tensor'] = ret['tensor'].astype(np.float32)
    return ret


def get_value_content(value_proto):
    name_info = parse_proto_name(value_proto.name)
    ret = {'type': get_type_content(value_proto.type)}
    ret.update(name_info)
    return ret


onnx_attr_convert_mapping = [
    ('UNDEFINED', lambda pb: pb),
    ('FLOAT', lambda pb: pb.f),
    ('INT', lambda pb: np.int64(pb.i)),
    ('STRING', lambda pb: pb.s.decode('utf-8')),
    ('TENSOR', lambda pb: get_tensor_content(pb.t)),
    ('GRAPH', lambda pb: get_graph_content(pb.g)),
    ('FLOATS', lambda pb: list(pb.floats)),
    ('INTS', lambda pb: list(pb.ints)),
    ('STRINGS', lambda pb: list(map(lambda string: string.decode('utf-8'), pb.strings))),
    ('TENSORS', lambda pb: list(map(get_tensor_content, pb.tensors))),
    ('GRAPHS', lambda pb: list(map(get_graph_content, pb.graphs))),
    ('SPARSE_TENSOR', lambda pb: pb.sparse_tensor),
    ('SPARSE_TENSORS', lambda pb: pb.sparse_tensors),
]


def get_attribute_content(attribute_proto):
    return onnx_attr_convert_mapping[attribute_proto.type][1](attribute_proto)


def get_node_content(node_proto):
    name_info = parse_proto_name(node_proto.name)

    output = []
    for i, o in enumerate(node_proto.output):
        meta_info = parse_proto_name(o)
        if meta_info.get('out_port', 0) != i:
            meta_info['out_port'] = i
        output.append(meta_info)

    ret = {'type': node_proto.op_type,
           'input': list(map(parse_proto_name, node_proto.input)),
           'output': output,
           'domain': node_proto.domain,
           }
    ret.update(name_info)
    ret.update({attr.name: get_attribute_content(attr)
                for attr in node_proto.attribute})
    return ret


def get_graph_content(graph_proto):
    const_values = list(parse_proto(
        graph_proto.initializer, get_tensor_content))
    inputs = list(parse_proto(graph_proto.input, get_value_content))
    outputs = list(parse_proto(graph_proto.output, get_value_content))
    nodes = list(parse_proto(graph_proto.node, get_node_content))

    output_index = {out.get('name', ''): i for i, out in enumerate(outputs)}
    for node in nodes:
        for node_out in node['output']:
            if node_out.get('name', '') in output_index:
                outputs[output_index[node_out.get(
                    'name', '')]]['out_port'] = node_out['out_port']

    return {'nodes': nodes, 'inputs': inputs, 'outputs': outputs, 'consts': const_values}


def get_opset_content(opset_proto):
    ret = {
        'version': getattr(opset_proto, 'version'),
        'domain': getattr(opset_proto, 'domain')
    }
    return ret


def parse_proto(container, func):
    for elem in container:
        value = func(elem)
        yield value


def get_model_content(model_proto):
    ret = {}
    for attr in ['ir_version', 'producer_name', 'producer_version', 'model_version', 'domain', 'opset_import']:
        if attr == 'opset_import':
            ret.update(
                {attr: list(parse_proto(model_proto.opset_import, get_opset_content))})
        else:
            ret.update({attr: getattr(model_proto, attr)})
    return ret
