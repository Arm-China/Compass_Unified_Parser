# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import os
import numpy as np
from functools import partial
import mmap
from ...common.defs import TensorType


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

ONNX_NP_TENSOR_MAP = {
    0: ('UNDEFINED', None),
    1: ('FLOAT', np.float32),
    2: ('UINT8', np.uint8),
    3: ('INT8', np.int8),
    4: ('UINT16', np.uint16),
    5: ('INT16', np.int16),
    6: ('INT32', np.int32),
    7: ('INT64', np.int64),
    8: ('STRING', str),
    9: ('BOOL', bool),
    10: ('FLOAT16', np.float16),
    11: ('DOUBLE', np.double),
    12: ('UINT32', np.uint32),
    13: ('UINT64', np.uint64),
    14: ('COMPLEX64', np.complex64),
    15: ('COMPLEX128', np.complex128)
}


def onnx_tensor_decoder(pb, data_dir=''):
    ret = None
    data_type = pb.data_type
    assert data_type in ONNX_NP_TENSOR_MAP
    data_type_name, np_type = ONNX_NP_TENSOR_MAP[data_type]
    if data_type_name != 'UNDEFINED':
        if pb.data_location == 1 and len(pb.external_data) > 0:
            data_file_name = pb.external_data[0].value
            data_file_path = os.path.join(data_dir, data_file_name)
            if len(pb.external_data) >= 2:
                offset = int(pb.external_data[1].value)
            else:
                offset = 0
            if len(pb.external_data) >= 3:
                length = int(pb.external_data[2].value)
            else:
                length = int(np.prod(pb.dims)) * np.dtype(np_type).itemsize
            with open(data_file_path, 'rb') as f:
                m = mmap.mmap(f.fileno(), length, access=mmap.ACCESS_READ, offset=offset)
                ret = np.frombuffer(m, dtype=np_type)
        else:
            if len(pb.raw_data) > 0:
                assert data_type_name not in ['STRING', 'UNDEFINED']
                ret = np.frombuffer(pb.raw_data, dtype=np_type)
            else:
                if data_type_name in ['INT32', 'INT16', 'INT8', 'UINT16', 'UINT8', 'BOOL', 'FLOAT16']:
                    ret = np.array(pb.int32_data, dtype=np_type)
                elif data_type_name == 'INT64':
                    ret = np.array(pb.int64_data, dtype=np_type)
                elif data_type_name in ['UINT32', 'UINT64']:
                    ret = np.array(pb.uint64_data, dtype=np_type)
                elif data_type_name == 'STRING':
                    ret = np.array(pb.string_data, dtype=np_type)
                elif data_type_name in ['FLOAT', 'COMPLEX64']:
                    ret = np.array(pb.float_data, dtype=np_type)
                elif data_type_name in ['DOUBLE', 'COMPLEX128']:
                    ret = np.array(pb.double_data, dtype=np_type)
                else:
                    pass
    return ret


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
            field_value = ONNX_NP_TENSOR_MAP[field_value][1].__name__
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


def get_tensor_content(tensor_proto, data_dir=''):
    ret = {}
    tensor_shape = np.array([dim for dim in tensor_proto.dims], dtype=np.int64)
    tensor_type = tensor_proto.data_type
    _, np_type = ONNX_NP_TENSOR_MAP[tensor_type]
    value = onnx_tensor_decoder(tensor_proto, data_dir=data_dir)
    if value is not None:
        name_info = parse_proto_name(tensor_proto.name)
        ret.update(name_info)
        if len(tensor_shape) == 0:
            value = np.array(value).copy()
            assert len(
                value) == 1, 'The length of Tensor is invalid in get_tensor_content.'
            if np.ndim(value[0]) == 0:
                ret.update({'tensor': value[0].astype(np_type) if hasattr(
                    value[0], 'astype') else value[0]})
            else:
                ret.update({'tensor': np.array(value[0], dtype=np_type)})
        else:
            flat = np.array(value, np_type)
            if len(flat) == tensor_shape.prod():
                ret.update({'tensor': flat.reshape(tensor_shape)})
            else:
                ret.update({'tensor': flat})
        if ret.get('tensor', None) is not None \
                and ret['tensor'].dtype == 'float16':
            ret['tensor'] = ret['tensor'].astype(np.float32)
    return ret


def get_tensor_name(tensor_proto):
    ret = {}
    tensor_type = tensor_proto.data_type
    tensor_type_name, _ = ONNX_NP_TENSOR_MAP[tensor_type]
    if tensor_type_name != 'UNDEFINED':
        name_info = parse_proto_name(tensor_proto.name)
        ret.update(name_info)
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


def get_graph_content(graph_proto, data_dir=''):
    const_values = parse_proto(graph_proto.initializer, partial(get_tensor_content, data_dir=data_dir))
    const_names = list(parse_proto(graph_proto.initializer, get_tensor_name))
    inputs = list(parse_proto(graph_proto.input, get_value_content))
    outputs = list(parse_proto(graph_proto.output, get_value_content))
    nodes = list(parse_proto(graph_proto.node, get_node_content))

    output_index = {out.get('name', ''): i for i, out in enumerate(outputs)}
    for node in nodes:
        for node_out in node['output']:
            if node_out.get('name', '') in output_index:
                outputs[output_index[node_out.get(
                    'name', '')]]['out_port'] = node_out['out_port']

    return {'nodes': nodes, 'inputs': inputs, 'outputs': outputs, 'consts': const_values, 'const_names': const_names}


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
