# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from collections.abc import Iterable
from functools import partial
from enum import Enum
import sys
import numpy as np
import os
import re
from ...common.utils import is_file
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL
try:
    from .caffe_pb2 import *
except TypeError as e:
    WARN('[Parser]: Fail to import caffe_pb2 because %s' % str(e))
    from caffe.proto.caffe_pb2 import *


def replace_caffe_pb(file_path):
    import importlib
    folder, name = os.path.split(os.path.realpath(file_path))
    path = sys.path
    sys.path = [folder]
    name = name.replace(".py", "")
    try:
        mdl = importlib.import_module(name)
    except TypeError:
        FATAL("When using customize proto, please export environment variable:\n export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python\n for using mutil-caffe.proto")
    sys.path = path
    if '__all__' in mdl.__dict__:
        names = mdl.__dict__['__all__']
    else:
        names = [x for x in mdl.__dict__ if not x.startswith("_")]
    globals().update({k: getattr(mdl, k) for k in names})


def trim_blob(blob):
    ret = None
    if 'data' in blob and 'shape' in blob:
        ret = np.reshape(blob['data'], newshape=blob['shape']['dim'].tolist())
    elif all([item in blob.keys() for item in ['num', 'channels', 'height', 'width', 'data']]):
        shape = [blob[item] for item in ['num', 'channels', 'height', 'width']]
        ret = np.reshape(blob['data'], newshape=shape)
    elif 'data' in blob and ('shape' not in blob or ('shape' in blob and 'dim' not in blob['shape'])):
        ret = np.squeeze(blob['data'])
    else:
        WARN('[Parser]: Blob type is not supportted in trim_blob!')
    return ret


def parse_number(np_type):
    return lambda pb: np.array(pb, dtype=np_type) if isinstance(pb, Iterable) else np_type(pb)


def parse_string(string_proto):
    return string_proto if isinstance(string_proto, str) else list(string_proto)


def parse_shape(shape_proto):
    return list(shape_proto.dim) if shape_proto is not None else []


def parse_enum(enum_proto):
    try:
        return enum_proto[0].enum_type.values_by_number[enum_proto[1]].name
    except:
        module_name = enum_proto[0].full_name.split('.')[-2]
        type_name = enum_proto[0].enum_type.name
        enum_value = enum_proto[1]
        try:
            enum_obj = eval(
                'getattr(sys.modules[__name__], module_name).' + type_name)
            index = enum_obj.values().index(enum_value)
            return enum_obj.keys()[index]
        except:
            enum_obj = eval(
                'getattr(sys.modules[__name__], module_name).' + enum_proto[0].name)
            return list(enum_obj.DESCRIPTOR.enum_type.values_by_name)[enum_value]


def parse_message(message_proto):
    ret = {}
    if hasattr(message_proto, 'ListFields'):
        for f in message_proto.ListFields():
            ret.update({f[0].name: get_attribute_content(f)})
    else:
        values = []
        for meta_proto in message_proto:
            meta_ret = {}
            if hasattr(meta_proto, 'ListFields'):
                for f in meta_proto.ListFields():
                    meta_ret.update({f[0].name: get_attribute_content(f)})
            values.append(meta_ret)
        if hasattr(message_proto, 'name'):
            ret.update({message_proto.name: values})
        else:
            ret = values
    return ret

# DOUBLE = 1
# FLOAT = 2
# INT64 = 3
# UINT64 = 4
# INT32 = 5
# FIXED64 = 6
# FIXED32 = 7
# BOOL = 8
# STRING = 9
# GROUP = 10
# MESSAGE = 11
# BYTES = 12
# UINT32 = 13
# ENUM = 14
# SFIXED32 = 15
# SFIXED64 = 16
# SINT32 = 17
# SINT64 = 18


caffe_attr_convert_mapping = [
    ('NONE', lambda pb: pb),
    ('DOUBLE', lambda pb: parse_number(np.float64)(pb[1])),
    ('FLOAT', lambda pb: parse_number(np.float32)(pb[1])),
    ('INT64', lambda pb: parse_number(np.int64)(pb[1])),
    ('UINT64', lambda pb: parse_number(np.uint64)(pb[1])),
    ('INT32', lambda pb: parse_number(np.int32)(pb[1])),
    ('FIXED64', lambda pb: pb),
    ('FIXED32', lambda pb: pb),
    ('BOOL', lambda pb: pb[1]),
    ('STRING', lambda pb: parse_string(pb[1])),
    ('GROUP', lambda pb: pb),
    ('MESSAGE', lambda pb: parse_message(pb[1])),
    ('BYTES', lambda pb: pb),
    ('UINT32', lambda pb: parse_number(np.uint32)(pb[1])),
    ('ENUM', lambda pb: parse_enum(pb)),
    ('SFIXED32', lambda pb: pb),
    ('SFIXED64', lambda pb: pb),
    ('SINT32', lambda pb: pb),
    ('SINT64', lambda pb: pb),
]


def get_attribute_content(attribute_proto):
    return caffe_attr_convert_mapping[attribute_proto[0].type][1](attribute_proto)


def parse_layer(layer_proto):
    ret = {}
    for field in layer_proto.ListFields():
        try:
            if field[0].name in ('include', 'phase'):
                continue
            value = get_attribute_content(field)
            if field[0].name == 'blobs':
                value = list(map(trim_blob, value))
            elif field[0].name == 'type':
                value = value.upper()
            ret.update({field[0].name: value})
        except Exception as e:
            WARN('[Parser]: Parsing layer meets error (%s : %s)' %
                 (field[0].name, str(e)))
    return ret


def parse_netparam(caffe_model):
    ret = {}
    name = caffe_model.name
    layers = list(map(parse_layer, caffe_model.layers if bool(
        caffe_model.layers) else caffe_model.layer))
    inputs = list(map(parse_string, caffe_model.input))
    shapes = list(map(parse_shape, getattr(caffe_model, 'input_shape', None)))
    if not shapes:
        shapes = list(getattr(caffe_model, 'input_dim', None))
    ret.update({'name': name, 'layers': layers,
                'inputs': inputs, 'shapes': shapes})
    return ret


def copy_trained_layers_from(dest, src):
    dest_layers = dest.layers if len(dest.layers) else dest.layer
    src_layers = src.layers if len(src.layers) else src.layer
    layers_names = {l.name: i for i, l in enumerate(dest_layers)}
    for layer in src_layers:
        if layer.name in layers_names:
            # copy
            dest_layer = dest_layers[layers_names[layer.name]]
            if len(dest_layer.blobs) == len(layer.blobs):
                for i in range(len(layer.blobs)):
                    dest_layer.blobs[i].FromProto(layer.blobs[i], False)
            else:
                # copy
                dest_layer.blobs.MergeFrom(layer.blobs)
        else:
            # ignoring
            pass
    return dest


def clean_caffe_net(net):
    # remove duplicate layers
    layers = net.layers if len(net.layers) else net.layer
    layers_names = [l.name for l in layers]
    duplicated_names = [
        name for name in layers_names if layers_names.count(name) >= 2]
    duplicated_names = list(set(duplicated_names))
    if duplicated_names:
        pop_index = []
        for i, l in enumerate(layers):
            if l.name in duplicated_names:
                m_include = getattr(l, 'include', [])
                if len(m_include) >= 1:
                    if len(re.findall(r'TRAIN', str(list(m_include)[0]))) == 1:
                        pop_index.append(i)
        pop_index = sorted(pop_index, reverse=True)
        for i in pop_index:
            WARN('[Parser]: Meets duplicated layer (%s, id=%d) that will be removed! Please check prototxt!' % (
                layers[i].name, i))
            layers.pop(i)
    return net


def read_caffe_model(caffe_path, prototxt='', params=None):
    ret = False, None
    net = None
    if prototxt:
        try:
            import google.protobuf
            import google.protobuf.text_format
            try:
                net = NetParameter()
                google.protobuf.text_format.Merge(open(prototxt).read(), net)
                net = clean_caffe_net(net)
            except google.protobuf.text_format.ParseError as e:
                if params is not None and "caffe_proto" in params:
                    replace_caffe_pb(params["caffe_proto"])
                    net = NetParameter()
                    google.protobuf.text_format.Merge(
                        open(prototxt).read(), net)
                    net = clean_caffe_net(net)
                else:
                    WARN(
                        '[Parser]: Reading Caffe prototxt meets error (%s)! Please check prototxt!', str(e))
                    net = None
            except Exception as e:
                WARN(
                    '[Parser]: Reading Caffe prototxt meets error (%s)! Please check prototxt!', str(e))
                net = None
        except Exception as e:
            WARN(
                '[Parser]: Reading Caffe prototxt meets error (%s)! Please check prototxt!', str(e))
            net = None

    try:
        with open(caffe_path, 'rb') as f:
            caffe_model = NetParameter()
            caffe_model.ParseFromString(f.read())
        if net is not None:
            caffe_model = copy_trained_layers_from(net, caffe_model)
        model = parse_netparam(caffe_model)
        ret = True, model
    except IOError as e:
        WARN('[Parser]: Reading caffe model meets error! ' + str(e))
    except Exception as e:
        WARN('[Parser]: Parsing caffe model meets error! ' + str(e))
    return ret
