# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import sys
from enum import Enum, unique
import numpy as np
import copy
from ..logger import WARN, ERROR


FLOAT_MIN = np.finfo(np.float32).min
FLOAT_MAX = np.finfo(np.float32).max

INT_MAX = sys.maxsize
INT_MIN = - sys.maxsize - 1


def FLOAT_EQUAL(x, y): return np.all(
    (np.abs(x - y)) < (np.finfo(np.float32).resolution))


def FLOAT64_EQUAL(x, y): return np.all(
    (np.abs(x - y)) < (np.finfo(np.float64).resolution))


def TYPE_MIN(x):
    if isinstance(x, str):
        x = np.dtype(x)
    if np.issubdtype(x, np.integer):
        return np.iinfo(x).min
    else:
        return np.finfo(x).min.astype(x)


def TYPE_MAX(x):
    if isinstance(x, str):
        x = np.dtype(x)
    if np.issubdtype(x, np.integer):
        return np.iinfo(x).max
    else:
        return np.finfo(x).max.astype(x)


@unique
class Framework(Enum):
    NONE = 0
    CAFFE = 1
    CAFFE2 = 2
    COREML = 3
    MXNET = 4
    ONNX = 5
    TENSORFLOW = 6
    TFLITE = 7
    TORCH = 8


def get_opset_version(version_number):
    ONNX_VERSION_OPSET_MAP = {
        '1.03': 8,
        '1.04': 9,
        '1.05': 10,
        '1.06': 11,
        '1.07': 12,
        '1.08': 13,
        '1.09': 14,
        '1.10': 15,
        '1.11': 16,
        '1.12': 17,
        '1.13': 18,
        '1.14': 19,
        '1.15': 20,
        '1.16': 21,
        '1.17': 22,
        '1.18': 23,
    }
    version_str = '%1.2f' % float(version_number)
    versions = sorted(
        [k for k in ONNX_VERSION_OPSET_MAP.keys()], key=lambda x: float(x))
    if versions:
        if float(version_str) < float(versions[0]):
            WARN('[Parser]: Meets too low Onnx version! Please upgrade!')
            return ONNX_VERSION_OPSET_MAP[versions[0]]
        elif float(version_str) > float(versions[-1]):
            WARN(
                '[Parser]: Meets too high Onnx version! Please downgrade to %s!' % versions[-1])
            increment = int(version_str.split(
                '.')[-1]) - int(versions[-1].split('.')[-1])
            return ONNX_VERSION_OPSET_MAP[versions[-1]] + increment
        else:
            return ONNX_VERSION_OPSET_MAP[version_str]
    else:
        ERROR('[Parser]: No Onnx version is supported in get_opset_version!')
        return None


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
@unique
class TensorType(Enum):
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16


class Tensor(object):
    DEFAULTS = {'name': '',
                'supported_types': [],
                'value': None,
                'shape': None,
                'required': False,
                'is_const': False,
                'min_max': tuple(),
                'scale_zp': tuple(),
                'activation_quantization_axis': None,
                'dtype': None,
                }

    def __init__(self, **kwargs):
        super(Tensor, self).__init__()
        for attr in Tensor.DEFAULTS.keys():
            setattr(self, attr, kwargs.get(attr, Tensor.DEFAULTS[attr]))
        st = getattr(self, 'supported_types', [])
        if getattr(self, 'value') is not None:
            setattr(self, 'shape', getattr(self, 'value').shape)
            if getattr(self, 'value').dtype.name not in st:
                st.append(getattr(self, 'value').dtype.name)
                setattr(self, 'supported_types', st)

    def get_dtype(self):
        value = getattr(self, 'value')
        if value is not None:
            ret = str(value.dtype)
            if ret == 'object':
                ret = None
            return ret
        else:
            dtype = getattr(self, 'dtype')
            if dtype is not None:
                return str(dtype)
            else:
                return None

    def get_shape(self):
        value = getattr(self, 'value')
        if value is not None:
            return value.shape
        else:
            return getattr(self, 'shape')


@unique
class AttrType(Enum):
    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSOR = 11
    SPARSE_TENSORS = 12
    BOOL = 13


class Attribute(object):
    DEFAULTS = {'key': None, 'type': AttrType.UNDEFINED, 'value': None,
                'options': [], 'default': None, 'required': False}

    def __init__(self, key, params):
        for k, v in Attribute.DEFAULTS.items():
            if k == 'key':
                setattr(self, k, key)
            else:
                setattr(self, k, params.get(k, copy.deepcopy(v)))
        if 'type' in params and params['type'].name == 'BOOL':
            setattr(self, 'options', [True, 1, False, 0])
        default_value = getattr(self, 'default', None)
        if getattr(self, 'value', None) is None and default_value is not None:
            setattr(self, 'value', default_value)

    def __getattr__(self, name):
        pass

    def __setattr__(self, key, value):
        super(Attribute, self).__setattr__(key, value)
        assert key in Attribute.DEFAULTS, (
            'Not supported attribute of [%s]!' % key)

    def update(self, params):
        for k, v in params.items():
            if k in Attribute.DEFAULTS.keys():
                setattr(self, k, copy.deepcopy(v))
