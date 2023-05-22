# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
import torch
import tensorflow as tf
import numpy as np
from ..op import *
from ...common.defs import FLOAT_MIN, FLOAT_MAX, FLOAT_EQUAL, TYPE_MIN, TYPE_MAX
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class AbsOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {},
                13: {}}

    def __init__(self, graph, attr_dict=None):
        super(AbsOp, self).__init__(graph, attr_dict)
        self.update_attributes(AbsOp, attr_dict)
        assert self.check_required(), 'AbsOp is missing a required parameter.'

    def infer_shape(self):
        super(AbsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.abs(inputs[0])
        self.set_out_tensor(out_tensor)


class AcosOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {7: {}}

    def __init__(self, graph, attr_dict=None):
        super(AcosOp, self).__init__(graph, attr_dict)
        self.update_attributes(AcosOp, attr_dict)
        assert self.check_required(), 'AcosOp is missing a required parameter.'

    def infer_shape(self):
        super(AcosOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arccos(*inputs)
        self.set_out_tensor(out_tensor)


class AcoshOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {}}

    def __init__(self, graph, attr_dict=None):
        super(AcoshOp, self).__init__(graph, attr_dict)
        self.update_attributes(AcoshOp, attr_dict)
        assert self.check_required(), 'AcoshOp is missing a required parameter.'

    def infer_shape(self):
        super(AcoshOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arccosh(*inputs)
        self.set_out_tensor(out_tensor)


class AddOp(OpNeedBroadcast, OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0},
                    'consumed_inputs': {'type': AttrType.INTS}},
                6: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0}},
                7: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(AddOp, self).__init__(graph, attr_dict)
        self.update_attributes(AddOp, attr_dict)
        assert self.check_required(), 'AddOp is missing a required parameter.'

    def infer_shape(self):
        super(AddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver <= 6:
            if self.broadcast:
                second_input = OpHasAxis.broadcast_to(
                    inputs[1], inputs[0].shape, int(self.axis))
            else:
                assert list(inputs[0].shape) == list(
                    inputs[1].shape), 'The lengths of input0 and input1 are not the same in AddOp infer shape.'
                second_input = inputs[1]
            out_tensor = np.add(inputs[0], second_input)
        else:
            out_tensor = np.add(*inputs)
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'broadcast':
                if cur_ver <= 6:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                          (cur_ver, type(self).__name__))
        except:
            ret = None
        if ret is None:
            ret = super(AddOp, self).__getattr__(item)
        return ret


class ArgMaxOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 0}, 'keepdims': {'default': 1}},
                11: {'axis': {'default': 0}, 'keepdims': {'default': 1}},
                12: {'axis': {'default': 0}, 'keepdims': {'default': 1}, 'select_last_index': {'type': AttrType.INT, 'default': 0}},
                13: {'axis': {'default': 0}, 'keepdims': {'default': 1}, 'select_last_index': {'type': AttrType.INT, 'default': 0}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArgMaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArgMaxOp, attr_dict)
        assert self.check_required(), 'ArgMaxOp is missing a required parameter.'

    def infer_shape(self):
        super(ArgMaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.argmax(inputs[0], axis=self.axis)
        if self.keepdims:
            out_tensor = np.expand_dims(out_tensor, axis=self.axis)
        self.set_out_tensor(out_tensor)


class ArgMinOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 0}, 'keepdims': {'default': 1}},
                11: {'axis': {'default': 0}, 'keepdims': {'default': 1}},
                12: {'axis': {'default': 0}, 'keepdims': {'default': 1}, 'select_last_index': {'type': AttrType.INT, 'default': 0}},
                13: {'axis': {'default': 0}, 'keepdims': {'default': 1}, 'select_last_index': {'type': AttrType.INT, 'default': 0}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArgMinOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArgMinOp, attr_dict)
        assert self.check_required(), 'ArgMinOp is missing a required parameter.'

    def infer_shape(self):
        super(ArgMinOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.argmin(inputs[0], axis=self.axis)
        if self.keepdims:
            out_tensor = np.expand_dims(out_tensor, axis=self.axis)
        self.set_out_tensor(out_tensor)


class AsinOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {7: {}}

    def __init__(self, graph, attr_dict=None):
        super(AsinOp, self).__init__(graph, attr_dict)
        self.update_attributes(AsinOp, attr_dict)
        assert self.check_required(), 'AsinOp is missing a required parameter.'

    def infer_shape(self):
        super(AsinOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arcsin(*inputs)
        self.set_out_tensor(out_tensor)


class AsinhOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {}}

    def __init__(self, graph, attr_dict=None):
        super(AsinhOp, self).__init__(graph, attr_dict)
        self.update_attributes(AsinhOp, attr_dict)
        assert self.check_required(), 'AsinhOp is missing a required parameter.'

    def infer_shape(self):
        super(AsinhOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arcsinh(*inputs)
        self.set_out_tensor(out_tensor)


class AtanhOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {}}

    def __init__(self, graph, attr_dict=None):
        super(AtanhOp, self).__init__(graph, attr_dict)
        self.update_attributes(AtanhOp, attr_dict)
        assert self.check_required(), 'AtanhOp is missing a required parameter.'

    def infer_shape(self):
        super(AtanhOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arctanh(*inputs)
        self.set_out_tensor(out_tensor)


class AtanOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {7: {}}

    def __init__(self, graph, attr_dict=None):
        super(AtanOp, self).__init__(graph, attr_dict)
        self.update_attributes(AtanOp, attr_dict)
        assert self.check_required(), 'AtanOp is missing a required parameter.'

    def infer_shape(self):
        super(AtanOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arctan(*inputs)
        self.set_out_tensor(out_tensor)


class BitwiseAndOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {18: {}}

    def __init__(self, graph, attr_dict=None):
        super(BitwiseAndOp, self).__init__(graph, attr_dict)
        self.update_attributes(BitwiseAndOp, attr_dict)
        assert self.check_required(), 'BitwiseAndOp is missing a required parameter.'

    def infer_shape(self):
        super(BitwiseAndOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.bitwise_and(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class BitwiseNotOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {18: {}}

    def __init__(self, graph, attr_dict=None):
        super(BitwiseNotOp, self).__init__(graph, attr_dict)
        self.update_attributes(BitwiseNotOp, attr_dict)
        assert self.check_required(), 'BitwiseNotOp is missing a required parameter.'

    def infer_shape(self):
        super(BitwiseNotOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.invert(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class BitwiseOrOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {18: {}}

    def __init__(self, graph, attr_dict=None):
        super(BitwiseOrOp, self).__init__(graph, attr_dict)
        self.update_attributes(BitwiseOrOp, attr_dict)
        assert self.check_required(), 'BitwiseOrOp is missing a required parameter.'

    def infer_shape(self):
        super(BitwiseOrOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.bitwise_or(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class BitwiseXorOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {18: {}}

    def __init__(self, graph, attr_dict=None):
        super(BitwiseXorOp, self).__init__(graph, attr_dict)
        self.update_attributes(BitwiseXorOp, attr_dict)
        assert self.check_required(), 'BitwiseXorOp is missing a required parameter.'

    def infer_shape(self):
        super(BitwiseXorOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.bitwise_xor(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class CeilOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 6: {}, 13: {}}

    def __init__(self, graph, attr_dict=None):
        super(CeilOp, self).__init__(graph, attr_dict)
        self.update_attributes(CeilOp, attr_dict)
        assert self.check_required(), 'CeilOp is missing a required parameter.'

    def infer_shape(self):
        super(CeilOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.ceil(inputs[0])
        self.set_out_tensor(out_tensor)


class ClipOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS},
                    'min': {'type': AttrType.FLOAT},
                    'max': {'type': AttrType.FLOAT}},
                6: {'min': {'type': AttrType.FLOAT, 'default': -3.402823e+38},
                    'max': {'type': AttrType.FLOAT, 'default': 3.402823e+38}},
                11: {},
                12: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(ClipOp, self).__init__(graph, attr_dict)
        self.update_attributes(ClipOp, attr_dict)
        assert self.check_required(), 'ClipOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item in ('min', 'max'):
                if cur_ver <= 6:
                    ret = self.__dict__['_attr'][item].value
                else:
                    inputs = self.get_input_tensors()
                    if item == 'min':
                        ret = inputs[1] if len(inputs) >= 2 else None
                        if ret is None:
                            ret = TYPE_MIN(inputs[0].dtype)
                    else:
                        ret = inputs[2] if len(inputs) >= 3 else None
                        if ret is None:
                            ret = TYPE_MAX(inputs[0].dtype)
        except:
            ret = None
        if ret is None:
            ret = super(ClipOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ClipOp, self).infer_shape()
        inputs = self.get_input_tensors()
        dtype = inputs[0].dtype
        if self.cur_version <= 6:
            out_tensor = np.clip(inputs[0], self.min, self.max).astype(dtype)
        else:
            min_v = inputs[1] if len(inputs) >= 2 else None
            max_v = inputs[2] if len(inputs) >= 3 else None
            if min_v is None:
                min_v = TYPE_MIN(inputs[0].dtype)
            if min_v is None:
                max_v = TYPE_MAX(inputs[0].dtype)
            out_tensor = np.clip(inputs[0], min_v, max_v).astype(dtype)
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            from ...front_end.onnx.passes.common_passes import insert_constant
            in_edges = self._graph.sorted_in_edges(self.name)
            inputs = self.get_input_tensors()
            min_val, max_val = None, None
            if cur_ver <= 6:
                min_val, max_val = self.min, self.max
            else:
                if len(inputs) == 2:
                    min_val = inputs[1]
                elif len(inputs) == 3:
                    min_val, max_val = inputs[1], inputs[2]
            if min_val is None:
                min_val = TYPE_MIN(inputs[0].dtype)
            if max_val is None:
                max_val = TYPE_MAX(inputs[0].dtype)
            self._graph.remove_edges_from(in_edges[1:])
            insert_constant(self._graph, self.name + '_min',
                            np.array(min_val), self.name, in_port=1)
            insert_constant(self._graph, self.name + '_max',
                            np.array(max_val), self.name, in_port=2)
            self.cur_version = max_ver


class CosOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {7: {}}

    def __init__(self, graph, attr_dict=None):
        super(CosOp, self).__init__(graph, attr_dict)
        self.update_attributes(CosOp, attr_dict)
        assert self.check_required(), 'CosOp is missing a required parameter.'

    def infer_shape(self):
        super(CosOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.cos(*inputs)
        self.set_out_tensor(out_tensor)


class CoshOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {}}

    def __init__(self, graph, attr_dict=None):
        super(CoshOp, self).__init__(graph, attr_dict)
        self.update_attributes(CoshOp, attr_dict)
        assert self.check_required(), 'CoshOp is missing a required parameter.'

    def infer_shape(self):
        super(CoshOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.cosh(*inputs)
        self.set_out_tensor(out_tensor)


class CumSumOp(OpHasOneOutPort, OpHasAxis, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {'exclusive': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                'reverse': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}},
                14: {'exclusive': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                'reverse': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}},
                }

    def __init__(self, graph, attr_dict=None):
        super(CumSumOp, self).__init__(graph, attr_dict)
        self.update_attributes(CumSumOp, attr_dict)
        assert self.check_required(), 'CumsumOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item in ('exclusive', 'reverse'):
                ret = bool(self.__dict__['_attr'][item].value)
            elif item in ('axis',):
                inputs = self.get_input_tensors()
                ret = int(inputs[1])
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.INT, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(CumSumOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item in ('exclusive', 'reverse'):
            self.__dict__['_attr'][item].value = int(value)
        else:
            super(CumSumOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(CumSumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.cumsum(
            inputs[0], axis=inputs[1], exclusive=self.exclusive, reverse=self.reverse).numpy()
        self.set_out_tensor(out_tensor)


class DivOp(OpNeedBroadcast, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0},
                    'consumed_inputs': {'type': AttrType.INTS}},
                6: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0}},
                7: {},
                13: {},
                14: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(DivOp, self).__init__(graph, attr_dict)
        self.update_attributes(DivOp, attr_dict)
        assert self.check_required(), 'DivOp is missing a required parameter.'

    def infer_shape(self):
        super(DivOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver <= 6:
            if self.broadcast:
                second_input = OpHasAxis.broadcast_to(
                    inputs[1], inputs[0].shape, int(self.axis))
            else:
                assert list(inputs[0].shape) == list(
                    inputs[1].shape), 'The lengths of input0 and input1 are not the same in DivOp infer shape.'
                second_input = inputs[1]
            if re.search(r'int', inputs[0].dtype.name) is not None and re.search(r'int', second_input.dtype.name) is not None:
                out_tensor = inputs[0] // second_input
            else:
                out_tensor = np.true_divide(inputs[0], second_input)
        else:
            if all([re.search(r'int', na.dtype.name) for na in inputs]):
                out_tensor = inputs[0] // inputs[1]
            else:
                out_tensor = np.true_divide(*inputs)
        self.set_out_tensor(out_tensor)


class EinsumOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {12: {'equation': {'type': AttrType.STRING, 'required': True}
                     }}

    def __init__(self, graph, attr_dict=None):
        super(EinsumOp, self).__init__(graph, attr_dict)
        self.update_attributes(EinsumOp, attr_dict)
        assert self.check_required(), 'EinsumOp is missing a required parameter.'

    def infer_shape(self):
        super(EinsumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 2, 'Currently only two inputs are supported.'
        out_tensor = torch.einsum(self.equation, torch.from_numpy(
            inputs[0]), torch.from_numpy(inputs[1])).numpy()
        self.set_out_tensor(out_tensor)


class EluOp(LayoutUnawareOp, BaseActivationOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 1.},
                    'consumed_inputs': {'type': AttrType.INTS}
                    },
                6: {'alpha': {'type': AttrType.FLOAT, 'default': 1.}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(EluOp, self).__init__(graph, attr_dict)
        self.update_attributes(EluOp, attr_dict)
        assert self.check_required(), 'EluOp is missing a required parameter.'

    def infer_shape(self):
        super(EluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0])
        mask = out_tensor <= 0
        out_tensor[mask] = self.alpha * (np.exp(out_tensor[mask]) - 1)
        self.set_out_tensor(out_tensor)


class ErfOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {}, 13: {}}

    def __init__(self, graph, attr_dict=None):
        super(ErfOp, self).__init__(graph, attr_dict)
        self.update_attributes(ErfOp, attr_dict)
        assert self.check_required(), 'ErfOp is missing a required parameter.'

    def infer_shape(self):
        super(ErfOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = torch.erf(torch.from_numpy(np.array(inputs[0]))).numpy()
        self.set_out_tensor(out_tensor)


class ExpOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 6: {}, 13: {}}

    def __init__(self, graph, attr_dict=None):
        super(ExpOp, self).__init__(graph, attr_dict)
        self.update_attributes(ExpOp, attr_dict)
        assert self.check_required(), 'ExpOp is missing a required parameter.'

    def infer_shape(self):
        super(ExpOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.exp(inputs[0])
        self.set_out_tensor(out_tensor)


class FloorOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 6: {}, 13: {}}

    def __init__(self, graph, attr_dict=None):
        super(FloorOp, self).__init__(graph, attr_dict)
        self.update_attributes(FloorOp, attr_dict)
        assert self.check_required(), 'FloorOp is missing a required parameter.'

    def infer_shape(self):
        super(FloorOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.floor(inputs[0]).astype(np.float32)
        self.set_out_tensor(out_tensor)


class GemmOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                    'beta': {'type': AttrType.FLOAT, 'default': 1.0},
                    'broadcast': {'type': AttrType.INT, 'default': 0},
                    'transA': {'type': AttrType.INT, 'default': 0},
                    'transB': {'type': AttrType.INT, 'default': 0}
                    },
                6: {'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                    'beta': {'type': AttrType.FLOAT, 'default': 1.0},
                    'broadcast': {'type': AttrType.INT, 'default': 0},
                    'transA': {'type': AttrType.INT, 'default': 0},
                    'transB': {'type': AttrType.INT, 'default': 0}
                    },
                7: {'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                    'beta': {'type': AttrType.FLOAT, 'default': 1.0},
                    'transA': {'type': AttrType.INT, 'default': 0},
                    'transB': {'type': AttrType.INT, 'default': 0}
                    },
                9: {'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                    'beta': {'type': AttrType.FLOAT, 'default': 1.0},
                    'transA': {'type': AttrType.INT, 'default': 0},
                    'transB': {'type': AttrType.INT, 'default': 0}
                    },
                11: {'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                     'beta': {'type': AttrType.FLOAT, 'default': 1.0},
                     'transA': {'type': AttrType.INT, 'default': 0},
                     'transB': {'type': AttrType.INT, 'default': 0}
                     },
                13: {'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                     'beta': {'type': AttrType.FLOAT, 'default': 1.0},
                     'transA': {'type': AttrType.INT, 'default': 0},
                     'transB': {'type': AttrType.INT, 'default': 0}
                     },
                }

    def __init__(self, graph, attr_dict=None):
        super(GemmOp, self).__init__(graph, attr_dict)
        self.update_attributes(GemmOp, attr_dict)
        assert self.check_required(), 'GemmOp is missing a required parameter.'

    def infer_shape(self):
        super(GemmOp, self).infer_shape()
        inputs = self.get_input_tensors()
        A = inputs[0] if not bool(self.transA) else np.transpose(inputs[0])
        B = inputs[1] if not bool(self.transB) else np.transpose(inputs[1])
        C = inputs[2] if len(inputs) == 3 else np.array(0, inputs[0].dtype)
        # alpha * A' * B' + beta * C
        out_tensor = self.alpha * np.matmul(A, B)
        if len(C.shape) == 1 and C.shape[0] != out_tensor.shape[-1]:
            C = np.reshape(C, (-1, 1))
        out_tensor = out_tensor + self.beta * C
        self.set_out_tensor(out_tensor)


class HardSigmoidOp(LayoutUnawareOp, BaseActivationOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 0.2},
                    'beta': {'type': AttrType.FLOAT, 'default': 0.5},
                    'consumed_inputs': {'type': AttrType.INTS}
                    },
                6: {'alpha': {'type': AttrType.FLOAT, 'default': 0.2},
                    'beta': {'type': AttrType.FLOAT, 'default': 0.5}}
                }

    def __init__(self, graph, attr_dict=None):
        super(HardSigmoidOp, self).__init__(graph, attr_dict)
        self.update_attributes(HardSigmoidOp, attr_dict)
        assert self.check_required(), 'HardSigmoidOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        if item in ('clip_max', 'clip_min'):
            try:
                ret = self.__dict__['_attr'][item].value
            except:
                pass
            if ret is None:
                ret = 0. if item == 'clip_min' else 1.
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.FLOAT, 'value': float(ret)})
        if ret is None:
            ret = super(HardSigmoidOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(HardSigmoidOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.maximum(self.clip_min,
                                np.minimum(self.clip_max,
                                           self.alpha * inputs[0] + self.beta))
        self.set_out_tensor(out_tensor)


class HardmaxOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 1}},
                11: {'axis': {'default': 1}},
                13: {'axis': {'default': -1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(HardmaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(HardmaxOp, attr_dict)
        assert self.check_required(), 'HardmaxOp is missing a required parameter.'

    def infer_shape(self):
        super(HardmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_shape = inputs[0].shape
        self.axis = OpHasAxis.make_axes_non_negative(
            self.axis, len(input_shape))
        if self.cur_version < 13:
            pre_reshape_dim = [np.prod(input_shape[:self.axis], dtype=np.uint32),
                               np.prod(input_shape[self.axis:], dtype=np.uint32)]
            indices = np.argmax(inputs[0].reshape(pre_reshape_dim), axis=1)
            out_tensor = tf.one_hot(indices,
                                    pre_reshape_dim[1],
                                    on_value=np.array(1, inputs[0].dtype),
                                    off_value=np.array(0, inputs[0].dtype),
                                    axis=-1).numpy()
            out_tensor = out_tensor.reshape(input_shape)
        else:
            indices = np.argmax(inputs[0], axis=self.axis)
            out_tensor = tf.one_hot(indices,
                                    input_shape[self.axis],
                                    on_value=np.array(1, inputs[0].dtype),
                                    off_value=np.array(0, inputs[0].dtype),
                                    axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            if cur_ver < 13:
                from ...front_end.onnx.passes.common_passes import insert_reshape, insert_reshape_after
                in_edges = self._graph.sorted_in_edges(self.name, data=True)
                inputs = self.get_input_tensors()
                if len(in_edges) == 1 \
                        and len(inputs) == 1 \
                        and inputs[0] is not None:
                    input_shape = inputs[0].shape
                    if self.axis < 0:
                        self.axis += len(input_shape)
                    pre_dim = [-1, 1] \
                        if self.axis == 0 \
                        else [int(np.prod(input_shape[:self.axis])), int(np.prod(input_shape[self.axis:]))]
                    post_dim = list(input_shape)
                    if self.axis > 1:
                        self.axis = 1

                    src, _, in_attr = in_edges[0]
                    insert_reshape(self._graph, src, self.name,
                                   in_attr, pre_dim, data_format='NCHW')
                    post_reshape = insert_reshape_after(
                        self._graph, self.name, post_dim)
                    if self.name in self._graph._attr['output_names']:
                        index = self._graph._attr['output_names'].index(
                            self.name)
                        self._graph._attr['output_names'][index] = post_reshape
                else:
                    ERROR(
                        '[Parser}: Meets invalid Hardmax (%s) in convert_version!' % self.name)
            self.cur_version = max_ver


class LogOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 6: {}, 13: {}}

    def __init__(self, graph, attr_dict=None):
        super(LogOp, self).__init__(graph, attr_dict)
        self.update_attributes(LogOp, attr_dict)
        assert self.check_required(), 'LogOp is missing a required parameter.'

    def infer_shape(self):
        super(LogOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.log(*inputs)
        self.set_out_tensor(out_tensor)


class LogSoftmaxOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 1}},
                11: {'axis': {'default': 1}},
                13: {'axis': {'default': -1}}}

    def __init__(self, graph, attr_dict=None):
        super(LogSoftmaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(LogSoftmaxOp, attr_dict)
        assert self.check_required(), 'LogSoftmaxOp is missing a required parameter.'

    def infer_shape(self):
        super(LogSoftmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.cur_version < 13:
            input_shape = inputs[0].shape
            inner_dim = [-1, 1] \
                if self.axis == 0 \
                else [int(np.prod(input_shape[:self.axis])), int(np.prod(input_shape[self.axis:]))]
            inp = np.reshape(inputs[0], inner_dim)
            out_tensor = torch.log_softmax(
                torch.from_numpy(inp), dim=0 if self.axis == 0 else -1).numpy()
            out_tensor = np.reshape(out_tensor, input_shape)
        else:
            out_tensor = torch.log_softmax(
                torch.from_numpy(inputs[0]), dim=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            if cur_ver < 13:
                from ...front_end.onnx.passes.common_passes import insert_constant, insert_reshape, insert_reshape_after
                in_edges = self._graph.sorted_in_edges(self.name, data=True)
                inputs = self.get_input_tensors()
                if len(in_edges) == 1 \
                        and len(inputs) == 1 \
                        and inputs[0] is not None:
                    input_shape = inputs[0].shape
                    if self.axis < 0:
                        self.axis += len(input_shape)
                    if len(input_shape) != 2 or self.axis != 1:
                        from ...front_end.onnx.passes.common_passes import insert_constant, insert_reshape, \
                            insert_reshape_after
                        pre_dim = [-1, 1] \
                            if self.axis == 0 \
                            else [int(np.prod(input_shape[:self.axis])), int(np.prod(input_shape[self.axis:]))]
                        post_dim = list(input_shape)
                        if self.axis > 1:
                            self.axis = 1
                        src, _, in_attr = in_edges[0]
                        insert_reshape(self._graph, src, self.name,
                                       in_attr, pre_dim, data_format='NCHW')
                        post_reshape = insert_reshape_after(
                            self._graph, self.name, post_dim)
                        if self.name in self._graph._attr['output_names']:
                            index = self._graph._attr['output_names'].index(
                                self.name)
                            self._graph._attr['output_names'][index] = post_reshape
                else:
                    ERROR(
                        '[Parser}: Meets invalid LogSoftmax (%s) in convert_version!' % self.name)
            self.cur_version = max_ver


class LpNormalizationOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': -1},
                    'p': {'type': AttrType.INT, 'default': 2, 'options': [1, 2]},
                    'epsilon': {'type': AttrType.FLOAT, 'default': 0.0},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(LpNormalizationOp, self).__init__(graph, attr_dict)
        self.update_attributes(LpNormalizationOp, attr_dict)
        assert self.check_required(), 'LpNormalizationOp is missing a required parameter.'
        if self.axes is None and self.axis is not None:
            self.axes = [self.axis]
            self.axis = None

    def infer_shape(self):
        super(LpNormalizationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        if self.p == 2:
            out_tensor = tf.math.l2_normalize(inputs[0], self.axes, self.epsilon).numpy()
        else:
            out_tensor = inputs[0] / np.sum(np.abs(inputs[0]), axis=tuple(self.axes), keepdims=True)
        self.set_out_tensor(out_tensor)


class MatMulOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 9: {}, 13: {}}

    def __init__(self, graph, attr_dict=None):
        super(MatMulOp, self).__init__(graph, attr_dict)
        self.update_attributes(MatMulOp, attr_dict)
        assert self.check_required(), 'MatMulOp is missing a required parameter.'

    def infer_shape(self):
        super(MatMulOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.matmul(*inputs)
        self.set_out_tensor(out_tensor)


class MatMulIntegerOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {10: {}}

    def __init__(self, graph, attr_dict=None):
        super(MatMulIntegerOp, self).__init__(graph, attr_dict)
        self.update_attributes(MatMulIntegerOp, attr_dict)
        assert self.check_required(), 'MatMulIntegerOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        if item in ('a_zero_point', 'b_zero_point'):
            try:
                ret = self.__dict__['_attr'][item].value
            except:
                pass
            if ret is None:
                try:
                    inputs = self.get_input_tensors()
                    ret = inputs[2] if item == 'a_zero_point' else inputs[3]
                except:
                    pass
                if ret is None:
                    dtype = inputs[0].dtype if item == 'a_zero_point' else inputs[1].dtype
                    ret = np.array(0, dtype)
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.INT, 'value': ret})
        if ret is None:
            ret = super(MatMulIntegerOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(MatMulIntegerOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.matmul(*inputs[0:2]).astype(np.int32)
        self.set_out_tensor(out_tensor)


class MaxOp(OpNeedBroadcast, LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {},
                8: {},
                12: {},
                13: {}}

    def __init__(self, graph, attr_dict=None):
        super(MaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(MaxOp, attr_dict)
        assert self.check_required(), 'MaxOp is missing a required parameter.'

    def infer_shape(self):
        super(MaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = reduce(lambda x, y: np.maximum(x, y), inputs)
        self.set_out_tensor(out_tensor)


class MeanOp(OpNeedBroadcast, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {},
                8: {},
                13: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(MeanOp, self).__init__(graph, attr_dict)
        self.update_attributes(MeanOp, attr_dict)
        assert self.check_required(), 'MeanOp is missing a required parameter.'

    def infer_shape(self):
        super(MeanOp, self).infer_shape()
        inputs = self.broad_casted
        stacked = np.stack(inputs, axis=0)
        out_tensor = np.squeeze(
            np.mean(stacked, axis=0, keepdims=True), axis=0)
        self.set_out_tensor(out_tensor)


class MeanVarianceNormalizationOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-9},
                    'axes': {'default': [0, 2, 3]}
                    },
                13: {
                    'epsilon': {'type': AttrType.FLOAT, 'default': 1e-9},
                    'axes': {'default': [0, 2, 3]}
        }
        }

    def __init__(self, graph, attr_dict=None):
        super(MeanVarianceNormalizationOp, self).__init__(graph, attr_dict)
        self.update_attributes(MeanVarianceNormalizationOp, attr_dict)
        assert self.check_required(), 'MeanVarianceNormalizationOp is missing a required parameter.'

    def infer_shape(self):
        super(MeanVarianceNormalizationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        data_mean = np.mean(inputs[0], axis=tuple(self.axes), keepdims=True)
        data_std = np.std(inputs[0], axis=tuple(self.axes), keepdims=True)
        out_tensor = (inputs[0] - data_mean) / (data_std + self.epsilon)
        self.set_out_tensor(out_tensor)


class MinOp(OpNeedBroadcast, LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {},
                8: {},
                12: {},
                13: {}}

    def __init__(self, graph, attr_dict=None):
        super(MinOp, self).__init__(graph, attr_dict)
        self.update_attributes(MinOp, attr_dict)
        assert self.check_required(), 'MinOp is missing a required parameter.'

    def infer_shape(self):
        super(MinOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = reduce(lambda x, y: np.minimum(x, y), inputs)
        self.set_out_tensor(out_tensor)


class ModOp(OpNeedBroadcast, LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {10: {'fmod': {'type': AttrType.INT, 'default': 0}},
                13: {'fmod': {'type': AttrType.INT, 'default': 0}}}

    def __init__(self, graph, attr_dict=None):
        super(ModOp, self).__init__(graph, attr_dict)
        self.update_attributes(ModOp, attr_dict)
        assert self.check_required(), 'ModOp is missing a required parameter.'

    def infer_shape(self):
        super(ModOp, self).infer_shape()
        inputs = self.get_input_tensors()
        with np.errstate(divide='ignore'):
            out_tensor = np.mod(
                *inputs) if self.fmod == 0 else np.fmod(*inputs)
        self.set_out_tensor(out_tensor)


class MulOp(OpNeedBroadcast, OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS},
                    'axis': {'type': AttrType.INT, 'default': None},
                    'broadcast': {'type': AttrType.INT, 'default': 0}
                    },
                6: {'axis': {'type': AttrType.INT, 'default': None},
                    'broadcast': {'type': AttrType.INT, 'default': 0}
                    },
                7: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(MulOp, self).__init__(graph, attr_dict)
        self.update_attributes(MulOp, attr_dict)
        assert self.check_required(), 'MulOp is missing a required parameter.'

    def infer_shape(self):
        super(MulOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 2, 'The number of inputs is invalid in MulOp.'
        cur_ver = self.cur_version
        if cur_ver <= 6:
            if self.broadcast:
                second_input = OpHasAxis.broadcast_to(
                    inputs[1], inputs[0].shape, int(self.axis))
            else:
                assert list(inputs[0].shape) == list(
                    inputs[1].shape), 'The lengths of input0 and input1 are not the same in MulOp infer shape.'
                second_input = inputs[1]
            out_tensor = np.multiply(inputs[0], second_input)
        else:
            out_tensor = np.multiply(*inputs)
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'broadcast':
                if cur_ver <= 6:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                          (cur_ver, type(self).__name__))
        except:
            ret = None
        if ret is None:
            ret = super(MulOp, self).__getattr__(item)
        return ret


class NegOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {},
                13: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(NegOp, self).__init__(graph, attr_dict)
        self.update_attributes(NegOp, attr_dict)
        assert self.check_required(), 'NegOp is missing a required parameter.'

    def infer_shape(self):
        super(NegOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = -inputs[0]
        self.set_out_tensor(out_tensor)


class NonZeroOp(LayoutUnawareOp, ConstLikeOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {},
                13: {}}

    def __init__(self, graph, attr_dict=None):
        super(NonZeroOp, self).__init__(graph, attr_dict)
        self.update_attributes(NonZeroOp, attr_dict)
        assert self.check_required(), 'NonZeroOp is missing a required parameter.'

    def infer_shape(self):
        super(NonZeroOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(np.nonzero(inputs[0]), dtype=np.int64)
        self.set_out_tensor(out_tensor)


class PowOp(OpNeedBroadcast, OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0}
                    },
                7: {},
                11: {},
                13: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(PowOp, self).__init__(graph, attr_dict)
        self.update_attributes(PowOp, attr_dict)
        assert self.check_required(), 'PowOp is missing a required parameter.'

    def infer_shape(self):
        super(PowOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver == 1:
            if self.broadcast:
                second_input = OpHasAxis.broadcast_to(
                    inputs[1], inputs[0].shape, int(self.axis))
            else:
                assert list(inputs[0].shape) == list(
                    inputs[1].shape), 'The lengths of input0 and input1 are not the same in PowOp infer shape.'
                second_input = inputs[1]
            out_tensor = np.power(inputs[0], second_input)
        else:
            out_tensor = np.power(*inputs)
        if out_tensor.dtype == 'float64':
            out_tensor = out_tensor.astype(np.float32)
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'broadcast':
                if cur_ver == 1:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                          (cur_ver, type(self).__name__))
        except:
            ret = None
        if ret is None:
            ret = super(PowOp, self).__getattr__(item)
        return ret


class ReciprocalOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReciprocalOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReciprocalOp, attr_dict)
        assert self.check_required(), 'ReciprocalOp is missing a required parameter.'

    def infer_shape(self):
        super(ReciprocalOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.reciprocal(inputs[0])
        self.set_out_tensor(out_tensor)


class ResizeOp(LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {10: {'mode': {'type': AttrType.STRING, 'default': 'nearest', 'options': ['nearest', 'linear', 'bilinear', 'trilinear']},
                     'nearest_mode': {'type': AttrType.STRING,
                                      'default': 'simple',
                                      'options': ['simple', 'round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil']},
                     },
                11: {'coordinate_transformation_mode': {'type': AttrType.STRING,
                                                        'default': 'half_pixel',
                                                        'options': ['half_pixel',
                                                                    'pytorch_half_pixel',
                                                                    'align_corners',
                                                                    'asymmetric',
                                                                    'tf_half_pixel_for_nn',
                                                                    'tf_crop_and_resize']
                                                        },
                     'cubic_coeff_a': {'type': AttrType.FLOAT, 'default': -0.75},
                     'exclude_outside': {'type': AttrType.INT, 'default': 0},
                     'extrapolation_value': {'type': AttrType.FLOAT, 'default': 0.0},
                     'mode': {'type': AttrType.STRING, 'default': 'nearest', 'options': ['nearest', 'linear', 'cubic']},
                     'nearest_mode': {'type': AttrType.STRING,
                                      'default': 'round_prefer_floor',
                                      'options': ['simple', 'round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil']},
                     },
                13: {'coordinate_transformation_mode': {'type': AttrType.STRING,
                                                        'default': 'half_pixel',
                                                        'options': ['half_pixel',
                                                                    'pytorch_half_pixel',
                                                                    'align_corners',
                                                                    'asymmetric',
                                                                    'tf_crop_and_resize']
                                                        },
                     'cubic_coeff_a': {'type': AttrType.FLOAT, 'default': -0.75},
                     'exclude_outside': {'type': AttrType.INT, 'default': 0},
                     'extrapolation_value': {'type': AttrType.FLOAT, 'default': 0.0},
                     'mode': {'type': AttrType.STRING, 'default': 'nearest', 'options': ['nearest', 'linear', 'cubic']},
                     'nearest_mode': {'type': AttrType.STRING,
                                      'default': 'round_prefer_floor',
                                      'options': ['simple', 'round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil']},
                     },
                18: {'antialias': {'type': AttrType.BOOL, 'default': False},
                     'axes': {'type': AttrType.INTS, 'default': None},
                     'coordinate_transformation_mode': {'type': AttrType.STRING,
                                                        'default': 'half_pixel',
                                                        'options': ['half_pixel',
                                                                    'pytorch_half_pixel',
                                                                    'align_corners',
                                                                    'asymmetric',
                                                                    'tf_crop_and_resize']
                                                        },
                     'cubic_coeff_a': {'type': AttrType.FLOAT, 'default': -0.75},
                     'exclude_outside': {'type': AttrType.INT, 'default': 0},
                     'extrapolation_value': {'type': AttrType.FLOAT, 'default': 0.0},
                     'keep_aspect_ratio_policy': {'type': AttrType.STRING,
                                                  'default': 'stretch',
                                                  'options': ['stretch', 'not_larger', 'not_smaller']},
                     'mode': {'type': AttrType.STRING, 'default': 'nearest', 'options': ['nearest', 'linear', 'cubic']},
                     'nearest_mode': {'type': AttrType.STRING,
                                      'default': 'round_prefer_floor',
                                      'options': ['simple', 'round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil']},
                     }
                }

    MODE_MAP = {'nearest': 'nearest',
                'linear': 'linear',
                'bilinear': 'linear',
                'trilinear': 'linear',
                'cubic': 'cubic',
                }

    def __init__(self, graph, attr_dict=None):
        super(ResizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ResizeOp, attr_dict)
        assert self.check_required(), 'ResizeOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = super(ResizeOp, self).__getattr__(item)
        except:
            ret = None
        if ret is None and item in ('roi', 'scales', 'sizes', 'coordinate_transformation_mode', 'extrapolation_value',
                                    'antialias', 'axes', 'keep_aspect_ratio_policy'):
            try:
                cur_ver = self.__dict__['_attr']['cur_version'].value
                if item == 'roi':
                    inputs = self.get_input_tensors()
                    if cur_ver >= 11:
                        try:
                            if inputs[1] is not None:
                                ret = np.array(inputs[1], np.float32)
                                axes = self.axes
                                if cur_ver >= 18 and axes is not None:
                                    input_length = len(inputs[0].shape)
                                    new_roi = np.array([0] * input_length + [1] * input_length, np.float32)
                                    complete_idx = axes + [(axis + input_length) for axis in axes]
                                    np.put(new_roi, complete_idx, np.array(ret[:(len(axes) * 2)]))
                                    ret = new_roi
                        except:
                            pass
                        if ret is None:
                            ret = np.array([], np.float32)
                elif item == 'scales':
                    inputs = self.get_input_tensors()
                    if cur_ver == 10:
                        ret = np.array(inputs[1], np.float32)
                    else:
                        try:
                            if inputs[2] is not None:
                                ret = np.array(inputs[2], np.float32)
                                axes = self.axes
                                if cur_ver >= 18 and axes is not None:
                                    input_length = len(inputs[0].shape)
                                    new_scales = np.ones([input_length], np.float32)
                                    complete_idx = axes + [(axis + input_length) for axis in axes]
                                    np.put(new_scales, complete_idx, np.array(ret[:len(axes)]))
                                    ret = new_scales
                        except:
                            pass
                        if ret is None:
                            ret = np.array([], np.float32)
                elif item == 'sizes':
                    inputs = self.get_input_tensors()
                    if cur_ver == 10:
                        ret = None
                    else:
                        try:
                            if inputs[3] is not None:
                                ret = np.array(inputs[3], np.int64)
                                axes = self.axes
                                if cur_ver >= 18 and axes is not None:
                                    input_length = len(inputs[0].shape)
                                    new_sizes = np.array(inputs[0].shape, np.int64)
                                    complete_idx = axes + [(axis + input_length) for axis in axes]
                                    np.put(new_sizes, complete_idx, np.array(ret[:len(axes)]))
                                    ret = new_sizes
                        except:
                            pass
                    if ret is None:
                        ret = np.array([], np.int64)
                elif item == 'coordinate_transformation_mode':
                    if cur_ver == 10:
                        ret = 'asymmetric'
                    else:
                        ret = self.__dict__['_attr'][item].value
                elif item == 'extrapolation_value':
                    if cur_ver >= 11:
                        ret = self.__dict__['_attr'][item].value
                    else:
                        ret = 0.0
                        self.__dict__['_attr'][item] = Attribute(
                            item, {'type': AttrType.FLOAT, 'value': ret})
                elif item == 'antialias':
                    if cur_ver >= 18:
                        ret = self.__dict__['_attr'][item].value
                    else:
                        ret = False
                        if item not in self.__dict__['_attr']:
                            self.__dict__['_attr'][item] = Attribute(
                                item, {'type': AttrType.BOOL, 'value': ret})
                        else:
                            self.__dict__['_attr'][item].value = ret
                elif item == 'axes':
                    if cur_ver >= 18:
                        ret = self.__dict__['_attr'][item].value
                        if ret is not None:
                            input_length = len(self.get_input_shapes()[0])
                            ret = [(axis + input_length) if axis < 0 else axis for axis in ret]
                    else:
                        ret = None
                elif item == 'keep_aspect_ratio_policy':
                    if cur_ver >= 18:
                        ret = self.__dict__['_attr'][item].value
                    else:
                        ret = 'stretch'
            except:
                ret = None
        return ret

    def __setattr__(self, item, value):
        if item in ('roi', 'scales', 'sizes'):
            dtype = np.int64 if item == 'sizes' else np.float32
            try:
                self.__dict__['_attr'][item].value = np.array(
                    value).astype(dtype)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.TENSOR, 'value': np.array(value).astype(dtype)})
        elif item in ('mode', 'coordinate_transformation_mode'):
            try:
                self.__dict__['_attr'][item].value = str(value)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.STRING, 'value': str(value)})
        else:
            super(ResizeOp, self).__setattr__(item, value)

    @staticmethod
    def get_nearest_pixel(nearest_mode, x_original, is_down_sample=False):
        # Use around to avoid errors between onnx runtime and this implementation because
        # most decimal fractions can't be represented exactly as a float, for example
        # 25.00000031370866 is used to represent 25. Then, we will get unexpected ceil
        # result because ceil(25.00000031370866)=26, not 25.
        x_original = np.around(x_original, 5)
        if nearest_mode == 'simple':
            ret = math.ceil(x_original) if is_down_sample else int(x_original)
        elif nearest_mode == 'round_prefer_ceil':
            ret = math.ceil(x_original) if np.isclose(x_original, (int(x_original) + 0.5)) else round(x_original)
        elif nearest_mode == 'floor':
            ret = math.floor(x_original)
        elif nearest_mode == 'ceil':
            ret = math.ceil(x_original)
        else:  # round_prefer_floor
            ret = math.floor(x_original) if np.isclose(x_original, (int(x_original) + 0.5)) else round(x_original)
        return int(ret)

    @staticmethod
    def get_original_coordinate(coordinate_transform_mode, x_resized, x_scale,
                                length_resized, length_original,
                                roi_start=None, roi_end=None):
        x_resized = float(x_resized)
        length_resized = float(length_resized)
        length_original = float(length_original)
        if coordinate_transform_mode == 'asymmetric':
            ret = x_resized / x_scale
        elif coordinate_transform_mode == 'pytorch_half_pixel':
            ret = ((x_resized + 0.5) / x_scale - 0.5) if length_resized > 1 else 0.
        elif coordinate_transform_mode == 'tf_half_pixel_for_nn':
            ret = (x_resized + 0.5) / x_scale
        elif coordinate_transform_mode == 'align_corners':
            ret = 0. if np.isclose(length_resized, 1) else (x_resized * (length_original - 1) / (length_resized - 1))
        elif coordinate_transform_mode == 'tf_crop_and_resize':
            assert roi_start is not None and roi_end is not None, \
                'roi_start and roi_end are required for tf_crop_and_resize, but got None'
            if length_resized > 1:
                ret = roi_start * (length_original - 1) + \
                    (x_resized * (roi_end - roi_start) * (length_original - 1)) / (length_resized - 1)
            else:
                ret = 0.5 * (roi_start + roi_end) * (length_original - 1)
        else:  # half_pixel
            ret = ((x_resized + 0.5) / x_scale) - 0.5
        return ret

    @staticmethod
    def get_cubic_coeffs(val, cubic_coeff_a=-0.75):
        abs_val = abs(val)
        coeffs_0 = ((cubic_coeff_a * (abs_val + 1) - 5 * cubic_coeff_a) * (abs_val + 1) + 8 * cubic_coeff_a) \
            * (abs_val + 1) - 4 * cubic_coeff_a
        coeffs_1 = ((cubic_coeff_a + 2) * abs_val - (cubic_coeff_a + 3)) \
            * abs_val * abs_val + 1
        coeffs_2 = ((cubic_coeff_a + 2) * (1 - abs_val) - (cubic_coeff_a + 3)) \
            * (1 - abs_val) * (1 - abs_val) + 1
        coeffs_3 = ((cubic_coeff_a * (2 - abs_val) - 5 * cubic_coeff_a) * (2 - abs_val) + 8 * cubic_coeff_a) \
            * (2 - abs_val) - 4 * cubic_coeff_a
        coeffs = [coeffs_0, coeffs_1, coeffs_2, coeffs_3]
        return coeffs

    @staticmethod
    def get_data_for_coordinate(data, data_idx, input_height, input_width, is_nchw=True):
        height_idx, width_idx = data_idx[2:4] if is_nchw else data_idx[1:3]
        valid_height_idx = np.clip(height_idx, 0, input_height - 1)
        valid_width_idx = np.clip(width_idx, 0, input_width - 1)
        if is_nchw:
            new_data_idx = data_idx[:2] + [valid_height_idx, valid_width_idx]
        else:
            new_data_idx = data_idx[:1] + [valid_height_idx, valid_width_idx] + data_idx[3:]
        return data[tuple(new_data_idx)]

    @staticmethod
    def setup_upsample_linear(input_spatial_sizes, output_spatial_sizes, spatial_scales, roi=None,
                              coordinate_transform_mode='asymmetric', is_nchw=True):
        ret_dict = {}
        spatial_len = len(input_spatial_sizes)
        assert spatial_len == len(output_spatial_sizes), \
            'Expect len(output_spatial_sizes)==%d, but got %d' % (spatial_len, len(output_spatial_sizes))
        assert spatial_len == len(spatial_scales), \
            'Expect len(spatial_scales)==%d, but got %d' % (spatial_len, len(spatial_scales))
        original_lists, d1_lists, d2_lists, inp1_lists, inp2_lists = [], [], [], [], []
        for idx in range(spatial_len):
            roi_start_value, roi_end_value = None, None
            if roi is not None and len(roi) != 0:
                roi = np.array(roi)
                assert np.ndim(roi) == 1 and len(roi) == 2 * (spatial_len + 2), \
                    'Meet invalid roi in setup_upsample_linear'
                if idx == 0:
                    roi_index = 1 if is_nchw else 2
                else:
                    roi_index = 0 if is_nchw else 1
                roi_start_index = roi.size / 2 - (roi_index + 1)
                roi_end_index = roi.size - (roi_index + 1)
                roi_start_value = roi[roi_start_index]
                roi_end_value = roi[roi_end_index]
            current_scale = spatial_scales[idx]
            current_input_size = input_spatial_sizes[idx]
            current_output_size = output_spatial_sizes[idx]
            original_list, d1_list, d2_list, inp1_list, inp2_list = [], [], [], [], []
            for output_idx in range(current_output_size):
                inp = output_idx if FLOAT_EQUAL(current_scale, 1) else \
                    ResizeOp.get_original_coordinate(coordinate_transform_mode,
                                                     output_idx,
                                                     current_scale,
                                                     current_output_size,
                                                     current_input_size,
                                                     roi_start_value,
                                                     roi_end_value)
                original_list.append(inp)
                inp = max(0, min(inp, (current_input_size - 1)))
                inp_1 = min(int(inp), current_input_size - 1)
                inp_2 = min(inp_1 + 1, current_input_size - 1)
                d_1 = 0.5 if inp_1 == inp_2 else abs(inp - inp_1)
                d_2 = 0.5 if inp_1 == inp_2 else abs(inp - inp_2)
                d1_list.append(d_1)
                d2_list.append(d_2)
                inp1_list.append(inp_1)
                inp2_list.append(inp_2)
            original_lists.append(original_list)
            d1_lists.append(d1_list)
            d2_lists.append(d2_list)
            inp1_lists.append(inp1_list)
            inp2_lists.append(inp2_list)
        ret_dict.update({'original': original_lists, 'd1': d1_lists,
                        'd2': d2_lists, 'in1': inp1_lists, 'in2': inp2_lists})
        return ret_dict

    @staticmethod
    def upsample_linear(x_data, output_spatial_sizes, spatial_scales, roi=None,
                        coordinate_transform_mode='asymmetric', is_nchw=True):
        import itertools
        input_shape = list(x_data.shape)
        batch_size = input_shape[0]
        num_channels = input_shape[1] if is_nchw else input_shape[-1]
        input_spatial_sizes = input_shape[2:] if is_nchw else input_shape[1:-1]

        init_dict = ResizeOp.setup_upsample_linear(input_spatial_sizes, output_spatial_sizes,
                                                   spatial_scales, roi,
                                                   coordinate_transform_mode, is_nchw)
        pre_perm = None
        if not is_nchw:
            pre_perm = [0, len(input_shape) - 1] + list(range(1, len(input_shape) - 1))
            x_data = np.transpose(x_data, pre_perm)
        y_data = np.zeros([batch_size, num_channels] + output_spatial_sizes)
        y_data_idx = [list(range(dim_len)) for dim_len in y_data.shape]
        spatial_len = len(y_data.shape) - 2
        for ele_idx in itertools.product(*y_data_idx):
            ret = 0
            for in_idx in itertools.product(list(range(1, 3)), repeat=spatial_len):
                in_names = ['in1' if idx == 1 else 'in2' for idx in in_idx]
                d_names = ['d1' if idx == 2 else 'd2' for idx in in_idx]
                data_idx = ele_idx[:2] + tuple([init_dict[in_names[idx]][idx][ele_idx[idx + 2]]
                                                for idx in range(spatial_len)])
                in_x = x_data[data_idx]
                ret = ret + np.prod([init_dict[d_names[idx]][idx][ele_idx[idx + 2]]
                                    for idx in range(spatial_len)]) * in_x
            y_data[ele_idx] = ret
        if pre_perm is not None:
            y_data = np.transpose(y_data, Op.cal_inverse_perm(pre_perm))
        return y_data.astype(x_data.dtype)

    @staticmethod
    def upsample_nearest(x_data, output_spatial_sizes, spatial_scales, roi=None,
                         coordinate_transform_mode='asymmetric', nearest_mode='simple',
                         extrapolation_value=0.0, is_nchw=True):
        assert isinstance(output_spatial_sizes, list), \
            'Expect output_spatial_sizes to be a list in upsample_nearest'
        assert isinstance(spatial_scales, list), \
            'Expect spatial_scales to be a list in upsample_nearest'
        input_shape = list(x_data.shape)
        n_dim = len(input_shape)
        input_dim_factor = [1] * n_dim
        for dim_idx in reversed(range(n_dim - 1)):
            input_dim_factor[dim_idx] = input_dim_factor[dim_idx + 1] * input_shape[dim_idx + 1]
        if is_nchw:
            full_scales = [1, 1] + spatial_scales
            output_shape = input_shape[:2] + output_spatial_sizes
        else:
            full_scales = [1] + spatial_scales + [1]
            output_shape = input_shape[:1] + output_spatial_sizes + input_shape[-1:]
        output_data = np.zeros(output_shape)

        use_extrapolation = (coordinate_transform_mode == 'tf_crop_and_resize')
        if use_extrapolation:
            assert roi is not None and len(roi) == 2 * n_dim, \
                'Expect roi in 1d and with size=%d in upsample_nearest' % len(roi)

        input_size = input_dim_factor[0] * input_shape[0]
        input_mappings = []
        for dim_idx in range(n_dim):
            input_mapping = []
            if FLOAT_EQUAL(full_scales[dim_idx], 1):
                for dim in range(output_shape[dim_idx]):
                    input_mapping.append(dim * input_dim_factor[dim_idx])
            else:
                roi_start = roi[dim_idx] if use_extrapolation else None
                roi_end = roi[dim_idx + n_dim] if use_extrapolation else None
                for dim in range(output_shape[dim_idx]):
                    original_dim = ResizeOp.get_original_coordinate(
                        coordinate_transform_mode, dim, full_scales[dim_idx],
                        output_shape[dim_idx], input_shape[dim_idx],
                        roi_start, roi_end)
                    need_extrapolation = (use_extrapolation and (
                        original_dim < 0 or original_dim >= input_shape[dim_idx]))
                    input_dim = ResizeOp.get_nearest_pixel(nearest_mode, original_dim, full_scales[dim_idx] < 1)
                    input_dim = max(0, min(input_dim, input_shape[dim_idx] - 1))
                    input_mapping_value = (-input_size) if need_extrapolation else (input_dim *
                                                                                    input_dim_factor[dim_idx])
                    input_mapping.append(input_mapping_value)
            input_mappings.append(input_mapping)

        flatten_output_data = []
        output_dim_counter = [0] * n_dim
        input_idx = 0
        for dim_idx in range(n_dim):
            input_idx = input_idx + input_mappings[dim_idx][0]
        out_size = np.prod(output_shape)
        for _ in range(0, out_size):
            ret = extrapolation_value if input_idx < 0 else x_data.flatten()[input_idx]
            flatten_output_data.append(ret)
            for dim_idx in reversed(range(0, n_dim)):
                input_idx = input_idx - input_mappings[dim_idx][output_dim_counter[dim_idx]]
                output_dim_counter[dim_idx] = output_dim_counter[dim_idx] + 1
                if output_dim_counter[dim_idx] < output_shape[dim_idx]:
                    input_idx = input_idx + input_mappings[dim_idx][output_dim_counter[dim_idx]]
                    break
                output_dim_counter[dim_idx] = 0
                input_idx = input_idx + input_mappings[dim_idx][0]
        output_data = np.reshape(flatten_output_data, output_shape)
        return output_data.astype(x_data.dtype)

    @staticmethod
    def cubic_interpolation_1d(data, data_idx, input_height, input_width,
                               coeff_array, coeff_sum, cache, is_nchw=True):
        assert isinstance(data_idx, list) and len(data_idx) == 4, \
            'Expect data_idx to be a list of 4 elements in cubic_interpolation_1d, but got %s' % str(data_idx)
        if not is_nchw:
            data = np.transpose(data, [0, 3, 1, 2])
            data_idx = [data_idx[0], data_idx[-1]] + data_idx[1:-1]
        n, c, y, x = data_idx
        grid_start_pos = y * input_width + (x - 1)
        if grid_start_pos in cache:
            return cache[grid_start_pos]

        result = 0.0
        for i, j in zip(range(4), range(-1, 3)):
            orig_data = ResizeOp.get_data_for_coordinate(data, [n, c, y, x + j], input_height, input_width)
            result = result + coeff_array[i] / coeff_sum * orig_data

        cache.update({grid_start_pos: result})
        return result

    @staticmethod
    def upsample_cubic(x_data, output_spatial_sizes, spatial_scales, roi=None,
                       coordinate_transform_mode='asymmetric', cubic_coeff_a=-0.75,
                       exclude_outside=False, extrapolation_value=0.0, is_nchw=True):
        input_shape = list(x_data.shape)
        if len(input_shape) != 4:
            ERROR('[Parser]: Resize op only supports cubic mode with 4d input, but got %dd!' % len(input_shape))
            return None
        if not is_nchw:
            x_data = np.transpose(x_data, [0, 3, 1, 2])
        cubic_coeffs = {}
        coeff_to_1Dinterpolation_map = {}
        original_lists = []
        batch_size, num_channels = input_shape[0:2]
        input_spatial_sizes = input_shape[2:]
        for idx in range(2):
            roi_start_value, roi_end_value = None, None
            if roi is not None and np.array(roi).size > 0:
                roi_size = np.array(roi).size
                assert roi_size == 8, 'Expect size of roi == 8, but got %d' % roi_size
                roi_index = 1 if idx == 0 else 0
                roi_start = int(roi_size / 2 - (roi_index + 1))
                roi_end = int(roi_size - (roi_index + 1))
                roi_start_value = roi[roi_start]
                roi_end_value = roi[roi_end]

            current_scale = spatial_scales[idx]
            current_output_size = output_spatial_sizes[idx]
            current_input_size = input_spatial_sizes[idx]
            original_list = []
            for output_idx in range(current_output_size):
                inp = output_idx if FLOAT_EQUAL(current_scale, 1) else \
                    ResizeOp.get_original_coordinate(coordinate_transform_mode,
                                                     output_idx,
                                                     current_scale,
                                                     current_output_size,
                                                     current_input_size,
                                                     roi_start_value,
                                                     roi_end_value)
                original_list.append(inp)
                diff = np.around(inp - math.floor(inp), 6)
                if diff not in cubic_coeffs:
                    cubic_coeffs[diff] = ResizeOp.get_cubic_coeffs(diff, cubic_coeff_a)
                    coeff_to_1Dinterpolation_map[diff] = {}
            original_lists.append(original_list)

        use_extrapolation = (coordinate_transform_mode == 'tf_crop_and_resize')
        output_height, output_width = output_spatial_sizes
        input_height, input_width = input_spatial_sizes
        output_shape = [batch_size, num_channels] + output_spatial_sizes
        y_data = np.zeros(output_shape)

        def get_coeff_and_sum(in_val, in_val_int, input_length, cubic_coeffs, exclude_outside):
            diff_val = np.around(in_val - in_val_int, 6)
            coeff_sum = 1.0
            if exclude_outside:
                coeff_sum = 0.0
                coeff_list = []
                for idx in range(0, 4):
                    val = in_val_int + idx - 1
                    coeff_holder = 0.0 if (
                        val < 0 or val >= input_length) else cubic_coeffs[diff_val][idx]
                    coeff_sum = coeff_sum + coeff_holder
                    coeff_list.append(coeff_holder)
            else:
                coeff_list = cubic_coeffs[diff_val]
            return coeff_list, coeff_sum

        for n in range(batch_size):
            for c in range(num_channels):
                for y in range(output_height):
                    in_y = original_lists[0][y]
                    if use_extrapolation and (in_y < 0 or in_y >= input_height):
                        y_data[n, c, y, :] = extrapolation_value
                        continue
                    y_int = int(math.floor(in_y))
                    coeff_y, y_coeff_sum = get_coeff_and_sum(in_y, y_int, input_height, cubic_coeffs, exclude_outside)
                    for x in range(output_width):
                        in_x = original_lists[1][x]
                        if use_extrapolation and (in_x < 0 or in_x >= input_width):
                            y_data[n, c, y, x] = extrapolation_value
                            continue
                        x_int = int(math.floor(in_x))
                        diff_x = np.around(in_x - x_int, 6)
                        cache = coeff_to_1Dinterpolation_map[diff_x] if diff_x in coeff_to_1Dinterpolation_map else {}
                        coeff_x, x_coeff_sum = get_coeff_and_sum(
                            in_x, x_int, input_width, cubic_coeffs, exclude_outside)
                        result = 0.
                        for idx in range(0, 4):
                            y_val = y_int + idx - 1
                            data_idx = [n, c, y_val, x_int]
                            x_interpolation_result = ResizeOp.cubic_interpolation_1d(
                                x_data, data_idx, input_height, input_width, coeff_x, x_coeff_sum, cache)
                            result = result + x_interpolation_result * coeff_y[idx] / y_coeff_sum
                        y_data[n, c, y, x] = result
                coeff_to_1Dinterpolation_map.clear()
        if not is_nchw:
            y_data = np.transpose(y_data, [0, 2, 3, 1])
        return y_data.astype(x_data.dtype)

    def infer_shape(self):
        super(ResizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if (self.scales is None or self.scales.size == 0) \
                and (self.sizes is None or self.sizes.size == 0):
            ERROR('[Parser]: At least one of scales and sizes of Resize Op (%s) should be valid!' % (
                self.name))
        input_dim_np = np.array(inputs[0].shape, np.float32)
        if self.scales is None or self.scales.size == 0:
            if self.keep_aspect_ratio_policy in ('not_larger', 'not_smaller'):
                axes = self.axes
                axes = list(range(len(input_dim_np))) if axes is None else axes
                if self.keep_aspect_ratio_policy == 'not_larger':
                    min_max_fn = min
                    scale_in_policy = np.finfo(np.float32).max
                else:
                    min_max_fn = max
                    scale_in_policy = np.finfo(np.float32).min
                original_out_size = self.sizes
                adjusted_out_size = copy.deepcopy(original_out_size)
                for axis in axes:
                    scale_in_policy = min_max_fn(original_out_size[axis]/input_dim_np[axis], scale_in_policy)
                for axis in axes:
                    adjusted_out_size[axis] = int(ResizeOp.get_nearest_pixel(
                        'round_prefer_ceil', scale_in_policy * input_dim_np[axis]))
                self.sizes = adjusted_out_size
            self.scales = np.array(self.sizes, np.float32) / input_dim_np
        if self.cur_version == 10:
            out_shape = np.floor(
                input_dim_np * self.scales).astype(np.int64).tolist()
        else:
            if self.sizes is not None and self.sizes.size > 0:
                out_shape = self.sizes.tolist()
            else:
                base_shape = input_dim_np * self.scales
                if self.coordinate_transformation_mode == 'tf_crop_and_resize':
                    isnan = np.isnan(self.roi).all()
                    if self.roi is not None and self.roi.size > 0 and not isnan:
                        base_shape = base_shape * \
                            (np.reshape(self.roi, (2, -1))
                             [1, :] - np.reshape(self.roi, (2, -1))[0, :])
                out_shape = np.floor(base_shape).astype(np.int64).tolist()
        if self.antialias:
            # TODO: Add the implementation of antialias to support const inputs.
            WARN('[Parser]: Meets unsupported attribute antialias of Resize Op (%s) in infer_shape!' % self.name)
        if self.is_all_inputs_const():
            if FLOAT_EQUAL(inputs[0], 0):
                out_tensor = np.zeros(out_shape).astype(inputs[0].dtype)
            else:
                assert self.mode in ResizeOp.MODE_MAP, \
                    'Meet unsupported mode (%s) of Resize Op (%s) in infer_shape' % (mode, self.name)
                mode = ResizeOp.MODE_MAP[self.mode]
                is_nchw = (self.data_format == 'NCHW')
                out_spatial_shape = out_shape[2:] if is_nchw else out_shape[1:-1]
                spatial_scales = list(self.scales[2:]) if is_nchw else list(self.scales[1:-1])
                if mode == 'linear':
                    out_tensor = ResizeOp.upsample_linear(inputs[0], out_spatial_shape, spatial_scales, self.roi,
                                                          self.coordinate_transformation_mode, is_nchw)
                elif mode == 'nearest':
                    out_tensor = ResizeOp.upsample_nearest(inputs[0], out_spatial_shape, spatial_scales, self.roi,
                                                           self.coordinate_transformation_mode, self.nearest_mode,
                                                           self.extrapolation_value, is_nchw)
                else:  # mode == 'cubic'
                    assert self.cur_version > 10, 'Mode cubic in Resize Op (%s) is not supported until opset 11' % self.name
                    out_tensor = ResizeOp.upsample_cubic(inputs[0], out_spatial_shape, spatial_scales, self.roi,
                                                         self.coordinate_transformation_mode, self.cubic_coeff_a,
                                                         self.exclude_outside, self.extrapolation_value, is_nchw)
        else:
            # Still use random output here to speedup parsing if inputs are non-const
            out_tensor = np.random.ranf(out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        from ...front_end.onnx.passes.common_passes import insert_constant
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            if cur_ver == 10:
                self.mode = ResizeOp.MODE_MAP[self.mode]
                self.coordinate_transformation_mode = 'asymmetric'
                in_edges = self._graph.sorted_in_edges(self.name, data=True)
                assert len(
                    in_edges) == 2, 'The number of in_edges is invalid in ResizeOp convert version.'
                scale_inp, _, in_attr = in_edges[1]
                self._graph.remove_edges_from(in_edges[1:])
                insert_constant(self._graph, self.name + '_roi',
                                np.array([], np.float32), self.name, in_port=1)
                new_in_attr = copy.deepcopy(in_attr)
                new_in_attr['dst_in_port'] = 2
                self._graph.add_edge(scale_inp, self.name, **new_in_attr)
            self.cur_version = max_ver

        in_edges = self._graph.sorted_in_edges(self.name, keys=True, data=True)
        if len(in_edges) < 2:
            insert_constant(self._graph, self.name + '_roi',
                            np.array([], np.float32), self.name, in_port=1)
        else:
            src, _, k, in_attr = in_edges[1]
            roi = self.roi
            if in_attr.get('tensor', None) is None \
                    or in_attr['tensor'].value is None \
                    or np.any(np.array(in_attr['tensor'].value) != roi):
                self._graph.remove_edge(src, self.name, key=k)
                insert_constant(self._graph, self.name + '_roi',
                                np.array(roi, np.float32), self.name, in_port=1)

        if len(in_edges) < 3:
            insert_constant(self._graph, self.name + '_scales', np.array(
                self.scales if self.scales is not None else [], np.float32), self.name, in_port=2)
        else:
            src, _, k, in_attr = in_edges[2]
            scales = self.scales
            if in_attr.get('tensor', None) is None \
                    or in_attr['tensor'].value is None \
                    or np.any(np.array(in_attr['tensor'].value) != scales):
                self._graph.remove_edge(src, self.name, key=k)
                insert_constant(self._graph, self.name + '_scales', np.array(
                    scales if scales is not None else [], np.float32), self.name, in_port=2)

        if len(in_edges) < 4:
            insert_constant(self._graph, self.name + '_sizes', np.array(
                self.sizes if self.sizes is not None else [], np.int64), self.name, in_port=3)
        else:
            src, _, k, in_attr = in_edges[3]
            sizes = self.sizes
            if in_attr.get('tensor', None) is None \
                    or in_attr['tensor'].value is None \
                    or np.any(np.array(in_attr['tensor'].value) != sizes):
                self._graph.remove_edge(src, self.name, key=k)
                insert_constant(self._graph, self.name + '_sizes', np.array(
                    sizes if sizes is not None else [], np.int64), self.name, in_port=3)


class RoundOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {}}

    def __init__(self, graph, attr_dict=None):
        super(RoundOp, self).__init__(graph, attr_dict)
        self.update_attributes(RoundOp, attr_dict)
        assert self.check_required(), 'RoundOp is missing a required parameter.'

    def infer_shape(self):
        super(RoundOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.round(inputs[0])
        self.set_out_tensor(out_tensor)


class SeluOp(LayoutUnawareOp, BaseActivationOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 1.6732},
                    'consumed_inputs': {'type': AttrType.INTS},
                    'gamma': {'type': AttrType.FLOAT, 'default': 1.0507},
                    },
                6: {'alpha': {'type': AttrType.FLOAT, 'default': 1.6732},
                    'gamma': {'type': AttrType.FLOAT, 'default': 1.0507},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(SeluOp, self).__init__(graph, attr_dict)
        self.update_attributes(SeluOp, attr_dict)
        assert self.check_required(), 'SeluOp is missing a required parameter.'

    def infer_shape(self):
        super(SeluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0])
        mask = out_tensor <= 0
        out_tensor[mask] = self.alpha * (np.exp(out_tensor[mask]) - 1)
        out_tensor = self.gamma * out_tensor
        self.set_out_tensor(out_tensor)


class ShrinkOp(LayoutUnawareOp, BaseActivationOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {'bias': {'type': AttrType.FLOAT, 'default': 0},
                    'lambd': {'type': AttrType.FLOAT, 'default': 0.5},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(ShrinkOp, self).__init__(graph, attr_dict)
        self.update_attributes(ShrinkOp, attr_dict)
        assert self.check_required(), 'ShrinkOp is missing a required parameter.'

    def infer_shape(self):
        super(ShrinkOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0])
        mask_neg = out_tensor < -self.lambd
        mask = out_tensor > self.lambd
        mask_0 = np.logical_not(mask | mask_neg)
        out_tensor[mask_neg] = out_tensor[mask_neg] + self.bias
        out_tensor[mask] = out_tensor[mask] - self.bias
        out_tensor[mask_0] = 0
        self.set_out_tensor(out_tensor)


class SigmoidOp(LayoutUnawareOp, BaseActivationOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(SigmoidOp, self).__init__(graph, attr_dict)
        self.update_attributes(SigmoidOp, attr_dict)
        assert self.check_required(), 'SigmoidOp is missing a required parameter.'

    def infer_shape(self):
        super(SigmoidOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_dtype = str(inputs[0].dtype)
        inp = inputs[0].astype(np.float32) if input_dtype != 'float32' else inputs[0]
        out_tensor = tf.sigmoid(inp).numpy()
        if input_dtype != 'float32':
            out_tensor = out_tensor.astype(input_dtype)
        self.set_out_tensor(out_tensor)


class SignOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {}, 13: {}}

    def __init__(self, graph, attr_dict=None):
        super(SignOp, self).__init__(graph, attr_dict)
        self.update_attributes(SignOp, attr_dict)
        assert self.check_required(), 'SignOp is missing a required parameter.'

    def infer_shape(self):
        super(SignOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.sign(inputs[0])
        self.set_out_tensor(out_tensor)


class SinOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {7: {}}

    def __init__(self, graph, attr_dict=None):
        super(SinOp, self).__init__(graph, attr_dict)
        self.update_attributes(SinOp, attr_dict)
        assert self.check_required(), 'SinOp is missing a required parameter.'

    def infer_shape(self):
        super(SinOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.sin(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class SinhOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {}}

    def __init__(self, graph, attr_dict=None):
        super(SinhOp, self).__init__(graph, attr_dict)
        self.update_attributes(SinhOp, attr_dict)
        assert self.check_required(), 'SinhOp is missing a required parameter.'

    def infer_shape(self):
        super(SinhOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.sinh(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class SoftmaxOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 1}},
                11: {'axis': {'default': 1}},
                13: {'axis': {'default': -1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(SoftmaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(SoftmaxOp, attr_dict)
        assert self.check_required(), 'SoftmaxOp is missing a required parameter.'

    def infer_shape(self):
        super(SoftmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.cur_version < 13:
            input_shape = inputs[0].shape
            inner_dim = [-1, 1] \
                if self.axis == 0 \
                else [int(np.prod(input_shape[:self.axis])), int(np.prod(input_shape[self.axis:]))]
            inp = np.reshape(inputs[0], inner_dim)
            out_tensor = torch.nn.functional.softmax(
                torch.from_numpy(inp), dim=0 if self.axis == 0 else -1).numpy()
            out_tensor = np.reshape(out_tensor, input_shape)
        else:
            out_tensor = torch.nn.functional.softmax(
                torch.from_numpy(inputs[0]), dim=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < 13:
            in_edges = self._graph.sorted_in_edges(self.name, data=True)
            inputs = self.get_input_tensors()
            if len(in_edges) != 1 \
                    or len(inputs) != 1 \
                    or inputs[0] is None:
                ERROR('[Parser}: Meets invalid Softmax (%s) in convert_version!' % self.name)
                return
            input_shape = inputs[0].shape
            if self.axis < 0:
                self.axis += len(input_shape)
            if self.axis != len(input_shape) - 1:
                from ...front_end.onnx.passes.common_passes import insert_reshape, insert_reshape_after
                pre_dim = [-1, 1] \
                    if self.axis == 0 \
                    else [int(np.prod(input_shape[:self.axis])), int(np.prod(input_shape[self.axis:]))]
                post_dim = list(input_shape)
                if self.axis > 1:
                    self.axis = 1
                src, _, in_attr = in_edges[0]
                insert_reshape(self._graph, src, self.name, in_attr, pre_dim)
                post_reshape = insert_reshape_after(self._graph, self.name, post_dim)
                if self.name in self._graph._attr['output_names']:
                    index = self._graph._attr['output_names'].index(self.name)
                    self._graph._attr['output_names'][index] = post_reshape
        self.cur_version = max_ver


class SoftplusOp(LayoutUnawareOp, BaseActivationOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(SoftplusOp, self).__init__(graph, attr_dict)
        self.update_attributes(SoftplusOp, attr_dict)
        assert self.check_required(), 'SoftplusOp is missing a required parameter.'

    def infer_shape(self):
        super(SoftplusOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0])
        out_tensor = tf.math.log(1 + tf.exp(out_tensor)).numpy()
        self.set_out_tensor(out_tensor)


class SoftsignOp(LayoutUnawareOp, BaseActivationOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}}}

    def __init__(self, graph, attr_dict=None):
        super(SoftsignOp, self).__init__(graph, attr_dict)
        self.update_attributes(SoftsignOp, attr_dict)
        assert self.check_required(), 'SoftsignOp is missing a required parameter.'

    def infer_shape(self):
        super(SoftsignOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0])
        out_tensor = out_tensor / (1 + tf.abs(out_tensor))
        self.set_out_tensor(out_tensor.numpy())


class SqrtOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(SqrtOp, self).__init__(graph, attr_dict)
        self.update_attributes(SqrtOp, attr_dict)
        assert self.check_required(), 'SqrtOp is missing a required parameter.'

    def infer_shape(self):
        super(SqrtOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.sqrt(inputs[0])
        self.set_out_tensor(out_tensor)


class SubOp(OpNeedBroadcast, OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0},
                    'consumed_inputs': {'type': AttrType.INTS}},
                6: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0}},
                7: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(SubOp, self).__init__(graph, attr_dict)
        self.update_attributes(SubOp, attr_dict)
        assert self.check_required(), 'SubOp is missing a required parameter.'

    def infer_shape(self):
        super(SubOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver <= 6:
            if self.broadcast:
                second_input = OpHasAxis.broadcast_to(
                    inputs[1], inputs[0].shape, int(self.axis))
            else:
                assert list(inputs[0].shape) == list(
                    inputs[1].shape), 'The lengths of input0 and input1 are not the same in SubOp infer shape.'
                second_input = inputs[1]
            out_tensor = np.subtract(inputs[0], second_input)
        else:
            out_tensor = np.subtract(*inputs)
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'broadcast':
                if cur_ver <= 6:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                          (cur_ver, type(self).__name__))
        except:
            ret = None
        if ret is None:
            ret = super(SubOp, self).__getattr__(item)
        return ret


class SumOp(OpNeedBroadcast, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {},
                8: {},
                13: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(SumOp, self).__init__(graph, attr_dict)
        self.update_attributes(SumOp, attr_dict)
        assert self.check_required(), 'SumOp is missing a required parameter.'

    def infer_shape(self):
        super(SumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = reduce(lambda x, y: x + y, inputs)
        self.set_out_tensor(out_tensor)


class TanhOp(LayoutUnawareOp, BaseActivationOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(TanhOp, self).__init__(graph, attr_dict)
        self.update_attributes(TanhOp, attr_dict)
        assert self.check_required(), 'TanhOp is missing a required parameter.'

    def infer_shape(self):
        super(TanhOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.tanh(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class TanOp(LayoutUnawareOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {7: {}}

    def __init__(self, graph, attr_dict=None):
        super(TanOp, self).__init__(graph, attr_dict)
        self.update_attributes(TanOp, attr_dict)
        assert self.check_required(), 'TanOp is missing a required parameter.'

    def infer_shape(self):
        super(TanOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.tan(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class ThresholdedReluOp(BaseActivationOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {10: {'alpha': {'type': AttrType.FLOAT, 'default': 1.}
                     }
                }

    def __init__(self, graph, attr_dict=None):
        super(ThresholdedReluOp, self).__init__(graph, attr_dict)
        self.update_attributes(ThresholdedReluOp, attr_dict)
        assert self.check_required(), 'ThresholdedReluOp is missing a required parameter.'

    def infer_shape(self):
        super(ThresholdedReluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = inputs[0]
        mask = out_tensor < self.alpha
        out_tensor[mask] = 0
        self.set_out_tensor(out_tensor)


class TopKOp(OpHasAxis, OpHasMultipleOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': -1}, 'k': {'type': AttrType.INT, 'required': True}},
                10: {'axis': {'default': -1}},
                11: {'axis': {'default': -1},
                     'largest': {'type': AttrType.INT, 'options': [0, 1], 'default': 1},
                     'sorted': {'type': AttrType.INT, 'options': [0, 1], 'default': 1}},
                }

    def __init__(self, graph, attr_dict=None):
        super(TopKOp, self).__init__(graph, attr_dict)
        self.update_attributes(TopKOp, attr_dict)
        assert self.check_required(), 'TopKOp is missing a required parameter.'

    def infer_shape(self):
        super(TopKOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if inputs[0].dtype == bool:
            inputs[0] = inputs[0].astype(np.int32)
        input_tensor = torch.from_numpy(inputs[0])
        cur_ver = self.cur_version
        k = self.k if cur_ver == 1 else int(inputs[1])
        largest = bool(self.largest) if cur_ver >= 11 else True
        need_sorted = bool(self.sorted) if cur_ver >= 11 else True
        out_tensor_list = torch.topk(
            input_tensor, k, dim=self.axis, largest=largest, sorted=need_sorted)
        out_tensor_list = [ot.numpy() if i == 0 else ot.numpy().astype(
            np.int32) for (i, ot) in enumerate(out_tensor_list)]
        self.set_out_tensor(out_tensor_list)


class UpsampleOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {7: {'mode': {'type': AttrType.STRING, 'default': 'nearest', 'options': ['nearest', 'linear', 'bilinear', 'trilinear']},
                    'scales': {'type': AttrType.FLOATS, 'required': True},
                    },
                9: {'mode': {'type': AttrType.STRING, 'default': 'nearest', 'options': ['nearest', 'linear', 'bilinear', 'trilinear']}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(UpsampleOp, self).__init__(graph, attr_dict)
        self.update_attributes(UpsampleOp, attr_dict)
        assert self.check_required(), 'UpsampleOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'scales':
                if cur_ver <= 7:
                    ret = self.__dict__['_attr'][item].value
                else:
                    if item not in self.__dict__['_attr']:
                        self.__dict__['_attr'][item] = Attribute(
                            item, {'type': AttrType.FLOATS, 'value': None})
                    inputs = self.get_input_tensors()
                    ret = np.array(inputs[1]).tolist()
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(UpsampleOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(UpsampleOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if np.all(np.array(self.scales) == 1):
            out_tensor = inputs[0].copy()
        else:
            input_dim_np = np.array(inputs[0].shape, np.float32)
            out_shape = np.floor(
                input_dim_np * self.scales).astype(np.int64).tolist()
            mode = ResizeOp.MODE_MAP[self.mode]
            is_nchw = True if self.data_format == 'NCHW' else False
            out_spatial_shape = out_shape[2:] if is_nchw else out_shape[1:-1]
            spatial_scales = self.scales[2:] if is_nchw else self.scales[1:-1]
            if not self.is_all_inputs_const():
                out_tensor = np.random.ranf(out_shape).astype(inputs[0].dtype)
            else:
                if mode == 'linear':
                    out_tensor = ResizeOp.upsample_linear(inputs[0], out_spatial_shape, spatial_scales)
                else:
                    out_tensor = ResizeOp.upsample_nearest(inputs[0], out_spatial_shape, spatial_scales)
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        from ...front_end.onnx.passes.common_passes import insert_constant
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            if cur_ver == 7:
                insert_constant(self._graph, self.name + '_scales',
                                np.array(self.scales, np.float32), self.name, in_port=1)
            self.cur_version = max_ver
