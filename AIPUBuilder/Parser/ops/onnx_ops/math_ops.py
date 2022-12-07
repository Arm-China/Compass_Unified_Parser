# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import tensorflow.compat.v1 as tf
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
                    WARN('[Parser]: Unsupported op version [%s] for %s!' %
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
                                           self.alpha * inputs[0] + self.beta)) \
            / self.clip_max
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
                    WARN(
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
                    WARN(
                        '[Parser}: Meets invalid LogSoftmax (%s) in convert_version!' % self.name)
            self.cur_version = max_ver


class LpNormalizationOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': -1},
                    'p': {'type': AttrType.INT, 'default': 2, 'options': [1, 2]}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(LpNormalizationOp, self).__init__(graph, attr_dict)
        self.update_attributes(LpNormalizationOp, attr_dict)
        assert self.check_required(), 'LpNormalizationOp is missing a required parameter.'

    def infer_shape(self):
        super(LpNormalizationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = inputs[0] / \
            np.linalg.norm(inputs[0], ord=self.p,
                           axis=self.axis, keepdims=True)
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
                    WARN('[Parser]: Unsupported op version [%s] for %s!' %
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
                    WARN('[Parser]: Unsupported op version [%s] for %s!' %
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
                     }
                }

    MODE_MAP = {'nearest': 'nearest',
                'linear': 'linear',
                'bilinear': 'linear',
                'trilinear': 'linear'
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
        if ret is None and item in ('roi', 'scales', 'sizes', 'coordinate_transformation_mode'):
            try:
                cur_ver = self.__dict__['_attr']['cur_version'].value
                if item == 'roi':
                    inputs = self.get_input_tensors()
                    if cur_ver >= 11:
                        try:
                            ret = np.array(inputs[1], np.float32)
                        except:
                            ret = np.array([], np.float32)
                elif item == 'scales':
                    inputs = self.get_input_tensors()
                    if cur_ver == 10:
                        ret = np.array(inputs[1], np.float32)
                    else:
                        try:
                            ret = np.array(inputs[2], np.float32)
                        except:
                            ret = None
                elif item == 'sizes':
                    inputs = self.get_input_tensors()
                    if cur_ver == 10:
                        ret = None
                    else:
                        try:
                            ret = np.array(inputs[3], np.int64)
                        except:
                            ret = None
                elif item == 'coordinate_transformation_mode':
                    if cur_ver == 10:
                        ret = 'asymmetric'
                    else:
                        ret = self.__dict__['_attr'][item].value
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

    def infer_shape(self):
        super(ResizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if not np.all(self.scales) and not np.all(self.sizes):
            WARN('[Parser]: At least one of scales and sizes of Resize Op (%s) should be valid!' % (
                self.name))
        input_dim_np = np.array(inputs[0].shape, np.float32)
        if self.scales is None or self.scales.size == 0 or not np.all(self.scales):
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
        if self.is_all_inputs_const():
            if FLOAT_EQUAL(inputs[0], 0):
                out_tensor = np.zeros(out_shape).astype(inputs[0].dtype)
            else:
                WARN('[Parser]: Non-zero constant inputs for Resize op is unsupported for now!')
                out_tensor = None
        else:
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
            if in_attr.get('tensor', None) is None \
                    or in_attr['tensor'].value is None \
                    or np.any(np.array(in_attr['tensor'].value) != self.roi):
                insert_constant(self._graph, self.name + '_roi',
                                np.array(self.roi, np.float32), self.name, in_port=1)
                self._graph.remove_edge(src, self.name, key=k)

        if len(in_edges) < 3:
            insert_constant(self._graph, self.name + '_scales', np.array(
                self.scales if self.scales is not None else [], np.float32), self.name, in_port=2)
        else:
            src, _, k, in_attr = in_edges[2]
            if in_attr.get('tensor', None) is None \
                    or in_attr['tensor'].value is None \
                    or np.any(np.array(in_attr['tensor'].value) != self.scales):
                self._graph.remove_edge(src, self.name, key=k)
                insert_constant(self._graph, self.name + '_scales', np.array(
                    self.scales if self.scales is not None else [], np.float32), self.name, in_port=2)

        if len(in_edges) < 4:
            insert_constant(self._graph, self.name + '_sizes', np.array(
                self.sizes if self.sizes is not None else [], np.int64), self.name, in_port=3)
        else:
            src, _, k, in_attr = in_edges[3]
            if in_attr.get('tensor', None) is None \
                    or in_attr['tensor'].value is None \
                    or np.any(np.array(in_attr['tensor'].value) != self.sizes):
                self._graph.remove_edge(src, self.name, key=k)
                insert_constant(self._graph, self.name + '_sizes', np.array(
                    self.sizes if self.sizes is not None else [], np.int64), self.name, in_port=3)


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
        out_tensor = tf.sigmoid(inputs[0]).numpy()
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
        out_tensor = torch.nn.functional.softmax(
            torch.from_numpy(inputs[0]), dim=self.axis).numpy()
        self.set_out_tensor(out_tensor)


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
        out_tensor = tf.log(1 + tf.exp(out_tensor)).numpy()
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
                    WARN('[Parser]: Unsupported op version [%s] for %s!' %
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
        if inputs[0].dtype == np.bool:
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
            out_shape = np.round(
                input_dim_np * self.scales).astype(np.int64).tolist()
            out_tensor = np.random.ranf(out_shape).astype(inputs[0].dtype)
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
