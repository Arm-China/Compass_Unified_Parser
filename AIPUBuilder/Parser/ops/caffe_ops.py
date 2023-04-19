# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import tensorflow as tf
from .op import *
from ..common.defs import FLOAT_EQUAL
from ..logger import INFO, DEBUG, WARN, ERROR, FATAL


class CaffeABSVALOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeABSVALOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeABSVALOp, attr_dict)
        assert self.check_required(), 'CaffeABSVALOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeABSVALOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.abs(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Abs', 'version': 6}


class CaffeARGMAXOp(OpHasAxis, OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'out_max_val': {'type': AttrType.INT, 'options': [0, 1], 'default': 0},
                    'k': {'type': AttrType.INT, 'default': 1},
                    'axis': {'type': AttrType.INT, 'default': None}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeARGMAXOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeARGMAXOp, attr_dict)
        assert self.check_required(), 'CaffeARGMAXOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeARGMAXOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = inputs[0] if self.axis is not None else np.reshape(
            inputs[0], newshape=(inputs[0].shape[0], -1, 1))
        if self.axis is not None:
            self.axis = OpHasAxis.make_axes_non_negative(
                self.axis, len(inputs[0].shape))
        values, indices = [t.numpy() for t in torch.topk(torch.from_numpy(inp),
                                                         self.k,
                                                         dim=self.axis if self.axis is not None else 1,
                                                         largest=True,
                                                         sorted=True)]
        if self.out_max_val:
            if self.axis is not None:
                out_tensor = values
            else:
                out_tensor = np.stack(
                    [indices.astype(values.dtype), values], axis=1)
        else:
            if self.axis is not None:
                out_tensor = indices.astype(values.dtype)
            else:
                out_tensor = np.stack([indices.astype(values.dtype)], axis=1)
        num_top_axes = np.ndim(inputs[0])
        if num_top_axes < 3:
            num_top_axes = 3
        shape = np.ones(num_top_axes).astype(np.int32)
        if self.axis is not None:
            shape = list(inputs[0].shape)
            shape[self.axis] = self.k
        else:
            shape[0] = inputs[0].shape[0]
            shape[2] = self.k
            if self.out_max_val:
                shape[1] = 2

        out_tensor = np.reshape(out_tensor, shape)
        self.set_out_tensor(out_tensor)


class CaffeAXPYOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeAXPYOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeAXPYOp, attr_dict)
        assert self.check_required(), 'CaffeAXPYOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeAXPYOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = inputs[0] * inputs[1] + inputs[2]
        self.set_out_tensor(out_tensor)


class CaffeBATCHNORMOp(BaseLinearOp, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-5},
                    'scale_factor': {'type': AttrType.TENSOR, 'default': None},
                    'use_global_stats': {'type': AttrType.INT, 'default': 1}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeBATCHNORMOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeBATCHNORMOp, attr_dict)
        assert self.check_required(), 'CaffeBATCHNORMOp is missing a required parameter.'
        if not self.use_global_stats:
            WARN(
                '[Parser]: Dose not support CaffeBATCHNORM (%s) with use_global_stats=False!' % self.name)

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'use_global_stats':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(CaffeBATCHNORMOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(CaffeBATCHNORMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        scale = 0 if (self.scale_factor is None or FLOAT_EQUAL(
            self.scale_factor, 0)) else (1 / self.scale_factor)
        mean = self.weights * scale
        var = self.biases * scale
        self.weights = 1 / (np.sqrt(var + self.epsilon))
        self.biases = - mean / (np.sqrt(var + self.epsilon))
        if self.data_format == 'NCHW':
            shape = [1] * len(inputs[0].shape)
            shape[1] = -1
            w = np.reshape(self.weights, shape)
            b = np.reshape(self.biases, shape)
        else:
            w, b = self.weights, self.biases
        out_tensor = inputs[0] * w + b
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BatchNormalization', 'version': 9}


class CaffeBATCHREINDEXOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeBATCHREINDEXOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeBATCHREINDEXOp, attr_dict)
        assert self.check_required(), 'CaffeBATCHREINDEXOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeBATCHREINDEXOp, self).infer_shape()
        inputs = self.get_input_tensors()
        # inputs[1] is a one dim vector of int for index
        assert len(
            inputs[1].shape) == 1, 'The length of inputs is invalid in CaffeBATCHREINDEXOp.'
        out_tensor = inputs[0][inputs[1].astype(np.int32)]
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Gather', 'version': 11}


class CaffeBIASOp(OpHasAxis, OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 1},
                    'num_axes': {'type': AttrType.INT, 'default': 1},
                    'bias_reshape_dim': {'type': AttrType.INTS, 'default': []},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeBIASOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeBIASOp, attr_dict)
        assert self.check_required(), 'CaffeBIASOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeBIASOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = int(OpHasAxis.make_axes_non_negative(
            self.axis, len(inputs[0].shape)))
        if len(inputs[1].shape) == len(inputs[0].shape):
            bias = inputs[1]
        else:
            expand_dims = len(inputs[0].shape) - \
                len(inputs[1].shape) - self.axis
            self.bias_reshape_dim = [1] * self.axis + \
                list(inputs[1].shape) + [1] * expand_dims
            bias = np.reshape(inputs[1], self.bias_reshape_dim)
        out_tensor = inputs[0] + bias
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Add', 'version': 7}


class CaffeBNOp(OpHasBiases, OpHasWeights, OpHasMultipleOutPorts, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'bn_mode': {'type': AttrType.STRING, 'options': ['LEARN', 'INFERENCE'], 'default': 'INFERENCE'},
                    'epsilon': {'type': AttrType.FLOAT, 'default': 1e-9}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeBNOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeBNOp, attr_dict)
        assert self.check_required(), 'CaffeBNOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeBNOp, self).infer_shape()
        self.weights = np.squeeze(self.weights)
        if np.ndim(self.weights) == 0:
            self.weights = np.array([self.weights])
        self.biases = np.squeeze(self.biases)
        if np.ndim(self.biases) == 0:
            self.biases = np.array([self.biases])
        inputs = self.get_input_tensors()
        out_tensor = inputs[0] * np.reshape(self.weights, list(self.weights.shape) + [1, 1]) \
            + np.reshape(self.biases, list(self.biases.shape) + [1, 1])
        self.set_out_tensor([out_tensor])

    @property
    def correspond_onnx_op(self):
        return {'type': 'BatchNormalization', 'version': 9}


class CaffeBNLLOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeBNLLOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeBNLLOp, attr_dict)
        assert self.check_required(), 'CaffeBNLLOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeBNLLOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.log(1. + np.exp(*inputs))
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BNLL', 'version': 1}


class CaffeCLIPOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'min': {'type': AttrType.FLOAT, 'required': True, 'default': 0.},
                    'max': {'type': AttrType.FLOAT, 'required': True, 'default': 6.}}
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeCLIPOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeCLIPOp, attr_dict)
        assert self.check_required(), 'CaffeCLIPOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeCLIPOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.clip(inputs[0], self.min, self.max)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Clip', 'version': 6}


class CaffeCONCATOp(OpHasAxis, OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 1}}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeCONCATOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeCONCATOp, attr_dict)
        if "axis" not in attr_dict and "concat_dim" in attr_dict:
            self.axis = int(attr_dict["concat_dim"])
        assert self.check_required(), 'CaffeCONCATOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeCONCATOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = OpHasAxis.make_axes_non_negative(
            self.axis, len(inputs[0].shape))
        out_tensor = reduce(lambda x, y: np.concatenate(
            [x, y], axis=self.axis), inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Concat', 'version': 4}


class CaffeCONVOLUTIONOp(BaseConvOp, CaffeHasBiasTerm, CaffeHasPad):
    @classmethod
    def attributes(cls):
        return {1: {}}

    @classmethod
    def perm_caffe_to_onnx(cls, dim=4):
        return list(range(dim))

    def __init__(self, graph, attr_dict=None):
        super(CaffeCONVOLUTIONOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeCONVOLUTIONOp, attr_dict)
        if not self.bias_term and self.biases is None and self.num_output is not None:
            self.biases = np.zeros(shape=(self.num_output,), dtype=np.float32)
        assert self.check_required(), 'CaffeCONVOLUTIONOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeCONVOLUTIONOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None or len(self.kernel_shape) == 0:
            self.kernel_shape = self.weights.shape[2:]
        out_shape = BaseConvOp.cal_out_shape(inputs[0].shape[2:],
                                             self.pads,
                                             self.strides,
                                             self.kernel_shape,
                                             self.auto_pad,
                                             dilations=self.dilations,
                                             data_format='NCHW')
        out_shape = [inputs[0].shape[0], self.num_output] + out_shape
        out_tensor = np.random.ranf(size=out_shape).astype(np.float32)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class CaffeCONVOLUTIONDEPTHWISEOp(BaseConvOp, CaffeHasBiasTerm):
    @classmethod
    def attributes(cls):
        return {1: {}}

    @classmethod
    def perm_caffe_to_onnx(cls, dim=4):
        return [0, 1, 2, 3]

    def __init__(self, graph, attr_dict=None):
        super(CaffeCONVOLUTIONDEPTHWISEOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeCONVOLUTIONDEPTHWISEOp, attr_dict)
        if not self.bias_term and self.biases is None and self.num_output is not None:
            self.biases = np.zeros(shape=(self.num_output,), dtype=np.float32)
        assert self.check_required(), 'CaffeCONVOLUTIONDEPTHWISEOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeCONVOLUTIONDEPTHWISEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None or len(self.kernel_shape) == 0:
            self.kernel_shape = self.weights.shape[2:4]
        out_shape = BaseConvOp.cal_out_shape(inputs[0].shape[2:4],
                                             self.pads,
                                             self.strides,
                                             self.kernel_shape,
                                             self.auto_pad,
                                             dilations=self.dilations,
                                             data_format='NCHW')
        out_shape = [inputs[0].shape[0], self.num_output] + out_shape
        out_tensor = np.random.ranf(size=out_shape).astype(np.float32)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class CaffeCROPOp(OpHasAxis, OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 2},
                    'offset': {'type': AttrType.INTS, 'required': True, 'default': []},
                    'size': {'type': AttrType.INTS, 'default': []}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeCROPOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeCROPOp, attr_dict)
        assert self.check_required(), 'CaffeCROPOp is missing a required parameter.'

    def infer_shape(self, input_tensor=None):
        super(CaffeCROPOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = OpHasAxis.make_axes_non_negative(
            self.axis, len(inputs[0].shape))
        input_dim = len(inputs[0].shape)
        offset = [0] * input_dim
        for i in range(input_dim):
            if i >= self.axis:
                if len(self.offset) == 1:
                    offset[i] = self.offset[0]
                else:
                    offset[i] = self.offset[i - self.axis]
        self.offset = offset
        self.size = list(inputs[0].shape[:self.axis]) + \
            list(inputs[1].shape[self.axis:])
        out_tensor = tf.slice(inputs[0], np.array(
            self.offset, np.int64), np.array(self.size, np.int64)).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Slice', 'version': 10}


class CaffeDATAOp(OpHasOneOutPort, InputLikeOp, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def infer_shape(self, input_tensor=None):
        super(CaffeDATAOp, self).infer_shape()
        assert input_tensor is not None, 'input_tensor is empty in CaffeDATAOp.'
        out_tensor = input_tensor.copy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Input', 'version': 1}


class CaffeDECONVOLUTIONOp(OpHasAxis, BaseConvOp, CaffeHasBiasTerm, CaffeHasPad):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 1}}}

    @classmethod
    def perm_caffe_to_onnx(cls, dim=4):
        return list(range(dim))

    def __init__(self, graph, attr_dict=None):
        super(CaffeDECONVOLUTIONOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeDECONVOLUTIONOp, attr_dict)
        if not self.bias_term and self.biases is None and self.num_output is not None:
            self.biases = np.zeros(shape=(self.num_output,), dtype=np.float32)
        assert self.check_required(), 'CaffeDECONVOLUTIONOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeDECONVOLUTIONOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = OpHasAxis.make_axes_non_negative(
            self.axis, len(inputs[0].shape))
        input_tensor = torch.from_numpy(inputs[0])
        torch_weights = torch.from_numpy(np.transpose(
            self.weights, axes=CaffeDECONVOLUTIONOp.perm_caffe_to_onnx()))
        torch_biases = torch.from_numpy(self.biases)
        out_tensor = torch.nn.functional.conv_transpose2d(input_tensor,
                                                          torch_weights,
                                                          bias=torch_biases,
                                                          stride=(
                                                              self.strides[0], self.strides[1]),
                                                          padding=(
                                                              self.pads[0], self.pads[1]),
                                                          groups=self.group,
                                                          dilation=(self.dilations[0], self.dilations[1])).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ConvTranspose', 'version': 1}


class CaffeDETECTIONOUTPUTOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'num_classes': {'type': AttrType.INTS, 'default': 21}}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeDETECTIONOUTPUTOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeDETECTIONOUTPUTOp, attr_dict)
        assert self.check_required(), 'CaffeDETECTIONOUTPUTOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeDETECTIONOUTPUTOp, self).infer_shape()


class CaffeDROPOUTOp(OpHasOneOutPort, CaffeOp):
    def infer_shape(self):
        super(CaffeDROPOUTOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.identity(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class CaffeDUMMYDATAOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'data_filler': {'type': AttrType.UNDEFINED, 'default': []},
                    'shape': {'type': AttrType.INTS, 'default': []},
                    'value': {'type': AttrType.TENSOR, 'default': None}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeDUMMYDATAOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeDUMMYDATAOp, attr_dict)
        assert self.check_required(), 'CaffeDUMMYDATAOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeDUMMYDATAOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs) >= 1 and inputs[0] is not None and (list(inputs[0].shape) == self.shape):
            out_tensor = inputs[0]
        else:
            if len(self.data_filler) == 0:
                out_tensor = np.zeros(self.shape, np.float32)
            else:
                out_tensor = np.ones(self.shape, np.float32) * \
                    self.data_filler[0].get('value', 1.0)
        self.value = out_tensor
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Constant', 'version': 9}


class CaffeELTWISEOp(OpHasMethod, OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'method': {'options': ['PROD', 'SUM', 'MAX'], 'default': 'SUM'},
                    'sequence': {'type': AttrType.INT, 'default': 0},
                    'coeff': {'type': AttrType.FLOATS, 'default': []}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeELTWISEOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeELTWISEOp, attr_dict)
        assert self.check_required(), 'CaffeELTWISEOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'sequence':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(CaffeELTWISEOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(CaffeELTWISEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs) > 2:
            self.sequence = True
        if self.method == 'PROD':
            out_tensor = reduce(lambda x, y: x * y, inputs)
        elif self.method == 'SUM':
            if not self.coeff:
                self.coeff = [1.] * len(inputs)
            inputs = [inp * c for (inp, c) in zip(inputs, self.coeff)]
            out_tensor = reduce(lambda x, y: x + y, inputs)
        else:
            out_tensor = reduce(lambda x, y: np.maximum(x, y), inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        if self.method == 'PROD':
            return {'type': 'Mul', 'version': 7}
        elif self.method == 'SUM':
            if self.sequence:
                return {'type': 'Sum', 'version': 8}
            else:
                return {'type': 'Add', 'version': 7}
        else:
            return {'type': 'Max', 'version': 6}


class CaffeEXPOp(OpHasOneOutPort, CaffeHasBaseScaleShift):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeEXPOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeEXPOp, attr_dict)
        assert self.check_required(), 'CaffeEXPOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeEXPOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.base < 0:
            out_tensor = np.exp(inputs[0] * self.scale + self.shift)
        else:
            out_tensor = np.power(
                self.base, inputs[0] * self.scale + self.shift)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Exp', 'version': 6}


class CaffeELUOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 1.}}
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeELUOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeELUOp, attr_dict)
        assert self.check_required(), 'CaffeELUOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeELUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0])
        mask = out_tensor <= 0
        out_tensor[mask] = self.alpha * (np.exp(out_tensor[mask]) - 1)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Elu', 'version': 6}


class CaffeFILTEROp(OpHasMultipleOutPorts, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeFILTEROp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeFILTEROp, attr_dict)
        assert self.check_required(), 'CaffeFILTEROp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeFILTEROp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert all([inp.shape[0] == inputs[-1].shape[0]
                    for inp in inputs[:-1]]), 'input shape is invalid in CaffeFILTEROp.'
        mask = inputs[-1].astype(bool)
        mask = np.squeeze(mask, axis=tuple(range(1, len(inputs[0].shape))))
        out_tensors = [np.array(inp)[mask] for inp in inputs[:-1]]
        self.set_out_tensor(out_tensors)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Filter', 'version': 1}


class CaffeFLATTENOp(OpHasAxis, OpHasOneOutPort, CaffeOp):
    @staticmethod
    def cal_dim(input_shape, axis, end_axis):
        ret = copy.deepcopy(input_shape)
        if input_shape and 0 <= axis < len(input_shape) and axis <= end_axis < len(input_shape):
            ret = []
            for i in range(len(input_shape)):
                if i < axis:
                    ret.append(input_shape[i])
            ret.append(int(np.prod(input_shape[axis: end_axis + 1])))
            for i in range(len(input_shape)):
                if i > end_axis:
                    ret.append(input_shape[i])
        return ret

    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT, 'default': 1},
                    'end_axis': {'type': AttrType.INT, 'default': -1}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeFLATTENOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeFLATTENOp, attr_dict)
        assert self.check_required(), 'CaffeFLATTENOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeFLATTENOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = int(OpHasAxis.make_axes_non_negative(
            self.axis, inputs[0].shape))
        self.end_axis = int(OpHasAxis.make_axes_non_negative(
            self.end_axis, len(inputs[0].shape)))
        dim = type(self).cal_dim(inputs[0].shape, self.axis, self.end_axis)
        out_tensor = np.reshape(inputs[0], newshape=dim)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reshape', 'version': 5}


class CaffeINNER_PRODUCTOp(OpHasAxis, BaseLinearOp, CaffeHasBiasTerm):
    @classmethod
    def attributes(cls):
        return {1: {'transpose': {'type': AttrType.INT, 'default': 0},
                    'axis': {'default': 1}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeINNER_PRODUCTOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeINNER_PRODUCTOp, attr_dict)
        assert self.check_required(), 'CaffeINNER_PRODUCTOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'transpose':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(CaffeINNER_PRODUCTOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(CaffeINNER_PRODUCTOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = OpHasAxis.make_axes_non_negative(
            self.axis, len(inputs[0].shape))
        new_shape = list(inputs[0].shape[:self.axis]) + [-1]
        inp = np.reshape(inputs[0], newshape=new_shape)
        out_tensor = np.matmul(inp, np.transpose(self.weights))
        self.set_out_tensor(out_tensor)


class CaffeINTERPOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'height': {'type': AttrType.INT, 'default': 0},
                    'width': {'type': AttrType.INT, 'default': 0},
                    'zoom_factor': {'type': AttrType.INT, 'default': 1},
                    'shrink_factor': {'type': AttrType.INT, 'default': 1},
                    'pad_beg': {'type': AttrType.INT, 'default': 0},
                    'pad_ebd': {'type': AttrType.INT, 'default': 0},
                    'mode': {'type': AttrType.STRING, 'default': 'linear'}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeINTERPOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeINTERPOp, attr_dict)
        assert self.check_required(), 'CaffeINTERPOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeINTERPOp, self).infer_shape()
        inputs = self.get_input_tensors()
        height_in, width_in = inputs[0].shape[2:4]
        if self.shrink_factor != 1 and self.zoom_factor == 1:
            height_out = int(
                np.floor((height_in - 1) / self.shrink_factor) + 1)
            width_out = int(np.floor((width_in - 1) / self.shrink_factor) + 1)
        elif self.shrink_factor == 1 and self.zoom_factor != 1:
            height_out = height_in + (height_in - 1) * (self.zoom_factor - 1)
            width_out = width_in + (width_in - 1) * (self.zoom_factor - 1)
        elif self.height != 0 and self.width != 0:
            height_out = self.height
            width_out = self.width
        elif self.shrink_factor != 1 and self.zoom_factor != 1:
            height_out = int(
                np.floor((height_in - 1) / self.shrink_factor) + 1)
            width_out = int(np.floor((width_in - 1) / self.shrink_factor) + 1)
            height_out = height_out + (height_out - 1) * (self.zoom_factor - 1)
            width_out = width_out + (width_out - 1) * (self.zoom_factor - 1)
        else:
            height_out, width_out = height_in, width_in
        self.height, self.width = height_out, width_out
        out_tensor = tf.compat.v1.image.resize_bilinear(np.transpose(inputs[0], axes=(
            0, 2, 3, 1)), size=np.array([height_out, width_out], np.int32)).numpy()
        out_tensor = np.transpose(out_tensor, (0, 3, 1, 2))
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Resize', 'version': 11}


class CaffeLOGOp(OpHasOneOutPort, CaffeHasBaseScaleShift):
    # LogLayer computes outputs y = log_base(shift + scale * x), for base > 0.
    # Or if base is set to the default (-1), base is set to e,
    # so y = ln(shift + scale * x) = log_e(shift + scale * x)
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeLOGOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeLOGOp, attr_dict)
        assert self.check_required(), 'CaffeLOGOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeLOGOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.base < 0:
            out_tensor = np.log(inputs[0] * self.scale + self.shift)
        else:
            out_tensor = np.log(
                inputs[0] * self.scale + self.shift) / np.log(self.base)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Log', 'version': 6}


class CaffeLRNOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'size': {'type': AttrType.INT, 'default': 5},
                    'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                    'beta': {'type': AttrType.FLOAT, 'default': 0.75},
                    'norm_region': {'type': AttrType.STRING, 'default': 'ACROSS_CHANNELS', 'options': ['ACROSS_CHANNELS', 'WITHIN_CHANNEL']},
                    'k': {'type': AttrType.FLOAT, 'default': 1.0},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeLRNOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeLRNOp, attr_dict)
        if self.norm_region == "WITHIN_CHANNEL":
            self.k = 1.0
        assert self.check_required(), 'CaffeLRNOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeLRNOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_tensor = torch.from_numpy(inputs[0])
        out_tensor = torch.nn.functional.local_response_norm(input_tensor,
                                                             self.size,
                                                             alpha=self.alpha,
                                                             beta=self.beta,
                                                             k=self.k).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LRN', 'version': 1}


class CaffeLSTMOp(CaffeRecurrent):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeLSTMOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeLSTMOp, attr_dict)
        assert self.check_required(), 'CaffeLSTMOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeLSTMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        time_steps, batch_size = inputs[0].shape[:2]
        Y = np.random.ranf(
            (time_steps, batch_size, self.num_output)).astype(np.float32)
        out_tensor_list = [Y]
        if self.expose_hidden:
            H = np.random.ranf(
                (1, batch_size, self.num_output)).astype(np.float32)
            C = np.random.ranf(
                (1, batch_size, self.num_output)).astype(np.float32)
            out_tensor_list.extend([H, C])
        self.set_out_tensor(out_tensor_list)


class CaffeMVNOp(OpHasAxis, OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-9},
                    'normalize_variance': {'type': AttrType.INT, 'default': 1},
                    'across_channels': {'type': AttrType.INT, 'default': 0},
                    'axes': {'default': [2, 3]}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeMVNOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeMVNOp, attr_dict)
        assert self.check_required(), 'CaffeMVNOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeMVNOp, self).infer_shape()
        self.axes = [1, 2, 3] if self.across_channels else [2, 3]
        inputs = self.get_input_tensors()
        dim = list(inputs[0].shape) + [1] * (4 - len(inputs[0].shape))
        reshaped = np.reshape(inputs[0], dim)
        data_mean = np.mean(reshaped, axis=tuple(self.axes), keepdims=True)
        out_tensor = reshaped - data_mean
        if self.normalize_variance:
            data_std = np.std(reshaped, axis=tuple(self.axes), keepdims=True)
            out_tensor = out_tensor / (data_std + self.epsilon)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        if self.normalize_variance:
            return {'type': 'MeanVarianceNormalization', 'version': 9}
        else:
            return super(CaffeMVNOp, self).correspond_onnx_op


class CaffeNORMALIZEOp(OpHasWeights, OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-10},
                    'across_spatial': {'type': AttrType.INT, 'default': 1},
                    'channel_shared': {'type': AttrType.INT, 'default': 1}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeNORMALIZEOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeNORMALIZEOp, attr_dict)
        assert self.check_required(), 'CaffeNORMALIZEOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeNORMALIZEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        ########### TODO ###################
        self.set_out_tensor(inputs[0])


class CaffePERMUTEOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'order': {'type': AttrType.INTS, 'default': []}}}

    def __init__(self, graph, attr_dict=None):
        super(CaffePERMUTEOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffePERMUTEOp, attr_dict)
        assert self.check_required(), 'CaffePERMUTEOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffePERMUTEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.transpose(
            inputs[0], axes=self.order if self.order else None)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Transpose', 'version': 1}


class CaffePOOLINGOp(OpHasMethod, CaffeHasPad, OpHasVariableOutPorts):
    @classmethod
    def attributes(cls):
        return {1: {'method': {'type': AttrType.STRING, 'default': 'MAX', 'options': ['MAX', 'AVE', 'STOCHASTIC']},
                    'round_mode': {'type': AttrType.STRING, 'default': 'CEIL', 'options': ['CEIL', 'FLOOR']},
                    'global_pooling': {'type': AttrType.INT, 'default': 0},
                    'dilations': {'default': [1, 1]}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffePOOLINGOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffePOOLINGOp, attr_dict)
        assert self.check_required(), 'CaffePOOLINGOp is missing a required parameter.'

    def __setattr__(self, item, value):
        if item == 'round_mode':
            try:
                if value in ['CEIL', 'FLOOR']:
                    self.__dict__['_attr'][item].value = value
            except:
                if value in ['CEIL', 'FLOOR']:
                    self.__dict__['_attr'][item] = Attribute(
                        item, {'type': AttrType.STRING, 'value': value})
        else:
            super(CaffePOOLINGOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(CaffePOOLINGOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.global_pooling and not self.kernel_shape:
            self.kernel_shape = self.get_input_shapes()[0][2:4]
        if self.method != 'STOCHASTIC':
            '''
            # input_tensor = torch.from_numpy(inputs[0])
            # paddings = self.torch_pads
            # input_tensor = torch.nn.functional.pad(input_tensor, paddings, mode='constant', value=0)
            # if self.method == 'MAX':
            #     out_tensor = torch.nn.functional.max_pool2d(input_tensor,
            #                                                 kernel_size=(self.kernel_shape[0], self.kernel_shape[1]),
            #                                                 stride=(self.strides[0], self.strides[1]),
            #                                                 dilation=(self.dilations[0], self.dilations[1]),
            #                                                 ceil_mode=True if self.round_mode == 'CEIL' else False
            #                                                 ).numpy()
            #     out_tensor_list = [out_tensor]
            #     if len(self.get_out_ports()) == 2:
            #         in_shape = inputs[0].shape[2:]
            #         argmax = np.random.randint(0, in_shape[1] * in_shape[0] + in_shape[1], size=out_tensor.shape, dtype=np.int32)
            #         out_tensor_list.append(argmax)
            # else:
            #     out_tensor = torch.nn.functional.avg_pool2d(input_tensor,
            #                                                 kernel_size=(self.kernel_shape[0], self.kernel_shape[1]),
            #                                                 stride=(self.strides[0], self.strides[1]),
            #                                                 ceil_mode=True if self.round_mode == 'CEIL' else False,
            #                                                 count_include_pad=False
            #                                                 ).numpy()
            #     out_tensor_list = [out_tensor]
            '''
            h, w = inputs[0].shape[2:]
            pad_h, pad_w = self.torch_pads[2], self.torch_pads[0]
            if self.round_mode == 'CEIL':
                out_h = int(
                    np.ceil((h + 2 * pad_h - self.kernel_shape[0]) / self.strides[0]) + 1)
                out_w = int(
                    np.ceil((w + 2 * pad_w - self.kernel_shape[1]) / self.strides[1]) + 1)
            else:
                out_h = int(
                    np.floor((h + 2 * pad_h - self.kernel_shape[0]) / self.strides[0]) + 1)
                out_w = int(
                    np.floor((w + 2 * pad_w - self.kernel_shape[1]) / self.strides[1]) + 1)
            if pad_h != 0 or pad_w != 0:
                if (out_h - 1) * self.strides[0] >= h + pad_h:
                    out_h -= 1
                    pad_in_height = (
                        out_h - 1) * self.strides[0] + (self.kernel_shape[0] - 1) * self.dilations[0] + 1 - h
                    pad_bottom = (abs(pad_in_height) // 2) * \
                        np.sign(pad_in_height)
                    pad_top = pad_in_height - pad_bottom
                    self.pads[0] = pad_top
                    self.pads[2] = pad_bottom
                if (out_w - 1) * self.strides[1] >= w + pad_w:
                    out_w -= 1
                    pad_in_width = (
                        out_w - 1) * self.strides[1] + (self.kernel_shape[1] - 1) * self.dilations[1] + 1 - w
                    pad_right = (abs(pad_in_width) // 2) * \
                        np.sign(pad_in_width)
                    pad_left = pad_in_width - pad_right
                    self.pads[1] = pad_left
                    self.pads[3] = pad_right

            out_shape = list(inputs[0].shape[0:2]) + [out_h, out_w]
            out_tensor = np.random.ranf(out_shape).astype(inputs[0].dtype)
            out_tensor_list = [out_tensor]
            if self.method == 'MAX' and len(self.get_out_ports()) == 2:
                in_shape = inputs[0].shape[2:]
                argmax = np.random.randint(
                    0, in_shape[1] * in_shape[0] + in_shape[1], size=out_tensor.shape, dtype=np.int32)
                out_tensor_list.append(argmax.astype(np.float32))
            self.set_out_tensor(out_tensor_list)
        else:
            WARN(
                '[Parser]: Pooling method STOCHASTIC for node(%s) is not supported!' % self.name)

    @property
    def correspond_onnx_op(self):
        if self.method == 'MAX':
            return {'type': 'MaxPool', 'version': 10}
        elif self.method == 'AVE':
            return {'type': 'AveragePool', 'version': 10}
        else:
            return None


class CaffePOWEROp(OpHasOneOutPort, CaffeHasBaseScaleShift):
    # PowerLayer computes outputs y = (shift + scale * x) ^ power
    @classmethod
    def attributes(cls):
        return {1: {'power': {'type': AttrType.FLOAT, 'default': 1}}}

    def __init__(self, graph, attr_dict=None):
        super(CaffePOWEROp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffePOWEROp, attr_dict)
        assert self.check_required(), 'CaffePOWEROp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'power':
                ret = float(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(CaffePOWEROp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(CaffePOWEROp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.power(inputs[0] * self.scale + self.shift, self.power)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pow', 'version': 7}


class CaffePRELUOp(OpHasWeights, BaseReluOp, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'channel_shared': {'type': AttrType.INT, 'default': 0}}}

    def __init__(self, graph, attr_dict=None):
        super(CaffePRELUOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffePRELUOp, attr_dict)
        assert self.check_required(), 'CaffePRELUOp is missing a required parameter.'
        self.activations = 'PRELU'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'channel_shared':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(CaffePRELUOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(CaffePRELUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if np.ndim(self.weights) == 0:
            channels = inputs[0].shape[1]
            self.weights = np.tile(self.weights, (channels,))
        out_tensor = torch.nn.functional.prelu(torch.from_numpy(
            inputs[0]), torch.from_numpy(self.weights)).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'PRelu', 'version': 9}


class CaffePRIORBOXOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'aspect_ratio': {'type': AttrType.FLOATS, 'required': False, 'default': [1.0]},
                    'min_size': {'type': AttrType.FLOATS, 'required': True, 'default': []},
                    'max_size': {'type': AttrType.FLOATS, 'required': True, 'default': []},
                    'flip': {'type': AttrType.INT, 'default': 1},
                    'clip': {'type': AttrType.INT, 'default': 0},
                    'num_priors': {'type': AttrType.INT, 'default': 6},
                    'variance': {'type': AttrType.FLOATS, 'required': True, 'default': []},
                    'step': {'type': AttrType.FLOAT, 'default': 0},
                    'step_w': {'type': AttrType.FLOAT, 'default': 0},
                    'step_h': {'type': AttrType.FLOAT, 'default': 0},
                    'offset': {'type': AttrType.FLOAT, 'default': 0.5},
                    'img_size': {'type': AttrType.INTS, 'default': []},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffePRIORBOXOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffePRIORBOXOp, attr_dict)
        assert self.check_required(), 'CaffePRIORBOXOp is missing a required parameter.'

        tmp_aspect_ratios = [1.0]
        for i, ar in enumerate(self.aspect_ratio):
            already_exist = False
            for j, tr in enumerate(tmp_aspect_ratios):
                if FLOAT_EQUAL(ar - tr, 1e-6):
                    already_exist = True
                    break
            if not already_exist:
                tmp_aspect_ratios.append(ar)
                if self.flip:
                    tmp_aspect_ratios.append(1. / ar)
        self.aspect_ratio = tmp_aspect_ratios
        self.num_priors = len(self.aspect_ratio) * \
            len(self.min_size) + len(self.max_size)
        if self.step != 0:
            if self.step_w == 0:
                self.step_w = self.step
            if self.step_h == 0:
                self.step_h = self.step

    def infer_shape(self):
        super(CaffePRIORBOXOp, self).infer_shape()
        inputs = self.get_input_tensors()
        grid_height, grid_width = list(inputs[0].shape[2:4])
        if not self.img_size:
            self.img_size = list(inputs[1].shape[2:4])[::-1]
        out_shape = [1, 2, grid_height * grid_width * self.num_priors * 4]
        if self.step_w == 0 or self.step_h == 0:
            self.step_w = self.img_size[0] / grid_width
            self.step_h = self.img_size[1] / grid_height
        out_tensor = np.zeros(out_shape, np.float32)
        np_img_size = np.array(self.img_size, np.float32)
        idx = 0
        for h in range(grid_height):
            for w in range(grid_width):
                meta_idx = idx
                center_x = (w + self.offset) * self.step_w
                center_y = (h + self.offset) * self.step_h
                np_center = np.array([center_x, center_y], np.float32)
                for min_s in self.min_size:
                    box_width, box_height = int(min_s), int(min_s)
                    out_tensor[0, 0, meta_idx: meta_idx + 2] = (
                        np_center - np.array([box_width, box_height]) / 2.0) / np_img_size
                    out_tensor[0, 0, meta_idx + 2: meta_idx + 4] = (
                        np_center + np.array([box_width, box_height]) / 2.0) / np_img_size
                    meta_idx += 4
                    for max_s in self.max_size:
                        box_width = np.sqrt(int(min_s) * int(max_s))
                        box_height = np.sqrt(int(min_s) * int(max_s))
                        out_tensor[0, 0, meta_idx: meta_idx + 2] \
                            = (np_center - np.array([box_width, box_height]) / 2.0) / np_img_size
                        out_tensor[0, 0, meta_idx + 2: meta_idx + 4] \
                            = (np_center + np.array([box_width, box_height]) / 2.0) / np_img_size
                        meta_idx += 4
                    for ar in self.aspect_ratio:
                        if np.fabs(ar - 1.) < 1e-6:
                            continue
                        box_width = min_s * np.sqrt(ar)
                        box_height = min_s / np.sqrt(ar)
                        out_tensor[0, 0, meta_idx: meta_idx + 2] \
                            = (np_center - np.array([box_width, box_height]) / 2.0) / np_img_size
                        out_tensor[0, 0, meta_idx + 2: meta_idx + 4] \
                            = (np_center + np.array([box_width, box_height]) / 2.0) / np_img_size
                        meta_idx += 4
                idx += self.num_priors * 4
        if self.clip:
            out_tensor = np.clip(out_tensor, 0, 1)
        self.set_out_tensor(out_tensor)


class CaffePROPOSALOp(OpHasMultipleOutPorts, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'pre_nms_topn': {'type': AttrType.INT, 'default': 6000},
                    'post_nms_topn': {'type': AttrType.INT, 'default': 300},
                    'nms_threshold': {'type': AttrType.FLOAT, 'default': 0.7},
                    'min_size': {'type': AttrType.INT, 'default': 16},
                    'scales': {'type': AttrType.INTS, 'default': [4, 8, 16, 32]},
                    'feat_stride': {'type': AttrType.INT, 'default': 16},
                    'anchors': {'type': AttrType.TENSOR, 'default': None}
                    }
                }

    @staticmethod
    def generate_anchors(anchor_scales, feat_stride):
        def _whctrs(anchor):
            """
            Return width, height, x center, and y center for an anchor (window).
            """
            w = anchor[2] - anchor[0] + 1
            h = anchor[3] - anchor[1] + 1
            x_ctr = anchor[0] + 0.5 * (w - 1)
            y_ctr = anchor[1] + 0.5 * (h - 1)
            return w, h, x_ctr, y_ctr

        def _mkanchors(ws, hs, x_ctr, y_ctr):
            """
            Given a vector of widths (ws) and heights (hs) around a center
            (x_ctr, y_ctr), output a set of anchors (windows).
            """
            ws = ws[:, np.newaxis]
            hs = hs[:, np.newaxis]
            anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                                 y_ctr - 0.5 * (hs - 1),
                                 x_ctr + 0.5 * (ws - 1),
                                 y_ctr + 0.5 * (hs - 1)))
            return anchors

        def _ratio_enum(anchor, ratios):
            """
            Enumerate a set of anchors for each aspect ratio wrt an anchor.
            """
            w, h, x_ctr, y_ctr = _whctrs(anchor)
            size = w * h
            size_ratios = size / ratios
            ws = np.round(np.sqrt(size_ratios))
            hs = np.round(ws * ratios)
            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
            return anchors

        def _scale_enum(anchor, scales):
            """
            Enumerate a set of anchors for each scale wrt an anchor.
            """
            w, h, x_ctr, y_ctr = _whctrs(anchor)
            ws = w * scales
            hs = h * scales
            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
            return anchors

        base_size = 16
        ratios = [0.5, 1, 2]
        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        ratio_anchors = _ratio_enum(base_anchor, ratios)
        anchors = np.vstack([_scale_enum(ratio_anchors[i, :], anchor_scales)
                             for i in range(ratio_anchors.shape[0])])
        return anchors.astype(np.float32)

    def __init__(self, graph, attr_dict=None):
        super(CaffePROPOSALOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffePROPOSALOp, attr_dict)
        assert self.check_required(), 'CaffePROPOSALOp is missing a required parameter.'
        self.anchors = CaffePROPOSALOp.generate_anchors(
            np.array(self.scales), self.feat_stride)

    def infer_shape(self):
        super(CaffePROPOSALOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = [np.random.ranf(
            (self.post_nms_topn, 5)).astype(np.float32)]
        self.set_out_tensor(out_tensors)


class CaffeREDUCTIONOp(OpHasAxis, OpHasMethod, OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'method': {'type': AttrType.STRING, 'default': 'SUM', 'options': ['SUM', 'ASUM', 'SUMSQ', 'MEAN']},
                    'axis': {'type': AttrType.INT, 'default': 0},
                    'coeff': {'type': AttrType.FLOAT, 'default': 1.0},
                    'keepdims': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeREDUCTIONOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeREDUCTIONOp, attr_dict)
        assert self.check_required(), 'CaffeREDUCTIONOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeREDUCTIONOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs) >= 1 and inputs[0] is not None:
            self.axis = OpHasAxis.make_axes_non_negative(
                self.axis, len(inputs[0].shape))
            axes = tuple(range(self.axis, len(inputs[0].shape)))
            if self.method == 'SUM':
                out_tensor = np.sum(
                    inputs[0], axis=axes, keepdims=bool(self.keepdims))
            elif self.method == 'ASUM':
                out_tensor = np.abs(inputs[0])
                out_tensor = np.sum(out_tensor, axis=axes,
                                    keepdims=bool(self.keepdims))
            elif self.method == 'SUMSQ':
                out_tensor = np.square(inputs[0])
                out_tensor = np.sum(out_tensor, axis=axes,
                                    keepdims=bool(self.keepdims))
            else:
                out_tensor = np.mean(
                    inputs[0], axis=axes, keepdims=bool(self.keepdims))
            if not FLOAT_EQUAL(self.coeff, 1.0):
                out_tensor *= self.coeff
        else:
            out_tensor = None
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        if self.method in ('SUM', 'ASUM'):
            return {'type': 'ReduceSum', 'version': 11}
        elif self.method == 'SUMSQ':
            return {'type': 'ReduceSumSquare', 'version': 11}
        elif self.method == 'MEAN':
            return {'type': 'ReduceMean', 'version': 11}
        else:
            return super(CaffeREDUCTIONOp, self).correspond_onnx_op


class CaffeREFINEDETECTIONOUTPUTOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'num_classes': {'type': AttrType.INTS, 'requied': True},
                    'share_location': {'type': AttrType.INT, 'default': 1},
                    'background_label_id': {'type': AttrType.INT, 'default': 0},
                    'variance_encoded_in_target': {'type': AttrType.INT, 'default': 0},
                    'keep_top_k': {'type': AttrType.INT, 'default': -1},
                    'confidence_threshold': {'type': AttrType.FLOAT, 'requied': True},
                    'visualize': {'type': AttrType.INT, 'default': 0},
                    'visualize_threshold': {'type': AttrType.FLOAT},
                    'objectness_score': {'type': AttrType.FLOAT, 'default': 0.01},
                    }}

    def __init__(self, graph, attr_dict=None):
        super(CaffeREFINEDETECTIONOUTPUTOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeREFINEDETECTIONOUTPUTOp, attr_dict)
        assert self.check_required(), 'CaffeREFINEDETECTIONOUTPUTOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeREFINEDETECTIONOUTPUTOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.random.ranf((1, 7)).astype(np.float32)
        self.set_out_tensor(out_tensor)


class CaffeRELUOp(BaseReluOp, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 0.0}}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeRELUOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeRELUOp, attr_dict)
        assert self.check_required(), 'CaffeRELUOp is missing a required parameter.'
        self.activations = 'RELU' if FLOAT_EQUAL(
            self.alpha, 0) else 'LEAKYRELU'
        self.negative_slope = self.alpha

    def infer_shape(self):
        super(CaffeRELUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = self.cal_activation(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        if FLOAT_EQUAL(self.alpha, 0):
            return {'type': 'Relu', 'version': 6}
        else:
            return {'type': 'LeakyRelu', 'version': 6}


class CaffeRESHAPEOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT, 'default': 0},
                    'num_axes': {'type': AttrType.INT, 'default': -1},
                    'shape': {'type': AttrType.INTS}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeRESHAPEOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeRESHAPEOp, attr_dict)
        assert self.check_required(), 'CaffeRESHAPEOp is missing a required parameter.'
        self.inferred_axis = -1
        self.copy_axes = []
        self.constant_count = 1
        top_num_axes = len(self.shape)
        for i in range(top_num_axes):
            top_dim = self.shape[i]
            if top_dim == 0:
                self.copy_axes.append(i)
            elif top_dim == -1:
                self.inferred_axis = i
            else:
                self.constant_count *= top_dim

    def infer_shape(self):
        super(CaffeRESHAPEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        in_shape = list(inputs[0].shape)
        start_axis = self.axis if self.axis >= 0 else (
            len(in_shape) + self.axis + 1)
        end_axis = len(in_shape) if self.num_axes == - \
            1 else (start_axis + self.num_axes)
        num_new_axes = len(self.shape)
        top_shape = in_shape[0: start_axis] + \
            self.shape[0:num_new_axes] + in_shape[end_axis:]
        for i in range(len(self.copy_axes)):
            copy_axis_index = self.copy_axes[i]
            top_shape[start_axis +
                      copy_axis_index] = in_shape[start_axis + copy_axis_index]
        if self.inferred_axis >= 0:
            explicit_count = self.constant_count
            explicit_count *= int(np.prod(in_shape[0:start_axis]))
            explicit_count *= int(np.prod(in_shape[end_axis:]))
            for i in range(len(self.copy_axes)):
                copy_axis_index = self.copy_axes[i]
                explicit_count *= top_shape[start_axis + copy_axis_index]
            inferred_dim = int(np.prod(in_shape)) // explicit_count
            top_shape[start_axis + self.inferred_axis] = inferred_dim
        self.shape = top_shape
        out_tensor = np.reshape(inputs[0], newshape=self.shape)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reshape', 'version': 5}


class CaffeRNNOp(OpHasVariableOutPorts, OpHasWeights, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeRNNOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeRNNOp, attr_dict)
        assert self.check_required(), 'CaffeRNNOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeRNNOp, self).infer_shape()
        inputs = self.get_input_tensors()
        # TODO
        return
        '''
        in_shape = list(inputs[0].shape)
        if self.axis < 0:
            self.axis = int(np.abs(self.axis)) - len(self.shape)
        self.shape = [s if s != 0 else in_shape[self.axis+i]
                      for i, s in enumerate(self.shape)]
        base_dim = list(in_shape[:self.axis]) + self.shape
        if self.num_axes >= 0:
            base_dim = base_dim + list(in_shape[self.axis+self.num_axes:])
        for i, d in enumerate(base_dim):
            if d == -1:
                base_dim[i] = int(
                    np.abs(int(np.prod(in_shape)) // int(np.prod(base_dim))))
                break
        self.shape = base_dim
        out_tensor = np.reshape(inputs[0], newshape=self.shape)
        self.set_out_tensor(out_tensor)
        '''


class CaffeROIPOOLINGOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'pooled_h': {'type': AttrType.INT, 'default': 0},
                    'pooled_w': {'type': AttrType.INT, 'default': 0},
                    'spatial_scale': {'type': AttrType.FLOAT, 'default': 1.0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeROIPOOLINGOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeROIPOOLINGOp, attr_dict)
        assert self.check_required(), 'CaffeROIPOOLINGOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeROIPOOLINGOp, self).infer_shape()
        inputs = self.get_input_tensors()
        channels = inputs[0].shape[1]
        rois = inputs[1].shape[0]
        out_tensor = np.random.ranf(
            (rois, channels, self.pooled_h, self.pooled_w)).astype(np.float32)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'MaxRoiPool', 'version': 1}


class CaffeSCALEOp(OpHasAxis, OpHasOneOutPort, CaffeHasBiasTerm):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 1},
                    'num_axes': {'type': AttrType.INT, 'default': 1},
                    'bias_term': {'default': 0},
                    'scale_reshape_dim': {'type': AttrType.INTS, 'default': []},
                    'bias_reshape_dim': {'type': AttrType.INTS, 'default': []},
                    'identity': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeSCALEOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeSCALEOp, attr_dict)
        assert self.check_required(), 'CaffeSCALEOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'identity':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(CaffeSCALEOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(CaffeSCALEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = int(OpHasAxis.make_axes_non_negative(
            self.axis, len(inputs[0].shape)))
        if len(inputs) == 1:
            out_tensor = inputs[0]
            self.identity = True
        else:
            if len(inputs[1].shape) == len(inputs[0].shape):
                scale = inputs[1]
            else:
                expand_dims = len(inputs[0].shape) - \
                    len(inputs[1].shape) - self.axis
                self.scale_reshape_dim = [1] * self.axis + \
                    list(inputs[1].shape) + [1] * expand_dims
                scale = np.reshape(inputs[1], self.scale_reshape_dim)
            out_tensor = inputs[0] * scale
            if self.bias_term and len(inputs) == 3:
                if len(inputs[2].shape) == len(inputs[0].shape):
                    bias = inputs[2]
                else:
                    expand_dims = len(inputs[0].shape) - \
                        len(inputs[2].shape) - self.axis
                    self.bias_reshape_dim = [
                        1] * self.axis + list(inputs[2].shape) + [1] * expand_dims
                    bias = np.reshape(inputs[2], self.bias_reshape_dim)
                out_tensor = out_tensor + bias
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        if self.identity:
            return {'type': 'Identity', 'version': 1}
        else:
            if not self.bias_term:
                return {'type': 'Mul', 'version': 7}
            else:
                return None


class CaffeSHUFFLECHANNELOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'group': {'type': AttrType.INT, 'default': 1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeSHUFFLECHANNELOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeSHUFFLECHANNELOp, attr_dict)
        assert self.check_required(), 'CaffeSHUFFLECHANNELOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeSHUFFLECHANNELOp, self).infer_shape()
        inputs = self.get_input_tensors()
        new_shape = [inputs[0].shape[0], self.group,
                     inputs[0].shape[1] // self.group] + list(inputs[0].shape[2:])
        reshape1 = np.reshape(inputs[0], newshape=new_shape)
        transpose = np.transpose(reshape1, axes=(0, 2, 1, 3, 4))
        out_tensor = np.reshape(transpose, newshape=inputs[0].shape)
        self.set_out_tensor(out_tensor)


class CaffeSIGMOIDOp(BaseActivationOp, CaffeOp):
    def infer_shape(self):
        super(CaffeSIGMOIDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = (1 / (1 + np.exp(-inputs[0]))).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sigmoid', 'version': 6}


class CaffeSILENCEOp(CaffeOp):
    def infer_shape(self):
        super(CaffeSILENCEOp, self).infer_shape()


class CaffeSLICEOp(OpHasAxis, OpHasMultipleOutPorts, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 1},
                    'slice_dim': {'type': AttrType.INT, 'default': 1},
                    'slice_point': {'type': AttrType.INTS, 'default': []}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeSLICEOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeSLICEOp, attr_dict)
        if "axis" not in attr_dict and "slice_dim" in attr_dict:
            self.axis = int(attr_dict["slice_dim"])
        assert self.check_required(), 'CaffeSLICEOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'slice_point':
                ret = self.__dict__['_attr'][item].value
                if ret is not None:
                    if isinstance(ret, np.ndarray):
                        ret = ret.tolist()
                    else:
                        ret = list(ret)
                else:
                    ret = []
        except:
            ret = None
        if ret is None:
            ret = super(CaffeSLICEOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(CaffeSLICEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = OpHasAxis.make_axes_non_negative(
            self.axis, len(inputs[0].shape))
        out_tensors = np.split(inputs[0], np.array(
            self.slice_point, np.int64), axis=self.axis)
        self.set_out_tensor(out_tensors)


class CaffeSOFTMAXOp(OpHasAxis, OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 1}}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeSOFTMAXOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeSOFTMAXOp, attr_dict)
        assert self.check_required(), 'CaffeSOFTMAXOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeSOFTMAXOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = OpHasAxis.make_axes_non_negative(
            self.axis, len(inputs[0].shape))
        out_tensor = torch.nn.functional.softmax(
            torch.from_numpy(inputs[0]), dim=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Softmax', 'version': 13}


class CaffeSOFTMAX_LOSSOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeSOFTMAX_LOSSOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeSOFTMAX_LOSSOp, attr_dict)
        assert self.check_required(), 'CaffeSOFTMAX_LOSSOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeSOFTMAX_LOSSOp, self).infer_shape()


class CaffeSPLITOp(OpHasMultipleOutPorts, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeSPLITOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeSPLITOp, attr_dict)
        assert self.check_required(), 'CaffeSPLITOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeSPLITOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.set_out_tensor([inputs[0]] * len(self.get_output_shapes()))


class CaffeSPPOp(OpHasOneOutPort, OpHasMethod, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'pyramid_height': {'type': AttrType.INT, 'default': 1}}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeSPPOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeSPPOp, attr_dict)
        assert self.check_required(), 'CaffeSPPOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeSPPOp, self).infer_shape()
        inputs = self.get_input_tensors()
        channel = inputs[0].shape[1]
        k = 0
        for i in range(self.pyramid_height):
            k += 2**(2 * i)
        out_shape = [inputs[0].shape[0], channel * k]
        out_tensor = np.random.ranf(size=out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class CaffeTANHOp(BaseActivationOp, CaffeOp):
    def infer_shape(self):
        super(CaffeTANHOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.tanh(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Tanh', 'version': 6}


class CaffeTHRESHOLDOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'threshold': {'type': AttrType.FLOAT, 'default': 0}}}

    def __init__(self, graph, attr_dict=None):
        super(CaffeTHRESHOLDOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeTHRESHOLDOp, attr_dict)
        assert self.check_required(), 'CaffeTHRESHOLDOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeTHRESHOLDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = (inputs[0] > self.threshold).astype(np.int32)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Greater', 'version': 9}


class CaffeUPSAMPLEOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'scale': {'type': AttrType.INT, 'default': 2},
                    'upsample_h': {'type': AttrType.INT, 'required': False},
                    'upsample_w': {'type': AttrType.INT, 'required': False}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeUPSAMPLEOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeUPSAMPLEOp, attr_dict)
        assert self.check_required(), 'CaffeUPSAMPLEOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeUPSAMPLEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = torch.nn.functional.upsample(torch.from_numpy(inputs[0]),
                                                  scale_factor=float(
                                                      self.scale),
                                                  mode='bilinear',
                                                  align_corners=None).numpy()
        self.set_out_tensor(out_tensor)


class CaffeUPSAMPLEDARKNETOp(OpHasOneOutPort, CaffeOp):
    @classmethod
    def attributes(cls):
        return {1: {'strides': {'type': AttrType.INTS, 'default': [1, 1]}}
                }

    def __init__(self, graph, attr_dict=None):
        super(CaffeUPSAMPLEDARKNETOp, self).__init__(graph, attr_dict)
        self.update_attributes(CaffeUPSAMPLEDARKNETOp, attr_dict)
        assert self.check_required(), 'CaffeUPSAMPLEDARKNETOp is missing a required parameter.'

    def infer_shape(self):
        super(CaffeUPSAMPLEDARKNETOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = torch.nn.functional.upsample(torch.from_numpy(inputs[0]),
                                                  scale_factor=self.strides,
                                                  mode='bilinear',
                                                  align_corners=None).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Resize', 'version': 11}
