# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import tensorflow as tf
from itertools import product
from functools import partial
from ..op import *
from ...common.defs import FLOAT_EQUAL
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class AffineGridOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {20: {'align_corners': {'type': AttrType.BOOL, 'default': False}},
                }

    def __init__(self, graph, attr_dict=None):
        super(AffineGridOp, self).__init__(graph, attr_dict)
        self.update_attributes(AffineGridOp, attr_dict)
        assert self.check_required(), 'AffineGridOp is missing a required parameter.'

    def infer_shape(self):
        super(AffineGridOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 2 and inputs[1].ndim == 1, \
            'Meets invalid inputs of AffineGridOp(%s) in infer_shape!' % self.name
        theta = torch.from_numpy(inputs[0].astype(np.float32))
        if self.data_format == 'NCHW':
            size = inputs[1].tolist()
        else:
            size = [inputs[1][0], inputs[1][-1]] + inputs[1][1:-1].tolist()
        out_tensor = torch.nn.functional.affine_grid(theta,
                                                     size=size,
                                                     align_corners=self.align_corners).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))


class AveragePoolOp(BaseOnnxPoolOp, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'kernel_shape': {'required': True}},
                7: {'count_include_pad': {'type': AttrType.INT, 'default': 0},
                    'kernel_shape': {'required': True},
                    },
                10: {'ceil_mode': {'type': AttrType.INT, 'default': 0},
                     'count_include_pad': {'type': AttrType.INT, 'default': 0},
                     'kernel_shape': {'required': True},
                     },
                11: {'ceil_mode': {'type': AttrType.INT, 'default': 0},
                     'count_include_pad': {'type': AttrType.INT, 'default': 0},
                     'kernel_shape': {'required': True},
                     },
                19: {'ceil_mode': {'type': AttrType.INT, 'default': 0},
                     'count_include_pad': {'type': AttrType.INT, 'default': 0},
                     'kernel_shape': {'required': True},
                     },
                }

    def __init__(self, graph, attr_dict=None):
        super(AveragePoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(AveragePoolOp, attr_dict)
        assert self.check_required(), 'AveragePoolOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'ceil_mode':
                if cur_ver >= 10:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ret = False
            elif item == 'count_include_pad':
                if cur_ver >= 7:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ret = False
        except:
            ret = None
        if ret is None:
            ret = super(AveragePoolOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(AveragePoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        in_shape = inputs[0].shape[1:-
                                   1] if self.data_format == 'NHWC' else inputs[0].shape[2:]
        out_shape = BaseOnnxPoolOp.cal_out_shape(
            in_shape, self.pads, self.strides, self.kernel_shape, self.auto_pad, dilations=self.dilations, ceil_mode=self.ceil_mode)
        out_tensor_shape = list(inputs[0].shape[0:1]) + out_shape + list(inputs[0].shape[-1:]) \
            if self.data_format == 'NHWC' \
            else list(inputs[0].shape[0:2]) + out_shape
        out_tensor = np.random.ranf(out_tensor_shape).astype(inputs[0].dtype)
        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            # re-calculate pads
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                in_shape,
                out_shape,
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
            )
            self.auto_pad = 'NOTSET'
        self.set_out_tensor(out_tensor)


class BatchNormalizationOp(LayoutConcernedOp, OpHasVariableOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS, 'default': [1], 'required': True},
                    'epsilon': {'type': AttrType.FLOAT, 'default': 1e-5},
                    'momentum': {'type': AttrType.FLOAT, 'default': 0.9},
                    'spatial': {'type': AttrType.INT, 'default': 1}
                    },
                6: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-5},
                    'momentum': {'type': AttrType.FLOAT, 'default': 0.9},
                    'spatial': {'type': AttrType.INT, 'default': 1}
                    },
                7: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-5},
                    'momentum': {'type': AttrType.FLOAT, 'default': 0.9},
                    'spatial': {'type': AttrType.INT, 'default': 1}
                    },
                9: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-5},
                    'momentum': {'type': AttrType.FLOAT, 'default': 0.9},
                    'spatial': {'type': AttrType.INT, 'default': 1}
                    },
                14: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-5},
                     'momentum': {'type': AttrType.FLOAT, 'default': 0.9},
                     'training_mode': {'type': AttrType.INT, 'default': 0}
                     },
                15: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-5},
                     'momentum': {'type': AttrType.FLOAT, 'default': 0.9},
                     'training_mode': {'type': AttrType.INT, 'default': 0}
                     }
                }

    def __init__(self, graph, attr_dict=None):
        super(BatchNormalizationOp, self).__init__(graph, attr_dict)
        self.update_attributes(BatchNormalizationOp, attr_dict)
        assert self.check_required(), 'BatchNormalizationOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'spatial':
                if self.cur_version <= 9:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ret = True
                    self.__dict__['_attr'][item] = Attribute(
                        item, {'type': AttrType.INT, 'value': int(ret)})
            elif item == 'epsilon':
                ret = float(self.__dict__['_attr'][item].value)
            elif item == 'momentum':
                ret = float(self.__dict__['_attr'][item].value)
            elif item == 'training_mode':
                if self.cur_version <= 9:
                    ret = (len(self.get_out_ports()) > 1)
                    self.__dict__['_attr'][item] = Attribute(
                        item, {'type': AttrType.INT, 'value': int(ret)})
                else:
                    ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(BatchNormalizationOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(BatchNormalizationOp, self).infer_shape()
        # X, scale, B, mean, var
        inputs = self.get_input_tensors()
        assert len(inputs) >= 5 and inputs[0].ndim >= 2, \
            'Meets invalid inputs of BatchNormalizationOp(%s) in infer_shape!' % self.name
        input_dtype = inputs[0].dtype
        mean_var_dtype = inputs[3].dtype
        inputs = [inp.astype(np.float32) for inp in inputs]
        is_training = self.training_mode
        if is_training:
            reshape_dim = None
            if inputs[0].ndim not in (4, 5):
                if self.data_format.startswith('NC'):
                    reshape_dim = list(inputs[0].shape[:2]) + [int(np.prod(inputs[0].shape[2:])), 1]
                else:
                    reshape_dim = [1, 1, int(np.prod(inputs[0].shape[:-1])), inputs[0].shape[-1]]
                inp = np.reshape(inputs[0], reshape_dim)
            else:
                inp = inputs[0]
            out_list = tf.compat.v1.nn.fused_batch_norm(
                x=inp,
                scale=inputs[1],
                offset=inputs[2],
                mean=inputs[3],
                variance=inputs[4],
                epsilon=self.epsilon,
                data_format=self.data_format,
                is_training=is_training,
                exponential_avg_factor=1.0
            )
            out_tensor_list = [o.numpy() for o in out_list]
            if reshape_dim is not None:
                out_tensor_list[0] = np.reshape(out_tensor_list[0], inputs[0].shape)
            out_tensor_list[0] = out_tensor_list[0].astype(input_dtype)
            out_tensor_list[1] = out_tensor_list[1].astype(mean_var_dtype)
            out_tensor_list[2] = out_tensor_list[2].astype(mean_var_dtype)
        else:
            if self.data_format[0] == 'N' and self.data_format[-1] == 'C':
                out_tensor = tf.nn.batch_normalization(inputs[0],
                                                       inputs[3],
                                                       inputs[4],
                                                       inputs[2],
                                                       inputs[1],
                                                       variance_epsilon=self.epsilon).numpy()
                out_tensor_list = [out_tensor]
            else:
                gamma = inputs[1]
                beta = inputs[2]
                mean = inputs[3]
                var = inputs[4]
                weights = gamma / np.sqrt(var + self.epsilon)
                biases = beta - gamma * mean / np.sqrt(var + self.epsilon)
                if len(weights.shape) == 1:
                    reshape_dim = [weights.shape[0]] + \
                        [1] * (len(inputs[0].shape) - 2)
                    weights = np.reshape(weights, reshape_dim)
                    biases = np.reshape(biases, reshape_dim)
                out_tensor = inputs[0] * weights + biases
                out_tensor_list = [out_tensor]
            out_tensor_list[0] = out_tensor_list[0].astype(input_dtype)
        self.set_out_tensor(out_tensor_list)


class CeluOp(LayoutUnawareOp, BaseActivationOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {12: {'alpha': {'type': AttrType.FLOAT, 'default': 1.0}}
                }

    def __init__(self, graph, attr_dict=None):
        super(CeluOp, self).__init__(graph, attr_dict)
        self.update_attributes(CeluOp, attr_dict)
        assert self.check_required(), 'CeluOp is missing a required parameter.'
        self.activations = 'CELU'

    def infer_shape(self):
        super(CeluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = self.cal_activation(inputs[0])
        self.set_out_tensor(out_tensor)


class Col2ImOp(OpHasPaddingStrides, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {18: {}}

    def __init__(self, graph, attr_dict=None):
        super(Col2ImOp, self).__init__(graph, attr_dict)
        self.update_attributes(Col2ImOp, attr_dict)
        assert self.check_required(), 'Col2ImOp is missing a required parameter.'

    def infer_shape(self):
        super(Col2ImOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 3, 'Col2ImOp expects 3 inputs, but got %d' % len(inputs)
        # Implementation: https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_col2im.py
        input_shape = list(inputs[0].shape)
        batch = input_shape[0]
        image_shape = inputs[1].tolist()
        block_shape = inputs[2].tolist()
        if not self.kernel_shape:
            self.kernel_shape = block_shape
        channels = input_shape[1] // int(np.prod(block_shape))
        out_shape = [batch, channels] + image_shape
        out_tensor = np.random.ranf(size=out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class ConvOp(BaseConvOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'strides': {'required': True}},
                11: {'strides': {'required': False}}
                }

    @classmethod
    def perm_onnx_to_tf(cls, is_2d=True):
        return [2, 3, 1, 0] if is_2d else [2, 1, 0]

    def __init__(self, graph, attr_dict=None):
        super(ConvOp, self).__init__(graph, attr_dict)
        self.update_attributes(ConvOp, attr_dict)
        assert self.check_required(), 'ConvOp is missing a required parameter.'

    def infer_shape(self):
        super(ConvOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_dtype = inputs[0].dtype
        if self.weights is not None:
            w = self.weights
        elif self.weights is None and len(inputs) >= 2:
            w = inputs[1]
        else:
            w = None
            ERROR('[Parser]: Meets invalid weights for Conv (%s)!' % self.name)

        if w.shape[1] * self.group != (inputs[0].shape[-1] if self.data_format == 'NHWC' else inputs[0].shape[1]):
            ERROR(
                '[Parser]: Meets invalid weights shape or input shape for Conv (%s)!' % self.name)

        if self.num_output is None:
            self.num_output = w.shape[0]

        if self.kernel_shape is None:
            self.kernel_shape = list(w.shape[2:])

        if self.biases is None and self.weights is not None and len(inputs) < 3:
            self.biases = np.zeros(self.num_output, np.float32)

        if self.data_format == 'NHWC':
            # from ..release_ops import ArmDepthwiseConvOp
            # inp = tf.pad(inputs[0], self.tf_pads) if self.auto_pad == 'NOTSET' else inputs[0]
            # if self.group == 1:
            #     conv_func = tf.nn.conv2d if is_2d else tf.nn.conv1d
            #     conv = conv_func(inp,
            #                      np.transpose(self.weights, axes=type(self).perm_onnx_to_tf(is_2d)),
            #                      [1] + self.strides + [1],
            #                      padding='VALID' if self.auto_pad in ('NOTSET', 'VALID') else 'SAME',
            #                      dilations=[1] + self.dilations + [1])
            # else:
            #     if self.weights.shape[0] == self.group:
            #         conv = tf.nn.depthwise_conv2d(inp,
            #                                       np.transpose(self.weights, axes=ArmDepthwiseConvOp.perm_onnx_to_tf()),
            #                                       strides=[1] + self.strides + [1],
            #                                       padding='VALID' if self.auto_pad in ('NOTSET', 'VALID') else 'SAME',
            #                                       data_format=self.data_format,
            #                                       rate=self.dilations)
            #     else:
            #         input_split = tf.split(inp, self.group, axis=3)
            #         weights_split = np.split(self.weights, self.group, axis=0)
            #         meta_conv_list = []
            #         for i in range(self.group):
            #             meta_conv = tf.nn.conv2d(input_split[i],
            #                                      np.transpose(weights_split[i], axes=ConvOp.perm_onnx_to_tf()),
            #                                      self.strides,
            #                                      padding='VALID' if self.auto_pad in ('NOTSET', 'VALID') else 'SAME',
            #                                      data_format='NHWC')
            #             meta_conv_list.append(meta_conv)
            #         conv = tf.concat(meta_conv_list, axis=3)
            # out_tensor = tf.nn.bias_add(conv, self.biases, data_format=self.data_format).numpy()

            out_shape = BaseConvOp.cal_out_shape(inputs[0].shape[1:-1],
                                                 self.pads,
                                                 self.strides,
                                                 self.kernel_shape,
                                                 self.auto_pad,
                                                 dilations=self.dilations,
                                                 data_format='NHWC')
            out_shape = [inputs[0].shape[0]] + out_shape + [self.num_output]
            out_tensor = np.random.ranf(size=out_shape).astype(input_dtype)

        else:
            out_shape = BaseConvOp.cal_out_shape(inputs[0].shape[2:],
                                                 self.pads,
                                                 self.strides,
                                                 self.kernel_shape,
                                                 self.auto_pad,
                                                 dilations=self.dilations,
                                                 data_format='NCHW')
            out_shape = [inputs[0].shape[0], self.num_output] + out_shape
            out_tensor = np.random.ranf(size=out_shape).astype(input_dtype)

        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                inputs[0].shape[1:-
                                1] if self.data_format == 'NHWC' else inputs[0].shape[2:],
                out_tensor.shape[1:-
                                 1] if self.data_format == 'NHWC' else out_tensor.shape[2:],
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
            )
            self.auto_pad = 'NOTSET'
        self.set_out_tensor(out_tensor)


class ConvIntegerOp(BaseConvOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {10: {}}

    def __init__(self, graph, attr_dict=None):
        super(ConvIntegerOp, self).__init__(graph, attr_dict)
        self.update_attributes(ConvIntegerOp, attr_dict)
        assert self.check_required(), 'ConvIntegerOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        if item in ('x_zero_point', 'w_zero_point'):
            try:
                ret = self.__dict__['_attr'][item].value
            except:
                pass
            if ret is None:
                try:
                    inputs = self.get_input_tensors()
                    ret = inputs[2] if item == 'x_zero_point' else inputs[3]
                except:
                    dtype = inputs[0].dtype if item == 'x_zero_point' else self.weights.dtype
                    ret = np.array(0, dtype)
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.INT, 'value': ret})
        if ret is None:
            ret = super(ConvIntegerOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ConvIntegerOp, self).infer_shape()
        inputs = self.get_input_tensors()
        in_shape = inputs[0].shape[1:-
                                   1] if self.data_format == 'NHWC' else inputs[0].shape[2:]
        out_shape = BaseConvOp.cal_out_shape(in_shape,
                                             self.pads,
                                             self.strides,
                                             self.kernel_shape,
                                             self.auto_pad,
                                             dilations=self.dilations,
                                             data_format=self.data_format)
        output_shape = [in_shape[0]] + out_shape + [self.num_output] \
            if self.data_format == 'NHWC' \
            else [in_shape[0], self.num_output] + out_shape
        out_tensor = np.random.ranf(size=output_shape).astype(np.int32)
        self.set_out_tensor(out_tensor)


class ConvTransposeOp(BaseDeconvOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'output_padding': {'type': AttrType.INTS, 'default': [], 'required': False},
                    # If output_shape is specified pads values are ignored
                    'output_shape': {'type': AttrType.INTS, 'default': [], 'required': False}
                    },
                11: {'output_padding': {'type': AttrType.INTS, 'default': [], 'required': False},
                     # If output_shape is specified pads values are ignored
                     'output_shape': {'type': AttrType.INTS, 'default': [], 'required': False}
                     }
                }

    @classmethod
    def perm_onnx_to_tf(cls, is_2d=True):
        return [2, 3, 1, 0] if is_2d else [2, 1, 0]

    def __init__(self, graph, attr_dict=None):
        super(ConvTransposeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ConvTransposeOp, attr_dict)
        assert self.check_required(), 'ConvTransposeOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'output_padding':
                ret = self.__dict__['_attr'][item].value
                kernel_shape = self.__dict__['_attr']['kernel_shape'].value
                if not ret and kernel_shape is not None:
                    ret = [0] * len(kernel_shape)
                    self.__dict__['_attr'][item].value = ret
                elif ret and kernel_shape is not None and len(ret) < len(kernel_shape):
                    ret = [0] * (len(kernel_shape) - len(ret)) + ret
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(ConvTransposeOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ConvTransposeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None:
            self.kernel_shape = list(self.weights.shape[2:])
        if len(inputs[0].shape) == 3 and len(self.output_padding) == 2:
            self.output_padding = self.output_padding[1:]
        self.update_pads(inputs[0].shape[1:-1] if self.data_format == 'NHWC' else inputs[0].shape[2:])
        if self.biases is None:
            self.biases = np.zeros(self.num_output, np.float32)
        if self.data_format == 'NHWC':
            out_shape = [inputs[0].shape[0]] + \
                self.output_shape + [self.num_output]
        else:
            # The number of channels in the output should be equal to W.shape[1] * group
            out_shape = [inputs[0].shape[0],
                         self.num_output] + self.output_shape
        out_tensor = np.random.ranf(size=out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class DeformConvOp(BaseConvOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {19: {'offset_group': {'type': AttrType.INT, 'default': 1}},
                }

    def __init__(self, graph, attr_dict=None):
        super(DeformConvOp, self).__init__(graph, attr_dict)
        self.update_attributes(DeformConvOp, attr_dict)
        assert self.check_required(), 'DeformConvOp is missing a required parameter.'

    @staticmethod
    def gen_offset_base(weights_shape, offset_shape, kernel_shape, strides, dilations, pads, offset_group):
        '''Calculate coordinates of sampling points within kernel.
        '''
        n = offset_shape[0]
        oc = weights_shape[0]
        oh, ow = offset_shape[-2:]
        kh, kw = kernel_shape
        sth, stw = strides
        dh, dw = dilations
        kh_new, kw_new = (kh - 1) * dh + 1, (kw - 1) * dw + 1
        bh, bw = -pads[0], -pads[1]

        kernel_pos_w, kernel_pos_h = np.meshgrid(np.arange(0, kw_new, dw), np.arange(0, kh_new, dh))
        kernel_pos = np.stack([kernel_pos_h, kernel_pos_w], axis=2)  # shape: [kh, kw, 2]

        kernel_offset = np.zeros([oh, ow, kh, kw, 2], dtype=np.int64)
        for i in range(oh):
            h_coord = bh + sth * i
            for j in range(ow):
                w_coord = bw + stw * j
                kernel_offset[i, j] = kernel_pos[:, :] + [h_coord, w_coord]

        # reshape from [oh, ow, kh, kw, 2] to [oh, ow, kh*kw, 2]
        kernel_offset = np.reshape(kernel_offset, [oh, ow, kh * kw, 2])
        kernel_offset = np.transpose(kernel_offset, [2, 3, 0, 1])  # shape: [kh*kw, 2, oh, ow]
        kernel_offset = np.tile(kernel_offset, [int(n * offset_group), 1, 1, 1]
                                )  # shape: [n*offset_group*kh*kw, 2, oh, ow]

        return kernel_offset

    def infer_shape(self):
        super(DeformConvOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 3, 'Expects at least 3 inputs for DeformConvOp but got %d' % len(inputs)
        input_dim = len(inputs[0].shape)
        in_c = inputs[0].shape[-1] if self.data_format == 'NHWC' else inputs[0].shape[1]
        self.weights = inputs[1]
        if self.weights.shape[1] * self.group != in_c:
            ERROR(
                '[Parser]: Meets invalid weights shape or input shape for DeformConvOp (%s)!' % self.name)
        self.num_output = self.weights.shape[0]
        self.biases = inputs[3] if (len(inputs) > 3 and inputs[3]
                                    is not None and inputs[3].size > 0) else np.zeros(self.num_output, np.float32)
        if self.kernel_shape is None:
            self.kernel_shape = list(self.weights.shape[2:])
        batch = inputs[0].shape[0]
        if self.data_format == 'NHWC':
            out_shape = [batch] + list(inputs[2].shape[1:-1]) + [self.num_output]
        else:
            out_shape = [batch, self.num_output] + list(inputs[2].shape[2:])
        out_tensor = np.random.ranf(size=out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class DropoutOp(OpHasVariableOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS},
                    'is_test': {'type': AttrType.BOOL, 'default': False},
                    'ratio': {'type': AttrType.FLOAT, 'default': 0.5}
                    },
                6: {'is_test': {'type': AttrType.BOOL, 'default': False},
                    'ratio': {'type': AttrType.FLOAT, 'default': 0.5}
                    },
                7: {'ratio': {'type': AttrType.FLOAT, 'default': 0.5}
                    },
                10: {'ratio': {'type': AttrType.FLOAT, 'default': 0.5}
                     },
                12: {'seed': {'type': AttrType.INT, 'default': 0, 'required': False}
                     },
                13: {'seed': {'type': AttrType.INT, 'default': 0, 'required': False}
                     }
                }

    def __init__(self, graph, attr_dict=None):
        super(DropoutOp, self).__init__(graph, attr_dict)
        self.update_attributes(DropoutOp, attr_dict)
        assert self.check_required(), 'DropoutOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'is_test':
                if self.cur_version <= 6:
                    ret = self.__dict__['_attr'][item].value
                else:
                    ret = None
                    self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.BOOL, 'value': ret})
            elif item == 'training_mode':
                if self.cur_version <= 6:
                    ret = not self.__dict__['_attr']['is_test'].value
                elif 6 < self.cur_version < 12:
                    ret = False
                else:
                    input_tensors = self.get_input_tensors()
                    if len(input_tensors) >= 3 and input_tensors[2] is not None:
                        training_mode_tensor = np.array(input_tensors[2])
                        if training_mode_tensor.size == 1:
                            ret = bool(training_mode_tensor)
                        else:
                            ret = None
                    else:
                        ret = False
                self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.BOOL, 'value': ret})
            elif item == 'ratio':
                if self.cur_version < 12:
                    ret = self.__dict__['_attr'][item].value
                else:
                    input_tensors = self.get_input_tensors()
                    if len(input_tensors) >= 2 and input_tensors[1] is not None:
                        ratio_tensor = np.array(input_tensors[1])
                        if ratio_tensor.size == 1 and (0.0 <= ratio_tensor < 1.0):
                            ret = ratio_tensor
                        else:
                            ret = None
                    else:
                        ret = np.array(0.5, np.float32)
                    self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.FLOAT, 'value': ret})
            elif item == 'seed':
                if self.cur_version < 12:
                    ret = 0
                    self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.INT, 'value': ret})
                else:
                    ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        if ret is None:
            ret = super(DropoutOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(DropoutOp, self).infer_shape()
        if self.training_mode and not FLOAT_EQUAL(self.ratio, 0.0):
            WARN('[Parser]: Dropout (%s) does not support training mode!' % self.name)
        inputs = self.get_input_tensors()
        in_shape = inputs[0].shape
        if FLOAT_EQUAL(self.ratio, 0.0) or (self.training_mode in (None, False)):
            output = inputs[0]
            if len(self.get_out_ports()) == 2:
                mask = np.tile(True, in_shape)
                if self.cur_version < 10:
                    mask = mask.astype(inputs[0].dtype)
                self.set_out_tensor([output, mask])
            else:
                self.set_out_tensor([output])
        else:
            rng = np.random.default_rng(self.seed)
            uniform = rng.uniform(0.0, 1.0, in_shape)
            mask = uniform >= self.ratio
            output = mask.astype(np.array(self.ratio).dtype) * inputs[0] / (1.0 - self.ratio)
            if self.cur_version < 10:
                mask = mask.astype(inputs[0].dtype)
            self.set_out_tensor([output, mask])

    def convert_version(self):
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if len(self.get_input_shapes()) < 1 \
                or any([s is None for s in self.get_input_shapes()[0]]):
            ERROR(
                '[Parser]: Meets invalid Dropout Node (%s) in convert_version!' % self.name)
            return

        in_edges = self._graph.sorted_in_edges(self.name, data=True)
        assert len(in_edges) >= 1
        if len(in_edges) < 2:
            from ...front_end.onnx.passes.common_passes import insert_constant
            insert_constant(self._graph, self.name + '_ratio', np.array(self.ratio), self.name, in_port=1)
            insert_constant(self._graph, self.name + '_training_mode',
                            np.array(self.training_mode), self.name, in_port=2)
        elif len(in_edges) < 3:
            from ...front_end.onnx.passes.common_passes import insert_constant
            insert_constant(self._graph, self.name + '_training_mode',
                            np.array(self.training_mode), self.name, in_port=2)
        if cur_ver < 10 and len(self.get_out_ports()) == 2:
            from ...front_end.onnx.passes.common_passes import insert_cast_after
            post_cast = insert_cast_after(self._graph, self.name, 'bool',
                                          in_edges[0][2]['tensor'].get_dtype(), out_port=1)
        if cur_ver < max_ver:
            self.cur_version = max_ver


class GeluOp(BaseActivationOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {20: {'approximate': {'type': AttrType.STRING, 'default': 'none', 'options': ['none', 'tanh']}},
                }

    def __init__(self, graph, attr_dict=None):
        super(GeluOp, self).__init__(graph, attr_dict)
        self.update_attributes(GeluOp, attr_dict)
        assert self.check_required(), 'GeluOp is missing a required parameter.'
        self.activations = 'GELU'

    def infer_shape(self):
        super(GeluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_tensor = torch.tensor(inputs[0], dtype=torch.float32)
        out_tensor = torch.nn.functional.gelu(input_tensor,
                                              approximate=self.approximate).numpy().astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class GlobalAveragePoolOp(LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(GlobalAveragePoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(GlobalAveragePoolOp, attr_dict)
        assert self.check_required(), 'GlobalAveragePoolOp is missing a required parameter.'

    def infer_shape(self):
        super(GlobalAveragePoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.data_format == 'NHWC':
            axes = list(range(1, len(inputs[0].shape) - 1))
        else:
            axes = list(range(2, len(inputs[0].shape)))
        out_tensor = np.mean(inputs[0], axis=tuple(axes), keepdims=True)
        self.set_out_tensor(out_tensor)


class GlobalLpPoolOp(LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'p': {'type': AttrType.FLOAT, 'default': '2.0'}
                    },
                2: {'p': {'type': AttrType.INT, 'default': '2'}
                    }}

    def __init__(self, graph, attr_dict=None):
        super(GlobalLpPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(GlobalLpPoolOp, attr_dict)
        assert self.check_required(), 'GlobalLpPoolOp is missing a required parameter.'

    def infer_shape(self):
        super(GlobalLpPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_shape = inputs[0].shape
        if self.data_format == 'NHWC':
            out_shape = [input_shape[0]] + \
                np.tile([1], len(input_shape) - 2).tolist() + [input_shape[-1]]
        else:
            out_shape = list(input_shape[:2]) + \
                np.tile([1], len(input_shape) - 2).tolist()
        out_tensor = np.random.ranf(out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class GlobalMaxPoolOp(LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(GlobalMaxPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(GlobalMaxPoolOp, attr_dict)
        assert self.check_required(), 'GlobalMaxPoolOp is missing a required parameter.'

    def infer_shape(self):
        super(GlobalMaxPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.data_format == 'NHWC':
            axes = list(range(1, len(inputs[0].shape) - 1))
        else:
            axes = list(range(2, len(inputs[0].shape)))
        out_tensor = np.max(inputs[0], axis=tuple(axes), keepdims=True)
        self.set_out_tensor(out_tensor)


class GridSampleOp(LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {16: {'align_corners': {'type': AttrType.BOOL, 'default': False},
                     'mode': {'type': AttrType.STRING, 'default': 'bilinear', 'options': ['bilinear', 'nearest', 'bicubic']},
                     'padding_mode': {'type': AttrType.STRING, 'default': 'zeros', 'options': ['zeros', 'border', 'reflection']}
                     },
                20: {'align_corners': {'type': AttrType.BOOL, 'default': False},
                     'mode': {'type': AttrType.STRING, 'default': 'linear', 'options': ['linear', 'nearest', 'cubic']},
                     'padding_mode': {'type': AttrType.STRING, 'default': 'zeros', 'options': ['zeros', 'border', 'reflection']}
                     },
                }

    def __init__(self, graph, attr_dict=None):
        super(GridSampleOp, self).__init__(graph, attr_dict)
        self.update_attributes(GridSampleOp, attr_dict)
        assert self.check_required(), 'GridSampleOp is missing a required parameter.'

    @staticmethod
    def denormalize(n, length, align_corners):
        if align_corners:
            x = (n + 1) / 2.0 * (length - 1)
        else:
            x = ((n + 1) * length - 1) / 2.0
        return x

    @staticmethod
    def denormalize_coordinates(n, dims, align_corners):
        assert len(n) == len(dims)
        return GridSampleOp.denormalize(np.array(n), np.array(dims), align_corners)

    @staticmethod
    def get_all_coords(data):
        x = [range(shape) for shape in data.shape]
        return list(product(*x))

    @staticmethod
    def reflect(x, x_min, x_max):
        rng = x_max - x_min
        if x >= x_min and x <= x_max:
            return x
        dx = (x_min - x) if x < x_min else (x - x_max)
        n = int(dx / rng)
        r = dx - n * rng
        if n % 2 == 0:
            x = (x_min + r) if x < x_min else (x_max - r)
        else:
            x = (x_max - r) if x < x_min else (x_min + r)
        return x

    @staticmethod
    def pixel_at_array(i, array, border, padding_mode):
        assert array.ndim == 1
        d = array.shape[0]
        if padding_mode == 'zeros':
            pixel = array[i] if (i >= 0 and i < d) else 0
        elif padding_mode == 'border':
            i = np.clip(i, 0, d - 1)
            pixel = array[i]
        else:  # reflection
            i = int(GridSampleOp.reflect(i, border[0], border[1]))
            pixel = array[i]
        return pixel

    @staticmethod
    def pixel_at_ndarray(x, ndarray, border, padding_mode):
        num_dims = ndarray.ndim
        if num_dims == 1:
            return GridSampleOp.pixel_at_array(x[0], ndarray, border, padding_mode)
        i = x[0]
        d = ndarray.shape[0]
        if padding_mode == 'zeros':
            ndarray = ndarray[i] if (i >= 0 and i < d) else np.zeros_like(ndarray[0])
        elif padding_mode == 'border':
            i = np.clip(i, 0, d - 1)
            ndarray = ndarray[i]
        else:  # reflection
            i = int(GridSampleOp.reflect(i, border[0], border[num_dims]))
            ndarray = ndarray[i]
        border = list(border[1:num_dims]) + list(border[1 + num_dims: 2 * num_dims])
        return GridSampleOp.pixel_at_ndarray(x[1:], ndarray, border, padding_mode)

    @staticmethod
    def prepare_border(dims, align_corners):
        num_dims = len(dims)
        borders = np.zeros(num_dims * 2)
        if align_corners:
            # borders[:num_dims] = 0.0
            borders[num_dims:] = np.array(dims) - 1.0
        else:
            borders[:num_dims] = -0.5
            borders[num_dims:] = np.array(dims) - 0.5
        return borders

    @staticmethod
    def get_linear_coeffs(x):
        x = abs(x)
        return [1 - x, x]

    @staticmethod
    def get_cubic_interpolate_1d_with_x(data, x, border, padding_mode):
        from .math_ops import ResizeOp
        x_0 = int(np.floor(x))
        x_1 = x_0 + 1
        x_2 = x_0 + 2
        x_minus_1 = x_0 - 1
        coeffs = ResizeOp.get_cubic_coeffs(x - x_0)
        v = [GridSampleOp.pixel_at_array(i, data, border, padding_mode) for i in [x_minus_1, x_0, x_1, x_2]]
        return coeffs @ np.array(v)

    @staticmethod
    def linear_interpolation_1d_with_x(data, x, border, padding_mode):
        x_0 = int(np.floor(x))
        x_1 = x_0 + 1
        coeffs = GridSampleOp.get_linear_coeffs(x - x_0)
        v = [GridSampleOp.pixel_at_array(i, data, border, padding_mode) for i in [x_0, x_1]]
        return coeffs @ np.array(v)

    @staticmethod
    def linear_interpolation_nd_with_x(x, data, border, padding_mode):
        num_dims = data.ndim
        if num_dims == 1:
            return GridSampleOp.linear_interpolation_1d_with_x(data, x[0], border, padding_mode)
        border_i = list(border[1: num_dims]) + list(border[1 + num_dims: 2 * num_dims])
        data_shape_0 = data.shape[0]
        res = [GridSampleOp.linear_interpolation_nd_with_x(
            x[1:], data[i], border_i, padding_mode) for i in range(data_shape_0)]
        res = np.array(res)
        return GridSampleOp.linear_interpolation_1d_with_x(res, x[0], [border[0], border[num_dims]], padding_mode)

    @staticmethod
    def cubic_interpolation_nd_with_x(x, data, border, padding_mode):
        num_dims = data.ndim
        if num_dims == 1:
            return GridSampleOp.get_cubic_interpolate_1d_with_x(data, x[0], border, padding_mode)
        border_i = list(border[1: num_dims]) + list(border[1 + num_dims: 2 * num_dims])
        data_shape_0 = data.shape[0]
        res = [GridSampleOp.cubic_interpolation_nd_with_x(
            x[1:], data[i], border_i, padding_mode) for i in range(data_shape_0)]
        res = np.array(res)
        return GridSampleOp.get_cubic_interpolate_1d_with_x(res, x[0], [border[0], border[num_dims]], padding_mode)

    @staticmethod
    def apply_padding(x, border, padding_mode, dims):
        num_dims = len(dims)
        for i, v in enumerate(x):
            x_min, x_max = border[i], border[i + num_dims]
            if v < x_min or v > x_max:
                x[i] = np.clip(v, 0, dims[i] - 1) if padding_mode == 'border' else GridSampleOp.reflect(v, x_min, x_max)
        return x

    @staticmethod
    def grid_sample(X, grid, mode, padding_mode, align_corners):
        # X must be in NCHW data format
        x_dims = X.shape
        N, C = x_dims[:2]
        grid_spatial_shape = list(grid.shape[1:-1])
        y_dims = [N, C] + grid_spatial_shape
        Y = np.empty(y_dims, dtype=X.dtype)

        for n in range(N):
            grid_data = grid[n]
            for c in range(C):
                X_data = X[n, c]
                dims = x_dims[2:]
                num_dims = len(dims)
                border = GridSampleOp.prepare_border(dims, align_corners)
                all_coords = GridSampleOp.get_all_coords(Y[n, c])
                x_list = []
                for ox in all_coords:
                    nx = grid_data[tuple(ox)][::-1]
                    x = GridSampleOp.denormalize_coordinates(nx, dims, align_corners)
                    x_list.append(x)
                if mode == 'nearest':
                    x_list = [np.rint(x) for x in x_list]
                x_list = map(partial(GridSampleOp.apply_padding, border=border,
                             padding_mode=padding_mode, dims=dims), x_list)
                if mode == 'nearest':
                    x_list = [x.astype(np.int32) for x in list(x_list)]
                    output = list(map(partial(GridSampleOp.pixel_at_ndarray, ndarray=X_data,
                                  border=border, padding_mode=padding_mode), x_list))
                elif mode in ('linear', 'bilinear'):
                    output = list(map(partial(GridSampleOp.linear_interpolation_nd_with_x,
                                  data=X_data, border=border, padding_mode=padding_mode), x_list))
                else:  # cubic, bicubic
                    output = list(map(partial(GridSampleOp.cubic_interpolation_nd_with_x,
                                  data=X_data, border=border, padding_mode=padding_mode), x_list))
                Y[n, c] = np.reshape(output, grid_spatial_shape)
        return Y

    def infer_shape(self):
        super(GridSampleOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_rank = len(inputs[0].shape)
        if self.cur_version == 16 and input_rank != 4:
            ERROR('[Parser]: GridSampleOp (%s) only supports spatial (4-D) inputs in Op version [%s]!' %
                  (self.name, self.cur_version))
        spatial_output_shape = list(inputs[1].shape[1:-1])
        inp = inputs[0].astype(np.float32) if inputs[0].dtype != np.float32 else inputs[0]
        perm = None
        if self.data_format == 'NHWC':
            perm = [0, input_rank - 1] + list(range(1, input_rank - 1))
        input_tensor = np.transpose(inp, perm) if perm is not None else inp
        is_cubic = self.mode.endswith('cubic')
        if (is_cubic and input_rank == 4) or (not is_cubic and input_rank in (4, 5)):
            # torch.nn.functional.grid_sample only supports 4-D and 5-D inputs(not bicubic)
            modes_map = {'linear': 'bilinear', 'cubic': 'bicubic'}
            mode = modes_map.get(self.mode, self.mode)
            out_tensor = torch.nn.functional.grid_sample(torch.from_numpy(input_tensor),
                                                         torch.from_numpy(
                inputs[1].astype(np.float32)),
                mode=mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners).numpy()
        else:
            if self.is_all_inputs_const():
                # from onnx.reference.ops._op_list import GridSample
                # modes_map = {'bilinear': 'linear', 'bicubic': 'cubic'}
                # mode = modes_map.get(self.mode, self.mode)
                # out_tensor = GridSample.eval(inputs[0].astype(np.float32), inputs[1].astype(np.float32),
                #                              mode=mode, padding_mode=self.padding_mode,
                #                              align_corners=self.align_corners)
                out_tensor = self.grid_sample(input_tensor, inputs[1].astype(np.float32),
                                              self.mode, self.padding_mode, self.align_corners)
            else:
                out_tensor = np.random.ranf(list(input_tensor.shape[:2]) + spatial_output_shape)
        out_tensor = np.transpose(
            out_tensor, Op.cal_inverse_perm(perm)) if perm is not None else out_tensor
        out_tensor = out_tensor.astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            if self.mode == 'bilinear':
                self.mode = 'linear'
            elif self.mode == 'bicubic':
                self.mode = 'cubic'


class GroupNormalizationOp(LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {18: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-5},
                     'num_groups': {'type': AttrType.INTS, 'required': True},
                     },
                }

    def __init__(self, graph, attr_dict=None):
        super(GroupNormalizationOp, self).__init__(graph, attr_dict)
        self.update_attributes(GroupNormalizationOp, attr_dict)
        assert self.check_required(), 'GroupNormalizationOp is missing a required parameter.'

    def infer_shape(self):
        super(GroupNormalizationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 3, 'GroupNormalizationOp expects 3 inputs, but got %d' % len(inputs)
        input_dim = len(inputs[0].shape)
        pre_perm = None
        if self.data_format.startswith('NC'):
            input_tensor = inputs[0]
            channel_axis = 1
        else:
            pre_perm = [0, input_dim - 1] + list(range(1, input_dim - 1))
            input_tensor = np.transpose(inputs[0], pre_perm)
            channel_axis = -1
        channels = inputs[0].shape[channel_axis]
        channel_num_per_group = channels / self.num_groups
        scale = np.repeat(inputs[1], channel_num_per_group, axis=0)
        bias = np.repeat(inputs[2], channel_num_per_group, axis=0)
        weight_bias_shape = [-1] + [1] * (input_dim - 2)
        weights = np.reshape(scale, weight_bias_shape)
        biases = np.reshape(bias, weight_bias_shape)

        groupnorm = torch.nn.GroupNorm(self.num_groups, channels, self.epsilon)
        out_tensor = groupnorm(torch.from_numpy(input_tensor)).detach().numpy() * weights + biases
        if pre_perm is not None:
            out_tensor = np.transpose(out_tensor, Op.cal_inverse_perm(pre_perm))
        self.set_out_tensor(out_tensor)


class GRUOp(BaseRnnOp, OpHasBiases, OpHasWeights, OnnxOp):
    @staticmethod
    def extract_weights(W, B, R, hidden_size, num_directions, linear_before_reset=False):
        ret = []
        for n in range(num_directions):
            update_w, reset_w, hidden_w = np.split(
                W[n, :, :], W.shape[1] // hidden_size, axis=0)
            update_r, reset_r, hidden_r = np.split(
                R[n, :, :], R.shape[1] // hidden_size, axis=0)
            update_wb, reset_wb, hidden_wb = np.split(
                B[n, 0:hidden_size * 3], 3, axis=0)
            update_rb, reset_rb, hidden_rb = np.split(
                B[n, hidden_size * 3:], 3, axis=0)
            if not linear_before_reset:
                update = np.concatenate((update_w, update_r), axis=1)
                reset = np.concatenate((reset_w, reset_r), axis=1)
                gate_weights = np.concatenate((reset, update), axis=0)
                candidate_weights = np.concatenate(
                    (hidden_w, hidden_r), axis=1)
                gate_biases = np.concatenate(
                    (reset_wb + reset_rb, update_wb + update_rb))
                candidate_biases = hidden_wb + hidden_rb
                ret.append({'gate_weights': gate_weights,
                            'gate_biases': gate_biases,
                            'candidate_weights': candidate_weights,
                            'candidate_biases': candidate_biases
                            })
            else:
                parameter_weights_rzh = np.concatenate(
                    [reset_w, update_w, hidden_w], axis=0)
                recurrence_weights_rzh = np.concatenate(
                    [reset_r, update_r, hidden_r], axis=0)
                ret.append({'parameter_weights_rzh': parameter_weights_rzh,
                            'recurrence_weights_rzh': recurrence_weights_rzh,
                            'biases_r': reset_wb + reset_rb,
                            'biases_z': update_wb + update_rb,
                            'parameter_biases_h': hidden_wb,
                            'recurrence_biases_h': hidden_rb
                            })
        return ret

    @staticmethod
    def gru_cell(x, initial_state, weights_dict, linear_before_reset=False):
        if not linear_before_reset:
            gates_kernel, gates_bias, candidate_kernel, candidate_bias \
                = weights_dict['gate_weights'], weights_dict['gate_biases'], weights_dict['candidate_weights'], weights_dict['candidate_biases']
            combined_input = tf.concat([x, initial_state], axis=0)
            combined_input = tf.expand_dims(combined_input, axis=0)
            fc1 = tf.sigmoid(
                tf.matmul(combined_input, gates_kernel, transpose_b=True) + gates_bias)
            fc1_reset, fc1_update = tf.split(fc1, 2, axis=1)
            combined_input2 = tf.concat(
                [tf.expand_dims(x, axis=0), fc1_reset * initial_state], axis=1)
            fc2 = tf.tanh(tf.matmul(combined_input2,
                                    candidate_kernel, transpose_b=True) + candidate_bias)
            ret = fc1_update * initial_state + (1 - fc1_update) * fc2
            ret = tf.squeeze(ret, axis=0)
        else:
            parameter_weights_rzh = weights_dict['parameter_weights_rzh']
            recurrence_weights_rzh = weights_dict['recurrence_weights_rzh']
            biases_r = weights_dict['biases_r']
            biases_z = weights_dict['biases_z']
            parameter_biases_h = weights_dict['parameter_biases_h']
            recurrence_biases_h = weights_dict['recurrence_biases_h']

            x = tf.expand_dims(x, 0)
            initial_state = tf.expand_dims(initial_state, 0)
            Wr, Wz, Wh = tf.split(parameter_weights_rzh, 3, axis=0)
            Rr, Rz, Rh = tf.split(recurrence_weights_rzh, 3, axis=0)
            zt = tf.sigmoid(x @ tf.transpose(Wz) +
                            initial_state @ tf.transpose(Rz) + biases_z)
            rt = tf.sigmoid(x @ tf.transpose(Wr) +
                            initial_state @ tf.transpose(Rr) + biases_r)
            ht = tf.tanh(x @ tf.transpose(Wh) + rt * (initial_state @
                                                      tf.transpose(Rh) + recurrence_biases_h) + parameter_biases_h)
            Ht = (1 - zt) * ht + zt * initial_state
            ret = tf.squeeze(Ht, 0)
        return ret

    @classmethod
    def attributes(cls):
        return {1: {'activation_alpha': {'type': AttrType.FLOATS, 'required': False},
                    'activation_beta': {'type': AttrType.FLOATS, 'required': False},
                    'activations': {'default': ['SIGMOID', 'TANH']},
                    'clip': {'type': AttrType.FLOAT, 'required': False},
                    'direction': {'type': AttrType.STRING, 'default': 'forward', 'options': ['forward', 'reverse', 'bidirectional']},
                    'hidden_size': {'type': AttrType.INT, 'required': True},
                    'output_sequence': {'type': AttrType.INT, 'default': 0, 'required': False},
                    'method': {'default': 'Y', 'options': ['Y', 'H', 'YH']}
                    },
                3: {'activation_alpha': {'type': AttrType.FLOATS, 'required': False},
                    'activation_beta': {'type': AttrType.FLOATS, 'required': False},
                    'activations': {'default': ['SIGMOID', 'TANH']},
                    'clip': {'type': AttrType.FLOAT, 'required': False},
                    'direction': {'type': AttrType.STRING, 'default': 'forward', 'options': ['forward', 'reverse', 'bidirectional']},
                    'hidden_size': {'type': AttrType.INT, 'required': True},
                    'linear_before_reset': {'type': AttrType.INT, 'default': 0, 'required': False},
                    'output_sequence': {'type': AttrType.INT, 'default': 0, 'required': False},
                    'method': {'default': 'Y', 'options': ['Y', 'H', 'YH']}
                    },
                7: {'activation_alpha': {'type': AttrType.FLOATS, 'required': False},
                    'activation_beta': {'type': AttrType.FLOATS, 'required': False},
                    'activations': {'default': ['SIGMOID', 'TANH']},
                    'clip': {'type': AttrType.FLOAT, 'required': False},
                    'direction': {'type': AttrType.STRING, 'default': 'forward', 'options': ['forward', 'reverse', 'bidirectional']},
                    'hidden_size': {'type': AttrType.INT, 'required': True},
                    'linear_before_reset': {'type': AttrType.INT, 'default': 0, 'required': False},
                    'method': {'default': 'Y', 'options': ['Y', 'H', 'YH']}
                    },
                14: {'activation_alpha': {'type': AttrType.FLOATS, 'required': False},
                     'activation_beta': {'type': AttrType.FLOATS, 'required': False},
                     'activations': {'default': ['SIGMOID', 'TANH']},
                     'clip': {'type': AttrType.FLOAT, 'required': False},
                     'direction': {'type': AttrType.STRING, 'default': 'forward', 'options': ['forward', 'reverse', 'bidirectional']},
                     'hidden_size': {'type': AttrType.INT, 'required': True},
                     'layout': {'type': AttrType.INT, 'default': 0},
                     'linear_before_reset': {'type': AttrType.INT, 'default': 0, 'required': False},
                     'method': {'default': 'Y', 'options': ['Y', 'H', 'YH']}
                     }
                }

    def __init__(self, graph, attr_dict=None):
        super(GRUOp, self).__init__(graph, attr_dict)
        self.update_attributes(GRUOp, attr_dict)
        assert self.check_required(), 'GRUOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'layout':
                if cur_ver >= 14:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ret = False
            elif item == 'linear_before_reset':
                if cur_ver >= 3:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ret = False
            elif item == 'output_sequence':
                if cur_ver <= 3:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ret = False
            elif item == 'activations':
                ret = self.__dict__['_attr'][item].value
                if self.direction == 'bidirectional' and len(ret) == 2:
                    ret.extend(['SIGMOID', 'TANH'])
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(GRUOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item in ('layout', 'linear_before_reset', 'output_sequence'):
            try:
                self.__dict__['_attr'][item].value = int(value)
            except:
                super(GRUOp, self).__setattr__(item, value)
        else:
            super(GRUOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(GRUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) in (
            3, 4, 5, 6), 'The length of inputs is invalid in GRUOp infer shape.'
        num_directions = 2 if self.direction == 'bidirectional' else 1
        if not self.layout:
            self.time_steps, batch_size, self.input_size = inputs[0].shape
            init_shape = (num_directions, batch_size, self.hidden_size)
        else:
            batch_size, self.time_steps, self.input_size = inputs[0].shape
            init_shape = (batch_size, num_directions, self.hidden_size)
        W, R = inputs[1:3]
        B = inputs[3] if (len(inputs) > 3 and inputs[3] is not None) else np.zeros(
            (num_directions, 6 * self.hidden_size)).astype(inputs[0].dtype)
        if len(inputs) == 6 and inputs[-1] is not None:
            initial_h = inputs[-1]
        else:
            initial_h = np.zeros(init_shape).astype(inputs[0].dtype)

        weights_list = GRUOp.extract_weights(
            W, B, R, self.hidden_size, num_directions, self.linear_before_reset)
        Y = []
        Y_h = []
        for b in range(batch_size):
            cur_input = inputs[0][:, b,
                                  :] if not self.layout else inputs[0][b, :, :]
            batch_out = []
            hidden_batch_out = []
            for n in range(num_directions):
                direction_out = []
                init_state = initial_h[n, b,
                                       :] if not self.layout else initial_h[b, n, :]
                if n == 1:
                    cur_input = tf.reverse(
                        cur_input, axis=np.array([0], np.int64))
                gru_cell = None
                for s in range(self.time_steps):
                    x = cur_input[s, :]
                    '''
                    gru_cell = GRUOp.gru_cell(x,
                                              init_state,
                                              weights_list[n],
                                              self.linear_before_reset
                                              )
                    '''
                    gru_cell = np.random.ranf(
                        size=(self.hidden_size,)).astype(np.float32)
                    direction_out.append(gru_cell)
                direction_out = tf.stack(direction_out, axis=0)
                hidden_batch_out.append(gru_cell)
                if n == 1:
                    direction_out = tf.reverse(
                        direction_out, axis=np.array([0], np.int64))
                batch_out.append(direction_out)
            batch_out = tf.stack(batch_out, axis=0)
            Y.append(batch_out)
            hidden_batch_out = tf.stack(hidden_batch_out, axis=0)
            Y_h.append(hidden_batch_out)

        ''' [batch_size, num_directions, seq_length, hidden_size]'''
        Y = tf.stack(Y, axis=0)
        ''' [batch_size, num_directions, hidden_size]'''
        Y_h = tf.stack(Y_h, axis=0)
        if self.layout:
            ''' [batch_size, seq_length, num_directions, hidden_size]'''
            Y = tf.transpose(Y, perm=[0, 2, 1, 3]).numpy()
        else:
            ''' [seq_length, num_directions, batch_size, hidden_size]'''
            Y = tf.transpose(Y, perm=[2, 1, 0, 3]).numpy()
            ''' [num_directions, batch_size, hidden_size]'''
            Y_h = tf.transpose(Y_h, perm=[1, 0, 2]).numpy()

        if not self.linear_before_reset:
            self.weights = np.stack([np.concatenate(
                (wl['gate_weights'], wl['candidate_weights'])) for wl in weights_list])
            self.biases = np.stack([np.concatenate(
                (wl['gate_biases'], wl['candidate_biases'])) for wl in weights_list])
        else:
            self.weights = np.stack([np.concatenate(
                (wl['parameter_weights_rzh'], wl['recurrence_weights_rzh']), axis=1) for wl in weights_list])
            self.biases = np.stack([np.concatenate((wl['biases_r'],
                                                    wl['biases_z'],
                                                    wl['parameter_biases_h'],
                                                    wl['recurrence_biases_h']), axis=0) for wl in weights_list])

        out_ports = self.get_out_ports()
        if sorted(out_ports) == [0, 1]:
            self.method = 'YH'
            out_tensor_list = [Y, Y_h]
        elif 0 in out_ports:
            self.method = 'Y'
            out_tensor_list = [Y]
        elif 1 in out_ports:
            self.method = 'H'
            out_tensor_list = [Y_h]
        else:
            ERROR('[Parser]: GRU Node (%s) has invalid output!' % self.name)
            out_tensor_list = []
        self.set_out_tensor(out_tensor_list)

    def convert_version(self):
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            out_ports = self.get_out_ports()
            if cur_ver <= 3:
                if not self.output_sequence and 0 in out_ports:
                    for _, dst, k, out_attr in self._graph.sorted_out_edges(self.name, keys=True, data=True):
                        if out_attr['src_out_port'] == 0:
                            self._graph.remove_edge(self.name, dst, key=k)
            if cur_ver <= 7:
                layout = False
                self._attr['layout'] = Attribute(
                    'layout', {'type': AttrType.INT, 'value': int(layout)})
            if cur_ver == 1:
                linear_before_reset = False
                self._attr['linear_before_reset'] = Attribute(
                    'linear_before_reset', {'type': AttrType.INT, 'value': int(linear_before_reset)})
            self.cur_version = max_ver

        from ...front_end.onnx.passes.common_passes import insert_constant
        in_edges = self._graph.sorted_in_edges(self.name, data=True)
        num_directions = 2 if self.direction == 'bidirectional' else 1
        batch_size = in_edges[0][2]['tensor'].shape[0] if self.layout else in_edges[0][2]['tensor'].shape[1]
        initial_state_shape = [batch_size, num_directions, self.hidden_size] \
            if self.layout \
            else [num_directions, batch_size, self.hidden_size]

        if len(in_edges) <= 5:
            initial_h = np.zeros(initial_state_shape, np.float32)
            insert_constant(self._graph, self.name + '_initial_h',
                            initial_h, self.name, in_port=5)
        if len(in_edges) <= 4:
            sequence_lens = np.array([self.time_steps] * batch_size, np.int32)
            insert_constant(self._graph, self.name + '_sequence_lens',
                            sequence_lens, self.name, in_port=4)
        if len(in_edges) <= 3:
            B = np.zeros((num_directions, 6 * self.hidden_size), np.float32)
            insert_constant(self._graph, self.name +
                            '_B', B, self.name, in_port=3)


class InstanceNormalizationOp(OpHasBiases, OpHasWeights, LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS},
                    'epsilon': {'type': AttrType.FLOAT, 'default': 1e-5},
                    'eps_scale': {'type': AttrType.FLOAT, 'default': None},
                    'eps_zp': {'type': AttrType.FLOAT, 'default': None},
                    'non_channel_axes': {'type': AttrType.INTS, 'default': None}
                    },
                6: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-5},
                    'eps_scale': {'type': AttrType.FLOAT, 'default': None},
                    'eps_zp': {'type': AttrType.FLOAT, 'default': None},
                    'non_channel_axes': {'type': AttrType.INTS, 'default': None}
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(InstanceNormalizationOp, self).__init__(graph, attr_dict)
        self.update_attributes(InstanceNormalizationOp, attr_dict)
        assert self.check_required(), 'InstanceNormalizationOp is missing a required parameter.'

    def infer_shape(self):
        super(InstanceNormalizationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.data_format == 'NHWC':
            self.non_channel_axes = list(range(1, len(inputs[0].shape) - 1))
            reshape_dim = [-1]
        else:
            self.non_channel_axes = list(range(2, len(inputs[0].shape)))
            reshape_dim = [-1] + [1] * len(self.non_channel_axes)
        mean = np.mean(inputs[0], axis=tuple(
            self.non_channel_axes), keepdims=True)
        variance = np.var(inputs[0], axis=tuple(
            self.non_channel_axes), keepdims=True)
        ngamma = 1.0 / ((variance + self.epsilon) ** (.5))
        normalized = (inputs[0] - mean) * ngamma
        out_tensor = np.reshape(self.weights, reshape_dim) * \
            normalized + np.reshape(self.biases, reshape_dim)
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))


class LayerNormalizationOp(OpHasAxis, OpHasVariableOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {17: {'epsilon': {'type': AttrType.FLOAT, 'required': True, 'default': 1e-5},
                     'axis': {'default': -1},
                     'stash_type': {'type': AttrType.BOOL, 'default': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LayerNormalizationOp, self).__init__(graph, attr_dict)
        self.update_attributes(LayerNormalizationOp, attr_dict)
        assert self.check_required(), 'LayerNormalizationOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        if item == 'axes':
            ret = self.__dict__['_attr'][item].value
            if ret is None and self.axis is not None:
                input_length = len(self.get_input_shapes()[0])
                start_axis = (self.axis + input_length) if self.axis < 0 else self.axis
                ret = [axis for axis in range(input_length) if axis >= start_axis]
                self.__dict__['_attr'][item].value = ret
        if ret is None:
            ret = super(LayerNormalizationOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(LayerNormalizationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) in (2, 3), 'LayerNormalizationOp expects 2 or 3 inputs, but got %d' % len(inputs)
        input_length = len(inputs[0].shape)
        input_dtype = inputs[0].dtype
        self.axes = OpHasAxis.make_axes_non_negative(self.axes, input_length)
        inp = np.array(inputs[0], np.float32) if self.stash_type else inputs[0]
        mean = np.mean(inp, axis=tuple(self.axes), keepdims=True)
        variance = np.var(inp, axis=tuple(self.axes), keepdims=True)
        ngamma = 1.0 / ((variance + self.epsilon) ** 0.5)
        normalized = (inputs[0] - mean) * ngamma
        if self.stash_type:
            normalized = np.array(normalized, inputs[0].dtype)
        weights = OpHasAxis.expand_to(inputs[1], self.axes, input_length)
        if len(inputs) == 3:
            biases = OpHasAxis.expand_to(inputs[2], self.axes, input_length)
        else:
            biases = np.zeros_like(weights)
        out_tensors = [np.array(normalized * weights + biases, dtype=input_dtype)]
        out_ports = self.get_out_ports()
        if 1 in out_ports:
            out_tensors.append(np.array(mean, np.float32))
        if 2 in out_ports:
            out_tensors.append(np.array(ngamma, np.float32))
        self.set_out_tensor(out_tensors)

    def convert_version(self):
        from ...front_end.onnx.passes.common_passes import insert_constant
        input_shapes = self.get_input_shapes()
        assert len(input_shapes) >= 2, 'Meets invalid inputs of LayerNormalizationOp(%s) in convert_version!' % self.name
        if len(input_shapes) == 2:
            np_dtype_str = self.get_inputs_info()[2][1]
            bias = np.zeros(input_shapes[1], dtype=getattr(np, np_dtype_str))
            insert_constant(self._graph, self.name + '_bias', bias, self.name, in_port=2)


class LeakyReluOp(BaseReluOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 0.01},
                    'consumed_inputs': {'type': AttrType.INTS}
                    },
                6: {'alpha': {'type': AttrType.FLOAT, 'default': 0.01}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LeakyReluOp, self).__init__(graph, attr_dict)
        self.update_attributes(LeakyReluOp, attr_dict)
        assert self.check_required(), 'LeakyReluOp is missing a required parameter.'
        self.activations = 'LEAKYRELU'
        self.negative_slope = self.alpha

    def infer_shape(self):
        super(LeakyReluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = self.cal_activation(inputs[0])
        self.set_out_tensor(out_tensor)


class LpPoolOp(BaseOnnxPoolOp, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'kernel_shape': {'type': AttrType.INTS, 'required': True},
                    'p': {'type': AttrType.FLOAT, 'default': '2.0'}
                    },
                2: {'kernel_shape': {'type': AttrType.INTS, 'required': True},
                    'p': {'type': AttrType.INT, 'default': '2'}
                    },
                11: {'kernel_shape': {'type': AttrType.INTS, 'required': True},
                     'p': {'type': AttrType.INT, 'default': '2'}
                     },
                18: {'kernel_shape': {'type': AttrType.INTS, 'required': True},
                     'p': {'type': AttrType.INT, 'default': '2'}
                     },
                }

    def __init__(self, graph, attr_dict=None):
        super(LpPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(LpPoolOp, attr_dict)
        assert self.check_required(), 'LpPoolOp is missing a required parameter.'

    def infer_shape(self):
        super(LpPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        in_shape = inputs[0].shape[1:-1] if self.data_format == 'NHWC' \
            else inputs[0].shape[2:]
        out_shape = BaseOnnxPoolOp.cal_out_shape(
            in_shape, self.pads, self.strides, self.kernel_shape, self.auto_pad, self.dilations, ceil_mode=self.ceil_mode)
        out_tensor_shape = list(inputs[0].shape[0:1]) + out_shape + list(inputs[0].shape[-1:]) \
            if self.data_format == 'NHWC' \
            else list(inputs[0].shape[0:2]) + out_shape
        out_tensor = np.random.ranf(out_tensor_shape).astype(inputs[0].dtype)
        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            # re-calculate pads
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                in_shape,
                out_shape,
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
            )
            self.auto_pad = 'NOTSET'
        self.set_out_tensor(out_tensor)


class LRNOp(OpHasMethod, LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 0.0001},
                    'beta': {'type': AttrType.FLOAT, 'default': 0.75},
                    'bias': {'type': AttrType.FLOAT, 'default': 1.0},
                    'size': {'type': AttrType.INT, 'required': True},
                    'method': {'required': False, 'default': 'ACROSS_CHANNELS'}
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(LRNOp, self).__init__(graph, attr_dict)
        self.update_attributes(LRNOp, attr_dict)
        assert self.check_required(), 'LRNOp is missing a required parameter.'

    def infer_shape(self):
        super(LRNOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_dim = len(inputs[0].shape)
        if self.data_format == 'NHWC':
            input_tensor = np.transpose(
                inputs[0], [0, input_dim - 1] + list(range(1, input_dim - 1)))
            '''
            out_tensor = tf.nn.local_response_normalization(inputs[0], (self.size - 1) // 2, self.bias, self.alpha / self.size, self.beta).numpy()
            '''
        else:
            input_tensor = inputs[0]
        input_dtype = input_tensor.dtype
        if input_dtype != np.float32:
            input_tensor = np.array(input_tensor, dtype=np.float32)
        input_tensor = torch.from_numpy(input_tensor)
        # torch.nn.LocalResponseNorm doesn't support int input
        lrn = torch.nn.LocalResponseNorm(
            self.size, self.alpha, self.beta, self.bias)
        out_tensor = lrn(input_tensor).numpy()
        if self.data_format == 'NHWC':
            out_tensor = np.transpose(
                out_tensor, [0] + list(range(2, input_dim)) + [1])
        if input_dtype != np.float32:
            out_tensor = np.array(out_tensor, dtype=input_dtype)
        self.set_out_tensor(out_tensor)


class LSTMOp(BaseRnnOp, OpHasBiases, OpHasWeights, OnnxOp):
    @staticmethod
    def extract_weights(W, B, R, hidden_size, num_directions):
        ret = []
        for n in range(num_directions):
            i_w, o_w, f_w, c_w = np.split(
                W[n, :, :], W.shape[1] // hidden_size, axis=0)
            i_r, o_r, f_r, c_r = np.split(
                R[n, :, :], R.shape[1] // hidden_size, axis=0)
            i_wb, o_wb, f_wb, c_wb, i_rb, o_rb, f_rb, c_rb = np.split(
                B[n, :], 8, axis=0)
            i_weights = np.concatenate([i_w, i_r], axis=1)
            o_weights = np.concatenate([o_w, o_r], axis=1)
            f_weights = np.concatenate([f_w, f_r], axis=1)
            c_weights = np.concatenate([c_w, c_r], axis=1)
            i_biases = i_wb + i_rb
            o_biases = o_wb + o_rb
            f_biases = f_wb + f_rb
            c_biases = c_wb + c_rb
            weights = np.concatenate(
                [i_weights, c_weights, f_weights, o_weights], axis=0)
            biases = np.concatenate(
                [i_biases, c_biases, f_biases, o_biases], axis=0)
            ret.append({'weights': weights, 'biases': biases})
        return ret

    @classmethod
    def attributes(cls):
        return {1: {'activation_alpha': {'type': AttrType.FLOATS, 'required': False},
                    'activation_beta': {'type': AttrType.FLOATS, 'required': False},
                    'activations': {'type': AttrType.STRINGS, 'required': False, 'default': ['SIGMOID', 'TANH', 'TANH']},
                    'clip': {'type': AttrType.FLOAT, 'required': False},
                    'cell_clip': {'type': AttrType.FLOAT, 'required': False},
                    'direction': {'type': AttrType.STRING, 'default': 'forward', 'options': ['forward', 'reverse', 'bidirectional']},
                    'hidden_size': {'type': AttrType.INT, 'required': True},
                    'output_sequence': {'type': AttrType.INT, 'default': 0, 'required': False},
                    'input_forget': {'type': AttrType.INT, 'default': 0, 'options': [0, 1], 'required': False},
                    'method': {'default': 'Y', 'options': ['Y', 'H', 'C', 'YHC', 'YH', 'YC', 'HC']}
                    },
                7: {'activation_alpha': {'type': AttrType.FLOATS, 'required': False},
                    'activation_beta': {'type': AttrType.FLOATS, 'required': False},
                    'activations': {'type': AttrType.STRINGS, 'required': False, 'default': ['SIGMOID', 'TANH', 'TANH']},
                    'clip': {'type': AttrType.FLOAT, 'required': False},
                    'cell_clip': {'type': AttrType.FLOAT, 'required': False},
                    'direction': {'type': AttrType.STRING, 'default': 'forward', 'options': ['forward', 'reverse', 'bidirectional']},
                    'hidden_size': {'type': AttrType.INT, 'required': True},
                    'input_forget': {'type': AttrType.INT, 'default': 0, 'options': [0, 1], 'required': False},
                    'method': {'default': 'Y', 'options': ['Y', 'H', 'C', 'YHC', 'YH', 'YC', 'HC']}
                    },
                14: {'activation_alpha': {'type': AttrType.FLOATS, 'required': False},
                     'activation_beta': {'type': AttrType.FLOATS, 'required': False},
                     'activations': {'type': AttrType.STRINGS, 'required': False, 'default': ['SIGMOID', 'TANH', 'TANH']},
                     'clip': {'type': AttrType.FLOAT, 'required': False},
                     'cell_clip': {'type': AttrType.FLOAT, 'required': False},
                     'direction': {'type': AttrType.STRING, 'default': 'forward',
                                   'options': ['forward', 'reverse', 'bidirectional']},
                     'hidden_size': {'type': AttrType.INT, 'required': True},
                     'input_forget': {'type': AttrType.INT, 'default': 0, 'options': [0, 1], 'required': False},
                     'layout': {'type': AttrType.INT, 'default': 0},
                     'method': {'default': 'Y', 'options': ['Y', 'H', 'C', 'YHC', 'YH', 'YC', 'HC']}
                     },
                }

    def __init__(self, graph, attr_dict=None):
        super(LSTMOp, self).__init__(graph, attr_dict)
        self.update_attributes(LSTMOp, attr_dict)
        assert self.check_required(), 'LSTMOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'input_forget':
                ret = bool(self.__dict__['_attr'][item].value)
            elif item == 'layout':
                if cur_ver <= 7:
                    ret = False
                else:
                    ret = bool(self.__dict__['_attr'][item].value)
            elif item == 'output_sequence':
                if cur_ver == 1:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ret = True
            elif item == 'activations':
                ret = self.__dict__['_attr'][item].value
                if self.__dict__['_attr']['direction'].value == 'bidirectional' and len(ret) == 3:
                    ret = ret * 2
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(LSTMOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item in ('input_forget', 'layout', 'output_sequence'):
            try:
                self.__dict__['_attr'][item].value = int(value)
            except:
                super(LSTMOp, self).__setattr__(item, value)
        else:
            super(LSTMOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(LSTMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        num_directions = 2 if self.direction == 'bidirectional' else 1
        input_dtype = inputs[0].dtype
        if not self.layout:
            self.time_steps, batch_size, self.input_size = inputs[0].shape
            init_shape = (num_directions, batch_size, self.hidden_size)
        else:
            batch_size, self.time_steps, self.input_size = inputs[0].shape
            init_shape = (batch_size, num_directions, self.hidden_size)
        W, R = inputs[1:3]
        B = inputs[3] if (len(inputs) >= 4 and inputs[3] is not None) else np.zeros(
            (num_directions, 8 * self.hidden_size), dtype=input_dtype)
        sequence_lens = inputs[4] if (len(inputs) >= 5 and inputs[4] is not None) else (
            np.ones((batch_size,), dtype=np.int32) * self.time_steps)
        initial_h = inputs[5] if (len(inputs) >= 6 and inputs[5] is not None) else np.zeros(
            init_shape, dtype=input_dtype)
        initial_c = inputs[6] if (len(inputs) >= 7 and inputs[6] is not None) else np.zeros(
            init_shape, dtype=input_dtype)
        p = inputs[7] if (len(inputs) == 8 and inputs[7] is not None) else np.zeros(
            (num_directions, 3 * self.hidden_size), dtype=input_dtype)

        weights_list = LSTMOp.extract_weights(
            W, B, R, self.hidden_size, num_directions)
        self.weights = np.stack([w['weights'] for w in weights_list])
        self.biases = np.stack([w['biases'] for w in weights_list])

        Y = []
        Y_h = []
        Y_c = []
        h_state = initial_h.copy()

        ''' [num_directions, batch_size, hidden_size] '''
        c_state = initial_c.copy()

        for n in range(num_directions):
            direction_out = []
            h_state_per_direction = h_state[n,
                                            ...] if not self.layout else h_state[:, n, ...]
            c_state_per_direction = c_state[n,
                                            ...] if not self.layout else c_state[:, n, ...]
            if n == 0:
                input_per_direction = inputs[0]
            else:
                input_per_direction = np.flip(inputs[0], axis=0)
            if self.layout:
                input_per_direction = np.transpose(
                    input_per_direction, [1, 0, 2])

            for s in range(self.time_steps):
                '''
                # combine_input = tf.concat([input_per_direction[s, ...], h_state_per_direction], axis=1)
                # fc = tf.matmul(combine_input, weights_list[n]['weights'], transpose_b=True) + weights_list[n]['biases']
                # i, j, f, o = tf.split(fc, 4, axis=1)
                # i = tf.nn.sigmoid(i)    # input_gate
                # j = tf.nn.tanh(j)       # cell_gate
                # f = tf.nn.sigmoid(f)    # forget_gate
                # o = tf.nn.sigmoid(o)    # output_gate
                # input_gate_res = i * j
                # forget_gate_res = f * c_state_per_direction
                # input_add_forget = input_gate_res + forget_gate_res
                # if cell_clip > 0:
                #    input_add_forget = np.clip(input_add_forget, -cell_clip, cell_clip)
                # output_gate_res = o * tf.nn.tanh(input_add_forget)
                '''

                output_gate_res = np.random.ranf(
                    (batch_size, self.hidden_size)).astype(input_dtype)
                input_add_forget = np.random.ranf(
                    (batch_size, self.hidden_size)).astype(input_dtype)

                direction_out.append(output_gate_res)
                if s == self.time_steps - 1:
                    Y_h.append(output_gate_res)
                    Y_c.append(input_add_forget)
                h_state_per_direction = output_gate_res
                c_state_per_direction = input_add_forget
            direction_out = np.stack(direction_out, 0)
            if n == 1:
                direction_out = np.flip(direction_out, axis=0)
            Y.append(direction_out)
        Y = np.stack(Y, axis=1)
        Y_h = np.stack(Y_h, axis=0)
        Y_c = np.stack(Y_c, axis=0)
        if self.layout:
            '''  [seq_length, num_directions, batch_size, hidden_size]
               => [batch_size, seq_length, num_directions, hidden_size]'''
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = np.transpose(Y_h, [1, 0, 2])
            Y_c = np.transpose(Y_c, [1, 0, 2])

        out_ports = self.get_out_ports()
        if out_ports == [1]:
            out_tensors = [Y_h]
            self.method = 'H'
        elif out_ports == [2]:
            out_tensors = [Y_c]
            self.method = 'C'
        elif out_ports == [0, 1, 2]:
            out_tensors = [Y, Y_h, Y_c]
            self.method = 'YHC'
        elif out_ports == [0, 1]:
            out_tensors = [Y, Y_h]
            self.method = 'YH'
        elif out_ports == [0, 2]:
            out_tensors = [Y, Y_c]
            self.method = 'YC'
        elif out_ports == [1, 2]:
            out_tensors = [Y_h, Y_c]
            self.method = 'HC'
        else:
            out_tensors = [Y]
            self.method = 'Y'
        self.set_out_tensor(out_tensors)

    def convert_version(self):
        def _need_insert_constant(in_edges, port):
            if len(in_edges) <= port:
                return True
            if len(in_edges[port]) > 3 \
                    and in_edges[port][3]['tensor'] is not None \
                    and in_edges[port][3]['tensor'].is_const \
                    and in_edges[port][3]['tensor'].value is None:
                src, _, k = in_edges[port][:3]
                self._graph.remove_edge(src, self.name, key=k)
                return True
            return False

        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            out_ports = self.get_out_ports()
            if cur_ver == 1:
                if not self.output_sequence and 0 in out_ports:
                    for _, dst, k, out_attr in self._graph.sorted_out_edges(self.name, keys=True, data=True):
                        if out_attr['src_out_port'] == 0:
                            self._graph.remove_edge(self.name, dst, key=k)
            if cur_ver <= 7:
                layout = False
                self._attr['layout'] = Attribute(
                    'layout', {'type': AttrType.INT, 'value': int(layout)})
            self.cur_version = max_ver

        from ...front_end.onnx.passes.common_passes import insert_constant
        input_dtype = self.get_input_dtypes()[0]
        in_edges = self._graph.sorted_in_edges(self.name, keys=True, data=True)
        num_directions = 2 if self.direction == 'bidirectional' else 1
        batch_size = in_edges[0][3]['tensor'].shape[0] if self.layout else in_edges[0][3]['tensor'].shape[1]
        initial_state_shape = [batch_size, num_directions, self.hidden_size] \
            if self.layout \
            else [num_directions, batch_size, self.hidden_size]
        if _need_insert_constant(in_edges, 7):
            P = np.zeros((num_directions, 3 * self.hidden_size), input_dtype)
            insert_constant(self._graph, self.name +
                            '_P', P, self.name, in_port=7)
        if _need_insert_constant(in_edges, 6):
            initial_c = np.zeros(initial_state_shape, input_dtype)
            insert_constant(self._graph, self.name + '_initial_c',
                            initial_c, self.name, in_port=6)
        if _need_insert_constant(in_edges, 5):
            initial_h = np.zeros(initial_state_shape, input_dtype)
            insert_constant(self._graph, self.name + '_initial_h',
                            initial_h, self.name, in_port=5)
        if _need_insert_constant(in_edges, 4):
            sequence_lens = np.array([self.time_steps] * batch_size, np.int32)
            insert_constant(self._graph, self.name + '_sequence_lens',
                            sequence_lens, self.name, in_port=4)
        if _need_insert_constant(in_edges, 3):
            B = np.zeros((num_directions, 8 * self.hidden_size), input_dtype)
            insert_constant(self._graph, self.name +
                            '_B', B, self.name, in_port=3)


class MaxPoolOp(BaseOnnxPoolOp, OpHasVariableOutPorts):
    @classmethod
    def attributes(cls):
        return {1: {'kernel_shape': {'required': True}},
                8: {'kernel_shape': {'required': True},
                    'storage_order': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}
                    },
                10: {'ceil_mode': {'type': AttrType.INT, 'default': 0},
                     'kernel_shape': {'required': True},
                     'storage_order': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}
                     },
                11: {'ceil_mode': {'type': AttrType.INT, 'default': 0},
                     'kernel_shape': {'required': True},
                     'storage_order': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}
                     },
                12: {'ceil_mode': {'type': AttrType.INT, 'default': 0},
                     'kernel_shape': {'required': True},
                     'storage_order': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}
                     }
                }

    def __init__(self, graph, attr_dict=None):
        super(MaxPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(MaxPoolOp, attr_dict)
        assert self.check_required(), 'MaxPoolOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'ceil_mode':
                if cur_ver >= 10:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ret = False
            elif item == 'storage_order':
                if cur_ver >= 8:
                    ret = int(self.__dict__['_attr'][item].value)
                else:
                    ret = 0
        except:
            ret = None
        if ret is None:
            ret = super(MaxPoolOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(MaxPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        in_shape = inputs[0].shape[1:-
                                   1] if self.data_format == 'NHWC' else inputs[0].shape[2:]
        out_shape = BaseOnnxPoolOp.cal_out_shape(
            in_shape, self.pads, self.strides, self.kernel_shape, self.auto_pad, dilations=self.dilations, ceil_mode=self.ceil_mode)
        out_tensor_shape = list(inputs[0].shape[0:1]) + out_shape + list(inputs[0].shape[-1:]) \
            if self.data_format == 'NHWC' \
            else list(inputs[0].shape[0:2]) + out_shape
        out_tensor = np.random.ranf(out_tensor_shape).astype(inputs[0].dtype)
        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            # re-calculate pads
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                in_shape,
                out_shape,
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
            )
            self.auto_pad = 'NOTSET'
        out_tensors = [out_tensor]
        if len(self.get_out_ports()) == 2:
            out_tensors.append(np.random.randint(
                0, 1, out_tensor.shape, np.int32))
        self.set_out_tensor(out_tensors)


class MaxRoiPoolOp(LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'pooled_shape': {'type': AttrType.INTS, 'required': True},
                    'spatial_scale': {'type': AttrType.FLOAT, 'default': 1.0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(MaxRoiPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(MaxRoiPoolOp, attr_dict)
        assert self.check_required(), 'MaxRoiPoolOp is missing a required parameter.'

    def infer_shape(self):
        super(MaxRoiPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        rois = inputs[1].shape[0]
        if self.data_format == 'NHWC':
            channels = inputs[0].shape[-1]
            out_tensor = np.random.ranf(
                (rois, *self.pooled_shape, channels)).astype(np.float32)
        else:
            channels = inputs[0].shape[1]
            out_tensor = np.random.ranf(
                (rois, channels, *self.pooled_shape)).astype(np.float32)
        self.set_out_tensor(out_tensor)


class MaxUnpoolOp(OpHasPaddingStrides, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            9: {'kernel_shape': {'required': True},
                'pads': {'required': False},
                'strides': {'required': False}},
            11: {'kernel_shape': {'required': True},
                 'pads': {'required': False},
                 'strides': {'required': False}}
        }

    def __init__(self, graph, attr_dict=None):
        super(MaxUnpoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(MaxUnpoolOp, attr_dict)
        assert self.check_required(), 'MaxUnpoolOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'output_shape':
                inputs = self.get_input_tensors()
                if len(inputs) in (2, 3):
                    if len(inputs) == 3 \
                            and inputs[2] is not None:
                        ret = inputs[2].tolist()
                        self.__dict__['_attr'][item] = Attribute(
                            item, {'type': AttrType.INTS, 'value': ret})
                    elif inputs[0] is not None and len(inputs[0].shape) >= 3:
                        input_shape = list(inputs[0].shape)
                        input_spatial_dim = input_shape[2:] if self.data_format == 'NCHW' else input_shape[1:-1]
                        pads = np.reshape(np.array(self.pads), (2, -1))
                        summed_pads = np.sum(pads, axis=0)
                        output_spatial = ((np.array(input_spatial_dim) - 1) * np.array(
                            self.strides) - summed_pads) + np.array(self.kernel_shape)
                        output_spatial_dim = output_spatial.tolist()
                        output_shape = (input_shape[0:2] + output_spatial_dim) \
                            if self.data_format == 'NCHW' \
                            else ([input_shape[0]] + output_spatial_dim + [input_shape[-1]])
                        ret = output_shape
                        self.__dict__['_attr'][item] = Attribute(
                            item, {'type': AttrType.INTS, 'value': ret})
                else:
                    ERROR(
                        '[Parser]: Meets invalid input number of MaxUnpool Op(%s)!' % self.name)
        except:
            ret = None
        if ret is None:
            ret = super(MaxUnpoolOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(MaxUnpoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.random.ranf(self.output_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            self.cur_version = max_ver
        from ...front_end.onnx.passes.common_passes import insert_constant
        in_edges = self._graph.sorted_in_edges(self.name)
        if len(in_edges) == 2:
            insert_constant(self._graph,
                            self.name + '_output_shape',
                            np.array(self.output_shape, np.int32),
                            self.name,
                            in_port=2)


class PReluOp(OpNeedUniBroadcast, BaseReluOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {},
                7: {},
                9: {},
                16: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(PReluOp, self).__init__(graph, attr_dict)
        self.update_attributes(PReluOp, attr_dict)
        assert self.check_required(), 'PReluOp is missing a required parameter.'
        self.activations = 'PRELU'

    def infer_shape(self):
        super(PReluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs) >= 2:
            self.negative_slope = inputs[1].astype(np.float32)
        broad_casted = OpNeedUniBroadcast.broad_cast(
            [inputs[0], self.negative_slope])
        out_tensor = self.cal_activation(*broad_casted).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class ReluOp(BaseReluOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'consumed_inputs': {'type': AttrType.INTS}},
                6: {},
                13: {},
                14: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReluOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReluOp, attr_dict)
        assert self.check_required(), 'ReluOp is missing a required parameter.'
        self.activations = 'RELU'

    def infer_shape(self):
        super(ReluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = self.cal_activation(inputs[0])
        self.set_out_tensor(out_tensor)
