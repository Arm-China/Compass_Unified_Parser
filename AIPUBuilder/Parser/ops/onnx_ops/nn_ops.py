# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import tensorflow as tf
from ..op import *
from ..release_ops import ArmDepthwiseConvOp
from ...common.defs import FLOAT_EQUAL
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


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
            out_tensor = np.random.ranf(size=out_shape).astype(np.float32)

        else:
            out_shape = BaseConvOp.cal_out_shape(inputs[0].shape[2:],
                                                 self.pads,
                                                 self.strides,
                                                 self.kernel_shape,
                                                 self.auto_pad,
                                                 dilations=self.dilations,
                                                 data_format='NCHW')
            out_shape = [inputs[0].shape[0], self.num_output] + out_shape
            out_tensor = np.random.ranf(size=out_shape).astype(np.float32)

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
        out_tensor = np.random.ranf(size=out_shape).astype(np.float32)
        self.set_out_tensor(out_tensor)


class DropoutOp(OpHasVariableOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'is_test': {'type': AttrType.INT, 'default': 0}},
                6: {'is_test': {'type': AttrType.INT, 'default': 0}},
                7: {},
                10: {},
                12: {'training_mode': {'type': AttrType.INT, 'default': 0}},
                13: {'training_mode': {'type': AttrType.INT, 'default': 0}},
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
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    if self.cur_version >= 12:
                        ret = not(
                            bool(self.__dict__['_attr']['training_mode'].value))
                    else:
                        ret = True
                    self.__dict__['_attr'][item] = Attribute(
                        item, {'type': AttrType.INT, 'value': int(ret)})
            elif item == 'training_mode':
                if self.cur_version >= 12:
                    input_tensors = self.get_input_tensors()
                    if len(input_tensors) >= 3:
                        ret = bool(input_tensors[2].item())
                    else:
                        ret = bool(self.__dict__['_attr'][item].value)
                else:
                    if self.cur_version <= 6:
                        ret = not(
                            bool(self.__dict__['_attr']['is_test'].value))
                    else:
                        ret = False
                    self.__dict__['_attr'][item] = Attribute(
                        item, {'type': AttrType.INT, 'value': int(ret)})
        except:
            ret = None
        if ret is None:
            ret = super(DropoutOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(DropoutOp, self).infer_shape()
        inputs = self.get_input_tensors()
        in_shape = inputs[0].shape
        if self.cur_version >= 12:
            mask_value = np.tile(True, in_shape)
        else:
            mask_value = np.tile(False, in_shape)
        if bool(self.training_mode):
            WARN('[Parser]: Dropout (%s) does not support training mode!' % self.name)
        if 1 not in self.get_out_ports() or (self.cur_version < 7 and bool(self.is_test)):
            self.set_out_tensor([inputs[0]])
        else:
            self.set_out_tensor([inputs[0], mask_value])

    def convert_version(self):
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if len(self.get_input_shapes()) < 1 \
                or any([s is None for s in self.get_input_shapes()[0]]):
            ERROR(
                '[Parser]: Meets invalid Dropout Node (%s) in convert_version!' % self.name)
            return

        out_edges = self._graph.sorted_out_edges(
            self.name, keys=True, data=True)
        if cur_ver < 7:
            for _, dst, k, out_attr in out_edges:
                if out_attr['src_out_port'] == 1:
                    self._graph.remove_edge(self.name, dst, key=k)
            return

        from ...graph.graph_algo import get_valid_node_name
        from ...graph.node_wrap import NodeWrap
        in_edges = self._graph.sorted_in_edges(self.name)
        in_shape = self.get_input_shapes()[0]
        mask = get_valid_node_name(self._graph, self.name + '_mask')
        mask_value = np.tile(True if cur_ver >= 12 else False, in_shape)
        mask_used = False
        for _, dst, k, out_attr in out_edges:
            if out_attr['src_out_port'] == 1:
                mask_used = True
                self._graph.remove_edge(self.name, dst, key=k)
                new_attr = copy.deepcopy(out_attr)
                new_attr.update(
                    {'src_out_port': 0, 'tensor': Tensor(value=mask_value)})
                self._graph.add_edge(mask, dst, **new_attr)
        self._graph.remove_edges_from(in_edges[1:])
        if mask_used:
            mask_attr = {'name': mask, 'value': mask_value, 'opset_version': 9}
            NodeWrap(self._graph, mask).replace_obj('Constant', mask_attr)
        if cur_ver < max_ver:
            self.cur_version = max_ver


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
        return {16: {'align_corners': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                     'mode': {'type': AttrType.STRING, 'default': 'bilinear', 'options': ['bilinear', 'nearest', 'bicubic']},
                     'padding_mode': {'type': AttrType.STRING, 'default': 'zeros', 'options': ['zeros', 'border', 'reflection']}
                     },
                }

    def __init__(self, graph, attr_dict=None):
        super(GridSampleOp, self).__init__(graph, attr_dict)
        self.update_attributes(GridSampleOp, attr_dict)
        assert self.check_required(), 'GridSampleOp is missing a required parameter.'

    def infer_shape(self):
        super(GridSampleOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.cur_version == 16 and len(inputs[0].shape) != 4:
            ERROR('[Parser]: GridSampleOp (%s) only supports spatial (4-D) inputs in Op version [%s]!' %
                  (self.name, self.cur_version))
        inp = inputs[0].astype(np.float32) if inputs[0].dtype != np.float32 else inputs[0]
        input_tensor = np.transpose(
            inp, [0, 3, 1, 2]) if self.data_format == 'NHWC' else inputs[0]
        out_tensor = torch.nn.functional.grid_sample(torch.from_numpy(input_tensor),
                                                     torch.from_numpy(
                                                         inputs[1].astype(np.float32)),
                                                     mode=self.mode,
                                                     padding_mode=self.padding_mode,
                                                     align_corners=bool(self.align_corners)).numpy()
        out_tensor = np.transpose(
            out_tensor, [0, 2, 3, 1]) if self.data_format == 'NHWC' else out_tensor
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))


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
        for b in range(batch_size):
            cur_input = inputs[0][:, b,
                                  :] if not self.layout else inputs[0][b, :, :]
            batch_out = []
            for n in range(num_directions):
                direction_out = []
                init_state = initial_h[n, b,
                                       :] if not self.layout else initial_h[b, n, :]
                if n == 1:
                    cur_input = tf.reverse(
                        cur_input, axis=np.array([0], np.int64))
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
                if n == 1:
                    direction_out = tf.reverse(
                        direction_out, axis=np.array([0], np.int64))
                batch_out.append(direction_out)
            batch_out = tf.stack(batch_out, axis=0)
            Y.append(batch_out)

        '''[batch_size, num_directions, seq_length, hidden_size]'''
        Y = tf.stack(Y, axis=0)
        if self.layout:
            ''' [batch_size, seq_length, num_directions, hidden_size]'''
            Y = tf.transpose(Y, perm=[0, 2, 1, 3]).numpy()
            Y_h = Y[:, -1, :, :]
        else:
            ''' [seq_length, num_directions, batch_size, hidden_size]'''
            Y = tf.transpose(Y, perm=[2, 1, 0, 3]).numpy()
            Y_h = Y[-1, :, :, :]

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
        self.set_out_tensor(out_tensor)


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
                input_length = len(self.get_input_tensors()[0].shape)
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
        out_tensors = [normalized * weights + biases]
        out_ports = self.get_out_ports()
        if 1 in out_ports:
            out_tensors.append(np.array(mean, np.float32))
        if 2 in out_ports:
            out_tensors.append(np.array(ngamma, np.float32))
        self.set_out_tensor(out_tensors)

    def convert_version(self):
        from ...front_end.onnx.passes.common_passes import insert_constant
        inputs = self.get_input_tensors()
        assert len(inputs) >= 2, 'Meets invalid inputs of LayerNormalizationOp(%s) in convert_version!' % self.name
        if len(inputs) == 2:
            bias = np.zeros_like(inputs[1])
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
        input_tensor = torch.from_numpy(input_tensor)
        lrn = torch.nn.LocalResponseNorm(
            self.size, self.alpha, self.beta, self.bias)
        out_tensor = lrn(input_tensor).numpy()
        if self.data_format == 'NHWC':
            out_tensor = np.transpose(
                out_tensor, [0] + list(range(2, input_dim)) + [1])
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
        if not self.layout:
            self.time_steps, batch_size, self.input_size = inputs[0].shape
            init_shape = (num_directions, batch_size, self.hidden_size)
        else:
            batch_size, self.time_steps, self.input_size = inputs[0].shape
            init_shape = (batch_size, num_directions, self.hidden_size)
        W, R = inputs[1:3]
        B = inputs[3] if (len(inputs) >= 4 and inputs[3] is not None) else np.zeros(
            (num_directions, 8 * self.hidden_size), dtype=np.float32)
        sequence_lens = inputs[4] if (len(inputs) >= 5 and inputs[4] is not None) else (
            np.ones((batch_size,), dtype=np.int32) * self.time_steps)
        initial_h = inputs[5] if (len(inputs) >= 6 and inputs[5] is not None) else np.zeros(
            init_shape, dtype=np.float32)
        initial_c = inputs[6] if (len(inputs) >= 7 and inputs[6] is not None) else np.zeros(
            init_shape, dtype=np.float32)
        p = inputs[7] if (len(inputs) == 8 and inputs[7] is not None) else np.zeros(
            (num_directions, 3 * self.hidden_size), dtype=np.float32)

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
                    (batch_size, self.hidden_size)).astype(np.float32)
                input_add_forget = np.random.ranf(
                    (batch_size, self.hidden_size)).astype(np.float32)

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
        in_edges = self._graph.sorted_in_edges(self.name, keys=True, data=True)
        num_directions = 2 if self.direction == 'bidirectional' else 1
        batch_size = in_edges[0][3]['tensor'].shape[0] if self.layout else in_edges[0][3]['tensor'].shape[1]
        initial_state_shape = [batch_size, num_directions, self.hidden_size] \
            if self.layout \
            else [num_directions, batch_size, self.hidden_size]
        if _need_insert_constant(in_edges, 7):
            P = np.zeros((num_directions, 3 * self.hidden_size), np.float32)
            insert_constant(self._graph, self.name +
                            '_P', P, self.name, in_port=7)
        if _need_insert_constant(in_edges, 6):
            initial_c = np.zeros(initial_state_shape, np.float32)
            insert_constant(self._graph, self.name + '_initial_c',
                            initial_c, self.name, in_port=6)
        if _need_insert_constant(in_edges, 5):
            initial_h = np.zeros(initial_state_shape, np.float32)
            insert_constant(self._graph, self.name + '_initial_h',
                            initial_h, self.name, in_port=5)
        if _need_insert_constant(in_edges, 4):
            sequence_lens = np.array([self.time_steps] * batch_size, np.int32)
            insert_constant(self._graph, self.name + '_sequence_lens',
                            sequence_lens, self.name, in_port=4)
        if _need_insert_constant(in_edges, 3):
            B = np.zeros((num_directions, 8 * self.hidden_size), np.float32)
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
        out_tensor = self.cal_activation(*broad_casted)
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
