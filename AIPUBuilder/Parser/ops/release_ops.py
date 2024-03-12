# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


from functools import partial
import itertools
import multiprocessing as mp
import tensorflow as tf
import numpy as np
from .op import *
from ..common.defs import FLOAT_EQUAL
from ..common.utils import list_list_to_string, get_random_array
from ..logger import INFO, DEBUG, WARN, ERROR, FATAL


class ArmAbsOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def __init__(self, graph, attr_dict=None):
        super(ArmAbsOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmAbsOp, attr_dict)
        assert self.check_required(), 'ArmAbsOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmAbsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.abs(inputs[0])
        self.set_out_tensor(out_tensor)


class ArmAccidentalHitsOp(OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: 'int32', 1: 'int32'}

    @classmethod
    def num_in_ports(cls):
        return 2

    def infer_shape(self):
        super(ArmAccidentalHitsOp, self).infer_shape()
        input_shape = self.get_input_shapes()[0]
        num_hits = min(np.prod(input_shape), 32768)
        output_indices = get_random_array([num_hits], 'int32')
        output_ids = get_random_array([num_hits], 'int32')
        output_effective_len = get_random_array([1], 'int32')
        self.set_out_tensor([output_indices, output_ids, output_effective_len])


class ArmAcosOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmAcosOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arccos(inputs[0])
        self.set_out_tensor(out_tensor)


class ArmAcoshOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmAcoshOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arccosh(*inputs)
        self.set_out_tensor(out_tensor)


class ArmActivationOp(LayoutUnawareOp, OpHasMethod, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8', 'int16', 'uint16', 'int32']}

    @classmethod
    def attributes(cls):
        return {'method': {'options': ['CELU', 'CLIP', 'ELU', 'GELU', 'HARDSIGMOID', 'HARDSWISH', 'LEAKYRELU', 'MISH', 'PRELU', 'RELU', 'RELU6', 'SELU', 'SHRINK', 'SIGMOID', 'SILU', 'SOFTPLUS', 'SOFTSIGN', 'SWISH', 'TANH', 'TANH_ABS', 'THRESHOLDEDRELU']},
                'clip_min': {'type': AttrType.FLOAT, 'default': None},
                'clip_max': {'type': AttrType.FLOAT, 'default': None},
                'alpha': {'type': AttrType.FLOAT, 'default': None},
                'beta': {'type': AttrType.FLOAT, 'default': None},
                'negative_slope': {'type': AttrType.TENSOR, 'default': None},
                'negative_slope_offset': {'type': AttrType.INT, 'default': -1},
                'negative_slope_range': {'type': AttrType.TENSOR, 'default': None},
                'negative_slope_range_offset': {'type': AttrType.INT, 'default': -1},
                'negative_slope_range_list': {'type': AttrType.TENSORS, 'default': []},
                'negative_slope_scale': {'type': AttrType.TENSOR, 'default': None},
                'negative_slope_scale_offset': {'type': AttrType.INT, 'default': -1},
                'negative_slope_zp': {'type': AttrType.TENSOR, 'default': None},
                'negative_slope_zp_offset': {'type': AttrType.INT, 'default': -1},
                'negative_slope_scale_zp_list': {'type': AttrType.TENSORS, 'default': []},
                'gamma': {'type': AttrType.FLOAT, 'default': None},
                'bias': {'type': AttrType.FLOAT, 'default': None},
                'lambd': {'type': AttrType.FLOAT, 'default': None},
                'approximate': {'type': AttrType.STRINGS, 'default': 'none'},
                }

    METHOD = {'CELU': lambda x, alpha: tf.math.maximum(0.0, x) + tf.math.minimum(0.0, alpha * (tf.exp(x / alpha) - 1.0)),
              'CLIP': lambda x, v1, v2: tf.clip_by_value(x, v1, v2),
              'ELU': tf.nn.elu,
              'GELU': lambda x: (x),
              'HARDSWISH': lambda x: (x * tf.nn.relu6(x + 3) / 6),
              'HARDSIGMOID': lambda x, alpha, beta, cmi, cma: tf.math.maximum(cmi, tf.math.minimum(cma, alpha * x + beta)),
              'LEAKYRELU': lambda x, y: tf.nn.leaky_relu(x, y),
              'MISH': lambda x: (x * tf.math.tanh(tf.math.log(tf.math.exp(x) + 1))),
              'PRELU': lambda x, y: tf.clip_by_value(x, 0, float('inf')) + tf.clip_by_value(x, float('-inf'), 0) * y,
              'RELU': tf.nn.relu,
              'RELU6': tf.nn.relu6,
              'SELU': lambda x: (x),
              'SHRINK': lambda x: (x),
              'SIGMOID': tf.sigmoid,
              'SILU': lambda x: (x) * tf.sigmoid(x),
              'SOFTPLUS': lambda x: tf.math.log(tf.exp(x) + 1),
              'SOFTSIGN': lambda x: x / (1 + tf.abs(x)),
              'SWISH': lambda x, alpha: (x) * tf.sigmoid(alpha * x),
              'TANH': tf.tanh,
              'TANH_ABS': lambda x: tf.abs(tf.tanh(x)),
              'THRESHOLDEDRELU': lambda x: (x),
              }

    def __init__(self, graph, attr_dict=None):
        super(ArmActivationOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmActivationOp, attr_dict)
        assert self.check_required(), 'ArmActivationOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmActivationOp, self).infer_shape()
        assert self.method in ArmActivationOp.METHOD, 'self.method is not in ArmActivationOp.METHOD in ArmActivationOp.'
        func = ArmActivationOp.METHOD[self.method]
        inputs = self.get_input_tensors()
        if self.method == 'CELU':
            inp = inputs[0].astype(np.float32) if np.issubdtype(inputs[0].dtype, np.integer) else inputs[0]
            out_tensor = func(inp, self.alpha).numpy().astype(inputs[0].dtype)
        elif self.method == 'CLIP':
            out_tensor = func(inputs[0], self.clip_min, self.clip_max).numpy().astype(inputs[0].dtype)
        elif self.method == 'GELU':
            out_tensor = self.gelu()
        elif self.method == 'HARDSIGMOID':
            out_tensor = func(
                inputs[0], self.alpha, self.beta, self.clip_min, self.clip_max).numpy().astype(inputs[0].dtype)
        elif self.method == 'LEAKYRELU':
            out_tensor = func(inputs[0], self.alpha).numpy().astype(inputs[0].dtype)
        elif self.method == 'PRELU':
            if not self.quantize:
                self.negative_slope = self.negative_slope.astype(np.float32)
            out_tensor = func(inputs[0], self.negative_slope).numpy().astype(inputs[0].dtype)
        elif self.method == 'SELU':
            out_tensor = self.selu()
        elif self.method == 'SHRINK':
            out_tensor = self.shrink()
        elif self.method in ('ELU', 'SIGMOID', 'TANH'):
            if np.issubdtype(inputs[0].dtype, np.integer):
                inp = inputs[0].astype(np.float32)
            else:
                inp = inputs[0]
            out_tensor = func(inp).numpy().astype(inputs[0].dtype)
        elif self.method == 'SILU':
            out_tensor = self.silu()
        elif self.method == 'SWISH':
            out_tensor = func(inputs[0], self.alpha).numpy().astype(inputs[0].dtype)
        elif self.method == 'THRESHOLDEDRELU':
            out_tensor = self.thresholded_relu()
        else:
            out_tensor = func(inputs[0]).numpy().astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    def gelu(self):
        inputs = self.get_input_tensors()
        input_tensor = torch.tensor(inputs[0], dtype=torch.float32)
        out_tensor = torch.nn.functional.gelu(input_tensor,
                                              approximate=self.approximate).numpy().astype(inputs[0].dtype)
        return out_tensor

    def silu(self):
        inputs = self.get_input_tensors()
        tor_arr = torch.from_numpy(inputs[0].astype(np.float32))
        m = torch.nn.SiLU()
        out_tensor = m(tor_arr)
        tor2numpy = out_tensor.numpy().astype(inputs[0].dtype)
        return tor2numpy

    def selu(self):
        inputs = self.get_input_tensors()
        out_tensor = inputs[0]
        mask = out_tensor <= 0
        out_tensor[mask] = self.alpha * (np.exp(out_tensor[mask]) - 1)
        out_tensor = self.gamma * out_tensor
        out_tensor = out_tensor.astype(inputs[0].dtype)
        return out_tensor

    def thresholded_relu(self):
        inputs = self.get_input_tensors()
        out_tensor = inputs[0]
        mask = out_tensor < self.alpha
        out_tensor[mask] = 0
        return out_tensor

    def shrink(self):
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0])
        mask_neg = out_tensor < -self.lambd
        mask = out_tensor > self.lambd
        mask_0 = np.logical_not(mask | mask_neg)
        out_tensor[mask_neg] = out_tensor[mask_neg] + self.bias
        out_tensor[mask] = out_tensor[mask] - self.bias
        out_tensor[mask_0] = 0
        self.set_out_tensor(out_tensor)
        return out_tensor

    def write_attrs(self, txt_file):
        ret = super(ArmActivationOp, self).write_attrs(txt_file)
        if ret:
            if self.method == 'CELU':
                txt_file.write('alpha=%1.6f\n' % float(self.alpha))
            elif self.method == 'CLIP':
                if self.quantize:
                    txt_file.write('clip_min=%d\n' % int(self.clip_min))
                    txt_file.write('clip_max=%d\n' % int(self.clip_max))
                else:
                    txt_file.write('clip_min=%1.6f\n' % float(self.clip_min))
                    txt_file.write('clip_max=%1.6f\n' % float(self.clip_max))
            elif self.method == 'ELU':
                txt_file.write('alpha=%1.8f\n' % float(self.alpha))
            elif self.method == 'GELU':
                txt_file.write('approximate=%s\n' %
                               str(self.approximate).upper())
            elif self.method == 'HARDSIGMOID':
                txt_file.write('alpha=%1.6f\n' % float(self.alpha))
                txt_file.write('beta=%1.6f\n' % float(self.beta))
                txt_file.write('clip_min=%1.6f\n' % float(self.clip_min))
                txt_file.write('clip_max=%1.6f\n' % float(self.clip_max))
            elif self.method == 'LEAKYRELU' and self.alpha is not None:
                txt_file.write('negative_slope_type=%s\n' % 'float32')
                txt_file.write('negative_slope_value=%1.6f\n' %
                               float(self.alpha))
            elif self.method == 'PRELU' and self.negative_slope is not None:
                txt_file.write('negative_slope_type=%s\n' % str(self.negative_slope.dtype))
                txt_file.write('negative_slope_offset=%d\n' %
                               self.negative_slope_offset)
                txt_file.write('negative_slope_size=%d\n' % (
                    self.negative_slope.size * self.negative_slope.dtype.itemsize))
                txt_file.write('negative_slope_shape=[%s]\n' % num_list_to_string(
                    list(self.negative_slope.shape)))
                if len(self.negative_slope_range_list) == 2:
                    txt_file.write('negative_slope_range=[%s]\n' %
                                   num_list_to_string(self.negative_slope_range_list))
                elif self.negative_slope_range is not None and len(self.negative_slope_range) == 2:
                    txt_file.write('negative_slope_range_type=%s\n' % str(self.negative_slope_range.dtype))
                    txt_file.write('negative_slope_range_offset=%d\n' % self.negative_slope_range_offset)
                    txt_file.write('negative_slope_range_size=%d\n' % (
                        self.negative_slope_range.size * self.negative_slope_range.dtype.itemsize))
                    txt_file.write('negative_slope_range_shape=[%s]\n' % num_list_to_string(
                        list(self.negative_slope_range.shape)))
                if self.quantize:
                    if len(self.negative_slope_scale_zp_list) == 2:
                        txt_file.write('negative_slope_scale=%s\n' %
                                       str(self.negative_slope_scale_zp_list[0]))
                        txt_file.write('negative_slope_zp=%s\n' %
                                       str(self.negative_slope_scale_zp_list[1]))
                    elif self.negative_slope_scale is not None \
                            and self.negative_slope_zp is not None:
                        txt_file.write('negative_slope_scale_type=%s\n' % str(self.negative_slope_scale.dtype))
                        txt_file.write('negative_slope_scale_offset=%d\n' % self.negative_slope_scale_offset)
                        txt_file.write('negative_slope_scale_size=%d\n' % (
                            self.negative_slope_scale.size * self.negative_slope_scale.dtype.itemsize))
                        txt_file.write('negative_slope_scale_shape=[%s]\n' % num_list_to_string(
                            list(self.negative_slope_scale.shape)))

                        txt_file.write('negative_slope_zp_type=%s\n' % str(self.negative_slope_zp.dtype))
                        txt_file.write('negative_slope_zp_offset=%d\n' % self.negative_slope_zp_offset)
                        txt_file.write('negative_slope_zp_size=%d\n' % (
                            self.negative_slope_zp.size * self.negative_slope_zp.dtype.itemsize))
                        txt_file.write('negative_slope_zp_shape=[%s]\n' % num_list_to_string(
                            list(self.negative_slope_zp.shape)))
            elif self.method == 'SELU':
                txt_file.write('alpha=%1.6f\n' % float(self.alpha))
                txt_file.write('gamma=%1.6f\n' % float(self.gamma))
            elif self.method == 'SWISH':
                txt_file.write('alpha=%1.6f\n' % float(self.alpha))
            elif self.method == 'SHRINK':
                txt_file.write('bias=%1.6f\n' % float(self.bias))
                txt_file.write('lambd=%1.6f\n' % float(self.lambd))
            elif self.method == 'THRESHOLDEDRELU':
                txt_file.write('alpha=%1.6f\n' % float(self.alpha))
        return ret

    def write_negative_slope(self, bin_file):
        ret = True
        if not bin_file.closed and bin_file.mode == 'wb':
            if self.negative_slope is not None \
                    and np.ndim(self.negative_slope) > 0 \
                    and self.negative_slope_offset >= 0:
                Op.numpy_to_bin(bin_file, self.negative_slope, self.negative_slope_offset, self.name)
            if self.negative_slope_range is not None \
                    and np.ndim(self.negative_slope_range) > 0 \
                    and self.negative_slope_range_offset >= 0:
                Op.numpy_to_bin(bin_file, self.negative_slope_range, self.negative_slope_range_offset, self.name)
        else:
            FATAL(
                '[Parser]: Invalid file to write negative_slope for Node(%s)!' % self.name)
        return ret


class ArmAdaptivePoolOp(OpHasMethod, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8']}

    @classmethod
    def attributes(cls):
        return {'output_size': {'type': AttrType.INTS, 'required': True},
                'method': {'options': ['AVG', 'MAX'], 'required': True}}

    def __init__(self, graph, attr_dict=None):
        super(ArmAdaptivePoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmAdaptivePoolOp, attr_dict)
        assert self.check_required(), 'ArmAdaptivePoolOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmAdaptivePoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 1 and len(inputs[0].shape) > 2, 'The input is invalid in ArmAdaptivePoolOp.'
        # input_tensor = torch.from_numpy(np.transpose(inputs[0], [0, 3, 1, 2]))
        # m = torch.nn.AdaptiveAvgPool2d(self.output_size) if self.method == 'AVG' \
        #     else torch.nn.AdaptiveMaxPool2d(self.output_size)
        # out_tensor = m(input_tensor).numpy()
        # out_tensor = np.transpose(out_tensor, [0, 2, 3, 1])
        output_shape = [inputs[0].shape[0]] + self.output_size + [inputs[0].shape[-1]]
        out_tensor = np.random.ranf(output_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmAdaptivePoolOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('output_size=[%s]\n' % list_list_to_string(self.output_size))
        return ret


class ArmArgMinMaxOp(OpHasMethod, OpHasAxis, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'method': {'options': ['MIN', 'MAX']},
                'axis': {'default': -1},
                'select_last_index': {'type': AttrType.INT, 'default': 0}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmArgMinMaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmArgMinMaxOp, attr_dict)
        assert self.check_required(), 'ArmArgMinMaxOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'select_last_index':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(ArmArgMinMaxOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ArmArgMinMaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        arg_func = np.argmin if self.method == 'MIN' else np.argmax
        out_tensor = np.expand_dims(
            arg_func(inputs[0], axis=self.axis), axis=self.axis).astype(np.int32)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmArgMinMaxOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('select_last_index=%s\n' %
                           str(self.select_last_index).lower())
        return ret


class ArmAsinOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmAsinOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arcsin(*inputs)
        self.set_out_tensor(out_tensor)


class ArmAsinhOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmAsinhOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arcsinh(*inputs)
        self.set_out_tensor(out_tensor)


class ArmAtanOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmAtanOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arctan(*inputs)
        self.set_out_tensor(out_tensor)


class ArmBasicLSTMOp(BaseRnnOp, OpHasBiases, OpHasWeights, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8'], 1: ['float32', 'float16', 'int8'], 2: ['float32', 'float16', 'int8']}

    @classmethod
    def num_in_ports(cls):
        return 3

    @classmethod
    def attributes(cls):
        return {'method': {'default': 'Y', 'options': ['Y', 'H', 'C', 'YHC', 'YH', 'YC', 'HC']},
                'activations': {'default': ['SIGMOID', 'TANH', 'TANH']},
                'threshold': {'type': AttrType.FLOAT, 'default': None},
                'cell_clip': {'type': AttrType.FLOAT, 'default': None},
                'activation_alpha': {'type': AttrType.FLOATS, 'default': []},
                'activation_beta': {'type': AttrType.FLOATS, 'default': []}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmBasicLSTMOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmBasicLSTMOp, attr_dict)
        assert self.check_required(), 'ArmBasicLSTMOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmBasicLSTMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        batch_size = self.get_input_tensors()[0].shape[0]

        '''
        # h_state = inputs[1].copy()
        # c_state = inputs[2].copy()
        # Y = []
        # Y_h = []
        # Y_c = []
        # direction_out = []
        # h_state_per_direction = h_state
        # c_state_per_direction = c_state
        # input_per_direction = inputs[0]
        # for s in range(self.time_steps):
        #     combine_input = tf.concat([input_per_direction[:, s, :], h_state_per_direction], axis=1)
        #     fc = tf.matmul(combine_input, self.weights, transpose_b=True) + self.biases
        #     i, j, f, o = tf.split(fc, 4, axis=1)
        #     i = tf.nn.sigmoid(i)    # input_gate
        #     j = tf.nn.tanh(j)       # cell_gate
        #     f = tf.nn.sigmoid(f)   # f = tf.nn.sigmoid(f + self.forget_bias)    # forget_gate
        #     o = tf.nn.sigmoid(o)    # output_gate
        #     input_gate_res = i * j
        #     forget_gate_res = f * c_state_per_direction
        #     input_add_forget = input_gate_res + forget_gate_res
        #     if self.threshold > 0:
        #         input_add_forget = tf.clip_by_value(input_add_forget, -self.threshold, self.threshold)
        #     output_gate_res = o * tf.nn.tanh(input_add_forget)
        #
        #     direction_out.append(output_gate_res)
        #     if s == self.time_steps - 1:
        #         Y_h.append(output_gate_res)
        #         Y_c.append(input_add_forget)
        #     h_state_per_direction = output_gate_res
        #     c_state_per_direction = input_add_forget

        # Y = tf.stack(direction_out, 0)
        # Y = tf.transpose(Y, perm=[1, 0, 2]).numpy()
        # Y_h = tf.stack(Y_h, axis=1).numpy()
        # Y_c = tf.stack(Y_c, axis=1).numpy()
        '''
        input_dtype = inputs[0].dtype
        Y = np.random.ranf((batch_size, self.time_steps,
                            self.hidden_size)).astype(input_dtype)
        Y_h = np.random.ranf((batch_size, self.hidden_size)).astype(input_dtype)
        Y_c = np.random.ranf((batch_size, self.hidden_size)).astype(input_dtype)

        if self.method == 'Y':
            self.set_out_tensor([Y])
        elif self.method == 'H':
            self.set_out_tensor([Y_h])
        elif self.method == 'C':
            self.set_out_tensor([Y_c])
        elif self.method == 'YH':
            self.set_out_tensor([Y, Y_h])
        elif self.method == 'YC':
            self.set_out_tensor([Y, Y_c])
        elif self.method == 'HC':
            self.set_out_tensor([Y_h, Y_c])
        elif self.method == 'YHC':
            self.set_out_tensor([Y, Y_h, Y_c])
        else:
            ERROR('[Parser]: BasicLSTM (%s) out-sequence type (%s) not supported!' %
                  (self.name, self.method))

    def write_attrs(self, txt_file):
        ret = super(ArmBasicLSTMOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('out_sequence=[%s]\n' %
                           string_list_to_string(list(self.method)))
            if self.threshold is not None:
                txt_file.write('threshold=%.12f\n' % self.threshold)
            if self.cell_clip is not None:
                txt_file.write('cell_clip=%.12f\n' % self.cell_clip)
            if self.activation_alpha:
                txt_file.write('activation_alpha=[%s]\n' % num_list_to_string(
                    self.activation_alpha))
            if self.activation_beta:
                txt_file.write('activation_beta=[%s]\n' % num_list_to_string(
                    self.activation_beta))
        return ret


class ArmBatchNormOp(BaseLinearOp, OpHasAxis, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8']}

    @classmethod
    def attributes(cls):
        return {'epsilon': {'type': AttrType.FLOAT, 'default': 0.0}}

    def __init__(self, graph, attr_dict=None):
        super(ArmBatchNormOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmBatchNormOp, attr_dict)
        assert self.check_required(), 'ArmBatchNormOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmBatchNormOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axis < 0:
            self.axis += len(inputs[0].shape)
        out_tensor = (inputs[0] * self.weights +
                      self.biases).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmBatchNormOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('epsilon=%.12f\n' % self.epsilon)
        return ret


class ArmBatchToSpaceOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'block_size_x': {'type': AttrType.INT, 'default': 2},
                'block_size_y': {'type': AttrType.INT, 'default': 2},
                'crops': {'type': AttrType.INTS, 'default': [0, 0, 0, 0], 'required': True}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmBatchToSpaceOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmBatchToSpaceOp, attr_dict)
        assert self.check_required(), 'ArmBatchToSpaceOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmBatchToSpaceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.compat.v1.batch_to_space_nd(inputs[0],
                                                    block_shape=np.array(
            [self.block_size_y, self.block_size_x], dtype=np.int64),
            crops=OpHasPaddingStrides.onnx_to_tf(self.crops)[1:3, :]).numpy()
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmBatchToSpaceOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('block_size_x=%d\n' % self.block_size_x)
            txt_file.write('block_size_y=%d\n' % self.block_size_y)
            txt_file.write('crop_left=%d\n' % self.crops[1])
            txt_file.write('crop_right=%d\n' % self.crops[3])
            txt_file.write('crop_top=%d\n' % self.crops[0])
            txt_file.write('crop_bottom=%d\n' % self.crops[2])
        return ret


class ArmBatchToSpaceNDOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'block_size': {'type': AttrType.INTS, 'required': True},
                'crops': {'type': AttrType.INTS, 'default': [0, 0, 0, 0], 'required': True}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmBatchToSpaceNDOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmBatchToSpaceNDOp, attr_dict)
        assert self.check_required(), 'ArmBatchToSpaceNDOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmBatchToSpaceNDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.compat.v1.batch_to_space_nd(
            inputs[0], self.block_size, self.crops).numpy()
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmBatchToSpaceNDOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('block_size=[%s]\n' % list_list_to_string(self.block_size))
            txt_file.write('crops=[%s]\n' % list_list_to_string(self.crops))
        return ret


class ArmBitShiftOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['uint32', 'uint8', 'int16', 'uint16', 'int8', 'int32'], 1: ['uint32', 'uint8', 'uint16', 'int16', 'int8', 'int32']}

    @classmethod
    def attributes(cls):
        return {'direction': {'type': AttrType.STRING, 'required': True, 'options': ['RIGHT', 'LEFT']},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmBitShiftOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmBitShiftOp, attr_dict)
        assert self.check_required(), 'ArmBitShiftOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmBitShiftOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.direction == 'LEFT':
            out_tensor = np.left_shift(inputs[0], inputs[1])
        else:
            out_tensor = np.right_shift(inputs[0], inputs[1])
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmBitShiftOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('direction=%s\n' % self.direction)
        return ret


class ArmBitwiseOp(OpHasOneOutPort, OpHasMethod, ArmOp):
    FUNC_MAP = {'AND': tf.bitwise.bitwise_and,
                'NOT': tf.bitwise.invert,
                'OR': tf.bitwise.bitwise_or,
                'XOR': tf.bitwise.bitwise_xor
                }

    @classmethod
    def cast_in_ports(cls):
        return {0: ['uint32', 'uint8', 'int16', 'uint16', 'int8', 'int32'], None: ['uint32', 'uint8', 'uint16', 'int16', 'int8', 'int32']}

    @classmethod
    def attributes(cls):
        return {'method': {'options': ['ADD', 'NOT', 'OR', 'XOR']},
                }

    def infer_shape(self):
        super(ArmBitwiseOp, self).infer_shape()
        inputs = self.get_input_tensors()
        exp_inputs_cnt = 1 if self.method == 'NOT' else 2
        if len(inputs) != exp_inputs_cnt \
                or any([inp is None for inp in inputs]) is None:
            ERROR(
                '[Parser]: Meets invalid inputs of Bitwise Op(%s) in infer_shape!' % self.name)

        bitwise_func = ArmBitwiseOp.FUNC_MAP[self.method]
        out_tensor = bitwise_func(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class ArmBNLLOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmBNLLOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.log(1. + np.exp(*inputs))
        out_tensor = out_tensor.astype(np.float32)
        self.set_out_tensor(out_tensor)


class ArmBoundingBoxOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def attributes(cls):
        return {'coordinate_order': {'type': AttrType.STRING, 'default': 'YX', 'options': ['XY', 'YX']},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmBoundingBoxOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmBoundingBoxOp, attr_dict)
        assert self.check_required(), 'ArmBoundingBoxOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmBoundingBoxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = copy.deepcopy(inputs[0])
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmBoundingBoxOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('coordinate_order=%s\n' % self.coordinate_order)
        return ret


class ArmCastOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'to_dtype': {'type': AttrType.STRING,
                             'required': True,
                             'options': ['int8', 'uint8', 'int16', 'uint32', 'uint16', 'int32', 'float32', 'float16', 'bfloat16']
                             },
                'clip_mode': {'type': AttrType.STRING, 'options': ['saturation', 'truncation'], 'default': 'saturation'},
                'ignore_scale_zp': {'type': AttrType.BOOL, 'default': True}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmCastOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmCastOp, attr_dict)
        assert self.check_required(), 'ArmCastOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmCastOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = inputs[0].astype(np.dtype(self.to_dtype))
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmCastOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('to_dtype=%s\n' % self.to_dtype)
            txt_file.write('clip_mode=%s\n' % self.clip_mode.upper())
            txt_file.write('ignore_scale_zp=%s\n' % str(self.ignore_scale_zp).lower())
        return ret


class ArmCeilOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmCeilOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.ceil(inputs[0])
        self.set_out_tensor(out_tensor)


class ArmChannelShuffleOp(LayoutConcernedOp, OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8']}

    @classmethod
    def attributes(cls):
        return {'group': {'type': AttrType.INT, 'required': True, 'default': 2},
                'splits': {'type': AttrType.INT, 'required': True, 'default': 1}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmChannelShuffleOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmChannelShuffleOp, attr_dict)
        assert self.check_required(), 'ArmChannelShuffleOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmChannelShuffleOp, self).infer_shape()
        inputs = self.get_input_tensors()
        max_dim = len(inputs[0].shape) - 1
        pre_perm = [0, max_dim] + list(range(1, max_dim))
        inp = np.transpose(inputs[0], pre_perm)
        out_tensor = torch.nn.functional.channel_shuffle(
            torch.from_numpy(inp), self.group).numpy()
        out_tensor = np.transpose(out_tensor, Op.cal_inverse_perm(pre_perm))
        out_tensors = np.split(out_tensor, self.splits, axis=-1)
        self.set_out_tensor(out_tensors)

    def write_attrs(self, txt_file):
        ret = super(ArmChannelShuffleOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('group=%d\n' % self.group)
            txt_file.write('splits=%d\n' % self.splits)
        return ret


class ArmCol2ImOp(OpHasPaddingStrides, OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 3

    @classmethod
    def cast_in_ports(cls):
        return {1: 'int32', 2: 'int32'}

    def infer_shape(self):
        super(ArmCol2ImOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_shape = list(inputs[0].shape)
        batch = input_shape[0]
        image_shape = inputs[1].tolist()
        block_shape = inputs[2].tolist()
        channels = input_shape[1] // int(np.prod(block_shape))
        out_shape = [batch, channels] + image_shape
        out_tensor = np.random.ranf(size=out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class ArmCompressOp(OpHasAxis, OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {1: 'uint8'}

    @classmethod
    def attributes(cls):
        return {'axis': {'default': 0}}

    def __init__(self, graph, attr_dict=None):
        super(ArmCompressOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmCompressOp, attr_dict)
        assert self.check_required(), 'ArmCompressOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmCompressOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.compress(inputs[1], inputs[0], axis=self.axis)
        self.set_out_tensor(out_tensor)


class ArmConcatOp(OpHasAxis, OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return -1

    @classmethod
    def cast_in_ports(cls):
        return {None: ['float32', 'float16', 'int32', 'uint8', 'int8']}

    def infer_shape(self):
        super(ArmConcatOp, self).infer_shape()
        inputs = self.get_input_tensors()
        shapes_list = self.get_input_shapes()

        if len(inputs) < 2 \
                or len(shapes_list) < 2 \
                or any((input_item is None for input_item in inputs))\
                or any((shape is None for shape in shapes_list))\
                or any((shape_item is None for shape in shapes_list for shape_item in shape)):
            ERROR('[Parser]: Invalid inputs shapes for ArmConcatOp!')

        shapes_noaxis = np.delete(shapes_list, self.axis, 1).tolist()
        if shapes_noaxis.count(shapes_noaxis[0]) != len(shapes_noaxis):  # Compare elements in shapes_list for equality
            ERROR('[Parser]: Unequal inputs shapes for ArmConcatOp!')

        out_tensor = np.concatenate(inputs, self.axis)
        in_type_list = [inp.dtype for inp in inputs]
        if in_type_list.count(in_type_list[0]) == len(in_type_list):
            dtype = in_type_list[0]
        elif np.float32 in in_type_list:
            dtype = np.float32
        elif np.int32 in in_type_list:
            dtype = np.int32
        else:
            dtype = None
        if dtype:
            out_tensor = out_tensor.astype(dtype)
        self.set_out_tensor(out_tensor)


class ArmCoshOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmCoshOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.cosh(*inputs)
        self.set_out_tensor(out_tensor)


class ArmConstantOp(OpHasWeights, OpHasOneOutPort, ConstLikeOp, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 0

    def infer_shape(self):
        super(ArmConstantOp, self).infer_shape()
        if self._graph._attr.get('quantize', False):
            out_edges = self._graph.sorted_out_edges(self.name, data=True)
            if len(out_edges) > 0:
                _, _, out_attr = out_edges[0]
                if out_attr.get('tensor', None) is not None \
                        and len(out_attr['tensor'].scale_zp) == 2:
                    self.quantize = True
                    self.weights_scale_zp = list(out_attr['tensor'].scale_zp)
        if self.weights.dtype == 'int64':
            self.weights = np.array(self.weights).astype(np.int32)
        elif self.weights.dtype == 'uint64':
            self.weights = np.array(self.weights).astype(np.uint32)
        elif self.weights.dtype in ['float64', ]:
            self.weights = np.array(self.weights).astype(np.float32)
        elif self.weights.dtype == 'bool':
            self.weights = np.array(self.weights).astype(np.uint8)
        self.set_out_tensor(self.weights)


class ArmConvolutionOp(BaseActivationOp, BaseConvOp, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8']}

    @classmethod
    def perm_onnx_to_ir(cls):
        return [0, 2, 3, 1]

    @classmethod
    def perm_onnx_to_tf(cls):
        pass

    @classmethod
    def perm_ir_to_tf(cls):
        return [1, 2, 3, 0]

    def infer_shape(self):
        super(ArmConvolutionOp, self).infer_shape()
        inputs = self.get_input_tensors()

        '''
        # inp = tf.pad(inputs[0], self.tf_pads) if self.auto_pad == 'NOTSET' else inputs[0]
        # input_split = tf.split(inp, self.group, axis=3)
        # weights_split = np.split(self.weights, self.group, axis=0)
        # meta_conv_list = []
        # for i in range(self.group):
        #     meta_conv = tf.nn.conv2d(input_split[i],
        #                              np.transpose(weights_split[i], axes=type(self).perm_ir_to_tf()),
        #                              [1] + self.strides + [1],
        #                              padding='VALID' if self.auto_pad in ('NOTSET', 'VALID') else 'SAME',
        #                              dilations=[1] + self.dilations + [1],
        #                              data_format='NHWC')
        #     meta_conv_list.append(meta_conv)
        # conv = tf.concat(meta_conv_list, axis=3)
        # out_tensor = tf.nn.bias_add(conv, self.biases, data_format=self.data_format).numpy()
        # out_tensor = self.cal_activation(out_tensor)
        '''

        out_shape = BaseConvOp.cal_out_shape(inputs[0].shape[1:-1],
                                             self.pads,
                                             self.strides,
                                             self.kernel_shape,
                                             self.auto_pad,
                                             dilations=self.dilations,
                                             data_format='NHWC')
        out_shape = [inputs[0].shape[0]] + out_shape + [self.num_output]
        out_tensor = np.random.ranf(size=out_shape).astype(np.float32)
        self.set_out_tensor(out_tensor)


class ArmConvolution3DOp(BaseActivationOp, BaseConvOp, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8']}

    @classmethod
    def perm_onnx_to_ir(cls):
        return [0, 3, 4, 2, 1]

    def infer_shape(self):
        super(ArmConvolution3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_shape = BaseConvOp.cal_out_shape(inputs[0].shape[1:-1],
                                             self.pads,
                                             self.strides,
                                             self.kernel_shape,
                                             self.auto_pad,
                                             dilations=self.dilations,
                                             data_format='NHWC')
        out_shape = [inputs[0].shape[0]] + out_shape + [self.num_output]
        out_tensor = np.random.ranf(size=out_shape).astype(np.float32)
        self.set_out_tensor(out_tensor)


class ArmConvIntegerOp(BaseConvOp, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: 'uint8'}

    @classmethod
    def perm_onnx_to_ir(cls):
        return [0, 2, 3, 1]

    @classmethod
    def attributes(cls):
        return {'x_zero_point': {'type': AttrType.INT, 'required': True},
                'w_zero_point': {'type': AttrType.INT, 'required': True}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmConvIntegerOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmConvIntegerOp, attr_dict)
        assert self.check_required(), 'ArmConvIntegerOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmConvIntegerOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_shape = BaseConvOp.cal_out_shape(inputs[0].shape[1:-1],
                                             self.pads,
                                             self.strides,
                                             self.kernel_shape,
                                             self.auto_pad,
                                             dilations=self.dilations,
                                             data_format='NHWC')
        out_shape = [inputs[0].shape[0]] + out_shape + [self.num_output]
        out_tensor = np.random.ranf(size=out_shape).astype(np.int32)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmConvIntegerOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('x_zero_point=%d\n' % int(self.x_zero_point))
            txt_file.write('w_zero_point=%d\n' % int(self.w_zero_point))
        return ret


class ArmConvTransposeOp(BaseActivationOp, BaseConvOp, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'uint8']}

    @classmethod
    def attributes(cls):
        return {'output_shape': {'type': AttrType.INTS, 'required': True},
                'output_padding': {'type': AttrType.INTS, 'default': [0, 0]}
                }

    @classmethod
    def perm_onnx_to_ir(cls):
        return [1, 2, 3, 0]

    @classmethod
    def perm_tf_to_ir(cls):
        return [2, 0, 1, 3]

    @classmethod
    def perm_ir_to_tf(cls):
        return Op.cal_inverse_perm(cls.perm_tf_to_ir())

    def __init__(self, graph, attr_dict=None):
        super(ArmConvTransposeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmConvTransposeOp, attr_dict)
        assert self.check_required(), 'ArmConvTransposeOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmConvTransposeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        '''
        # out_tensor = tf.nn.conv2d_transpose(inputs[0],
        #                                     np.transpose(self.weights, axes=ArmConvTransposeOp.perm_ir_to_tf()),
        #                                     [inputs[0].shape[0]] + self.output_shape + [self.num_output],
        #                                     strides=[1] + self.strides + [1],
        #                                     padding='VALID' if (self.tf_pads == 0).all() else 'SAME')
        # out_tensor = tf.nn.bias_add(out_tensor, self.biases).numpy()
        # out_tensor = self.cal_activation(out_tensor)
        '''
        out_shape = [inputs[0].shape[0]] + \
            self.output_shape + [self.num_output]
        out_tensor = np.random.ranf(size=out_shape).astype(np.float32)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmConvTransposeOp, self).write_attrs(txt_file)
        if ret:
            # TODO: Remove output_padding_x and output_padding_y
            txt_file.write('output_padding_x=%d\n' % self.output_padding[1])
            txt_file.write('output_padding_y=%d\n' % self.output_padding[0])
            assert(all(p == 0 for p in self.output_padding)
                   ), 'Meet non-zero output_padding in ArmConvTransposeOp!'
        return ret


class ArmConvTranspose3DOp(BaseActivationOp, BaseConvOp, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'output_shape': {'type': AttrType.INTS, 'required': True}}

    @classmethod
    def perm_onnx_to_ir(cls):
        return [1, 3, 4, 2, 0]

    def __init__(self, graph, attr_dict=None):
        super(ArmConvTranspose3DOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmConvTranspose3DOp, attr_dict)
        assert self.check_required(), 'ArmConvTranspose3DOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmConvTranspose3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_shape = [inputs[0].shape[0]] + \
            self.output_shape + [self.num_output]
        out_tensor = np.random.ranf(size=out_shape).astype(np.float32)
        self.set_out_tensor(out_tensor)


class ArmCosineOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmCosineOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.cos(inputs[0])
        self.set_out_tensor(out_tensor)


class ArmCountOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'min': {'type': AttrType.FLOAT, 'required': True},
                'max': {'type': AttrType.FLOAT, 'required': True},
                'nbins': {'type': AttrType.INT, 'required': True},
                'discrete': {'type': AttrType.INT, 'required': True},
                'in_type': {'type': AttrType.STRING, 'required': False, 'default': 'float32'},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmCountOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmCountOp, attr_dict)
        assert self.check_required(), 'ArmCountOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'discrete':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(ArmCountOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ArmCountOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if str(inputs[0].dtype) not in ('int32', 'int64', 'float32', 'float16', 'float64'):
            if np.issubdtype(inputs[0].dtype, np.integer) or np.issubdtype(inputs[0].dtype, bool):
                inp = inputs[0].astype(np.int32)
            else:
                inp = inputs[0].astype(np.float32)
        else:
            inp = inputs[0]
        self.in_type = str(inp.dtype)

        tensor_list = []
        for b in range(inp.shape[0]):
            meta_tensor = tf.histogram_fixed_width(inp[b],
                                                   value_range=np.array(
                                                       [self.min, self.max], inp.dtype),
                                                   nbins=self.nbins).numpy()
            tensor_list.append(meta_tensor)
        out_tensor = tf.stack(tensor_list, axis=0).numpy()
        out_tensor = out_tensor.astype(np.int32)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmCountOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('min=%s\n' % str(
                np.array(self.min).astype(np.dtype(self.in_type))))
            txt_file.write('max=%s\n' % str(
                np.array(self.max).astype(np.dtype(self.in_type))))
            txt_file.write('nbins=%d\n' % self.nbins)
            txt_file.write('discrete=%s\n' % str(self.discrete).lower())
        return ret


class ArmCropAndResizeOp(OpHasMethod, LayoutConcernedOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16'], 2: 'int32'}

    @classmethod
    def num_in_ports(cls):
        return 3

    @classmethod
    def attributes(cls):
        return {'crop_size': {'type': AttrType.INTS, 'default': []},
                'method': {'options': ['BILINEAR', 'NEAREST']},
                'extrapolation_value': {'type': AttrType.FLOAT, 'default': 0.0}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmCropAndResizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmCropAndResizeOp, attr_dict)
        assert self.check_required(), 'ArmCropAndResizeOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmCropAndResizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        depth = inputs[0].shape[-1]
        box_num = inputs[2].size
        out_tensor = np.random.ranf(
            [box_num] + self.crop_size + [depth]).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmCropAndResizeOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('crop_size=[%s]\n' %
                           list_list_to_string(self.crop_size))
            txt_file.write('extrapolation_value=%f\n' %
                           self.extrapolation_value)
        return ret


class ArmCTCGreedyDecoderOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: 'int32'}

    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def attributes(cls):
        return {'merge_repeated': {'type': AttrType.INT, 'default': 1},
                'sequence_lens': {'type': AttrType.INT},
                'input_size': {'type': AttrType.INT}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmCTCGreedyDecoderOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmCTCGreedyDecoderOp, attr_dict)
        assert self.check_required(), 'ArmCTCGreedyDecoderOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'merge_repeated':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(ArmCTCGreedyDecoderOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ArmCTCGreedyDecoderOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = get_random_array(
            [inputs[0].shape[0], 4096, 1, 1], 'int32')
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmCTCGreedyDecoderOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('merge_repeated=%s\n' %
                           str(self.merge_repeated).lower())
        return ret


class ArmCumulateOp(OpHasOneOutPort, OpHasMethod, OpHasAxis, ArmOp):
    FUNC_MAP = {'PROD': tf.math.cumprod,
                'SUM': tf.math.cumsum
                }

    @classmethod
    def cast_in_ports(cls):
        return {0: ['uint32', 'uint8', 'int16', 'uint16', 'int8', 'int32', 'float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'method': {'options': ['PROD', 'SUM']},
                'exclusive': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                'reverse': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmCumulateOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmCumulateOp, attr_dict)
        assert self.check_required(), 'Cumulative is missing a required parameter.'

    def infer_shape(self):
        super(ArmCumulateOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cum_func = ArmCumulateOp.FUNC_MAP[self.method]
        out_tensor = cum_func(inputs[0], axis=self.axis, exclusive=bool(
            self.exclusive), reverse=bool(self.reverse)).numpy()
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmCumulateOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('exclusive=%s\n' % str(bool(self.exclusive)).lower())
            txt_file.write('reverse=%s\n' % str(bool(self.reverse)).lower())
        return ret


class ArmDecodeBoxOp(OpHasAnchors, OpHasMultipleOutPorts, ArmOp):
    @staticmethod
    def tile_anchors(grid_height,
                     grid_width,
                     scales,
                     aspect_ratios,
                     base_anchor_size,
                     anchor_stride,
                     anchor_offset):
        scales = np.array(scales, np.float32)
        ratio_sqrts = np.sqrt(aspect_ratios)
        heights = scales / ratio_sqrts * base_anchor_size[0]
        widths = scales * ratio_sqrts * base_anchor_size[1]

        # Get a grid of box centers
        y_centers = np.array(range(grid_height), dtype=np.float32)
        y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
        x_centers = np.array(range(grid_width), dtype=np.float32)
        x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
        x_centers, y_centers = np.meshgrid(x_centers, y_centers)

        widths_grid, x_centers_grid = np.meshgrid(widths, x_centers)
        heights_grid, y_centers_grid = np.meshgrid(heights, y_centers)

        bbox_centers = np.stack([y_centers_grid[:, :, np.newaxis],
                                 x_centers_grid[:, :, np.newaxis]], axis=3)
        bbox_sizes = np.stack(
            [heights_grid[:, :, np.newaxis], widths_grid[:, :, np.newaxis]], axis=3)
        bbox_centers = np.reshape(bbox_centers, [-1, 2])
        bbox_sizes = np.reshape(bbox_sizes, [-1, 2])

        bbox_corners = np.concatenate(
            [bbox_centers - .5 * bbox_sizes, bbox_centers + .5 * bbox_sizes], 1)
        return bbox_corners

    @staticmethod
    def generate_anchors(feature_map_shape_list,
                         min_scale=0.2,
                         max_scale=0.95,
                         aspect_ratios=(1.0, 2.0, 1.0 / 2, 3.0, 1.0 / 3)):
        num_layers = len(feature_map_shape_list)
        origin_scales = np.linspace(
            min_scale, max_scale, num_layers).tolist() + [1.0]
        box_specs_list = []

        for layer, scale, scale_next in zip(range(num_layers),
                                            origin_scales[:-1],
                                            origin_scales[1:]):
            layer_box_specs = []
            if layer == 0:
                layer_box_specs.extend(
                    [(0.1, 1.0), (scale, 2.0), (scale, 0.5), ])
            else:
                for aspect_ratio in aspect_ratios:
                    layer_box_specs.append((scale, aspect_ratio))
                layer_box_specs.append((np.sqrt(scale * scale_next), 1.0))
            box_specs_list.append(layer_box_specs)

        scales_list = [[bs[0] for bs in box_specs]
                       for box_specs in box_specs_list]
        aspect_ratios_list = [[bs[1] for bs in box_specs]
                              for box_specs in box_specs_list]
        anchor_grid_list = []
        anchor_strides = [(1.0 / pair[0], 1.0 / pair[1])
                          for pair in feature_map_shape_list]
        anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1])
                          for stride in anchor_strides]
        base_anchor_size = [1.0, 1.0]
        for grid_size, scales, aspect_ratios, stride, offset in \
                zip(feature_map_shape_list, scales_list, aspect_ratios_list, anchor_strides, anchor_offsets):
            anchor_grid_list.extend(ArmDecodeBoxOp.tile_anchors(
                grid_size[0],
                grid_size[1],
                scales,
                aspect_ratios,
                base_anchor_size,
                stride,
                offset
            ))
        anchors = np.stack(anchor_grid_list, axis=1)
        ymin, xmin, ymax, xmax = np.split(anchors, 4, axis=0)
        width = xmax - xmin
        height = ymax - ymin
        ycenter = ymin + height / 2.
        xcenter = xmin + width / 2.
        return np.squeeze(np.stack([ycenter, xcenter, height, width], axis=2), axis=0).astype(np.float32)

    @staticmethod
    def generate_anchors_for_resnet(fig_size,
                                    feat_size,
                                    steps=None,
                                    scales=None,
                                    aspect_ratios=[[2], [2, 3], [
                                        2, 3], [2, 3], [2], [2]],
                                    scale_xy=0.1,
                                    scale_wh=0.2):
        fig_size_w, fig_size_h = fig_size
        if steps is None:
            steps = [(int(fig_size_w / fs[0]), int(fig_size_h / fs[1]))
                     for fs in feat_size]
        if scales is None:
            scales = [(int(s * fig_size_w / 300), int(s * fig_size_h / 300))
                      for s in [21, 45, 99, 153, 207, 261, 315]]

        steps_w = [st[0] for st in steps]
        steps_h = [st[1] for st in steps]

        fkw = fig_size_w // np.array(steps_w)
        fkh = fig_size_h // np.array(steps_h)

        default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(feat_size):
            sfeat_w, sfeat_h = sfeat
            sk1 = scales[idx][0] / fig_size_w
            sk2 = scales[idx + 1][1] / fig_size_h
            sk3 = np.sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * np.sqrt(alpha), sk1 / np.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat_w), range(sfeat_h)):
                    cx, cy = (j + 0.5) / fkh[idx], (i + 0.5) / fkw[idx]
                    # default_boxes.append((cx, cy, w, h))
                    default_boxes.append([cy, cx, h, w])
        default_boxes = np.clip(default_boxes, 0, 1)
        return np.array(default_boxes, dtype=np.float32)

    @classmethod
    def num_in_ports(cls):
        return 2

    @staticmethod
    def convert_to_nested_list(src_list):
        assert len(
            src_list) % 2 == 0, 'The length of src_list is invalid in convert_to_nested_list of ArmDecodeBoxOp.'
        return [[src_list[i * 2], src_list[i * 2 + 1]] for i in range(len(src_list) // 2)]

    @classmethod
    def attributes(cls):
        return {'feature_map': {'type': AttrType.INTS},
                'image_width': {'type': AttrType.INT, 'default': 300},
                'image_height': {'type': AttrType.INT, 'default': 300},
                'max_box_num': {'type': AttrType.INT, 'default': 5000},
                'class_num': {'type': AttrType.INT, 'default': 90},
                'score_threshold': {'type': AttrType.FLOAT, 'default': 0.5},
                'variance': {'type': AttrType.FLOATS, 'default': []},
                'firstbox_scale': {'type': AttrType.FLOATS, 'default': []},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmDecodeBoxOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmDecodeBoxOp, attr_dict)
        assert self.check_required(), 'ArmDecodeBoxOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmDecodeBoxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp_shapes = self.get_input_shapes()
        assert len(inputs) == 2 and len(
            inp_shapes) == 2, 'The length of input is invalid in ArmDecodeBoxOp.'
        inp1_shape = inp_shapes[0]
        inp2_shape = inp_shapes[1]
        anchors_shape = self.anchors.shape
        if not inp1_shape or not inp2_shape \
                or not anchors_shape \
                or any((shape is None for shape in inp1_shape)) \
                or any((shape is None for shape in inp2_shape)) \
                or any((shape is None for shape in anchors_shape)) \
                or len(inp1_shape) != 3 or len(inp2_shape) != 3 \
                or len(anchors_shape) != 2 \
                or inp1_shape[:2] != inp2_shape[:2] \
                or inp1_shape[1] != anchors_shape[0] \
                or inp2_shape[-1] != 4 \
                or anchors_shape[-1] != 4:
            ERROR('[Parser]: Invalid inputs shape of ArmDecodeBoxOp!')

        self.max_box_num = max(inputs[1].shape[1], 5000)
        batch_size = inputs[0].shape[0]
        out_tensor1 = np.random.ranf(
            size=(batch_size, self.max_box_num, 4)).astype(np.float32)
        out_tensor2 = np.random.ranf(
            size=(batch_size, self.max_box_num)).astype(np.int32)
        out_tensor3 = np.random.ranf(size=(batch_size, 1)).astype(np.int32)
        out_tensor4 = np.random.ranf(
            size=(batch_size, self.max_box_num)).astype(np.float32)
        out_tensor5 = np.random.ranf(
            size=(batch_size, self.max_box_num)).astype(np.int32)
        out_tensor_list = [out_tensor1, out_tensor2,
                           out_tensor3, out_tensor4, out_tensor5]
        self.set_out_tensor(out_tensor_list)

    def write_attrs(self, txt_file):
        ret = super(ArmDecodeBoxOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('image_width=%d\n' % self.image_width)
            txt_file.write('image_height=%d\n' % self.image_height)
            txt_file.write('max_box_num=%d\n' % self.max_box_num)
            txt_file.write('class_num=%d\n' % self.class_num)
            txt_file.write('score_threshold=%f\n' % self.score_threshold)
            txt_file.write('feature_map=[%s]\n' %
                           list_list_to_string(self.feature_map))
            if self.variance:
                txt_file.write('variance=[%s]\n' %
                               list_list_to_string(self.variance))
            if (self.xcenter is None or self.ycenter is None or self.ha is None or self.wa is None) \
                    and self.firstbox_scale:
                txt_file.write(
                    'firstbox_scale=[%s]\n' % list_list_to_string(self.firstbox_scale))
        return ret


class ArmDepthToSpaceOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'blocksize': {'type': AttrType.INT, 'required': True},
                'mode': {'type': AttrType.STRING, 'default': 'DCR', 'options': ['DCR', 'CRD']}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmDepthToSpaceOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmDepthToSpaceOp, attr_dict)
        assert self.check_required(), 'ArmDepthToSpaceOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmDepthToSpaceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        n, h, w, c = inputs[0].shape
        if self.mode == 'DCR':
            reshape_dim = [n, h, w, self.blocksize, self.blocksize, c // self.blocksize ** 2]
            trans_perm = [0, 1, 3, 2, 4, 5]
        else:
            reshape_dim = [n, h, w, c // self.blocksize ** 2, self.blocksize, self.blocksize]
            trans_perm = [0, 1, 4, 2, 5, 3]
        inp = np.reshape(inputs[0], reshape_dim)
        out = np.transpose(inp, trans_perm)
        out_tensor = np.reshape(
            out, (n, h * self.blocksize, w * self.blocksize, c // self.blocksize ** 2))
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmDepthToSpaceOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('mode=%s\n' % self.mode)
            txt_file.write('block_size_x=%d\n' % self.blocksize)
            txt_file.write('block_size_y=%d\n' % self.blocksize)
        return ret


class ArmDepthwiseConvOp(BaseActivationOp, BaseConvOp, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8']}

    @classmethod
    def attributes(cls):
        return {'multiplier': {'type': AttrType.INT, 'default': 1}}

    @classmethod
    def perm_onnx_to_tf(cls):
        return [2, 3, 0, 1]

    @classmethod
    def perm_onnx_to_ir(cls):
        return [0, 2, 3, 1]

    @classmethod
    def perm_ir_to_tf(cls):
        return [1, 2, 0, 3]

    def __init__(self, graph, attr_dict=None):
        super(ArmDepthwiseConvOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmDepthwiseConvOp, attr_dict)
        assert self.check_required(), 'ArmDepthwiseConvOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmDepthwiseConvOp, self).infer_shape()
        inputs = self.get_input_tensors()

        # inp = tf.pad(inputs[0], self.tf_pads) if self.auto_pad == 'NOTSET' else inputs[0]
        # out_tensor = tf.nn.depthwise_conv2d(inp,
        #                                     np.transpose(self.weights, axes=type(self).perm_ir_to_tf()),
        #                                     strides=[1] + self.strides + [1],
        #                                     padding='VALID' if self.auto_pad in ('NOTSET', 'VALID') else 'SAME',
        #                                     data_format=self.data_format,
        #                                     rate=self.dilations)
        # out_tensor = tf.nn.bias_add(out_tensor, self.biases, data_format=self.data_format).numpy()
        # out_tensor = self.cal_activation(out_tensor)

        out_shape = BaseConvOp.cal_out_shape(inputs[0].shape[1:-1],
                                             self.pads,
                                             self.strides,
                                             self.kernel_shape,
                                             self.auto_pad,
                                             dilations=self.dilations,
                                             data_format='NHWC')
        out_shape = [inputs[0].shape[0]] + out_shape + [self.num_output]
        out_tensor = np.random.ranf(size=out_shape).astype(np.float32)

        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmDepthwiseConvOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('multiplier=%d\n' % self.multiplier)
        return ret


class ArmDeQuantizeOp(OpHasAxis, BaseQuantizeDequantizeOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'from_dtype': {'type': AttrType.STRING,
                               'required': True,
                               'options': ['int8', 'uint8', 'int32', 'uint32']}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmDeQuantizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmDeQuantizeOp, attr_dict)
        assert self.check_required(), 'ArmDeQuantizeOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmDeQuantizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axis is None:
            out_tensor = ((inputs[0].astype(np.int32) -
                           self.zero_point) * self.scale).astype(np.float32)
        else:
            expand_len = len(inputs[0].shape) - self.axis - 1
            scale = np.reshape(self.scale, list(
                self.scale.shape) + [1] * expand_len)
            zero_point = np.reshape(self.zero_point, list(
                self.zero_point.shape) + [1] * expand_len)

            out_tensor = ((inputs[0].astype(np.int32) -
                           zero_point) * scale).astype(np.float32)
        self.set_out_tensor(out_tensor)


class ArmDilationOp(OpHasPaddingStrides, OpHasWeights, OpHasOneOutPort, LayoutConcernedOp, ArmOp):
    @classmethod
    def attributes(cls):
        return {'dilations': {'default': [1, 1, 1, 1]}
                }

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def __init__(self, graph, attr_dict=None):
        super(ArmDilationOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmDilationOp, attr_dict)
        assert self.check_required(), 'ArmDilationOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmDilationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = tf.pad(inputs[0], self.tf_pads) if self.auto_pad == 'NOTSET' else inputs[0]
        out_tensor = tf.nn.dilation2d(inp,
                                      np.transpose(np.reshape(self.weights, self.weights.shape[:3]), [1, 2, 0]),
                                      strides=[1] + self.strides + [1],
                                      padding='VALID',
                                      dilations=[1] + self.dilations + [1],
                                      data_format='NHWC').numpy()
        self.set_out_tensor(out_tensor)


class ArmDetectionOutputOp(OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 3

    @classmethod
    def attributes(cls):
        return {'image_width': {'type': AttrType.INT, 'required': True},
                'image_height': {'type': AttrType.INT, 'required': True},
                'class_num': {'type': AttrType.INT, 'required': True},
                'score_threshold': {'type': AttrType.FLOAT, 'default': 0.7},
                'max_box_num': {'type': AttrType.INT, 'default': 5000},
                'anchor_mode': {'type': AttrType.STRING, 'default': None},
                'variance': {'type': AttrType.FLOATS, 'default': []},
                'bbox_xform_clip': {'type': AttrType.FLOAT, 'default': None},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmDetectionOutputOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmDetectionOutputOp, attr_dict)
        assert self.check_required(), 'ArmDetectionOutputOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmDetectionOutputOp, self).infer_shape()
        inputs = self.get_input_tensors()
        batch_size, box_num = inputs[0].shape[:2]
        valid_max_box_num = int(box_num * self.class_num)
        if self.max_box_num > valid_max_box_num:
            self.max_box_num = valid_max_box_num
        out_tensor1 = np.random.ranf(
            size=(batch_size, self.max_box_num)).astype(np.float32)
        out_tensor2 = np.random.ranf(
            size=(batch_size, self.max_box_num, 4)).astype(np.float32)
        out_tensor3 = np.random.ranf(
            size=(batch_size, self.class_num)).astype(np.int32)
        out_tensor4 = np.random.ranf(
            size=(batch_size, self.class_num)).astype(np.int32)
        out_tensor5 = np.random.ranf(size=(batch_size, 1)).astype(np.int32)
        out_tensor_list = [out_tensor1, out_tensor2,
                           out_tensor3, out_tensor4, out_tensor5]
        self.set_out_tensor(out_tensor_list)

    def write_attrs(self, txt_file):
        ret = super(ArmDetectionOutputOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('image_width=%d\n' % self.image_width)
            txt_file.write('image_height=%d\n' % self.image_height)
            txt_file.write('score_threshold=%f\n' % self.score_threshold)
            if self.anchor_mode:
                txt_file.write('anchor_mode=%s\n' % self.anchor_mode)
            if self.variance:
                txt_file.write('variance=[%s]\n' %
                               list_list_to_string(self.variance))
            if self.bbox_xform_clip is not None:
                txt_file.write('bbox_xform_clip=%f\n' % self.bbox_xform_clip)
        return ret


class ArmDivOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16']}

    @classmethod
    def num_in_ports(cls):
        return 2

    def infer_shape(self):
        super(ArmDivOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = np.true_divide(*inputs)
        self.set_out_tensor(out_tensors)


class ArmDivModOp(LayoutUnawareOp, OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['int8', 'uint8', 'int32', 'uint32'], 1: ['int8', 'uint8', 'int32', 'uint32']}

    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def attributes(cls):
        return {'mode': {'type': AttrType.STRING, 'default': 'floor', 'options': ['trunc', 'floor']}}

    def __init__(self, graph, attr_dict=None):
        super(ArmDivModOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmDivModOp, attr_dict)
        assert self.check_required(), 'ArmDivModOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmDivModOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 2, 'Expects 2 inputs for ArmDivMod op (%s), but got %d' % (self.name, len(inputs))
        try:
            out0 = torch.div(torch.tensor(inputs[0]), torch.tensor(inputs[1]), rounding_mode=self.mode).numpy()
        except:
            in_consts = self.sorted_in_consts()
            if len(in_consts) == 2 \
                    or (len(in_consts) == 1 and in_consts[0][1] == 1):
                ERROR('[Parser]: Meets invalid second input of ArmDivMod Op (%s) in infer_shape!' % self.name)
                out0 = None
            else:
                WARN('[Parser]: The second input of ArmDivMod Op (%s) is replaced by ones in infer_shape!' % self.name)
                second_input = np.ones_like(inputs[1])
                out0 = torch.div(torch.tensor(inputs[0]), torch.tensor(second_input), rounding_mode=self.mode).numpy()
        out1 = (inputs[0] - out0 * inputs[1]) if out0 is not None else None
        self.set_out_tensor([out0, out1])

    def write_attrs(self, txt_file):
        ret = super(ArmDivModOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('mode=%s\n' % self.mode.upper())


class ArmEltwiseOp(LayoutUnawareOp, OpHasMethod, BaseActivationOp, ArmOp):
    FUNC_MAP = {'ADD': np.add,
                'SUB': np.subtract,
                'MUL': np.multiply,
                'MAX': np.maximum,
                'MIN': np.minimum,
                }

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8', 'int32'], 1: ['float32', 'float16', 'int8', 'uint8', 'int32']}

    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def attributes(cls):
        return {'method': {'options': ['ADD', 'SUB', 'MUL', 'MAX', 'MIN']}}

    def __init__(self, graph, attr_dict=None):
        super(ArmEltwiseOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmEltwiseOp, attr_dict)
        assert self.check_required(), 'ArmEltwiseOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmEltwiseOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs) != 2 \
                or any([inp is None for inp in inputs]) is None:
            ERROR(
                '[Parser]: Meets invalid inputs of Eltwise Op(%s) in infer_shape!' % self.name)
        if list(inputs[0].shape) != list(inputs[1].shape):
            ERROR('[Parser]: Shapes of two inputs of Eltwise Op(%s) should be equal in infer_shape!' % self.name)
        eltwise_func = ArmEltwiseOp.FUNC_MAP[self.method]
        out_tensor = eltwise_func(*inputs)
        out_tensor = self.cal_activation(out_tensor)
        self.set_out_tensor(out_tensor)


class ArmEmbeddingLookupSparseOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 4

    @classmethod
    def cast_in_ports(cls):
        return {1: 'int32', 2: 'int32', 3: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'combiner': {'type': AttrType.STRING, 'default': 'MEAN', 'options': ['MEAN', 'SQRTN', 'SUM']},
                'max_norm': {'type': AttrType.FLOAT, 'default': 'NONE'}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmEmbeddingLookupSparseOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmEmbeddingLookupSparseOp, attr_dict)
        assert self.check_required(), 'ArmEmbeddingLookupSparseOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmEmbeddingLookupSparseOp, self).infer_shape()
        inputs = self.get_input_tensors()
        data, ids, values, weights = inputs[:4]
        ids = np.expand_dims(ids, -1)
        sp_ids = tf.sparse.SparseTensor(ids, values, [ids.shape[-1]])
        sp_weights = tf.sparse.SparseTensor(ids, weights, [ids.shape[-1]])
        out_tensor = tf.nn.embedding_lookup_sparse(data,
                                                   sp_ids,
                                                   sp_weights,
                                                   combiner=self.combiner.lower(),
                                                   max_norm=None if self.max_norm == 'NONE' else float(self.max_norm)).numpy()
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmEmbeddingLookupSparseOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('combiner=%s\n' % self.combiner.upper())
            txt_file.write('max_norm=%s\n' % str(self.max_norm))
        return ret


class ArmErfOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmErfOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = torch.erf(torch.from_numpy(inputs[0])).numpy()
        self.set_out_tensor(out_tensor)


class ArmErosionOp(OpHasPaddingStrides, OpHasWeights, OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'dilations': {'default': [1, 1, 1, 1]},
                }

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def __init__(self, graph, attr_dict=None):
        super(ArmErosionOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmErosionOp, attr_dict)
        assert self.check_required(), 'ArmErosionOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmErosionOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = tf.pad(inputs[0], self.tf_pads) if self.auto_pad == 'NOTSET' else inputs[0]

        out_tensor = tf.compat.v1.nn.erosion2d(inp,
                                               np.transpose(np.reshape(
                                                   self.weights, self.weights.shape[:3]), [1, 2, 0]),
                                               strides=[1] + self.strides + [1],
                                               rates=[1] + self.dilations + [1],
                                               padding='VALID').numpy()

        self.set_out_tensor(out_tensor)


class ArmExpOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmExpOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.exp(inputs[0])
        self.set_out_tensor(out_tensor)


class ArmFakeQuantWithMinMaxVarsOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'num_bits': {'type': AttrType.INT, 'default': 8},
                'narrow_range': {'type': AttrType.INT, 'default': 0},
                'min_val': {'type': AttrType.FLOAT, 'required': True},
                'max_val': {'type': AttrType.FLOAT, 'required': True},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmFakeQuantWithMinMaxVarsOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmFakeQuantWithMinMaxVarsOp, attr_dict)
        assert self.check_required(), 'ArmFakeQuantWithMinMaxVarsOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmFakeQuantWithMinMaxVarsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        quant_min = 1 if bool(self.narrow_range) else 0
        quant_max = 2 ** self.num_bits - 1
        from .tf_ops.array_ops import TfFakeQuantWithMinMaxVarsOp
        nudged_min, nudged_max, nudged_scale \
            = TfFakeQuantWithMinMaxVarsOp.nudge(self.min_val, self.max_val, quant_min, quant_max)
        out_tensor = TfFakeQuantWithMinMaxVarsOp.cal_output(
            inputs[0], nudged_min, nudged_max, nudged_scale)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmFakeQuantWithMinMaxVarsOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('num_bits=%d\n' % int(self.num_bits))
            txt_file.write('narrow_range=%s\n' %
                           str(bool(self.narrow_range)).lower())
            txt_file.write('min=%f\n' % float(self.min_val))
            txt_file.write('max=%f\n' % float(self.max_val))
        return ret


class ArmFilterOp(OpHasAxis, OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {-1: 'uint8', None: ['float32', 'float16']}

    @classmethod
    def num_in_ports(cls):
        return -1

    @classmethod
    def attributes(cls):
        return {'axis': {'default': 0},
                'num': {'type': AttrType.INT, 'required': False}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmFilterOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmFilterOp, attr_dict)
        assert self.check_required(), 'ArmFilterOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmFilterOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.num = len(inputs) - 1
        mask = inputs[-1].astype(bool)
        out_tensors = [np.zeros_like(inp) for inp in inputs[:-1]]
        for i, ot in enumerate(inputs[:-1]):
            true_indices = np.where(mask)[0]
            out_tensors[i][mask] = np.take(ot, true_indices, axis=0)
        valid_num = np.array(int(np.sum(mask)), np.int32)
        out_tensors.append(valid_num)
        self.set_out_tensor(out_tensors)

    def write_attrs(self, txt_file):
        ret = super(ArmFilterOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('num=%d\n' % self.num)
        return ret


class ArmFilterBoxesOp(OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16'], 2: 'int32', 3: 'int32', 4: 'int32'}

    @classmethod
    def num_in_ports(cls):
        return 5

    @classmethod
    def attributes(cls):
        return {'min_size': {'type': AttrType.FLOATS, 'required': True, 'default': []},
                'maxnum': {'type': AttrType.INT, 'default': 5000},
                'score_threshold': {'type': AttrType.FLOAT, 'default': -float('inf')}, }

    def __init__(self, graph, attr_dict=None):
        super(ArmFilterBoxesOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmFilterBoxesOp, attr_dict)
        assert self.check_required(), 'ArmFilterBoxesOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmFilterBoxesOp, self).infer_shape()
        inputs = self.get_input_tensors()
        # input/output: boxes, scores, box_num_perClass, label_perclass, total_class_num
        boxes_shape = inputs[0].shape
        assert len(boxes_shape) == 3 and boxes_shape[-1] == 4, \
            'Meets invalid shape(%s) of first input(boxes)!' % str(boxes_shape)
        batch_size, box_num = boxes_shape[:2]
        class_num = inputs[2].shape[1]
        if self.maxnum > box_num:
            self.maxnum = box_num
        out_tensor1 = np.random.ranf(
            size=(batch_size, self.maxnum, 4)).astype(np.float32)  # boxes
        out_tensor2 = np.random.ranf(
            size=(batch_size, self.maxnum)).astype(np.float32)  # scores
        out_tensor3 = np.random.ranf(
            size=(batch_size, class_num)).astype(np.int32)  # box_num_perClass
        out_tensor4 = np.random.ranf(
            size=(batch_size, class_num)).astype(np.int32)  # label_perclass
        out_tensor5 = np.random.ranf(size=(batch_size, 1)).astype(np.int32)  # total_class_num
        out_tensor_list = [out_tensor1, out_tensor2,
                           out_tensor3, out_tensor4, out_tensor5]
        self.set_out_tensor(out_tensor_list)

    def write_attrs(self, txt_file):
        ret = super(ArmFilterBoxesOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('score_threshold=%f\n' % self.score_threshold)
            txt_file.write('min_size=[%s]\n' % num_list_to_string(self.min_size))
            txt_file.write('maxnum=%d\n' % self.maxnum)
        return ret


class ArmFloorOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def __init__(self, graph, attr_dict=None):
        super(ArmFloorOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmFloorOp, attr_dict)
        assert self.check_required(), 'ArmFloorOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmFloorOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.floor(inputs[0])
        self.set_out_tensor(out_tensor)


class ArmTruncOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def __init__(self, graph, attr_dict=None):
        super(ArmTruncOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmTruncOp, attr_dict)
        assert self.check_required(), 'ArmTruncOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmTruncOp, self).infer_shape()
        inputs = self.get_input_tensors()
        torch_input = torch.from_numpy(inputs[0])
        out_tensor = torch.trunc(torch_input).numpy()
        self.set_out_tensor(out_tensor)


class ArmFractionalPoolOp(OpHasMethod, OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def attributes(cls):
        return {'method': {'options': ['AVG', 'MAX'], 'required': True},
                'pooling_ratio': {'type': AttrType.FLOATS, 'required': True},
                'pseudo': {'type': AttrType.BOOL, 'default': False},
                'overlap': {'type': AttrType.BOOL, 'default': False},
                'seed': {'type': AttrType.INT, 'default': 0}
                }

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def __init__(self, graph, attr_dict=None):
        super(ArmFractionalPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmFractionalPoolOp, attr_dict)
        assert self.check_required(), 'ArmFractionalPoolOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmFractionalPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tuple = tf.nn.fractional_avg_pool(
            inputs[0],
            self.pooling_ratio,
            pseudo_random=self.pseudo,
            overlapping=self.overlap,
            seed=self.seed
        )
        out_tensors = [t.numpy() if i == 0 else t.numpy().astype(np.int32) for i, t in enumerate(out_tuple)]
        self.set_out_tensor(out_tensors)

    def write_attrs(self, txt_file):
        ret = super(ArmFractionalPoolOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('pseudo=%s\n' % str(self.pseudo).lower())
            txt_file.write('overlap=%s\n' % str(self.overlap).lower())
            txt_file.write('seed=%d\n' % int(self.seed))
        return ret


class ArmFullyConnectedOp(BaseActivationOp, BaseLinearOp, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8']}

    @classmethod
    def perm_onnx_to_ir(cls):
        return [0, 1]

    @classmethod
    def perm_onnx_to_tf(cls):
        return [1, 0]

    @classmethod
    def perm_ir_to_tf(cls):
        return [1, 0]

    def infer_shape(self):
        super(ArmFullyConnectedOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs[0].shape) == 2, 'The shape of input is invalid in ArmFullyConnectedOp.'

        if np.issubdtype(inputs[0].dtype, np.integer):
            inp = inputs[0].astype(np.float32)
        else:
            inp = inputs[0]
        out_tensor = (tf.matmul(inp,
                                np.transpose(self.weights, axes=type(
                                    self).perm_onnx_to_tf())
                                ) + self.biases).numpy()
        out_tensor = self.cal_activation(out_tensor)
        self.set_out_tensor(out_tensor)


class ArmGatherOp(OpHasAxis, OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {1: 'int32'}

    @classmethod
    def attributes(cls):
        return {'batch_dims': {'type': AttrType.INT, 'default': 0}}

    def __init__(self, graph, attr_dict=None):
        super(ArmGatherOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmGatherOp, attr_dict)
        assert self.check_required(), 'ArmGatherOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmGatherOp, self).infer_shape()
        inputs = self.get_input_tensors()
        indices = inputs[1].tolist()
        ref_shape = inputs[0].shape
        indices = np.array(indices, np.int64)
        negative_axes = indices < 0
        if np.any(negative_axes):
            len_shape = ref_shape[self.axis]
            indices[negative_axes] += len_shape
        indices = indices.tolist()
        out_tensor = tf.gather(inputs[0],
                               indices,
                               axis=self.axis,
                               batch_dims=self.batch_dims).numpy()
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmGatherOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('batch_dims=%d\n' % self.batch_dims)
        return ret


class ArmGatherElementsOp(OpHasAxis, OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {1: 'int32'}

    @classmethod
    def attributes(cls):
        return {'axis': {'default': 0},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmGatherElementsOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmGatherElementsOp, attr_dict)
        assert self.check_required(), 'ArmGatherElementsOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmGatherElementsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        indices = inputs[1]
        from .onnx_ops.array_ops import GatherElementsOp
        indices = GatherElementsOp.make_indices_non_negative(
            indices, inputs[0].shape[self.axis])
        torch_input = torch.from_numpy(inputs[0])
        torch_indices = torch.from_numpy(np.array(indices, np.int64))
        out_tensor = torch.gather(torch_input, self.axis, torch_indices)
        out_tensor = out_tensor.numpy()
        self.set_out_tensor(out_tensor)


class ArmGatherNDOp(OpHasOneOutPort, ArmOp):
    @staticmethod
    def slice_indexing(indices, data, batch_dim, dtype):
        gather_index = tuple(indices)
        try:
            ret = data[(batch_dim,) + gather_index]
        except:
            ret = np.zeros_like(data[batch_dim, 0], dtype)
        return ret

    @classmethod
    def cast_in_ports(cls):
        return {1: 'int32'}

    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def attributes(cls):
        return {'batch_dims': {'type': AttrType.INT, 'default': 0}}

    def __init__(self, graph, attr_dict=None):
        super(ArmGatherNDOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmGatherNDOp, attr_dict)
        assert self.check_required(), 'ArmGatherNDOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmGatherNDOp, self).infer_shape()
        inputs = self.get_input_tensors()

        in_rank = len(inputs[0].shape)
        assert inputs[1].shape[-1] <= in_rank, 'The shape of the input is invalid in ArmGatherNDOp.'
        batch_dims_shape = []
        batch_dims_size = 1
        for i in range(self.batch_dims):
            batch_dims_shape.append(inputs[1].shape[i])
            batch_dims_size *= inputs[1].shape[i]
        output_shape = batch_dims_shape + list(inputs[1].shape)[self.batch_dims:-1] if (inputs[1].shape[-1] == in_rank - self.batch_dims) \
            else batch_dims_shape + list(inputs[1].shape)[self.batch_dims:-1] + list(inputs[0].shape)[self.batch_dims + inputs[1].shape[-1]:]

        out_data = []
        reshaped_indices = inputs[1].reshape(
            batch_dims_size, -1, inputs[1].shape[-1])
        reshaped_data = inputs[0].reshape(
            (batch_dims_size,) + inputs[0].shape[self.batch_dims:])
        for batch_dim in range(reshaped_indices.shape[0]):
            cur_reshaped_indices = reshaped_indices[batch_dim]
            func = partial(ArmGatherNDOp.slice_indexing,
                           data=reshaped_data,
                           batch_dim=batch_dim,
                           dtype=inputs[0].dtype)
            with mp.Pool(mp.cpu_count()) as pool:
                meta_out_data = pool.map(func, cur_reshaped_indices)
            out_data.extend(meta_out_data)
        out_tensor = np.asarray(
            out_data, dtype=inputs[0].dtype).reshape(output_shape)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmGatherNDOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('batch_dims=%d\n' % self.batch_dims)
        return ret


class ArmGemmOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8'], 1: ['float32', 'float16', 'int8', 'uint8'], 2: ['float32', 'float16', 'int32']}

    @classmethod
    def num_in_ports(cls):
        return 3

    @classmethod
    def attributes(cls):
        return {'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                'beta': {'type': AttrType.FLOAT, 'default': 1.0},
                'trans_a': {'type': AttrType.INT, 'default': 0},
                'trans_b': {'type': AttrType.INT, 'default': 0}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmGemmOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmGemmOp, attr_dict)
        assert self.check_required(), 'ArmGemmOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmGemmOp, self).infer_shape()
        inputs = self.get_input_tensors()
        A = inputs[0] if not bool(self.trans_a) else np.transpose(inputs[0])
        B = inputs[1] if not bool(self.trans_b) else np.transpose(inputs[1])
        # alpha * A' * B' + beta * C
        C = inputs[2]
        out_tensor = self.alpha * np.matmul(A, B) + self.beta * C
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmGemmOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('trans_a=%s\n' % str(bool(self.trans_a)).lower())
            txt_file.write('trans_b=%s\n' % str(bool(self.trans_b)).lower())
            txt_file.write('alpha=%f\n' % self.alpha)
            txt_file.write('beta=%f\n' % self.beta)
        return ret


class ArmGenerateProposalsOp(LayoutConcernedOp, OpHasWeights, OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 3

    @classmethod
    def attributes(cls):
        return {'pre_nms_topn': {'type': AttrType.INT, 'default': 6000},
                'post_nms_topn': {'type': AttrType.INT, 'default': 300},
                'min_size': {'type': AttrType.INT, 'default': 16},
                'iou_threshold': {'type': AttrType.FLOAT, 'default': 0.7},
                'image_width': {'type': AttrType.INT, 'default': 600},
                'image_height': {'type': AttrType.INT, 'default': 600},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmGenerateProposalsOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmGenerateProposalsOp, attr_dict)
        assert self.check_required(), 'ArmGenerateProposalsOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmGenerateProposalsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        batch_size = inputs[0].shape[0]
        scores = np.random.ranf(
            (batch_size, self.post_nms_topn)).astype(np.float32)
        boxes = np.random.ranf(
            (batch_size, self.post_nms_topn, 4)).astype(np.float32)
        indices = np.random.ranf(
            (batch_size, self.post_nms_topn, 1)).astype(np.float32)
        box_num = np.random.ranf((batch_size, 1)).astype(np.float32)
        self.set_out_tensor([scores, boxes, indices, box_num])

    def write_attrs(self, txt_file):
        ret = super(ArmGenerateProposalsOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('pre_nms_topn=%d\n' % self.pre_nms_topn)
            txt_file.write('post_nms_topn=%d\n' % self.post_nms_topn)
            txt_file.write('min_size=%d\n' % self.min_size)
            txt_file.write('iou_threshold=%f\n' % self.iou_threshold)
            txt_file.write('image_width=%d\n' % self.image_width)
            txt_file.write('image_height=%d\n' % self.image_height)
        return ret


class ArmGridSampleOp(LayoutConcernedOp, OpHasMethod, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16']}

    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def attributes(cls):
        return {'align_corners': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                'method': {'type': AttrType.STRING, 'default': 'BILINEAR', 'options': ['BILINEAR', 'NEAREST', 'BICUBIC']},
                'padding_mode': {'type': AttrType.STRING, 'default': 'zeros', 'options': ['zeros', 'border', 'reflection']}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmGridSampleOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmGridSampleOp, attr_dict)
        assert self.check_required(), 'ArmGridSampleOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmGridSampleOp, self).infer_shape()
        inputs = self.get_input_tensors()

        inp = np.transpose(inputs[0], (0, 3, 1, 2))
        out_tensor = torch.nn.functional.grid_sample(torch.from_numpy(inp),
                                                     torch.from_numpy(
                                                         inputs[1]),
                                                     mode=self.method.lower(),
                                                     padding_mode=self.padding_mode,
                                                     align_corners=bool(self.align_corners)).numpy()
        out_tensor = np.transpose(out_tensor, (0, 2, 3, 1))
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmGridSampleOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('padding_mode=%s\n' % self.padding_mode.upper())
            txt_file.write('align_corners=%s\n' %
                           str(bool(self.align_corners)).lower())
        return ret


class ArmGroupNormOp(OpHasAxis, OpHasBiases, OpHasWeights, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'epsilon': {'type': AttrType.FLOAT, 'required': True, 'default': 1e-5},
                'group': {'type': AttrType.INT, 'required': True},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmGroupNormOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmGroupNormOp, attr_dict)
        assert self.check_required(), 'ArmGroupNormOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmGroupNormOp, self).infer_shape()
        inputs = self.get_input_tensors()
        src_perm = [i for i in range(len(inputs[0].shape)) if i != self.axis]
        src_perm.insert(1, self.axis)
        inp = np.transpose(inputs[0], src_perm)
        m = torch.nn.GroupNorm(self.group, inp.shape[1], self.epsilon)
        normalized = m(torch.from_numpy(inp)).detach().numpy()
        weight_bias_shape = [-1 if i ==
                             1 else 1 for i in range(len(inputs[0].shape))]
        weights = np.reshape(self.weights, weight_bias_shape)
        biases = np.reshape(self.biases, weight_bias_shape)
        normalized = (normalized * weights + biases).astype(np.float32)
        out_tensor = np.transpose(normalized, Op.cal_inverse_perm(src_perm))
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmGroupNormOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('group=%d\n' % self.group)
            txt_file.write('epsilon=%1.12f\n' % self.epsilon)
        return ret


class ArmGRUv1Op(BaseRnnOp, OpHasBiases, OpHasWeights, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16']}

    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def attributes(cls):
        return {'method': {'default': 'Y', 'options': ['Y', 'H', 'YH']},
                'threshold': {'type': AttrType.FLOAT, 'default': None},
                'activation_alpha': {'type': AttrType.FLOATS, 'default': []},
                'activation_beta': {'type': AttrType.FLOATS, 'default': []}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmGRUv1Op, self).__init__(graph, attr_dict)
        self.update_attributes(ArmGRUv1Op, attr_dict)
        assert self.check_required(), 'ArmGRUv1Op is missing a required parameter.'

    def infer_shape(self):
        super(ArmGRUv1Op, self).infer_shape()
        inputs = self.get_input_tensors()
        input_shape = list(inputs[0].shape)
        seq_out_shape = input_shape[0:-1] + [self.hidden_size]
        state_out_shape = [input_shape[0], self.hidden_size]
        seq_out_tensor = get_random_array(seq_out_shape, 'float32')
        state_out_tensor = get_random_array(state_out_shape, 'float32')
        out_tensor_list = [seq_out_tensor, state_out_tensor] \
            if self.method == 'YH' \
            else ([seq_out_tensor] if self.method == 'Y' else [state_out_tensor])
        self.set_out_tensor(out_tensor_list)

    def write_attrs(self, txt_file):
        ret = super(ArmGRUv1Op, self).write_attrs(txt_file)
        if ret:
            if self.method == 'Y':
                out_sequence = 'H'
            elif self.method == 'H':
                out_sequence = 'Hn'
            else:
                out_sequence = 'H,Hn'
            txt_file.write('out_sequence=[%s]\n' % out_sequence)
            if self.threshold is not None:
                txt_file.write('threshold=%.12f\n' % self.threshold)
            if self.activation_alpha:
                txt_file.write('activation_alpha=[%s]\n' % num_list_to_string(
                    self.activation_alpha))
            if self.activation_beta:
                txt_file.write('activation_beta=[%s]\n' % num_list_to_string(
                    self.activation_beta))
        return ret


class ArmGRUv3Op(BaseRnnOp, OpHasBiases, OpHasWeights, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16']}

    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def attributes(cls):
        return {'method': {'default': 'Y', 'options': ['Y', 'H', 'YH']},
                'threshold': {'type': AttrType.FLOAT, 'default': None},
                'activation_alpha': {'type': AttrType.FLOATS, 'default': []},
                'activation_beta': {'type': AttrType.FLOATS, 'default': []}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmGRUv3Op, self).__init__(graph, attr_dict)
        self.update_attributes(ArmGRUv3Op, attr_dict)
        assert self.check_required(), 'ArmGRUv3Op is missing a required parameter.'

    def infer_shape(self):
        super(ArmGRUv3Op, self).infer_shape()
        inputs = self.get_input_tensors()
        input_shape = list(inputs[0].shape)
        seq_out_shape = input_shape[0:-1] + [self.hidden_size]
        state_out_shape = [input_shape[0], self.hidden_size]
        seq_out_tensor = get_random_array(seq_out_shape, 'float32')
        state_out_tensor = get_random_array(state_out_shape, 'float32')
        out_tensor_list = [seq_out_tensor, state_out_tensor] \
            if self.method == 'YH' \
            else ([seq_out_tensor] if self.method == 'Y' else [state_out_tensor])
        self.set_out_tensor(out_tensor_list)

    def write_attrs(self, txt_file):
        ret = super(ArmGRUv3Op, self).write_attrs(txt_file)
        if ret:
            if self.method == 'Y':
                out_sequence = 'H'
            elif self.method == 'H':
                out_sequence = 'Hn'
            else:
                out_sequence = 'H,Hn'
            txt_file.write('out_sequence=[%s]\n' % out_sequence)
            if self.threshold is not None:
                txt_file.write('threshold=%.12f\n' % self.threshold)
            if self.activation_alpha:
                txt_file.write('activation_alpha=[%s]\n' % num_list_to_string(
                    self.activation_alpha))
            if self.activation_beta:
                txt_file.write('activation_beta=[%s]\n' % num_list_to_string(
                    self.activation_beta))
        return ret


class ArmInputOp(OpHasOneOutPort, InputLikeOp, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 0

    def infer_shape(self, input_tensor=None):
        super(ArmInputOp, self).infer_shape()
        assert input_tensor is not None, 'input tensor is empty in ArmInputOp.'
        out_tensor = input_tensor.copy()
        self.set_out_tensor(out_tensor)


class ArmInstanceNormOp(OpHasBiases, OpHasWeights, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'non_channel_axes': {'type': AttrType.INTS, 'default': None},
                'epsilon': {'type': AttrType.FLOAT, 'default': 1e-5},
                'eps_scale': {'type': AttrType.FLOAT, 'default': None},
                'eps_zp': {'type': AttrType.FLOAT, 'default': None},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmInstanceNormOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmInstanceNormOp, attr_dict)
        assert self.check_required(), 'ArmInstanceNormOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmInstanceNormOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.non_channel_axes is None:
            self.non_channel_axes = list(range(1, len(inputs[0].shape) - 1))
        mean = np.mean(inputs[0], axis=tuple(
            self.non_channel_axes), keepdims=True)
        variance = np.var(inputs[0], axis=tuple(
            self.non_channel_axes), keepdims=True)
        ngamma = 1.0 / ((variance + self.epsilon) ** (.5))
        normalized = (inputs[0] - mean) * ngamma
        out_tensor = (normalized * self.weights +
                      self.biases).astype(np.float32)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmInstanceNormOp, self).write_attrs(txt_file)
        if ret:
            if self.quantize \
                    and self.eps_scale is not None \
                    and self.eps_zp is not None:
                txt_file.write('epsilon=%d\n' % self.epsilon)
                txt_file.write('eps_scale=%1.12f\n' % self.eps_scale)
                txt_file.write('eps_zp=%1.12f\n' % self.eps_zp)
            else:
                txt_file.write('epsilon=%1.12f\n' % self.epsilon)
        return ret


class ArmInTopKOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: 'int32'}

    @classmethod
    def attributes(cls):
        return {'k': {'type': AttrType.INT, 'required': True}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmInTopKOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmInTopKOp, attr_dict)
        assert self.check_required(), 'ArmInTopKOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmInTopKOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.InTopK(
            predictions=inputs[0], targets=inputs[1], k=self.k).numpy()
        out_tensor = out_tensor.astype(np.uint8)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmInTopKOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('k=%d\n' % self.k)
        return ret


class ArmLayerNormOp(OpHasAxis, OpHasBiases, OpHasWeights, OpHasVariableOutPorts, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8']}

    @classmethod
    def attributes(cls):
        return {'epsilon': {'type': AttrType.FLOAT, 'required': True, 'default': 1e-5}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmLayerNormOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmLayerNormOp, attr_dict)
        assert self.check_required(), 'ArmLayerNormOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmLayerNormOp, self).infer_shape()
        inputs = self.get_input_tensors()
        mean = np.mean(inputs[0], axis=tuple(self.axes), keepdims=True)
        variance = np.var(inputs[0], axis=tuple(self.axes), keepdims=True)
        ngamma = 1.0 / ((variance + self.epsilon) ** 0.5)
        normalized = (inputs[0] - mean) * ngamma
        axes = OpHasAxis.make_axes_non_negative(
            self.axes, len(inputs[0].shape))
        weights = OpHasAxis.expand_to(self.weights, axes, len(inputs[0].shape))
        biases = OpHasAxis.expand_to(self.biases, axes, len(inputs[0].shape))
        out_tensor = (normalized * weights + biases).astype(np.float32)
        out_tensors = [out_tensor]
        out_ports = self.get_out_ports()
        if 1 in out_ports or 2 in out_ports:
            out_tensors.append(np.array(mean, np.float32))
            out_tensors.append(np.array(ngamma, np.float32))
        self.set_out_tensor(out_tensors)

    def write_attrs(self, txt_file):
        ret = super(ArmLayerNormOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('epsilon=%1.12f\n' % self.epsilon)
        return ret


class ArmLogOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmLogOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.log(inputs[0])
        self.set_out_tensor(out_tensor)


class ArmLogSoftmaxOp(OpHasAxis, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'axis': {'default': -1}}

    def __init__(self, graph, attr_dict=None):
        super(ArmLogSoftmaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmLogSoftmaxOp, attr_dict)
        assert self.check_required(), 'ArmLogSoftmaxOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmLogSoftmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = torch.log_softmax(
            torch.from_numpy(inputs[0]), dim=self.axis).numpy()
        self.set_out_tensor(out_tensor)


class ArmLogicalOp(LayoutUnawareOp, OpHasMethod, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {}

    @classmethod
    def num_in_ports(cls):
        return 2

    FUNC_MAP = {'EQUAL': np.equal,
                'NOT_EQUAL': np.not_equal,
                'GREATER': np.greater,
                'GREATER_EQUAL': np.greater_equal,
                'LESS': np.less,
                'LESS_EQUAL': np.less_equal,
                'NOT': np.logical_not,
                'AND': np.logical_and,
                'OR': np.logical_or,
                'XOR': np.logical_xor,
                }

    @classmethod
    def attributes(cls):
        return {'method': {'options': ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS', 'GREATER_EQUAL', 'LESS_EQUAL', 'NOT', 'AND', 'OR', 'XOR'], 'required': True},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmLogicalOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmLogSoftmaxOp, attr_dict)
        assert self.check_required(), 'ArmLogSoftmaxOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmLogicalOp, self).infer_shape()
        inputs = self.get_input_tensors()
        logical_func = ArmLogicalOp.FUNC_MAP[self.method]
        out_tensor = logical_func(*inputs).astype(np.uint8)
        self.set_out_tensor(out_tensor)


class ArmLRNOp(OpHasMethod, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'alpha': {'type': AttrType.FLOAT, 'default': 0.0001},
                'beta': {'type': AttrType.FLOAT, 'default': 0.75},
                'bias': {'type': AttrType.FLOAT, 'default': 1.0},
                'size': {'type': AttrType.INT, 'required': True}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmLRNOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmLRNOp, attr_dict)
        assert self.check_required(), 'ArmLRNOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmLRNOp, self).infer_shape()
        inputs = self.get_input_tensors()
        perm1 = [0, len(inputs[0].shape) - 1] + \
            list(range(1, len(inputs[0].shape) - 1))
        perm2 = [0] + list(range(2, len(inputs[0].shape))) + [1]
        inp = np.transpose(inputs[0], axes=perm1)
        lrn = torch.nn.functional.local_response_norm(torch.from_numpy(inp),
                                                      self.size,
                                                      alpha=self.alpha,
                                                      beta=self.beta,
                                                      k=self.bias).numpy()
        out_tensor = np.transpose(lrn, axes=perm2)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmLRNOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('size=%d\n' % self.size)
            txt_file.write('bias=%f\n' % self.bias)
            txt_file.write('alpha=%f\n' % self.alpha)
            txt_file.write('beta=%f\n' % self.beta)
        return ret


class ArmMatMulOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'uint8', 'int16', 'int8'],
                1: ['float32', 'float16', 'uint8', 'int16', 'int8']}

    @classmethod
    def attributes(cls):
        return {'trans_a': {'type': AttrType.INT, 'default': 0},
                'trans_b': {'type': AttrType.INT, 'default': 0},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmMatMulOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmMatMulOp, attr_dict)
        assert self.check_required(), 'ArmMatMulOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'trans_a':
                ret = bool(self.__dict__['_attr'][item].value)
            elif item == 'trans_b':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(ArmMatMulOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ArmMatMulOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs[0].shape) != 4 or len(inputs[1].shape) != 4:
            ERROR('[Parser]: Currently only 4 dim input are supported in ArmMatMulOp.!')
        A = inputs[0] if not bool(self.trans_a) else np.transpose(
            inputs[0], (0, 1, 3, 2))
        B = inputs[1] if not bool(self.trans_b) else np.transpose(
            inputs[1], (0, 1, 3, 2))
        out_tensor = np.matmul(A, B)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmMatMulOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('trans_a=%s\n' % str(self.trans_a).lower())
            txt_file.write('trans_b=%s\n' % str(self.trans_b).lower())
        return ret


class ArmMatMulIntegerOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {0: ['uint8', 'int8'], 1: ['uint8', 'int8']}

    @classmethod
    def attributes(cls):
        return {'a_zero_point': {'type': AttrType.INT, 'default': 0},
                'b_zero_point': {'type': AttrType.INT, 'default': 0}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmMatMulIntegerOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmMatMulIntegerOp, attr_dict)
        assert self.check_required(), 'ArmMatMulIntegerOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmMatMulIntegerOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.matmul(*inputs).astype(np.int32)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmMatMulIntegerOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('a_zero_point=%d\n' % int(self.a_zero_point))
            txt_file.write('b_zero_point=%d\n' % int(self.b_zero_point))
        return ret


class ArmMaxPoolingWithArgMaxOp(OpHasPaddingStrides, OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'flatten_dim': {'type': AttrType.STRING, 'default': 'HWC', 'options': ['NHWC', 'HWC', 'HW', 'NCHW']},
                'ceil_mode': {'type': AttrType.INT, 'default': 0},
                'storage_order': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmMaxPoolingWithArgMaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmMaxPoolingWithArgMaxOp, attr_dict)
        assert self.check_required(), 'ArmMaxPoolingWithArgMaxOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmMaxPoolingWithArgMaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_tensor = torch.from_numpy(
            np.transpose(inputs[0], axes=(0, 3, 1, 2)))
        inp = torch.nn.functional.pad(input_tensor, self.torch_pads, mode='constant', value=0) \
            if self.auto_pad == 'NOTSET' \
            else input_tensor
        out_tensors = torch.nn.functional.max_pool2d(inp,
                                                     kernel_size=(
                                                         self.kernel_shape[0], self.kernel_shape[1]),
                                                     stride=(
                                                         self.strides[0], self.strides[1]),
                                                     dilation=(
                                                         self.dilations[0], self.dilations[1]),
                                                     return_indices=True,
                                                     ceil_mode=bool(
                                                         self.ceil_mode)
                                                     )
        out_tensors = [t.numpy() if i == 0 else t.numpy().astype(np.int32)
                       for i, t in enumerate(out_tensors)]
        out_tensors = [np.transpose(t, axes=(0, 2, 3, 1)) for t in out_tensors]
        self.set_out_tensor(out_tensors)

    def write_attrs(self, txt_file):
        ret = super(ArmMaxPoolingWithArgMaxOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('ceil_mode=%s\n' %
                           str(bool(self.ceil_mode)).lower())
            txt_file.write('flatten_dim=%s\n' % self.flatten_dim)
            txt_file.write('storage_order=%d\n' % self.storage_order)


class ArmMaxRoiPoolOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'spatial': {'type': AttrType.FLOATS, 'default': [1.0, 1.0]},
                'pooled_shape': {'type': AttrType.INTS, 'required': True}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmMaxRoiPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmMaxRoiPoolOp, attr_dict)
        assert self.check_required(), 'ArmMaxRoiPoolOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmMaxRoiPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        roi_num = inputs[1].shape[0]
        channels = inputs[0].shape[-1]
        out_tensor = np.random.ranf(
            (roi_num, *self.pooled_shape, channels)).astype(np.float32)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmMaxRoiPoolOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('pooled_shape=[%s]\n' %
                           list_list_to_string(self.pooled_shape))
            txt_file.write('spatial=[%s]\n' %
                           list_list_to_string(self.spatial))
        return ret


class ArmMaxUnpoolOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: 'int32'}

    @classmethod
    def attributes(cls):
        return {'output_shape': {'type': AttrType.INTS, 'required': True},
                'flatten_dim': {'type': AttrType.STRING, 'default': 'HW', 'options': ['HW', 'NCHW']},
                'storage_order': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmMaxUnpoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmMaxUnpoolOp, attr_dict)
        assert self.check_required(), 'ArmMaxUnpoolOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmMaxUnpoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.random.ranf(self.output_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmMaxUnpoolOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('output_shape=[%s]\n' %
                           list_list_to_string(self.output_shape))
            txt_file.write('flatten_dim=%s\n' % self.flatten_dim)
            txt_file.write('storage_order=%d\n' % self.storage_order)


class ArmMeshgridOp(OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def attributes(cls):
        return {'indexing': {'type': AttrType.STRING, 'options': ['ij', 'xy'], 'default': 'xy'},
                'sparse': {'type': AttrType.INT, 'options': [0, 1], 'default': 0},
                'copy': {'type': AttrType.INT, 'options': [0, 1], 'default': 1}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmMeshgridOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmMeshgridOp, attr_dict)
        assert self.check_required(), 'ArmMeshgridOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmMeshgridOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = tf.meshgrid(*inputs, indexing=self.indexing)
        out_tensors = [t.numpy() for t in out_tensors]
        self.set_out_tensor(out_tensors)

    def write_attrs(self, txt_file):
        ret = super(ArmMeshgridOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('indexing=%s\n' % str(self.indexing))
            txt_file.write('sparse=%s\n' % str(bool(self.sparse)).lower())
            txt_file.write('copy=%s\n' % str(bool(self.copy)).lower())
        return ret


class ArmModOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8', 'int32', 'uint32'], 1: ['float32', 'float16', 'int8', 'uint8', 'int32', 'uint32']}

    @classmethod
    def attributes(cls):
        return {'fmod': {'type': AttrType.INT, 'default': 0}}

    def __init__(self, graph, attr_dict=None):
        super(ArmModOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmModOp, attr_dict)
        assert self.check_required(), 'ArmModOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmModOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if str(inputs[0].dtype) == 'float32' and not bool(self.fmod):
            ERROR(
                '[Parser]: Mod Op(%s) with fmod=0 does not comply with float inputs!' % self.name)
        with np.errstate(divide='ignore'):
            out_tensor = np.mod(
                *inputs) if self.fmod == 0 else np.fmod(*inputs)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmModOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('fmod=%s\n' % str(bool(self.fmod)).lower())
        return ret


class ArmMomentsOp(OpHasMultipleOutPorts, OpHasAxis, ArmOp):
    def infer_shape(self):
        super(ArmMomentsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = tf.nn.moments(
            inputs[0], self.axes, keepdims=self.keepdims)
        out_tensors = [out_tensor.numpy() for out_tensor in out_tensors]
        self.set_out_tensor(out_tensors)


class ArmMVNOp(OpHasOneOutPort, OpHasAxis, ArmOp):
    '''
    (X-EX)/sqrt(E(X-EX)^2)
    '''
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-9},
                'axes': {'default': [0, 2, 3]}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmMVNOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmMVNOp, attr_dict)
        assert self.check_required(), 'ArmMVNOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmMVNOp, self).infer_shape()
        inputs = self.get_input_tensors()
        data_mean = np.mean(inputs[0], axis=tuple(self.axes), keepdims=True)
        data_std = np.std(inputs[0], axis=tuple(self.axes), keepdims=True)
        out_tensor = (inputs[0] - data_mean) / (data_std + self.epsilon)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmMVNOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('epsilon=%1.12f\n' % self.epsilon)
        return ret


class ArmNormalizedMomentsOp(OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 3

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16'], 2: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'counts': {'type': AttrType.FLOAT, 'required': True},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmNormalizedMomentsOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmNormalizedMomentsOp, attr_dict)
        assert self.check_required(), 'ArmNormalizedMomentsOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmNormalizedMomentsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        count = np.array(self.counts).astype(np.float32)
        out_tensors = tf.nn.normalize_moments(counts=count,
                                              mean_ss=inputs[0],
                                              variance_ss=inputs[1],
                                              shift=inputs[2])
        out_tensors = [out_tensor.numpy() for out_tensor in out_tensors]
        self.set_out_tensor(out_tensors)

    def write_attrs(self, txt_file):
        ret = super(ArmNormalizedMomentsOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('counts=%f\n' % self.counts)
        return ret


class ArmNegativeOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmNegativeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = -inputs[0]
        self.set_out_tensor(out_tensor)


class ArmNMSOp(OpHasMethod, OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 4

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: 'int32', 2: 'int32', 3: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'image_width': {'type': AttrType.INT, 'required': True, 'default': 300},
                'image_height': {'type': AttrType.INT, 'required': True, 'default': 300},
                'center_point_box': {'type': AttrType.INT, 'required': True, 'options': [0, 1]},
                'max_box_num': {'type': AttrType.INT, 'default': 5000},
                'method': {'options': ['HARD', 'GAUSSIAN', 'LINEAR'], 'default': 'HARD'},
                'iou_threshold': {'type': AttrType.FLOAT, 'default': 0.6},
                'score_threshold': {'type': AttrType.FLOAT, 'default': -float('inf')},
                'soft_nms_sigma': {'type': AttrType.FLOAT, 'required': False, 'default': 0.0},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmNMSOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmNMSOp, attr_dict)
        assert self.check_required(), 'ArmNMSOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmNMSOp, self).infer_shape()
        # proposal_boxes, box_num_per_class, total_class_num, proposal_scores
        inputs = self.get_input_tensors()
        if len(inputs) != 4:
            ERROR('[Parser]: NMS (%s) inputs number error, not equal to 4!' % self.name)
        batch_size = inputs[0].shape[0]
        num_classes = inputs[1].shape[1]
        out_tensor1 = np.random.ranf(
            size=(batch_size, self.max_box_num, 4)).astype(np.float32)
        out_tensor2 = np.random.ranf(
            size=(batch_size, num_classes)).astype(np.int32)
        out_tensor3 = np.random.ranf(
            size=(batch_size, self.max_box_num)).astype(np.float32)
        out_tensor4 = np.random.ranf(
            size=(batch_size, self.max_box_num)).astype(np.int32)
        out_tensor_list = [out_tensor1, out_tensor2, out_tensor3, out_tensor4]
        self.set_out_tensor(out_tensor_list)

    def write_attrs(self, txt_file):
        ret = super(ArmNMSOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('image_width=%d\n' % self.image_width)
            txt_file.write('image_height=%d\n' % self.image_height)
            txt_file.write('center_point_box=%d\n' % self.center_point_box)
            txt_file.write('max_output_size=%d\n' % self.max_box_num)
            txt_file.write('iou_threshold=%f\n' % self.iou_threshold)
            txt_file.write('score_threshold=%f\n' % self.score_threshold)
            txt_file.write('soft_nms_sigma=%f\n' % self.soft_nms_sigma)
        return ret


class ArmNormalizationOp(OpHasMethod, OpHasAxis, OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'method': {'options': ['L1', 'L2'], 'default': 'L2'},
                'epsilon': {'type': AttrType.FLOAT, 'default': 0.0}}

    def __init__(self, graph, attr_dict=None):
        super(ArmNormalizationOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmNormalizationOp, attr_dict)
        assert self.check_required(), 'ArmNormalizationOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmNormalizationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        ord = 1 if self.method == 'L1' else 2
        if ord == 2:
            out_tensor = tf.math.l2_normalize(inputs[0], self.axes, self.epsilon).numpy()
        else:
            out_tensor = inputs[0] / np.sum(np.abs(inputs[0]), axis=tuple(self.axes), keepdims=True)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmNormalizationOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('epsilon=%.8e\n' % self.epsilon)
        return ret


class ArmOneHotOp(OpHasOneOutPort, OpHasAxis, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: 'int32'}

    @classmethod
    def attributes(cls):
        return {
            'values': {'type': AttrType.TENSOR, 'default': np.array([0, 1], np.float32)},
            'depth': {'type': AttrType.INT, 'required': True},
            'axis': {'type': AttrType.INT, 'default': -1},
        }

    def __init__(self, graph, attr_dict=None):
        super(ArmOneHotOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmOneHotOp, attr_dict)
        assert self.check_required(), 'ArmOneHotOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmOneHotOp, self).infer_shape()
        inputs = self.get_input_tensors()
        indices = inputs[0].astype(np.int64)
        reps = [1] * (len(indices.shape) + 1)
        reps[self.axis] = self.depth
        tiled_indices = np.tile(np.expand_dims(indices, axis=self.axis), reps)
        out_tensor = (np.ones_like(tiled_indices) *
                      self.values[1]).astype(self.values.dtype)
        off_mask = np.logical_and(
            tiled_indices >= -self.depth, tiled_indices < self.depth - 1)
        out_tensor[off_mask] = self.values[0]
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmOneHotOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('values=[%s]\n' %
                           list_list_to_string(self.values.tolist()))
            txt_file.write('depth=%d\n' % self.depth)
        return ret


class ArmOverlapAddOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'uint32', 'uint8', 'int16', 'uint16', 'int8', 'int32']}

    @classmethod
    def attributes(cls):
        return {'frame_step': {'type': AttrType.INT}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmOverlapAddOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmOverlapAddOp, attr_dict)
        assert self.check_required(), 'ArmOverlapAddOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmOverlapAddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs[0].shape) >= 2, 'The rank of ArmOverlapAddOp inp0 must be at least 2., but got %d' % len(
            inputs[0].shape)
        out_tensor = tf.signal.overlap_and_add(
            signal=inputs[0], frame_step=self.frame_step).numpy()
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmOverlapAddOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('frame_step=%d\n' % self.frame_step)
        return ret


class ArmPadOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8']}

    @classmethod
    def attributes(cls):
        return {'pads': {'type': AttrType.INTS, 'required': True},
                'constant_value': {'type': AttrType.FLOAT, 'required': False, 'default': 0},
                'mode': {'type': AttrType.STRING, 'default': 'constant', 'options': ['constant', 'reflect', 'symmetric', 'edge', 'wrap']}
                }

    @staticmethod
    def convert_pads_to_tf(onnx_pads):
        pads = np.reshape(np.array(onnx_pads, np.int32), (2, -1))
        return np.transpose(pads)

    def __init__(self, graph, attr_dict=None):
        super(ArmPadOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmPadOp, attr_dict)
        assert self.check_required(), 'ArmPadOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmPadOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.pad(inputs[0], ArmPadOp.convert_pads_to_tf(
            self.pads), mode=self.mode)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmPadOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('constant_value=%f\n' % self.constant_value)
            txt_file.write('mode=%s\n' % self.mode.upper())
            txt_file.write('pads=[%s]\n' % list_list_to_string(
                ArmPadOp.convert_pads_to_tf(self.pads).tolist()))
        return ret


class ArmPoolingOp(OpHasMethod, OpHasPaddingStrides, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8']}

    @classmethod
    def attributes(cls):
        return {'method': {'options': ['AVG', 'MAX', 'L1', 'L2']},
                'ceil_mode': {'type': AttrType.INT, 'default': 0},
                'count_include_pad': {'type': AttrType.INT, 'default': 0}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmPoolingOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmPoolingOp, attr_dict)
        assert self.check_required(), 'ArmPoolingOp is missing a required parameter.'

    def infer_shape(self):
        '''
        # input_tensor = torch.from_numpy(np.transpose(inputs[0], axes=(0, 3, 1, 2)))
        # input_tensor = torch.nn.functional.pad(input_tensor, self.torch_pads, mode='constant', value=0)
        # if self.method == 'AVG':
        #     out_tensor = torch.nn.functional.avg_pool2d(input_tensor,
        #                                                 kernel_size=(self.kernel_shape[0], self.kernel_shape[1]),
        #                                                 stride=(self.strides[0], self.strides[1]),
        #                                                 count_include_pad=bool(self.count_include_pad),
        #                                                 ceil_mode=bool(self.ceil_mode)
        #                                                 ).numpy()
        # else:
        #     out_tensor = torch.nn.functional.max_pool2d(input_tensor,
        #                                                 kernel_size=(self.kernel_shape[0], self.kernel_shape[1]),
        #                                                 stride=(self.strides[0], self.strides[1]),
        #                                                 dilation=(self.dilations[0], self.dilations[1]),
        #                                                 ceil_mode=bool(self.ceil_mode)
        #                                                 ).numpy()
        # out_tensor = np.transpose(out_tensor, (0, 2, 3, 1))
        # self.set_out_tensor(out_tensor)

        # pool_func = tf.nn.avg_pool if self.method == 'AVG' else tf.nn.max_pool
        # inp = tf.pad(inputs[0], self.tf_pads) if self.auto_pad == 'NOTSET' else inputs[0]
        # out_tensor = pool_func(inp,
        #                        ksize=[1] + self.kernel_shape + [1],
        #                        strides=[1] + self.strides + [1],
        #                        padding='VALID' if self.auto_pad in ('NOTSET', 'VALID') else 'SAME',
        #                        data_format=self.data_format).numpy()
        #
        # if self.method == 'AVG' and self.count_include_pad == 0:
        #     scale_input = inputs[0]
        #     scale_tensor = tf.ones_like(scale_input)
        #     scale_tensor = tf.pad(scale_tensor, self.tf_pads, constant_values=0) if self.auto_pad == 'NOTSET' else scale_tensor
        #     scale = tf.nn.avg_pool(scale_tensor,
        #                            ksize=[1] + self.kernel_shape + [1],
        #                            strides=[1] + self.strides + [1],
        #                            padding='VALID' if self.auto_pad in ('NOTSET', 'VALID') else 'SAME',
        #                            data_format=self.data_format).numpy()
        #     scale_mask = scale * (self.kernel_shape[0] * self.kernel_shape[1])
        #     out_tensor = out_tensor * (self.kernel_shape[0] * self.kernel_shape[1]) / scale_mask
        # self.set_out_tensor(out_tensor)
        '''

        super(ArmPoolingOp, self).infer_shape()
        inputs = self.get_input_tensors()
        batch, spatial_shape, channel = inputs[0].shape[0], inputs[0].shape[1:-
                                                                            1], inputs[0].shape[-1]
        out_shape = BaseOnnxPoolOp.cal_out_shape(spatial_shape,
                                                 self.pads,
                                                 self.strides,
                                                 self.kernel_shape,
                                                 self.auto_pad,
                                                 dilations=self.dilations,
                                                 ceil_mode=self.ceil_mode)
        out_tensor = np.random.ranf(
            [batch] + out_shape + [channel]).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmPoolingOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('ceil_mode=%s\n' %
                           str(bool(self.ceil_mode)).lower())
            if self.method == 'AVG':
                txt_file.write('count_include_pad=%s\n' %
                               str(bool(self.count_include_pad)).lower())
        return ret


class ArmPooling3DOp(OpHasMethod, OpHasPaddingStrides, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8']}

    @classmethod
    def attributes(cls):
        return {'method': {'options': ['AVG', 'MAX', 'L1', 'L2']},
                'ceil_mode': {'type': AttrType.INT, 'default': 0},
                'count_include_pad': {'type': AttrType.INT, 'default': 0}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmPooling3DOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmPooling3DOp, attr_dict)
        assert self.check_required(), 'ArmPooling3DOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmPooling3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        # input_tensor = torch.from_numpy(inputs[0]).permute(0, 4, 1, 2, 3)
        # input_tensor = torch.nn.functional.pad(
        #     input_tensor, self.torch_pads, mode='constant', value=0)
        # if self.method == 'AVG':
        #     # NOTE: torch.nn.functional.avg_pool3d doesn't support dilations
        #     out_tensor = torch.nn.functional.avg_pool3d(input_tensor,
        #                                                 kernel_size=tuple(
        #                                                     self.kernel_shape),
        #                                                 stride=tuple(
        #                                                     self.strides),
        #                                                 ceil_mode=bool(
        #                                                     self.ceil_mode),
        #                                                 count_include_pad=bool(
        #                                                     self.count_include_pad)
        #                                                 ).numpy()
        # else:
        #     out_tensor = torch.nn.functional.max_pool3d(input_tensor,
        #                                                 kernel_size=tuple(
        #                                                     self.kernel_shape),
        #                                                 stride=tuple(
        #                                                     self.strides),
        #                                                 dilation=tuple(
        #                                                     self.dilations),
        #                                                 ceil_mode=bool(
        #                                                     self.ceil_mode)
        #                                                 ).numpy()
        # out_tensor = np.transpose(out_tensor, [0, 2, 3, 4, 1])
        input_shape = inputs[0].shape
        batch, spatial_shape, channel = input_shape[0], input_shape[1:-1], input_shape[-1]
        out_shape = BaseOnnxPoolOp.cal_out_shape(spatial_shape,
                                                 self.pads,
                                                 self.strides,
                                                 self.kernel_shape,
                                                 self.auto_pad,
                                                 dilations=self.dilations,
                                                 ceil_mode=self.ceil_mode)
        out_tensor = np.random.ranf(
            [batch] + out_shape + [channel]).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmPooling3DOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('ceil_mode=%s\n' %
                           str(bool(self.ceil_mode)).lower())
            if self.method == 'AVG':
                txt_file.write('count_include_pad=%s\n' %
                               str(bool(self.count_include_pad)).lower())
        return ret


class ArmPostNMS1Op(OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 3

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: 'int32', 2: 'int32'}

    @classmethod
    def attributes(cls):
        return {'image_width': {'type': AttrType.INT, 'default': 600},
                'image_height': {'type': AttrType.INT, 'default': 600},
                'proposal_cnt': {'type': AttrType.INT, 'default': 100}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmPostNMS1Op, self).__init__(graph, attr_dict)
        self.update_attributes(ArmPostNMS1Op, attr_dict)
        assert self.check_required(), 'ArmPostNMS1Op is missing a required parameter.'

    def infer_shape(self):
        super(ArmPostNMS1Op, self).infer_shape()
        inputs = self.get_input_tensors()
        batch_size = inputs[0].shape[0]
        out_tensor1 = np.random.ranf(
            size=(batch_size, self.proposal_cnt, 4)).astype(np.float32)
        out_tensor2 = np.random.ranf(
            size=(batch_size, self.proposal_cnt, 4)).astype(np.float32)
        self.set_out_tensor([out_tensor1, out_tensor2])

    def write_attrs(self, txt_file):
        ret = super(ArmPostNMS1Op, self).write_attrs(txt_file)
        if ret:
            txt_file.write('image_width=%d\n' % self.image_width)
            txt_file.write('image_height=%d\n' % self.image_height)
            txt_file.write('proposal_cnt=%d\n' % self.proposal_cnt)
        return ret


class ArmPowOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16', 'int32']}

    def __init__(self, graph, attr_dict=None):
        super(ArmPowOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmPowOp, attr_dict)
        assert self.check_required(), 'ArmPowOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmPowOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.pow(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class ArmPreprocessOp(OpHasVariableOutPorts, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return -1

    def infer_shape(self):
        super(ArmPreprocessOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = [inp.copy() for inp in inputs]
        self.set_out_tensor(out_tensors)


class ArmProposalOp(OpHasAnchors, OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'score_threshold': {'type': AttrType.FLOAT, 'default': 0.45},
                'width': {'type': AttrType.INT, 'default': 600},
                'height': {'type': AttrType.INT, 'default': 600},
                'scale_anchor': {'type': AttrType.FLOATS, 'default': [0.25, 0.5, 1.0, 2.0]},
                'max_box_num': {'type': AttrType.INT, 'default': 5000},
                'class_num': {'type': AttrType.INT, 'default': 91}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmProposalOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmProposalOp, attr_dict)
        assert self.check_required(), 'ArmProposalOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmProposalOp, self).infer_shape()
        inputs = self.get_input_tensors()
        batch_size = inputs[0].shape[0]
        out_tensor1 = np.random.ranf(
            size=(batch_size, self.max_box_num)).astype(np.float32)
        out_tensor2 = np.random.ranf(
            size=(batch_size, self.max_box_num, 4)).astype(np.float32)
        out_tensor3 = np.random.ranf(
            size=(batch_size, self.class_num)).astype(np.int32)
        out_tensor4 = np.random.ranf(size=(batch_size, 1)).astype(np.int32)
        self.set_out_tensor(
            [out_tensor1, out_tensor2, out_tensor3, out_tensor4])

    def write_attrs(self, txt_file):
        ret = super(ArmProposalOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('width=%d\n' % self.width)
            txt_file.write('height=%d\n' % self.height)
            txt_file.write('score_threshold=%f\n' % self.score_threshold)
            txt_file.write('scale_anchor=[%s]\n' %
                           list_list_to_string(self.scale_anchor))
        return ret


class ArmPyramidROIAlignOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'resize_width': {'type': AttrType.INT, 'required': True},
                'resize_height': {'type': AttrType.INT, 'required': True},
                'image_width': {'type': AttrType.INT, 'required': True},
                'image_height': {'type': AttrType.INT, 'required': True},
                'sample_ratio': {'type': AttrType.INTS, 'default': [0, 0]},
                'coordinate_transformation_mode': {'type': AttrType.STRING, 'default': 'output_half_pixel',
                                                   'options': ['half_pixel', 'output_half_pixel']},
                'spatial_scale': {'type': AttrType.FLOATS, 'default': [1.0, 1.0, 1.0, 1.0]},
                'proposal_normalized': {'type': AttrType.BOOL, 'default': True},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmPyramidROIAlignOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmPyramidROIAlignOp, attr_dict)
        assert self.check_required(), 'ArmPyramidROIAlignOp is missing a required parameter.'

    @staticmethod
    def setup_scales(feature_heights, image_height):
        scales = []
        for height in feature_heights:
            approx_scale = float(height) / float(image_height)
            scale = 2 ** np.log2(approx_scale).round()
            scales.append(scale)
        return scales

    def infer_shape(self):
        super(ArmPyramidROIAlignOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 5, 'Inputs number of ArmPyramidROIAlignOp should be equal to 5!'
        if not self.spatial_scale:
            feature_heights = [inp.shape[1] for inp in inputs[1:]]
            self.spatial_scale = self.setup_scales(feature_heights, self.image_height)
        roi_num = inputs[0].shape[1]
        channels = inputs[1].shape[-1]
        out_tensor = np.random.ranf((roi_num,
                                     self.resize_height,
                                     self.resize_width,
                                     channels)).astype(np.float32)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmPyramidROIAlignOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('resize_width=%d\n' % self.resize_width)
            txt_file.write('resize_height=%d\n' % self.resize_height)
            txt_file.write('image_width=%d\n' % self.image_width)
            txt_file.write('image_height=%d\n' % self.image_height)
            txt_file.write('spatial_scale_value=[%s]\n' %
                           list_list_to_string(self.spatial_scale))
            txt_file.write('sample=[%s]\n' %
                           list_list_to_string(self.sample_ratio))
            txt_file.write('coordinate_transformation_mode=%s\n' %
                           self.coordinate_transformation_mode.upper())
            txt_file.write('proposal_normalized=%s\n' % str(self.proposal_normalized).lower())
        return ret


class ArmQuantizeOp(OpHasAxis, BaseQuantizeDequantizeOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'to_dtype': {'type': AttrType.STRING,
                             'required': True,
                             'options': ['int8', 'uint8', 'int32', 'uint32']}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmQuantizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmQuantizeOp, attr_dict)
        assert self.check_required(), 'ArmQuantizeOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmQuantizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        zp_min = np.iinfo(self.zero_point.dtype).min
        zp_max = np.iinfo(self.zero_point.dtype).max
        if self.axis is None:
            out_tensor = np.clip(
                (inputs[0] / self.scale + self.zero_point), zp_min, zp_max).astype(self.to_dtype)
        else:
            expand_len = len(inputs[0].shape) - self.axis - 1
            scale = np.reshape(self.scale, list(
                self.scale.shape) + [1] * expand_len)
            zero_point = np.reshape(self.zero_point, list(
                self.zero_point.shape) + [1] * expand_len)
            out_tensor = np.clip(
                (inputs[0] / scale + zero_point), zp_min, zp_max).astype(self.to_dtype)

        self.set_out_tensor(out_tensor)


class ArmReduceOp(OpHasMethod, OpHasAxis, OpHasOneOutPort, ArmOp):
    FUNC_MAP = {
        'ALL': lambda x, y, z: np.all(x, axis=y, keepdims=z),
        'ANY': lambda x, y, z: np.any(x, axis=y, keepdims=z),
        'MEAN': lambda x, y, z: np.mean(x, axis=y, keepdims=z),
        'MIN': lambda x, y, z: np.min(x, axis=y, keepdims=z),
        'MAX': lambda x, y, z: np.max(x, axis=y, keepdims=z),
        'PROD': lambda x, y, z: np.prod(x, axis=y, keepdims=z),
        'SUM': lambda x, y, z: np.sum(x, axis=y, keepdims=z),
        'L1': lambda x, y, z: np.sum(np.abs(x), axis=y, keepdims=z),
        'L2': lambda x, y, z: np.sqrt(np.sum(np.square(x), axis=y, keepdims=z)),
        'VARIANCE': lambda x, y, z: np.var(x, axis=y, keepdims=z),
        'UNBIASED_VARIANCE': lambda x, y, z: np.var(x, axis=y, keepdims=z, ddof=1),
    }

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8']}

    @classmethod
    def attributes(cls):
        return {'method': {'options': ['ALL', 'ANY', 'MEAN', 'MIN', 'MAX', 'SUM', 'PROD', 'L1', 'L2', 'VARIANCE', 'UNBIASED_VARIANCE']}}

    def __init__(self, graph, attr_dict=None):
        super(ArmReduceOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmReduceOp, attr_dict)
        assert self.check_required(), 'ArmReduceOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmReduceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = ArmReduceOp.FUNC_MAP[self.method](
            inputs[0], tuple(self.axes), bool(self.keepdims))
        if out_tensor.dtype == bool:
            out_tensor = out_tensor.astype(np.uint8)
        self.set_out_tensor(out_tensor)


class ArmReciprocalOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmReciprocalOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.reciprocal(inputs[0]).astype(np.float32)
        self.set_out_tensor(out_tensor)


class ArmRefineDetDetectionOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 5

    @classmethod
    def attributes(cls):
        return {'obj_thresh': {'type': AttrType.FLOAT, 'default': 0.1, 'required': True},
                'conf_thresh': {'type': AttrType.FLOAT, 'default': 0.1, 'required': True},
                'pre_nms_topk': {'type': AttrType.INT, 'default': 1000, 'required': True},
                'post_nms_topk': {'type': AttrType.INT, 'default': 200, 'required': True},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmRefineDetDetectionOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmRefineDetDetectionOp, attr_dict)
        assert self.check_required(), 'ArmRefineDetDetectionOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmRefineDetDetectionOp, self).infer_shape()
        inputs = self.get_input_tensors()
        batch = inputs[0].shape[0]
        out_tensor = np.random.ranf(
            (batch, self.post_nms_topk, 4)).astype(np.float32)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmRefineDetDetectionOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('obj_thresh=%f\n' % self.obj_thresh)
            txt_file.write('conf_thresh=%f\n' % self.conf_thresh)
            txt_file.write('pre_nms_topk=%d\n' % self.pre_nms_topk)
            txt_file.write('post_nms_topk=%d\n' % self.post_nms_topk)
        return ret


class ArmRegionOp(OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def attributes(cls):
        return {'anchors': {'type': AttrType.FLOATS,
                            'default': [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]},
                'max_box_num': {'type': AttrType.INT, 'default': 5000},
                'class_num': {'type': AttrType.INT, 'default': 20},
                'grid_width': {'type': AttrType.INT, 'default': 13},
                'grid_height': {'type': AttrType.INT, 'default': 13},
                'box_per_grid': {'type': AttrType.INT, 'default': 5},
                'obj_threshold': {'type': AttrType.FLOAT, 'default': 0.3},
                'grid_compensate': {'type': AttrType.INT, 'required': False, 'default': 1}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmRegionOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmRegionOp, attr_dict)
        assert self.check_required(), 'ArmRegionOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmRegionOp, self).infer_shape()
        inputs = self.get_input_tensors()
        batch_size = inputs[0].shape[0]
        out_tensor1 = np.random.ranf(
            size=(batch_size, self.max_box_num)).astype(np.float32)
        out_tensor2 = np.random.ranf(
            size=(batch_size, self.max_box_num, 4)).astype(np.float32)
        out_tensor3 = np.random.ranf(
            size=(batch_size, self.class_num)).astype(np.int32)
        out_tensor4 = np.random.ranf(
            size=(batch_size, self.class_num)).astype(np.int32)
        out_tensor5 = np.random.ranf(size=(batch_size, 1)).astype(np.int32)
        out_tensor_list = [out_tensor1, out_tensor2,
                           out_tensor3, out_tensor4, out_tensor5]
        self.set_out_tensor(out_tensor_list)

    def write_attrs(self, txt_file):
        ret = super(ArmRegionOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('grid_width=%d\n' % self.grid_width)
            txt_file.write('grid_height=%d\n' % self.grid_height)
            txt_file.write('box_per_grid=%d\n' % self.box_per_grid)
            txt_file.write('max_box_num=%d\n' % self.max_box_num)
            txt_file.write('class_num=%d\n' % self.class_num)
            txt_file.write('obj_thresh=%f\n' % self.obj_threshold)
            txt_file.write('anchors=[%s]\n' %
                           list_list_to_string(self.anchors))
            txt_file.write('grid_compensate=%s\n' %
                           str(self.grid_compensate).lower())
        return ret


class ArmRegionFuseOp(OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 10

    @classmethod
    def cast_in_ports(cls):
        return {k: ['float32', 'float16'] if k < 4 else 'int32' for k in range(ArmRegionFuseOp.num_in_ports())}

    @classmethod
    def attributes(cls):
        return {'class_num': {'type': AttrType.INT, 'required': True}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmRegionFuseOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmRegionFuseOp, attr_dict)
        assert self.check_required(), 'ArmRegionFuseOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmRegionFuseOp, self).infer_shape()
        inputs = self.get_input_tensors()
        batch_size = inputs[0].shape[0]
        out1 = np.random.ranf((batch_size, 10000)).astype(np.float32)
        out2 = np.random.ranf((batch_size, 10000, 4)).astype(np.float32)
        out3 = np.random.ranf((batch_size, self.class_num)).astype(np.int32)
        out4 = np.random.ranf((batch_size, self.class_num)).astype(np.int32)
        out5 = np.random.ranf((batch_size, 1)).astype(np.int32)
        self.set_out_tensor([out1, out2, out3, out4, out5])

    def write_attrs(self, txt_file):
        ret = super(ArmRegionFuseOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('class_num=%d\n' % self.class_num)
        return ret


class ArmRepeatOp(OpHasAxis, OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def attributes(cls):
        return {'max_dim': {'type': AttrType.INT, 'default': 1000}}

    def __init__(self, graph, attr_dict=None):
        super(ArmRepeatOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmRepeatOp, attr_dict)
        assert self.check_required(), 'ArmRepeatOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmRepeatOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.repeat(*inputs, axis=self.axis)
        out_shape = list(out_tensor.shape)
        if self.axis is not None and out_shape[self.axis] > self.max_dim:
            obj = tuple(slice(0, e if i != self.axis else self.max_dim)
                        for (i, e) in enumerate(out_shape))
            out_tensor = out_tensor[obj]
        elif self.axis is not None and out_shape[self.axis] < self.max_dim:
            shape_diff = self.max_dim - out_shape[self.axis]
            zeros_shape = copy.deepcopy(out_shape)
            zeros_shape[self.axis] = shape_diff
            zeros = np.zeros(zeros_shape, inputs[0].dtype)
            out_tensor = np.concatenate([out_tensor, zeros], axis=self.axis)
        self.set_out_tensor(out_tensor)


class ArmReshapeOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'dim': {'type': AttrType.INTS, 'required': True}}

    def __init__(self, graph, attr_dict=None):
        super(ArmReshapeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmReshapeOp, attr_dict)
        assert self.check_required(), 'ArmReshapeOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmReshapeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.reshape(inputs[0], self.dim)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmReshapeOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('shape=[%s]\n' % num_list_to_string(self.dim))
        return ret


class ArmResizeOp(OpHasMethod, OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'factors': {'type': AttrType.FLOATS, 'default': [1.0, 1.0]},
                'sizes': {'type': AttrType.INTS, 'default': None},
                'method': {'options': ['NEAREST', 'BILINEAR'], 'default': 'BILINEAR'},
                'mode': {'type': AttrType.STRING,
                         'default': 'half_pixel',
                         'options': ['half_pixel', 'half_pixel_symmetric', 'align_corners', 'asymmetric', 'pytorch_half_pixel', 'tf_half_pixel_for_nn']},
                'nearest_mode': {'type': AttrType.STRING,
                                 'default': 'round_prefer_floor',
                                 'options': ['simple', 'round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil']},
                'antialias': {'type': AttrType.BOOL, 'default': False},
                'exclude_outside': {'type': AttrType.BOOL, 'default': False},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmResizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmResizeOp, attr_dict)
        assert self.check_required(), 'ArmResizeOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmResizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.sizes:
            size = self.sizes
        else:
            spatial_shape = inputs[0].shape[1:-1]
            size = np.floor(np.array(spatial_shape) *
                            np.array(self.factors)).astype(np.int32).tolist()
        perm = [0, len(inputs[0].shape) - 1] + \
            list(range(1, len(inputs[0].shape) - 1))
        inverse_perm = Op.cal_inverse_perm(perm)
        # torch.nn.functional.interpolate doesn't support int(not implemented)
        out_tensor = torch.nn.functional.interpolate(torch.from_numpy(np.transpose(inputs[0].astype(np.float32), perm)),
                                                     size=tuple(size),
                                                     mode='nearest'
                                                     ).numpy()
        out_tensor = np.transpose(out_tensor, inverse_perm)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmResizeOp, self).write_attrs(txt_file)
        if ret:
            if self.sizes:
                input_shape = self.get_input_shapes()[0]
                size = [input_shape[0]] + self.sizes + [input_shape[-1]]
                txt_file.write('size=[%s]\n' % list_list_to_string(size))
            else:
                txt_file.write('ratio_x=%.8f\n' % self.factors[-1])
                txt_file.write('ratio_y=%.8f\n' % self.factors[-2])
                if len(self.factors) == 3:
                    txt_file.write('ratio_z=%.8f\n' % self.factors[-3])
            txt_file.write('mode=%s\n' % self.mode.upper())
            txt_file.write('antialias=%s\n' % str(bool(self.antialias)).lower())
            txt_file.write('exclude_outside=%s\n' % str(bool(self.exclude_outside)).lower())
            if self.method.upper() == 'NEAREST':
                txt_file.write('nearest_mode=%s\n' % self.nearest_mode.upper())
        return ret


class ArmReverseSequenceOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {0: 'float32', 1: 'int32'}

    @classmethod
    def attributes(cls):
        return {'batch_axis': {'type': AttrType.INT, 'required': False, 'default': 0},
                'time_axis': {'type': AttrType.INT, 'required': False, 'default': 1}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmReverseSequenceOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmReverseSequenceOp, attr_dict)
        assert self.check_required(), 'ArmReverseSequenceOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmReverseSequenceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs) != 2:
            ERROR('[Parser]: Invalid inputs number of ReverseSequence (%s)!' % self.name)
        out_tensor = tf.reverse(inputs[0], axis=np.array(
            [self.time_axis], np.int64)).numpy()
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmReverseSequenceOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('batch_axis=%d\n' % self.batch_axis)
            txt_file.write('time_axis=%d\n' % self.time_axis)
        return ret


class ArmRMSNormOp(OpHasAxis, OpHasWeights, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'epsilon': {'type': AttrType.FLOAT, 'required': True, 'default': 1e-5}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmRMSNormOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmRMSNormOp, attr_dict)
        assert self.check_required(), 'ArmRMSNormOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmRMSNormOp, self).infer_shape()
        inputs = self.get_input_tensors()
        square = np.power(inputs[0], 2)
        mean = np.mean(square, axis=tuple(self.axes), keepdims=True)
        normalized = inputs[0] / np.sqrt(mean + self.epsilon)
        axes = OpHasAxis.make_axes_non_negative(
            self.axes, len(inputs[0].shape))
        weights = OpHasAxis.expand_to(self.weights, axes, len(inputs[0].shape))
        out_tensor = (normalized * weights).astype(np.float32)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmRMSNormOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('epsilon=%1.12f\n' % self.epsilon)
        return ret


class ArmRoiAlignOp(OpHasMethod, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16']}

    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def attributes(cls):
        return {'coordinate_transformation_mode': {'type': AttrType.STRING, 'default': 'half_pixel', 'options': ['half_pixel', 'output_half_pixel']},
                'spatial_scale': {'type': AttrType.FLOATS, 'default': [1.0, 1.0]},
                'pooled_shape': {'type': AttrType.INTS, 'required': True},
                'method': {'default': 'AVG'},
                'sample_ratio': {'type': AttrType.INTS, 'default': [0, 0]},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmRoiAlignOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmRoiAlignOp, attr_dict)
        assert self.check_required(), 'ArmRoiAlignOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmRoiAlignOp, self).infer_shape()
        inputs = self.get_input_tensors()
        roi_num = inputs[1].shape[0]
        channels = inputs[0].shape[-1]
        out_tensor = np.random.ranf(
            (roi_num, *self.pooled_shape, channels)).astype(np.float32)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmRoiAlignOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('pooled_shape=[%s]\n' %
                           list_list_to_string(self.pooled_shape))
            txt_file.write(
                'spatial_scale_value=[%s]\n' % list_list_to_string(self.spatial_scale))
            txt_file.write('sample=[%s]\n' %
                           list_list_to_string(self.sample_ratio))
            txt_file.write('coordinate_transformation_mode=%s\n' %
                           self.coordinate_transformation_mode.upper())
        return ret


class ArmRoundOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmRoundOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.round(inputs[0]).astype(np.float32)
        self.set_out_tensor(out_tensor)


class ArmRsqrtOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmRsqrtOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = np.sqrt(inputs[0])
        self.set_out_tensor(out_tensors)


class ArmScatterNDOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 3

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: 'int32', 2: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'reduction': {'type': AttrType.STRING, 'options': ['NONE', 'MUL', 'ADD', 'MAX', 'MIN'], 'default': 'NONE'}}

    def __init__(self, graph, attr_dict=None):
        super(ArmScatterNDOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmScatterNDOp, attr_dict)
        assert self.check_required(), 'ArmScatterNDOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmScatterNDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        data, indices, updates = inputs
        from .onnx_ops.array_ops import ScatterNDOp
        out_tensor = ScatterNDOp.scatternd(data, indices, updates, reduction=self.reduction)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmScatterNDOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('reduction=%s\n' % self.reduction)
        return ret


class ArmScatterElementsOp(OpHasOneOutPort, OpHasAxis, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 3

    @classmethod
    def cast_in_ports(cls):
        return {1: 'int32'}

    @classmethod
    def attributes(cls):
        return {'reduction': {'type': AttrType.STRING, 'options': ['NONE', 'MUL', 'ADD', 'MAX', 'MIN'], 'default': 'NONE'}}

    def __init__(self, graph, attr_dict=None):
        super(ArmScatterElementsOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmScatterElementsOp, attr_dict)
        assert self.check_required(), 'ArmScatterElementsOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmScatterElementsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        data, indices, updates = inputs
        from .onnx_ops.array_ops import GatherElementsOp
        indices = GatherElementsOp.make_indices_non_negative(
            indices, inputs[0].shape[self.axis])
        data_torch = torch.from_numpy(np.array(data))
        index_torch = torch.from_numpy(np.array(indices).astype(np.int64))
        update_torch = torch.from_numpy(updates)
        if self.reduction == 'NONE':
            out_tensor = torch.Tensor.scatter_(
                data_torch, src=update_torch, dim=self.axis, index=index_torch).numpy()
        else:
            reduction_map = {'MUL': 'prod', 'ADD': 'sum', 'MAX': 'amax', 'MIN': 'amin'}
            assert self.reduction in reduction_map, 'Meets invalid reduction %s in infer_shape of ArmScatterElementsOp!' % self.reduction
            out_tensor = torch.Tensor.scatter_reduce(
                data_torch, src=update_torch, dim=self.axis, index=index_torch, reduce=reduction_map[self.reduction]).numpy()
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmScatterElementsOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('reduction=%s\n' % self.reduction)
        return ret


class ArmSegmentReduceOp(OpHasMethod, OpHasOneOutPort, ArmOp):
    FUNC_MAP = {'SUM': tf.math.segment_sum,
                'PROD': tf.math.segment_prod,
                'MIN': tf.math.segment_min,
                'MAX': tf.math.segment_max,
                'MEAN': tf.math.segment_mean,
                }

    @classmethod
    def attributes(cls):
        return {'method': {'options': ['SUM', 'PROD', 'MIN', 'MAX', 'MEAN'], 'default': 'SUM'}}

    @classmethod
    def cast_in_ports(cls):
        return {1: 'int32'}

    @classmethod
    def num_in_ports(cls):
        return 2

    def __init__(self, graph, attr_dict=None):
        super(ArmSegmentReduceOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmSegmentReduceOp, attr_dict)
        assert self.check_required(), 'ArmSegmentReduceOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmSegmentReduceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = ArmSegmentReduceOp.FUNC_MAP[self.method](*inputs).numpy()
        self.set_out_tensor(out_tensor)


class ArmSignOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmSignOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.sign(inputs[0])
        self.set_out_tensor(out_tensor)


class ArmSineOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmSineOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.sin(inputs[0])
        self.set_out_tensor(out_tensor)


class ArmSinhOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmSinhOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.sinh(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class ArmSliceOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'starts': {'type': AttrType.INTS, 'required': True},
                'ends': {'type': AttrType.INTS, 'required': True},
                'steps': {'type': AttrType.INTS, 'required': True}}

    def __init__(self, graph, attr_dict=None):
        super(ArmSliceOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmSliceOp, attr_dict)
        assert self.check_required(), 'ArmSliceOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmSliceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        obj = tuple(slice(s, None if (p < 0 and e < 0) else e, p)
                    for s, e, p in zip(self.starts, self.ends, self.steps))
        out_tensor = inputs[0][obj]
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmSliceOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('begin=[%s]\n' % list_list_to_string(self.starts))
            txt_file.write('end=[%s]\n' % list_list_to_string(self.ends))
            txt_file.write('strides=[%s]\n' % list_list_to_string(self.steps))
        return ret


class ArmSoftmaxOp(OpHasAxis, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8']}

    @classmethod
    def attributes(cls):
        return {'axis': {'default': -1}}

    def __init__(self, graph, attr_dict=None):
        super(ArmSoftmaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmSoftmaxOp, attr_dict)
        assert self.check_required(), 'ArmSoftmaxOp is missing a required parameter.'

    def infer_shape(self, input_tensor: np.ndarray = None):
        super(ArmSoftmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.softmax(inputs[0].astype(np.float32), axis=self.axis).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))


class ArmSpaceToBatchOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'block_size_x': {'type': AttrType.INT, 'default': 2},
                'block_size_y': {'type': AttrType.INT, 'default': 2},
                'pads': {'type': AttrType.INTS, 'default': [0, 0, 0, 0], 'required': True}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmSpaceToBatchOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmSpaceToBatchOp, attr_dict)
        assert self.check_required(), 'ArmSpaceToBatchOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            pad_size = len(self._attr['pads'].value)
            if item == 'pad_top':
                if pad_size in (4, 6):
                    ret = self.pads[0]
                elif pad_size == 8:
                    ret = self.pads[1]
                else:
                    ERROR('[Parser]: Node(%s) pads size not supported!' %
                          self.name)
            elif item == 'pad_bottom':
                if pad_size == 4:
                    ret = self.pads[2]
                elif pad_size == 6:
                    ret = self.pads[3]
                elif pad_size == 8:
                    ret = self.pads[5]
                else:
                    ERROR('[Parser]: Node(%s) pads size not supported!' %
                          self.name)
            elif item == 'pad_left':
                if pad_size in (4, 6):
                    ret = self.pads[1]
                elif pad_size == 8:
                    ret = self.pads[2]
                else:
                    ERROR('[Parser]: Node(%s) pads size not supported!' %
                          self.name)
            elif item == 'pad_right':
                if pad_size == 4:
                    ret = self.pads[3]
                elif pad_size == 6:
                    ret = self.pads[4]
                elif pad_size == 8:
                    ret = self.pads[6]
                else:
                    ERROR('[Parser]: Node(%s) pads size not supported!' %
                          self.name)
        except:
            ret = None
        if ret is None:
            ret = super(ArmSpaceToBatchOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ArmSpaceToBatchOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.space_to_batch_nd(inputs[0],
                                          block_shape=np.array(
                                              [self.block_size_y, self.block_size_x], dtype=np.int64),
                                          paddings=OpHasPaddingStrides.onnx_to_tf(self.pads)[1:3, :]).numpy()
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmSpaceToBatchOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('block_size_x=%d\n' % self.block_size_x)
            txt_file.write('block_size_y=%d\n' % self.block_size_y)
            txt_file.write('pad_left=%d\n' % self.pad_left)
            txt_file.write('pad_right=%d\n' % self.pad_right)
            txt_file.write('pad_top=%d\n' % self.pad_top)
            txt_file.write('pad_bottom=%d\n' % self.pad_bottom)
        return ret


class ArmSpaceToDepthOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'blocksize': {'type': AttrType.INT, 'required': True}}

    def __init__(self, graph, attr_dict=None):
        super(ArmSpaceToDepthOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmSpaceToDepthOp, attr_dict)
        assert self.check_required(), 'ArmSpaceToDepthOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmSpaceToDepthOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.blocksize == 1:
            out_tensor = inputs[0]
        else:
            out_tensor = tf.nn.space_to_depth(inputs[0], self.blocksize).numpy()
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmSpaceToDepthOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('block_size_x=%d\n' % self.blocksize)
            txt_file.write('block_size_y=%d\n' % self.blocksize)
        return ret


class ArmSplitOp(OpHasAxis, OpHasMultipleOutPorts, ArmOp):

    @classmethod
    def attributes(cls):
        return {'split': {'type': AttrType.INTS, 'required': True}}

    def __init__(self, graph, attr_dict=None):
        super(ArmSplitOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmSplitOp, attr_dict)
        assert self.check_required(), 'ArmSplitOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'splits':
                ret = self.__dict__['_attr'][item].value if self.__dict__[
                    '_attr'][item].value is not None else []
        except:
            ret = None
        if ret is None:
            ret = super(ArmSplitOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ArmSplitOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = tf.split(inputs[0], np.array(self.split), axis=self.axis)
        out_tensors = [o.numpy() for o in out_tensors]
        self.set_out_tensor(out_tensors)

    def write_attrs(self, txt_file):
        ret = super(ArmSplitOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('splits=[%s]\n' % list_list_to_string(self.split))
        return ret


class ArmSqrtOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmSqrtOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = np.sqrt(inputs[0])
        self.set_out_tensor(out_tensors)


class ArmSquareOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'int8', 'uint8']}

    def infer_shape(self):
        super(ArmSquareOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = np.square(inputs[0])
        self.set_out_tensor(out_tensors)


class ArmSquaredDifferenceOp(OpNeedBroadcast, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmSquaredDifferenceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.power(np.subtract(*inputs), 2)
        self.set_out_tensor(out_tensor)


class ArmSufficientStatisticsOp(OpHasAxis, OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 2

    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16', 'uint32', 'uint8', 'int16', 'uint16', 'int8', 'int32'],
                1: ['float32', 'float16', 'uint32', 'uint8', 'int16', 'uint16', 'int8', 'int32']}

    def __init__(self, graph, attr_dict=None):
        super(ArmSufficientStatisticsOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmSufficientStatisticsOp, attr_dict)
        assert self.check_required(), 'ArmSufficientStatisticsOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmSufficientStatisticsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor_list = tf.nn.sufficient_statistics(
            inputs[0], self.axes, shift=inputs[1], keepdims=True)
        out_tensor_list = [ot.numpy() for ot in out_tensor_list[1:3]]
        self.set_out_tensor(out_tensor_list)


class ArmTanOp(LayoutUnawareOp, OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmTanOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.tan(inputs[0])
        self.set_out_tensor(out_tensor)


class ArmTileOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {'reps': {'type': AttrType.INTS, 'required': True}}

    def __init__(self, graph, attr_dict=None):
        super(ArmTileOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmTileOp, attr_dict)
        assert self.check_required(), 'ArmTileOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmTileOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs[0].shape) != len(self.reps):
            ERROR(
                '[Parser]: Input shape of ArmTile(%s) does not comply with repeats!' % self.name)
        out_tensors = np.tile(inputs[0], self.reps)
        self.set_out_tensor(out_tensors)

    def write_attrs(self, txt_file):
        ret = super(ArmTileOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('repeats=[%s]\n' % list_list_to_string(self.reps))
        return ret


class ArmTopKOp(OpHasAxis, OpHasMultipleOutPorts, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16']}

    @classmethod
    def attributes(cls):
        return {'k': {'type': AttrType.INT, 'default': 1},
                'axis': {'default': -1},
                'sorted': {'type': AttrType.INT, 'options': [0, 1], 'default': 1},
                'largest': {'type': AttrType.INT, 'options': [0, 1], 'default': 1},
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmTopKOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmTopKOp, attr_dict)
        assert self.check_required(), 'ArmTopKOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmTopKOp, self).infer_shape()
        inputs = self.get_input_tensors()
        # "topk_cpu" not implemented for 'Half'
        input_tensor = torch.from_numpy(inputs[0].astype(np.float64))
        out_tensor_list = torch.topk(input_tensor, self.k, dim=self.axis, largest=bool(
            self.largest), sorted=bool(self.sorted))
        out_tensor_list = [ot.numpy() for ot in out_tensor_list]
        out_tensor_list[0] = out_tensor_list[0].astype(inputs[0].dtype)
        out_tensor_list[1] = out_tensor_list[1].astype(np.int32)
        self.set_out_tensor(out_tensor_list)

    def write_attrs(self, txt_file):
        ret = super(ArmTopKOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('k=%d\n' % self.k)
            txt_file.write('sorted=%s\n' % str(bool(self.sorted)).lower())
            txt_file.write('largest=%s\n' % str(bool(self.largest)).lower())
        return ret


class ArmTransposeOp(OpHasOneOutPort, ArmOp):
    @staticmethod
    def cal_merged_perm(perm1, perm2):
        assert len(perm1) == len(
            perm2), 'The length of perm1 is not equal to the length of perm2 in ArmTransposeOp.'
        base = np.array(list(range(len(perm1))), np.int64)
        final = base[np.array(perm1)][np.array(perm2)]
        return final.tolist()

    @classmethod
    def attributes(cls):
        return {'perm': {'type': AttrType.INTS, 'default': [], 'required': True}}

    def __init__(self, graph, attr_dict=None):
        super(ArmTransposeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmTransposeOp, attr_dict)
        assert self.check_required(), 'ArmTransposeOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmTransposeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.transpose(
            inputs[0], axes=self.perm if self.perm else None)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmTransposeOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('perm=[%s]\n' % num_list_to_string(self.perm))
        return ret


class ArmUpsampleByIndexOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def cast_in_ports(cls):
        return {0: ['float32', 'float16'], 1: 'int32'}

    @classmethod
    def attributes(cls):
        return {'scale': {'type': AttrType.INT, 'default': 2},
                'flatten_dim': {'type': AttrType.STRING, 'default': 'HWC', 'options': ['NHWC', 'HWC', 'HW']},
                'shape': {'type': AttrType.INTS, 'default': []}
                }

    def __init__(self, graph, attr_dict=None):
        super(ArmUpsampleByIndexOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmUpsampleByIndexOp, attr_dict)
        assert self.check_required(), 'ArmUpsampleByIndexOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmUpsampleByIndexOp, self).infer_shape()
        inputs = self.get_input_tensors()
        multiplier = [1, self.scale, self.scale, 1]
        out_shape = np.array(inputs[0].shape, np.int64) * \
            np.array(multiplier, np.int64)
        self.shape = out_shape.tolist()
        out_tensor = np.random.ranf(self.shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmUpsampleByIndexOp, self).write_attrs(txt_file)
        if ret:
            txt_file.write('flatten_dim=%s\n' % self.flatten_dim)
            txt_file.write('shape=[%s]\n' % list_list_to_string(self.shape))
        return ret


class ArmWhereOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def num_in_ports(cls):
        return 3

    @classmethod
    def cast_in_ports(cls):
        return {0: 'uint8', 1: ['float32', 'float16'], 2: ['float32', 'float16']}

    def infer_shape(self):
        super(ArmWhereOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.where(*inputs)
        self.set_out_tensor(out_tensor)


class ArmYuvToRgbOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {
            'format': {'type': AttrType.STRING},
            'bits': {'type': AttrType.INT},
            'conversion': {'type': AttrType.STRING, 'default': 'BT709', 'options': ['SELF', 'BT709']},
            'coefficient': {'type': AttrType.FLOATS},
            'coefficient_dtype': {'type': AttrType.STRING},
            'coefficient_shift': {'type': AttrType.INT},
            'shape': {'type': AttrType.INTS},
        }

    def __init__(self, graph, attr_dict=None):
        super(ArmYuvToRgbOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmYuvToRgbOp, attr_dict)
        self.out_dtype = attr_dict.get('dtype', np.uint8)
        assert self.check_required(), 'ArmYuvToRgbOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmYuvToRgbOp, self).infer_shape()
        # input_shape = re.findall("\[[\s*\d+,]*\d+\]|\[\s*\]", self.shape)
        # input_shape = [[int(i) for i in re.findall("\d+", shape)] for shape in input_shape]
        out_tensor = np.random.randn(*tuple(self.shape)).astype(self.out_dtype)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmYuvToRgbOp, self).write_attrs(txt_file)
        if ret:
            for k in ['format', 'bits', 'conversion', 'coefficient', 'coefficient_dtype', 'coefficient_shift', 'shape']:
                if k in ['coefficient', 'shape']:
                    txt_file.write('%s=[%s]\n' % (
                        k, num_list_to_string(getattr(self, k))))
                else:
                    txt_file.write('%s=%s\n' % (k, str(getattr(self, k))))
        return ret


class ArmRgbToYuvOp(OpHasOneOutPort, ArmOp):
    @classmethod
    def attributes(cls):
        return {
            'format': {'type': AttrType.STRING},
            'bits': {'type': AttrType.INT},
            'conversion': {'type': AttrType.STRING, 'default': 'BT709', 'options': ['SELF', 'BT709']},
            'coefficient': {'type': AttrType.FLOATS},
            'coefficient_dtype': {'type': AttrType.STRING},
            'coefficient_shift': {'type': AttrType.INT},
        }

    def __init__(self, graph, attr_dict=None):
        super(ArmRgbToYuvOp, self).__init__(graph, attr_dict)
        self.update_attributes(ArmYuvToRgbOp, attr_dict)
        self.out_dtype = attr_dict.get('dtype', np.uint8)
        assert self.check_required(), 'ArmYuvToRgbOp is missing a required parameter.'

    def infer_shape(self):
        super(ArmRgbToYuvOp, self).infer_shape()
        input_shape = self.get_input_tensors()[0].shape
        out_tensor = np.random.randn(input_shape[0], int(
            input_shape[1] * input_shape[2] * 1.5)).astype(self.out_dtype)
        self.set_out_tensor(out_tensor)

    def write_attrs(self, txt_file):
        ret = super(ArmRgbToYuvOp, self).write_attrs(txt_file)
        if ret:
            for k in ['format', 'bits', 'conversion', 'coefficient', 'coefficient_dtype', 'coefficient_shift']:
                if k in ['coefficient']:
                    txt_file.write('%s=[%s]\n' % (
                        k, num_list_to_string(getattr(self, k))))
                else:
                    txt_file.write('%s=%s\n' % (k, str(getattr(self, k))))
        return ret


class ArmZeroFractionOp(OpHasOneOutPort, ArmOp):
    def infer_shape(self):
        super(ArmZeroFractionOp, self).infer_shape()
        input_tensor = self.get_input_tensors()[0]
        out_tensor = np.array(tf.math.zero_fraction(input_tensor).numpy())
        self.set_out_tensor(out_tensor)
