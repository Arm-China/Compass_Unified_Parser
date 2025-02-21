# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.
import torch

from ..op import *
import numpy as np


class MultiHeadAttentionMsOp(OpHasVariableOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'mask_filter_value': {
                    'type': AttrType.FLOAT, 'default': -10000.0
                },
                'num_heads': {
                    'type': AttrType.INT, 'required': True
                },
                'scale': {
                    'type': AttrType.FLOAT,
                },
                'unidirectional': {
                    'type': AttrType.INT, 'default': 0
                },
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(MultiHeadAttentionMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(MultiHeadAttentionMsOp, attr_dict)
        assert self.check_required(), 'MultiHeadAttentionMsOp is missing a required parameter.'

    def infer_shape(self):
        super(MultiHeadAttentionMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) <= 8, f'MultiHeadAttentionMsOp should have 1~8 inputs, but got {len(inputs)}'

        query = inputs[0]
        self.head_dim = query.shape[-1] // self.num_heads
        key = inputs[1] if len(inputs) > 1 else None
        value = inputs[2] if len(inputs) > 2 else None
        bias = inputs[3] if len(inputs) > 3 else None
        key_padding_mask = inputs[4] if len(inputs) > 4 else None
        attention_bias = inputs[5] if len(inputs) > 5 else None
        past_key = inputs[6] if len(inputs) > 6 else None
        past_value = inputs[7] if len(inputs) > 7 else None

        # TODO, add infer if bias/key_padding_mask/past_key/past_value is not None
        # TODO, add present_key/value output
        def split_heads(x):
            x = x.reshape(x.shape[0], -1, self.num_heads, self.head_dim)
            return x.transpose(0, 2, 1, 3)

        def concat_heads(x):
            x = np.transpose(x, [0, 2, 1, 3])
            return x.reshape(x.shape[0], -1, self.num_heads * self.head_dim)

        q_split = split_heads(query)
        k_split = split_heads(key)
        qk_matmul = np.matmul(q_split, np.transpose(k_split, [0, 1, 3, 2]))
        scores = qk_matmul * self.scale
        attention_out = scores + attention_bias
        attention = torch.softmax(torch.tensor(attention_out), dim=-1)
        v_split = split_heads(value)
        output = np.matmul(attention.cpu().numpy(), v_split)

        # concat heads
        out_tensors = [concat_heads(output)]
        out_ports = self.get_out_ports()
        # if 1 in out_ports:
        #     out_tensors.append(np.array(std_var, np.float32))
        # if 2 in out_ports:
        #     out_tensors.append(np.array(std_var, np.float32))
        self.set_out_tensor(out_tensors)


class QGemmMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                    'transA': {'type': AttrType.BOOL, 'default': False},
                    'transB': {'type': AttrType.BOOL, 'default': False}}}

    def __init__(self, graph, attr_dict=None):
        super(QGemmMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QGemmMsOp, attr_dict)
        assert self.check_required(), 'QGemmMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            input_names = ['A', 'a_scale', 'a_zero_point', 'B',
                           'b_scale', 'b_zero_point', 'C', 'y_scale', 'y_zero_point']
            if item in input_names:
                item_idx = input_names.index(item)
                inputs = self.get_input_tensors()
                if len(inputs) > item_idx:
                    ret = inputs[item_idx]
                    if ret is not None:
                        ret = np.array(ret, dtype=np.float32) if 'scale' in item else np.array(ret)
                self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QGemmMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QGemmMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 6, 'Meets invalid inputs length of QGemmMsOp(%s)' % self.name
        A = self.A if not self.transA else np.transpose(self.A)
        float_A = (A.astype(np.int32) - self.a_zero_point) * self.a_scale
        B = self.B if not self.transB else np.transpose(self.B)
        float_B = (B.astype(np.int32) - self.b_zero_point) * self.b_scale
        C = np.array(0, dtype=np.int32) if self.C is None else self.C
        c_scale = self.alpha * self.a_scale * self.b_scale
        float_C = C.astype(np.int32) * c_scale
        # alpha * A' * B' + C
        out_tensor = self.alpha * np.matmul(float_A, float_B) + float_C
        if self.y_scale is not None:
            assert self.y_zero_point is not None, \
                'y_zero_point of QGemmMsOp(%s) must not be null if y_scale is not null' % self.name
            out_min = np.iinfo(self.y_zero_point.dtype).min
            out_max = np.iinfo(self.y_zero_point.dtype).max
            out_tensor = np.clip(np.around(out_tensor / self.y_scale) + self.y_zero_point,
                                 out_min, out_max).astype(self.y_zero_point.dtype)
        else:
            assert self.y_zero_point is None, \
                'y_zero_point of QGemmMsOp(%s) must be null if y_scale is null' % self.name
        self.set_out_tensor(out_tensor)


class QLinearAddMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearAddMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearAddMsOp, attr_dict)
        assert self.check_required(), 'QLinearAddMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            if ret is None:
                input_names = ['A', 'A_scale', 'A_zero_point', 'B',
                               'B_scale', 'B_zero_point', 'C_scale', 'C_zero_point']
                if item in input_names:
                    item_idx = input_names.index(item)
                    inputs = self.get_input_tensors()
                    if len(inputs) > item_idx:
                        ret = inputs[item_idx]
                        if 'scale' in item:
                            ret = np.array(ret).astype(np.float32)
                        self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
                if ret is None and item in ('A_zero_point', 'B_zero_point', 'C_zero_point') and self.A is not None:
                    ret = np.array(0, dtype=self.A.dtype)
                    self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QLinearAddMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearAddMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 7, 'Meets invalid inputs length of QLinearAddMs op (%s)' % self.name
        float_a = (self.A.astype(np.int32) - self.A_zero_point) * self.A_scale
        float_b = (self.B.astype(np.int32) - self.B_zero_point) * self.B_scale
        float_y = np.add(float_a, float_b)
        out_min = np.iinfo(self.C_zero_point.dtype).min
        out_max = np.iinfo(self.C_zero_point.dtype).max
        out_tensor = np.clip(np.around(float_y / self.C_scale) + self.C_zero_point,
                             out_min, out_max).astype(self.C_zero_point.dtype)
        self.set_out_tensor(out_tensor)


class QLinearAveragePoolMsOp(BaseOnnxPoolOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'channels_last': {'type': AttrType.INT, 'default': 0},
                    'kernel_shape': {'type': AttrType.INTS, 'required': True},
                    'auto_pad': {'type': AttrType.STRING, 'default': 'NOTSET'},
                    'ceil_mode': {'type': AttrType.INT, 'default': 0},
                    'count_include_pad': {'type': AttrType.INT, 'default': 0}}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearAveragePoolMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearAveragePoolMsOp, attr_dict)
        assert self.check_required(), 'QLinearAveragePoolMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            input_names = ['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point']
            if item in input_names:
                item_idx = input_names.index(item)
                inputs = self.get_input_tensors()
                if len(inputs) > item_idx:
                    ret = inputs[item_idx]
                    if 'scale' in item:
                        ret = np.array(ret).astype(np.float32)
                    self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
            if item == 'data_format':
                if self.channels_last == 1:
                    self.__dict__['_attr'][item] = 'NHWC'
                    ret = 'NHWC'
                else:
                    self.__dict__['_attr'][item] = 'NCHW'
                    ret = 'NCHW'
        except:
            ret = None
        if ret is None:
            ret = super(QLinearAveragePoolMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearAveragePoolMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 5, 'Meets invalid inputs length of QLinearAveragePoolMsOp(%s)' % self.name
        pading_value = np.nan if self.count_include_pad == 0 else 0
        in_shape = inputs[0].shape[1:-1] if self.data_format == 'NHWC' else inputs[0].shape[2:]
        spatial_dim = len(inputs[0].shape) - 2
        out_shape = BaseOnnxPoolOp.cal_out_shape(
            in_shape, self.pads, self.strides, self.kernel_shape, self.auto_pad, dilations=self.dilations,
            ceil_mode=self.ceil_mode)
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
        float_x = (self.x.astype(np.int32) - self.x_zero_point) * self.x_scale
        n_dims = len(self.pads) // 2
        pads_np = [(self.pads[i], self.pads[i + n_dims]) for i in range(n_dims)]
        if self.data_format == 'NHWC':
            perm = [0, spatial_dim + 1] + list(range(1, spatial_dim + 1))
            float_x = np.transpose(float_x, perm)
        padded_input = np.pad(
            float_x,
            ((0, 0), (0, 0), *pads_np),
            mode="constant",
            constant_values=pading_value,
        )
        float_y = BaseOnnxPoolOp.pool(
            padded_input,
            float_x.shape,
            self.kernel_shape,
            self.strides,
            out_shape,
            'AVG',
            self.pads,
            self.dilations,
            self.count_include_pad
        )
        if self.data_format == 'NHWC':
            perm = [0] + list(range(2, spatial_dim + 2)) + [1]
            float_y = np.transpose(float_y, perm)
        if self.y_scale is not None:
            assert self.y_zero_point is not None, \
                'y_zero_point of QLinearAveragePoolMsOp(%s) must not be null if y_scale is not null' % self.name
            out_min = np.iinfo(self.y_zero_point.dtype).min
            out_max = np.iinfo(self.y_zero_point.dtype).max
            out_tensor = np.clip(np.around(float_y / self.y_scale) + self.y_zero_point,
                                 out_min, out_max).astype(self.y_zero_point.dtype)
        else:
            assert self.y_zero_point is None, \
                'y_zero_point of QLinearAveragePoolMsOp(%s) must be null if y_scale is null' % self.name
        self.set_out_tensor(out_tensor)


class QLinearConcatMsOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            1: {'axis': {'required': True}}
        }

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            input_names = ['y_scale', 'y_zero_point']
            if item in input_names:
                item_idx = input_names.index(item)
                inputs = self.get_input_tensors()
                if len(inputs) > item_idx:
                    ret = inputs[item_idx]
                    if 'scale' in item:
                        ret = np.array(ret).astype(np.float32)
                    self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QLinearConcatMsOp, self).__getattr__(item)
        return ret

    def __init__(self, graph, attr_dict=None):
        super(QLinearConcatMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearConcatMsOp, attr_dict)
        assert self.check_required(), 'QLinearConcatMsOp is missing a required parameter.'

    def infer_shape(self):
        super(QLinearConcatMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 5, 'Meets invalid inputs length of QLinearConcatMsOp op (%s)' % self.name
        fp_inputs = []
        for i in range(2, len(inputs), 3):
            inp_t = inputs[i]
            inp_scale = inputs[i + 1]
            inp_zp = inputs[i + 2]
            fp_inp = (inp_t.astype(np.int32) - inp_zp) * inp_scale
            fp_inputs.append(fp_inp)
        float_y = np.concatenate(fp_inputs, axis=self.axis)
        out_min = np.iinfo(self.y_zero_point.dtype).min
        out_max = np.iinfo(self.y_zero_point.dtype).max
        out_tensor = np.clip(np.around(float_y / self.y_scale) + self.y_zero_point,
                             out_min, out_max).astype(self.y_zero_point.dtype)
        self.set_out_tensor(out_tensor)


class QLinearGlobalAveragePoolMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'channels_last': {'type': AttrType.INT, 'required': True}}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearGlobalAveragePoolMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearGlobalAveragePoolMsOp, attr_dict)
        assert self.check_required(), 'QLinearGlobalAveragePoolMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            input_names = ['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point']
            if item in input_names:
                item_idx = input_names.index(item)
                inputs = self.get_input_tensors()
                if len(inputs) > item_idx:
                    ret = inputs[item_idx]
                    if 'scale' in item:
                        ret = np.array(ret).astype(np.float32)
                    self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QLinearGlobalAveragePoolMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearGlobalAveragePoolMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 5, 'Meets invalid inputs length of QLinearGlobalAveragePoolMsOp(%s)' % self.name
        if self.channels_last == 0:
            axes = list(range(2, len(inputs[0].shape)))
        else:
            axes = list(range(1, len(inputs[0].shape) - 1))
        float_x = (self.x.astype(np.int32) - self.x_zero_point) * self.x_scale
        float_y = np.mean(float_x, axis=tuple(axes), keepdims=True)
        out_min = np.iinfo(self.y_zero_point.dtype).min
        out_max = np.iinfo(self.y_zero_point.dtype).max
        out_tensor = np.clip(np.around(float_y / self.y_scale) + self.y_zero_point,
                             out_min, out_max).astype(self.y_zero_point.dtype)
        self.set_out_tensor(out_tensor)


class QLinearLeakyReluMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 0.01}}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearLeakyReluMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearLeakyReluMsOp, attr_dict)
        assert self.check_required(), 'QLinearLeakyReluMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            if ret is None:
                input_names = ['X', 'X_scale', 'X_zero_point', 'Y_scale', 'Y_zero_point']
                if item in input_names:
                    item_idx = input_names.index(item)
                    inputs = self.get_input_tensors()
                    if len(inputs) > item_idx:
                        ret = inputs[item_idx]
                        if 'scale' in item:
                            ret = np.array(ret).astype(np.float32)
                        self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QLinearLeakyReluMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearLeakyReluMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 4, 'Meets invalid inputs length of QLinearLeakyReluMsOp op (%s)' % self.name
        float_x = (self.X.astype(np.int32) - self.X_zero_point) * self.X_scale
        float_y = tf.nn.leaky_relu(float_x.astype(np.float32), alpha=self.alpha).numpy()
        out_min = np.iinfo(self.Y_zero_point.dtype).min
        out_max = np.iinfo(self.Y_zero_point.dtype).max
        out_tensor = np.clip(np.around(float_y / self.Y_scale) + self.Y_zero_point,
                             out_min, out_max).astype(self.Y_zero_point.dtype)
        self.set_out_tensor(out_tensor)


class QLinearSigmoidMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearSigmoidMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearSigmoidMsOp, attr_dict)
        assert self.check_required(), 'QLinearSigmoidMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            if ret is None:
                input_names = ['X', 'X_scale', 'X_zero_point', 'Y_scale', 'Y_zero_point']
                if item in input_names:
                    item_idx = input_names.index(item)
                    inputs = self.get_input_tensors()
                    if len(inputs) > item_idx:
                        ret = inputs[item_idx]
                        if 'scale' in item:
                            ret = np.array(ret).astype(np.float32)
                        self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QLinearSigmoidMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearSigmoidMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 4, 'Meets invalid inputs length of QLinearSigmoidMsOp op (%s)' % self.name
        float_x = (self.X.astype(np.int32) - self.X_zero_point) * self.X_scale
        float_y = tf.sigmoid(float_x.astype(np.float32)).numpy()
        out_min = np.iinfo(self.Y_zero_point.dtype).min
        out_max = np.iinfo(self.Y_zero_point.dtype).max
        out_tensor = np.clip(np.around(float_y / self.Y_scale) + self.Y_zero_point,
                             out_min, out_max).astype(self.Y_zero_point.dtype)
        self.set_out_tensor(out_tensor)


class RotaryEmbeddingMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'interleaved': {
                    'type': AttrType.INT, 'default': 0
                },
                'is_packed_batching': {
                    'type': AttrType.INT, 'default': 0
                },
                'num_heads': {
                    'type': AttrType.INT, 'default': 0
                },
                'rotary_embedding_dim': {
                    'type': AttrType.INT, 'default': 0
                },
                'scale': {
                    'type': AttrType.FLOAT, 'default': 1.0
                },
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(RotaryEmbeddingMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(RotaryEmbeddingMsOp, attr_dict)
        assert self.check_required(), 'RotaryEmbeddingMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            if ret is None:
                input_names = ['input', 'position_ids', 'cos_cache', 'sin_cache']
                if item in input_names:
                    item_idx = input_names.index(item)
                    inputs = self.get_input_tensors()
                    if len(inputs) > item_idx:
                        ret = inputs[item_idx]
                        if 'cache' in item:
                            ret = np.array(ret)
                        self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(RotaryEmbeddingMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(RotaryEmbeddingMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 4, f'Expected 4 inputs of RotaryEmbeddingMsOp op ({self.name}), but got {len(inputs)} inputs'
        assert len(inputs[0].shape) in (
            3, 4), f'Input x is expected to be 3 or 4 dimensional, but got {len(inputs[0].shape)}'
        assert len(inputs[1].shape) in (
            0, 1, 2), f'Input position_ids is expected to be 0, 1 or 2 dimensional, but got {len(inputs[1].shape)}'
        assert len(inputs[2].shape) == 2 and len(inputs[3].shape) == 2, \
            f'Input cos_cache/sin_cache is expected to be 2 dimensional, but got {len(inputs[2].shape)}/{len(inputs[3].shape)}'
        if self.rotary_embedding_dim > 0 and self.num_heads == 0:
            ERROR('[Parser]: num_heads must be provided if rotary_embedding_dim > 0 in (%s).' % self.name)

        bs = inputs[0].shape[0]
        seq_len = inputs[0].shape[1]
        hidden_size = inputs[0].shape[2]

        if len(inputs[0].shape) == 4:
            seq_len = inputs[0].shape[2]
            hidden_size = inputs[0].shape[1] * inputs[0].shape[3]
        max_seq_len = self.cos_cache.shape[0]
        head_size = self.cos_cache.shape[1] * 2 if self.rotary_embedding_dim == 0 else hidden_size // self.num_heads
        if self.rotary_embedding_dim > 0 and self.rotary_embedding_dim > head_size:
            ERROR('[Parser]: rotary_embedding_dim must be less than or equal to head_size in %s.' % self.name)

        if self.is_packed_batching == 0 and seq_len > max_seq_len:
            ERROR('[Parser]: Updating cos_cache and sin_cache in RotaryEmbedding is not currently supported')
        if self.interleaved == 1:
            raise NotImplementedError('interleaved still not supported yet.')
        postion_ids = inputs[1]
        input = inputs[0].reshape(bs, seq_len, -1, head_size)
        cos_cache = np.repeat(self.cos_cache, 2, axis=-1)
        sin_cache = np.repeat(self.sin_cache, 2, axis=-1)
        cos_emb = np.expand_dims(np.take(cos_cache, np.array(
            postion_ids, np.int32), axis=0), axis=2)
        cos_mul = input * np.tile(cos_emb, reps=(1, 1, int(input.shape[2]), 1))

        rotate_input = np.concatenate([np.negative(input[..., head_size // 2:]),
                                       input[..., : head_size // 2]], axis=-1)

        sin_emb = np.expand_dims(np.take(sin_cache, np.array(postion_ids, np.int32), axis=0), axis=2)
        sin_mul = rotate_input * np.tile(sin_emb, reps=(1, 1, int(input.shape[2]), 1))
        out_tensor = np.reshape((cos_mul + sin_mul), (bs, seq_len, -1)).astype(input.dtype)
        self.set_out_tensor(out_tensor)


class SkipSimplifiedLayerNormalizationMsOp(OpHasVariableOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'epsilon': {
                    'type': AttrType.FLOAT, 'required': True
                },
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(SkipSimplifiedLayerNormalizationMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(SkipSimplifiedLayerNormalizationMsOp, attr_dict)
        assert self.check_required(), 'SkipSimplifiedLayerNormalizationMsOp is missing a required parameter.'

    def infer_shape(self):
        super(SkipSimplifiedLayerNormalizationMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) in (
            3, 4), f'SkipSimplifiedLayerNormalizationMsOp should have 3 or 4 inputs, but got {len(inputs)}'

        input, skip, gamma = inputs[:3]
        inp_skip_bias_sum = input + skip
        if len(inputs) == 4:
            bias = inputs[3]
            inp_skip_bias_sum += bias

        # RMSNorm
        inp_skip_bias_sum = inp_skip_bias_sum.astype(np.float32)
        square = np.power(inp_skip_bias_sum, 2)
        mean = np.mean(square, axis=(-1, ), keepdims=True)
        variance = np.var(square, axis=(-1, ), keepdims=True)
        normalized = inp_skip_bias_sum / np.sqrt(mean + self.epsilon)
        normalized = normalized.astype(input.dtype)
        axes = OpHasAxis.make_axes_non_negative(
            [-1], len(inp_skip_bias_sum.shape))
        weights = OpHasAxis.expand_to(gamma, axes, len(inp_skip_bias_sum.shape))
        rms_output = (normalized * weights).astype(input.dtype)
        out_ports = self.get_out_ports()
        output_tensors = [rms_output]
        if 1 in out_ports:
            output_tensors.append(mean.astype(np.float32))
        if 2 in out_ports:
            output_tensors.append(variance.astype(np.float32))
        if 3 in out_ports:
            output_tensors.append(inp_skip_bias_sum.astype(input.dtype))
        self.set_out_tensor(output_tensors)
