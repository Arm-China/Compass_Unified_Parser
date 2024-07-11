# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.
import torch

from ..op import *
import numpy as np


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
