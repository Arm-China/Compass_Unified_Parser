# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from ..op import *
import numpy as np


class DequantizeLinearOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {10: {'axis': {'default': None, 'required': False}},
                13: {'axis': {'default': None, 'required': False}}
                }

    def __init__(self, graph, attr_dict=None):
        super(DequantizeLinearOp, self).__init__(graph, attr_dict)
        self.update_attributes(DequantizeLinearOp, attr_dict)
        assert self.check_required(), 'DequantizeLinearOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axis':
                if self.cur_version == 10:
                    ret = None
                else:
                    inputs = self.get_input_tensors()
                    if self.__dict__['_attr'][item].value is None:
                        if len(inputs[1].shape) != 0:
                            if len(inputs[0].shape) == 1:
                                ret = 0
                            else:
                                ret = 1
                            self.__dict__['_attr'][item].value = ret
                    else:
                        ret = self.__dict__['_attr'][item].value
            elif item == 'x_scale':
                inputs = self.get_input_tensors()
                ret = np.array(inputs[1]).astype(np.float32)
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.TENSOR, 'value': ret})
            elif item == 'x_zero_point':
                inputs = self.get_input_tensors()
                try:
                    ret = np.array(inputs[2])
                except:
                    ret = np.array(0, inputs[0].dtype)
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(DequantizeLinearOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(DequantizeLinearOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) in (2, 3), 'DequantizeLinearOp expects 2 or 3 inputs, but got %d.' % len(inputs)
        if len(inputs) > 2:
            assert self.x_scale.shape == self.x_zero_point.shape, 'x_scale and x_zero_point in DequantizeLinearOp must have same shape.'
        if self.axis is None:
            out_tensor = ((inputs[0] - self.x_zero_point) * self.x_scale).astype(np.float32)
        else:
            self.axis = OpHasAxis.make_axes_non_negative(self.axis, len(inputs[0].shape))
            axis_dim = inputs[0].shape[self.axis]
            if self.x_scale.size == 1:
                self.x_scale = np.tile(self.x_scale, axis_dim)
            if self.x_zero_point.size == 1:
                self.x_zero_point = np.tile(self.x_zero_point, axis_dim)
            expand_dims = len(inputs[0].shape) - 1 - self.axis
            x_scale_shape = [self.x_scale.size] + [1] * expand_dims
            x_zero_point_shape = [self.x_zero_point.size] + [1] * expand_dims
            out_tensor = ((inputs[0] - np.reshape(self.x_zero_point,
                                                  x_zero_point_shape)) * np.reshape(self.x_scale, x_scale_shape)).astype(np.float32)
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        inputs = self.get_input_tensors()
        if len(inputs) < 3:
            zero_point = np.zeros_like(inputs[1], inputs[0].dtype)
            from ...front_end.onnx.passes.common_passes import insert_constant
            insert_constant(self._graph, self.name + '_x_zero_point',
                            zero_point, self.name, in_port=2, data_format='NHWC')
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            self.cur_version = max_ver


class DynamicQuantizeLinearOp(OpHasMultipleOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {}}

    def __init__(self, graph, attr_dict=None):
        super(DynamicQuantizeLinearOp, self).__init__(graph, attr_dict)
        self.update_attributes(DynamicQuantizeLinearOp, attr_dict)
        assert self.check_required(), 'DynamicQuantizeLinearOp is missing a required parameter.'

    def infer_shape(self):
        super(DynamicQuantizeLinearOp, self).infer_shape()
        inputs = self.get_input_tensors()
        X = inputs[0]
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float32((x_max - x_min) / (255 - 0))
        Y_ZeroPoint = np.clip(round((0 - x_min) / Y_Scale),
                              0, 255).astype(np.uint8)
        Y = np.clip(np.round(X / Y_Scale) + Y_ZeroPoint,
                    0, 255).astype(np.uint8)
        outputs = [Y, Y_Scale, Y_ZeroPoint]
        self.set_out_tensor(outputs)


class QLinearConvOp(BaseConvOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {10: {}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearConvOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearConvOp, attr_dict)
        assert self.check_required(), 'QLinearConvOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            need_set_attr = False
            if item == 'x_scale':
                inputs = self.get_input_tensors()
                ret = np.array(inputs[1]).astype(np.float32)
                need_set_attr = True
            elif item == 'x_zero_point':
                inputs = self.get_input_tensors()
                ret = np.array(inputs[2])
                need_set_attr = True
            elif item == 'w':
                inputs = self.get_input_tensors()
                ret = np.array(inputs[3])
                need_set_attr = True
            elif item == 'w_scale':
                inputs = self.get_input_tensors()
                ret = np.array(inputs[4]).astype(np.float32)
                need_set_attr = True
            elif item == 'w_zero_point':
                inputs = self.get_input_tensors()
                ret = np.array(inputs[5])
                need_set_attr = True
            elif item == 'y_scale':
                inputs = self.get_input_tensors()
                ret = np.array(inputs[6]).astype(np.float32)
                need_set_attr = True
            elif item == 'y_zero_point':
                inputs = self.get_input_tensors()
                ret = np.array(inputs[7])
                need_set_attr = True
            elif item == 'B':
                inputs = self.get_input_tensors()
                if len(inputs) == 9:
                    ret = np.array(inputs[8]).astype(np.int32)
                else:
                    ret = np.zeros((self.num_output,), np.int32)
                need_set_attr = True
            if need_set_attr:
                self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QLinearConvOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearConvOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) in (8, 9), ('Invalid inputs length of QLinearConv(%s)' % self.name)
        if len(inputs[0].shape) < 3:
            ERROR('[Parser]: Meets invalid input dimension for QLinearConv(%s)' % self.name)
            return
        if len(inputs[0].shape) > 5:
            WARN('[Parser]: Meets unsupported input dimension for QLinearConv(%s)' % self.name)
            return

        spatial_len = len(inputs[0].shape) - 2
        if self.data_format == 'NHWC':
            inp = inputs[0]
        else:
            inp = np.transpose(inputs[0], [0] + list(range(2, spatial_len + 2)) + [1])
        if self.auto_pad == 'NOTSET' and any(p != 0 for p in self.pads):
            pads = np.reshape(np.array(self.pads, np.int32), (2, -1))
            pads = np.transpose(pads)
            pads = np.concatenate([np.zeros((1, 2), np.int32), pads, np.zeros((1, 2), np.int32)], axis=0)
            inp = np.pad(inp, pads)
        float_x = (inp.astype(np.int32) - self.x_zero_point) * self.x_scale
        float_w = (self.w.astype(np.int32) - np.reshape(self.w_zero_point, [-1] + [1] * (len(self.w.shape) - 1))) \
            * np.reshape(self.w_scale, [-1] + [1] * (len(self.w.shape) - 1))
        float_w = np.transpose(float_w, list(range(2, spatial_len + 2)) + [1, 0])
        float_B = self.B.astype(np.int32) * (self.x_scale * self.w_scale)
        conv = eval('tf.keras.layers.Conv%sD' % spatial_len)
        padding = 'valid' if self.auto_pad in ('VALID', 'NOTSET') else 'same'
        out_tensor = conv(self.num_output,
                          self.kernel_shape,
                          strides=self.strides,
                          padding=padding,
                          data_format='channels_last',
                          dilation_rate=self.dilations,
                          groups=self.group,
                          activation=None,
                          use_bias=True,
                          kernel_initializer=tf.constant_initializer(float_w),
                          bias_initializer=tf.constant_initializer(float_B))(float_x).numpy()
        out_min = np.iinfo(self.y_zero_point.dtype).min
        out_max = np.iinfo(self.y_zero_point.dtype).max
        out_tensor = np.clip(out_tensor / self.y_scale + self.y_zero_point,
                             out_min, out_max).astype(self.y_zero_point.dtype)
        if self.data_format == 'NCHW':
            out_tensor = np.transpose(out_tensor, [0, spatial_len + 1] + list(range(1, spatial_len + 1)))
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        inputs = self.get_input_tensors()
        if len(inputs) < 9:
            from ...front_end.onnx.passes.common_passes import insert_constant
            insert_constant(self._graph, self.name + '_B',
                            self.B, self.name, in_port=8, data_format=self.data_format)


class QLinearMatMulOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {10: {}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearMatMulOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearMatMulOp, attr_dict)
        assert self.check_required(), 'QLinearMatMulOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            input_names = ['a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point', 'y_scale', 'y_zero_point']
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
            ret = super(QLinearMatMulOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearMatMulOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 8, 'Meets invalid inputs length of QLinearMatMul(%s)' % self.name
        float_a = (self.a.astype(np.int32) - self.a_zero_point) * self.a_scale
        float_b = (self.b.astype(np.int32) - self.b_zero_point) * self.b_scale
        float_y = np.matmul(float_a, float_b)
        out_min = np.iinfo(self.y_zero_point.dtype).min
        out_max = np.iinfo(self.y_zero_point.dtype).max
        out_tensor = np.clip(np.around(float_y / self.y_scale) + self.y_zero_point,
                             out_min, out_max).astype(self.y_zero_point.dtype)
        self.set_out_tensor(out_tensor)


class QuantizeLinearOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {10: {'axis': {'default': None, 'required': False}},
                13: {'axis': {'default': None, 'required': False}}
                }

    def __init__(self, graph, attr_dict=None):
        super(QuantizeLinearOp, self).__init__(graph, attr_dict)
        self.update_attributes(QuantizeLinearOp, attr_dict)
        assert self.check_required(), 'QuantizeLinearOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axis':
                if self.cur_version == 10:
                    ret = None
                else:
                    inputs = self.get_input_tensors()
                    if self.__dict__['_attr'][item].value is None:
                        if len(inputs[1].shape) != 0:
                            if len(inputs[0].shape) == 1:
                                ret = 0
                            else:
                                ret = 1
                            self.__dict__['_attr'][item].value = ret
                    else:
                        ret = self.__dict__['_attr'][item].value
            elif item == 'y_scale':
                inputs = self.get_input_tensors()
                ret = np.array(inputs[1]).astype(np.float32)
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.TENSOR, 'value': ret})
            elif item == 'y_zero_point':
                inputs = self.get_input_tensors()
                try:
                    ret = np.array(inputs[2])
                except:
                    ret = np.array(0, dtype=np.uint8)
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QuantizeLinearOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QuantizeLinearOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) in (2, 3), 'QuantizeLinearOp expects 2 or 3 inputs, but got %d.' % len(inputs)
        if len(inputs) > 2:
            assert self.y_scale.shape == self.y_zero_point.shape, 'y_scale and y_zero_point in QuantizeLinearOp must have same shape.'
        if self.axis is None:
            out_tensor = np.round(inputs[0] / self.y_scale) + self.y_zero_point
        else:
            axis_dim = inputs[0].shape[self.axis]
            if self.y_scale.size == 1:
                self.y_scale = np.tile(self.y_scale, axis_dim)
            if self.y_zero_point.size == 1:
                self.y_zero_point = np.tile(self.y_zero_point, axis_dim)
            expand_dims = len(inputs[0].shape) - 1 - self.axis
            y_scale_shape = [self.y_scale.size] + [1] * expand_dims
            y_zero_point_shape = [self.y_zero_point.size] + [1] * expand_dims
            out_tensor = np.round(inputs[0] / np.reshape(self.y_scale, y_scale_shape)) \
                + np.reshape(self.y_zero_point, y_zero_point_shape)
        zp_dtype = self.y_zero_point.dtype
        out_tensor = np.clip(out_tensor, np.iinfo(zp_dtype).min, np.iinfo(
            zp_dtype).max).astype(zp_dtype)
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        inputs = self.get_input_tensors()
        if len(inputs) < 3:
            from ...front_end.onnx.passes.common_passes import insert_constant
            zp_value = np.zeros_like(inputs[1], np.uint8)
            insert_constant(self._graph, self.name + '_y_zero_point',
                            zp_value, self.name, in_port=2, data_format='NHWC')
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            self.cur_version = max_ver
