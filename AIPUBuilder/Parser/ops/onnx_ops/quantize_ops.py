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
