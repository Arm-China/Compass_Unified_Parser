# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import tensorflow as tf
from .op import *
from ..common.defs import FLOAT_EQUAL
from ..logger import INFO, DEBUG, WARN, ERROR, FATAL


class LiteADDOp(BaseActivationOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}, 3: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteADDOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteADDOp, attr_dict)
        assert self.check_required(), 'LiteADDOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteADDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 2, 'The input length is invalid in LiteADDOp infer shape.'
        out_tensor = tf.add(*inputs).numpy()
        out_tensor = self.cal_activation(out_tensor)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Add', 'version': 7}


class LiteADD_NOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteADD_NOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteADD_NOp, attr_dict)
        assert self.check_required(), 'LiteADD_NOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteADD_NOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = reduce(lambda x, y: x + y, inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sum', 'version': 13}


class LiteARG_MAXOp(OpHasAxis, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 0}, 'to': {'type': AttrType.STRING, 'required': True}},
                2: {'keepdims': {'default': 0}, 'to': {'type': AttrType.STRING, 'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteARG_MAXOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteARG_MAXOp, attr_dict)
        assert self.check_required(), 'LiteARG_MAXOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'to':
                ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        if ret is None:
            ret = super(LiteARG_MAXOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(LiteARG_MAXOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = int(inputs[1])
        out_tensor = np.argmax(
            inputs[0], axis=self.axis).astype(np.dtype(self.to))
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ArgMax', 'version': 13}


class LiteARG_MINOp(OpHasAxis, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 0}, 'to': {'type': AttrType.STRING, 'required': True}},
                2: {'keepdims': {'default': 0}, 'to': {'type': AttrType.STRING, 'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteARG_MINOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteARG_MINOp, attr_dict)
        assert self.check_required(), 'LiteARG_MINOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'to':
                ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        if ret is None:
            ret = super(LiteARG_MINOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(LiteARG_MINOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = int(inputs[1])
        out_tensor = np.argmin(
            inputs[0], axis=self.axis).astype(np.dtype(self.to))
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ArgMin', 'version': 13}


class LiteAVERAGE_POOL_2DOp(BaseActivationOp, OpHasPaddingStrides, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteAVERAGE_POOL_2DOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteAVERAGE_POOL_2DOp, attr_dict)
        assert self.check_required(), 'LiteAVERAGE_POOL_2DOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteAVERAGE_POOL_2DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.avg_pool(inputs[0].astype(np.float32),
                                    ksize=[1] + self.kernel_shape + [1],
                                    strides=[1] + self.strides + [1],
                                    padding='VALID' if self.auto_pad in (
                                        'VALID', 'NOTSET') else 'SAME',
                                    data_format='NHWC').numpy()
        out_tensor = self.cal_activation(out_tensor).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)
        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                inputs[0].shape[1:3],
                out_tensor.shape[1:3],
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
                zero_minimum=True
            )
            self.auto_pad = 'NOTSET'

    @property
    def correspond_onnx_op(self):
        return {'type': 'AveragePool', 'version': 10}


class LiteABSOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteABSOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteABSOp, attr_dict)
        assert self.check_required(), 'LiteABSOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteABSOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.abs(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Abs', 'version': 13}


class LiteBATCH_MATMULOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'adj_x': {'type': AttrType.STRING, 'default': False},
                    'adj_y': {'type': AttrType.STRING, 'default': False}
                    },
                2: {'adj_x': {'type': AttrType.STRING, 'default': False},
                    'adj_y': {'type': AttrType.STRING, 'default': False}
                    },
                3: {'adj_x': {'type': AttrType.STRING, 'default': False},
                    'adj_y': {'type': AttrType.STRING, 'default': False}
                    },
                4: {'adj_x': {'type': AttrType.STRING, 'default': False},
                    'adj_y': {'type': AttrType.STRING, 'default': False}
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteBATCH_MATMULOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteBATCH_MATMULOp, attr_dict)
        assert self.check_required(), 'LiteBATCH_MATMULOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteBATCH_MATMULOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.BatchMatMulV2(
            x=inputs[0].astype(np.float32), y=inputs[1].astype(np.float32),
            adj_x=self.adj_x, adj_y=self.adj_y).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'MatMul', 'version': 9}


class LiteBATCH_TO_SPACE_NDOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteBATCH_TO_SPACE_NDOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteBATCH_TO_SPACE_NDOp, attr_dict)
        assert self.check_required(), 'LiteBATCH_TO_SPACE_NDOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteBATCH_TO_SPACE_NDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.compat.v1.batch_to_space_nd(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class LiteBITWISE_XOROp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def infer_shape(self):
        super(LiteBITWISE_XOROp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.bitwise_xor(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitwiseXor', 'version': 18}


class LiteBROADCAST_TOOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteBROADCAST_TOOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteBROADCAST_TOOp, attr_dict)
        assert self.check_required(), 'LiteBROADCAST_TOOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteBROADCAST_TOOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.broadcast_to(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class LiteBROADCAST_ARGSOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteBROADCAST_ARGSOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteBROADCAST_ARGSOp, attr_dict)
        assert self.check_required(), 'LiteBROADCAST_ARGSOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteBROADCAST_ARGSOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 2, 'Meets invalid inputs of LiteBROADCAST_ARGSOp(%s) in infer_shape' % self.name
        out_tensor = tf.raw_ops.BroadcastArgs(s0=inputs[0], s1=inputs[1]).numpy()
        self.set_out_tensor(out_tensor)


class LiteCASTOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'to': {'type': AttrType.STRING, 'required': False}},
                2: {'to': {'type': AttrType.STRING, 'required': False}},
                3: {'to': {'type': AttrType.STRING, 'required': False}},
                4: {'to': {'type': AttrType.STRING, 'required': False}},
                5: {'to': {'type': AttrType.STRING, 'required': False}},
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteCASTOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteCASTOp, attr_dict)
        assert self.check_required(), 'LiteCASTOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = super(LiteCASTOp, self).__getattr__(item)
        except:
            ret = None
        if ret is None and item in ('to',):
            try:
                outputs = self.get_output_tensors()
                ret = str(outputs[0].dtype)
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.STRING, 'value': str(ret)})
            except:
                ret = None
        return ret

    def infer_shape(self):
        super(LiteCASTOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = inputs[0].astype(np.dtype(self.to))
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Cast', 'version': 19}


class LiteCEILOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteCEILOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteCEILOp, attr_dict)
        assert self.check_required(), 'LiteCEILOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteCEILOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.ceil(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Ceil', 'version': 13}


class LiteCONCATENATIONOp(OpHasAxis, BaseActivationOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteCONCATENATIONOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteCONCATENATIONOp, attr_dict)
        assert self.check_required(), 'LiteCONCATENATIONOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteCONCATENATIONOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.concatenate(inputs, self.axis)
        out_tensor = self.cal_activation(out_tensor)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Concat', 'version': 4}


class LiteCONV_2DOp(BaseActivationOp, BaseConvOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}

    @classmethod
    def perm_lite_to_onnx(cls):
        return [0, 3, 1, 2]

    @classmethod
    def perm_lite_to_tf(cls):
        return [1, 2, 3, 0]

    def __init__(self, graph, attr_dict=None):
        super(LiteCONV_2DOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteCONV_2DOp, attr_dict)
        assert self.check_required(), 'LiteCONV_2DOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteCONV_2DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None:
            self.kernel_shape = self.weights.shape[1:3]
        if self.biases is None:
            self.biases = np.zeros((self.num_output,), np.float32)
        if inputs[0].dtype != 'float32':
            inp = inputs[0].astype(np.float32)
        else:
            inp = inputs[0]
        if self.opcode_version == 6 and inputs[0].shape[-1] != self.weights.shape[-1]:
            # Group Conv
            assert inputs[0].shape[-1] % self.weights.shape[-1] == 0, 'ic or group_num not match.'
            self.group = inputs[0].shape[-1] // self.weights.shape[-1]
        else:
            self.group = 1
        conv_layer = tf.keras.layers.Conv2D(filters=self.weights.shape[0],
                                            kernel_size=self.kernel_shape,
                                            strides=self.strides,
                                            padding='VALID' if self.auto_pad in ('VALID', 'NOTSET') else 'SAME',
                                            dilation_rate=self.dilations,
                                            groups=self.group,
                                            use_bias=False,
                                            activation=None)
        out_tensor = conv_layer(inp).numpy()
        conv_layer.set_weights([np.transpose(self.weights, axes=type(self).perm_lite_to_tf())])
        out_tensor = conv_layer(inp).numpy()

        out_tensor = tf.nn.bias_add(
            out_tensor, self.biases, data_format='NHWC').numpy()
        out_tensor = self.cal_activation(out_tensor).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)
        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                inputs[0].shape[1:3],
                out_tensor.shape[1:3],
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
                zero_minimum=True
            )
            self.auto_pad = 'NOTSET'

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class LiteCONV_3DOp(BaseActivationOp, BaseConvOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}, 3: {}}

    @classmethod
    def perm_lite_to_onnx(cls):
        return [4, 3, 0, 1, 2]

    @classmethod
    def perm_lite_to_tf(cls):
        return [0, 1, 2, 3, 4]

    def __init__(self, graph, attr_dict=None):
        super(LiteCONV_3DOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteCONV_3DOp, attr_dict)
        assert self.check_required(), 'LiteCONV_3DOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'num_output':
                ret = self.__dict__['_attr'][item].value
                if ret is None:
                    if self.weights is not None:
                        ret = self.weights.shape[-1]
                    elif self.biases is not None:
                        ret = self.biases.shape[0]
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(LiteCONV_3DOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(LiteCONV_3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        # Filter has shape of [filter_depth, filter_height, filter_width, in_channels, out_channels].
        if self.kernel_shape is None:
            self.kernel_shape = self.weights.shape[0:3]
        if self.biases is None:
            self.biases = np.zeros((self.num_output,), np.float32)
        out_tensor = tf.nn.conv3d(inputs[0],
                                  np.transpose(self.weights, axes=type(
                                      self).perm_lite_to_tf()),
                                  strides=[1] + self.strides + [1],
                                  padding='VALID' if self.auto_pad in (
                                      'VALID', 'NOTSET') else 'SAME',
                                  dilations=[1] + self.dilations + [1],
                                  data_format='NDHWC')
        out_tensor = tf.nn.bias_add(
            out_tensor, self.biases, data_format='NHWC').numpy()
        out_tensor = self.cal_activation(out_tensor)
        out_tensor = out_tensor.astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)
        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                inputs[0].shape[1:-1],
                out_tensor.shape[1:-1],
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
                zero_minimum=True
            )
            self.auto_pad = 'NOTSET'

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class LiteCONV_3D_TRANSPOSEOp(BaseActivationOp, BaseDeconvOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'output_shape': {'type': AttrType.INTS}},
                2: {'output_shape': {'type': AttrType.INTS}},
                3: {'output_shape': {'type': AttrType.INTS}}
                }

    @classmethod
    def main_in_port(cls):
        return 1

    @classmethod
    def perm_lite_to_onnx(cls):
        return [4, 3, 0, 1, 2]

    @classmethod
    def perm_lite_to_tf(cls):
        return [0, 1, 2, 3, 4]

    def __init__(self, graph, attr_dict=None):
        super(LiteCONV_3D_TRANSPOSEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteCONV_3D_TRANSPOSEOp, attr_dict)
        assert self.check_required(), 'LiteCONV_3D_TRANSPOSEOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'num_output':
                ret = self.__dict__['_attr'][item].value
                if ret is None:
                    if self.weights is not None:
                        ret = self.weights.shape[3]
                    elif self.biases is not None:
                        ret = self.biases.shape[0]
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(LiteCONV_3D_TRANSPOSEOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(LiteCONV_3D_TRANSPOSEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        # Filter has shape of [filter_depth, filter_height, filter_width, out_channels, in_channels].
        if self.kernel_shape is None:
            self.kernel_shape = self.weights.shape[0:3]
        if self.biases is None:
            self.biases = np.zeros((self.num_output,), np.float32)
        out_tensor = tf.nn.conv3d_transpose(inputs[1],
                                            np.transpose(self.weights, axes=type(
                                                self).perm_lite_to_tf()),
                                            inputs[0],
                                            strides=[1] + self.strides + [1],
                                            padding='VALID' if self.auto_pad in ('VALID', 'NOTSET') else 'SAME',
                                            dilations=self.dilations)
        out_tensor = tf.nn.bias_add(
            out_tensor, self.biases, data_format='NHWC').numpy()
        out_tensor = out_tensor.astype(inputs[1].dtype)
        self.set_out_tensor(out_tensor)
        self.output_shape = inputs[0].tolist()[1:-1]


class LiteCUMSUMOp(OpHasOneOutPort, OpHasAxis, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'exclusive': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                    'reverse': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}, }
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteCUMSUMOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteCUMSUMOp, attr_dict)
        assert self.check_required(), 'LiteCUMSUMOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item in ('exclusive', 'reverse'):
                ret = bool(self.__dict__['_attr'][item].value)
            elif item in ('axis',):
                inputs = self.get_input_tensors()
                ret = int(inputs[1])
        except:
            ret = None
        if ret is None:
            ret = super(LiteCUMSUMOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item in ('exclusive', 'reverse'):
            self.__dict__['_attr'][item].value = int(value)
        else:
            super(LiteCUMSUMOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(LiteCUMSUMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.cumsum(inputs[0], axis=self.axis, exclusive=self.exclusive, reverse=self.reverse).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'CumSum', 'version': 14}


class LiteCUSTOMOp(OpHasMethod, OpHasVariableOutPorts, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteCUSTOMOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteCUSTOMOp, attr_dict)
        assert self.check_required(), 'LiteCUSTOMOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteCUSTOMOp, self).infer_shape()


class LiteCOSOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteCOSOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteCOSOp, attr_dict)
        assert self.check_required(), 'LiteCOSOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteCOSOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.cos(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Cos', 'version': 7}


class LiteDEPTH_TO_SPACEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'blocksize': {'type': AttrType.INT, 'required': True}},
                2: {'blocksize': {'type': AttrType.INT, 'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteDEPTH_TO_SPACEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteDEPTH_TO_SPACEOp, attr_dict)
        assert self.check_required(), 'LiteDEPTH_TO_SPACEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteDEPTH_TO_SPACEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.depth_to_space(inputs[0], self.blocksize).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'DepthToSpace', 'version': 1}


class LiteDEPTHWISE_CONV_2DOp(BaseActivationOp, BaseConvOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'multiplier': {'type': AttrType.INT, 'default': 1, 'required': True}},
                2: {'multiplier': {'type': AttrType.INT, 'default': 1, 'required': True}},
                3: {'multiplier': {'type': AttrType.INT, 'default': 1, 'required': True}}}

    @classmethod
    def perm_lite_to_tf(cls, multiplier=1):
        return [1, 2, 3, 0] if multiplier == 1 else [1, 2, 0, 3]

    @classmethod
    def perm_lite_to_onnx(cls):
        return [3, 0, 1, 2]

    def __init__(self, graph, attr_dict=None):
        super(LiteDEPTHWISE_CONV_2DOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteDEPTHWISE_CONV_2DOp, attr_dict)
        assert self.check_required(), 'LiteDEPTHWISE_CONV_2DOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteDEPTHWISE_CONV_2DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None:
            self.kernel_shape = self.weights.shape[1:3]
        if self.multiplier == 1:
            self.group = self.weights.shape[-1]
        if self.biases is None:
            self.biases = np.zeros((self.num_output,), np.float32)

        '''
        # out_tensor = tf.nn.depthwise_conv2d(inputs[0],
        #                                     np.transpose(self.weights, axes=type(self).perm_lite_to_tf(self.multiplier)),
        #                                     strides=[1] + self.strides + [1],
        #                                     padding='VALID' if self.auto_pad in ('VALID', 'NOTSET') else 'SAME',
        #                                     data_format='NHWC',
        #                                     rate=self.dilations)
        # out_tensor = tf.nn.bias_add(out_tensor, self.biases, data_format='NHWC').numpy()
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
        out_tensor = np.random.ranf(size=out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                inputs[0].shape[1:3],
                out_tensor.shape[1:3],
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
                zero_minimum=True
            )
            self.auto_pad = 'NOTSET'

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'num_output':
                if self.__dict__['_attr'][item].value is not None:
                    ret = self.__dict__['_attr'][item].value
                elif self.weights is not None:
                    ret = self.weights.shape[-1]
                    self.__dict__['_attr'][item].value = ret
                elif self.biases is not None:
                    ret = self.biases.shape[0]
                    self.__dict__['_attr'][item].value = ret
                else:
                    ret = None
            elif item == 'multiplier':
                if self.__dict__['_attr'][item].value is not None:
                    ret = int(self.__dict__['_attr'][item].value)
                else:
                    ret = 1
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(LiteDEPTHWISE_CONV_2DOp, self).__getattr__(item)
        return ret


class LiteDEQUANTIZEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'scale': {'type': AttrType.TENSOR, 'default': np.array(1., np.float32)},
                    'zero_point': {'type': AttrType.TENSOR, 'default': np.array(0, np.int32)}},
                2: {'scale': {'type': AttrType.TENSOR, 'default': np.array(1., np.float32)},
                    'zero_point': {'type': AttrType.TENSOR, 'default': np.array(0, np.int32)}},
                3: {'scale': {'type': AttrType.TENSOR, 'default': np.array(1., np.float32)},
                    'zero_point': {'type': AttrType.TENSOR, 'default': np.array(0, np.int32)}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteDEQUANTIZEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteDEQUANTIZEOp, attr_dict)
        assert self.check_required(), 'LiteDEQUANTIZEOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'scale':
                in_edges = self._graph.sorted_in_edges(self.name, data=True)
                ret, _ = in_edges[0][2]['tensor'].scale_zp
                self.__dict__['_attr'][item].value = ret
            elif item == 'zero_point':
                in_edges = self._graph.sorted_in_edges(self.name, data=True)
                _, ret = in_edges[0][2]['tensor'].scale_zp
                self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(LiteDEQUANTIZEOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        # input: uint8, int8, int16, float16
        super(LiteDEQUANTIZEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if np.issubdtype(inputs[0].dtype, np.integer):
            x = inputs[0].astype(np.int32)
        else:
            x = inputs[0]
        out_tensor = (self.scale * (x - self.zero_point)).astype(np.float32)
        self.set_out_tensor(out_tensor)


class LiteDIVOp(BaseActivationOp, OpHasDivisor, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}, 3: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteDIVOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteDIVOp, attr_dict)
        assert self.check_required(), 'LiteDIVOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteDIVOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.truediv(*inputs).numpy()
        out_tensor = self.cal_activation(out_tensor)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Div', 'version': 7}


class LiteEQUALOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def infer_shape(self):
        super(LiteEQUALOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.equal(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Equal', 'version': 11}


class LiteELUOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def infer_shape(self):
        super(LiteELUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0])
        mask = out_tensor <= 0
        out_tensor[mask] = np.exp(out_tensor[mask]) - 1
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Elu', 'version': 6}


class LiteEXPOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def infer_shape(self):
        super(LiteEXPOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.exp(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Exp', 'version': 13}


class LiteEXPAND_DIMSOp(OpHasAxis, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axis':
                inputs = self.get_input_tensors()
                if len(inputs) >= 2 \
                        and inputs[1] is not None \
                        and inputs[1].size == 1:
                    ret = int(inputs[1].item(0))
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(LiteEXPAND_DIMSOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(LiteEXPAND_DIMSOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.expand_dims(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reshape', 'version': 13}


class LiteFILLOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}, 3: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteFILLOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteFILLOp, attr_dict)
        assert self.check_required(), 'LiteFILLOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteFILLOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.full(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Fill', 'version': 1}


class LiteFLOOROp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteFLOOROp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteFLOOROp, attr_dict)
        assert self.check_required(), 'LiteFLOOROp is missing a required parameter.'

    def infer_shape(self):
        super(LiteFLOOROp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.floor(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Floor', 'version': 13}


class LiteFLOOR_DIVOp(OpHasOneOutPort, OpHasDivisor, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def infer_shape(self):
        super(LiteFLOOR_DIVOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.floordiv(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class LiteFLOOR_MODOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def infer_shape(self):
        super(LiteFLOOR_MODOp, self).infer_shape()
        inputs = self.get_input_tensors()
        with np.errstate(divide='ignore'):
            out_tensor = np.mod(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Mod', 'version': 13}


class LiteFULLY_CONNECTEDOp(BaseActivationOp, BaseLinearOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                2: {'weights_format': {'type': AttrType.INT, 'default': 0}},
                3: {'weights_format': {'type': AttrType.INT, 'default': 0}},
                4: {'weights_format': {'type': AttrType.INT, 'default': 0}},
                5: {'weights_format': {'type': AttrType.INT, 'default': 0},
                    'keepdims': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                    },
                9: {'weights_format': {'type': AttrType.INT, 'default': 0},
                    'keepdims': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteFULLY_CONNECTEDOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteFULLY_CONNECTEDOp, attr_dict)
        assert self.check_required(), 'LiteFULLY_CONNECTEDOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'keepdims':
                if self.cur_version < 5:
                    ret = False
                    self.__dict__['_attr'][item] = Attribute(
                        item, {'type': AttrType.INT, 'value': int(ret)})
                else:
                    ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(LiteFULLY_CONNECTEDOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(LiteFULLY_CONNECTEDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.weights is not None:
            last_out_dim = self.weights.shape[-2]
            inp = np.reshape(inputs[0], (-1, self.weights.shape[-1]))
            out_tensor = np.matmul(inp, np.transpose(
                self.weights, axes=type(self).perm_lite_to_tf()))
        else:
            last_out_dim = inputs[1].shape[-2]
            inp = np.reshape(inputs[0], (-1, inputs[1].shape[-1]))
            out_tensor = np.matmul(inp, np.transpose(inputs[1]))
        if self.biases is not None:
            out_tensor = out_tensor + self.biases
        out_tensor = self.cal_activation(out_tensor)
        if self.keepdims:
            out_shape = list(inputs[0].shape[:-1]) + [last_out_dim]
        else:
            out_shape = [int(np.prod(inputs[0].shape[:-1])), last_out_dim]
        out_tensor = np.reshape(out_tensor, out_shape)
        self.set_out_tensor(out_tensor)


class LiteGATHEROp(OpHasAxis, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'batch_dims': {'type': AttrType.INT, 'default': 0, 'required': True}},
                2: {'batch_dims': {'type': AttrType.INT, 'default': 0, 'required': True}},
                5: {'batch_dims': {'type': AttrType.INT, 'default': 0, 'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteGATHEROp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteGATHEROp, attr_dict)
        assert self.check_required(), 'LiteGATHEROp is missing a required parameter.'

    def infer_shape(self):
        super(LiteGATHEROp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.gather(*inputs, axis=self.axis,
                               batch_dims=self.batch_dims).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        if self.batch_dims == 0:
            return {'type': 'Gather', 'version': 11}
        else:
            return {'type': 'BatchGather', 'version': 1}


class LiteGATHER_NDOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteGATHER_NDOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteGATHER_NDOp, attr_dict)
        assert self.check_required(), 'LiteGATHER_NDOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteGATHER_NDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.gather_nd(*inputs, batch_dims=0).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'GatherND', 'version': 11}


class LiteGELUOp(ActivationOnlyOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'approximate': {'type': AttrType.BOOL, 'default': False},
                    },
                2: {'approximate': {'type': AttrType.BOOL, 'default': False},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteGELUOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteGELUOp, attr_dict)
        assert self.check_required(), 'LiteGELUOp is missing a required parameter.'
        self.activations = 'GELU'

    def infer_shape(self):
        super(LiteGELUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.gelu(inputs[0].astype(np.float32), approximate=self.approximate).numpy()
        out_tensor = out_tensor.astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Gelu', 'version': 20}


class LiteGREATEROp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteGREATEROp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteGREATEROp, attr_dict)
        assert self.check_required(), 'LiteGREATEROp is missing a required parameter.'

    def infer_shape(self):
        super(LiteGREATEROp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.greater(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Greater', 'version': 9}


class LiteGREATER_EQUALOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteGREATER_EQUALOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteGREATER_EQUALOp, attr_dict)
        assert self.check_required(), 'LiteGREATER_EQUALOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteGREATER_EQUALOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.greater_equal(*inputs)
        self.set_out_tensor(out_tensor)


class LiteHARD_SWISHOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteHARD_SWISHOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteHARD_SWISHOp, attr_dict)
        assert self.check_required(), 'LiteHARD_SWISHOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteHARD_SWISHOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = (inputs[0] * tf.nn.relu6(inputs[0] + 3) / 6).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'HardSwish', 'version': 1}


class LiteL2_NORMALIZATIONOp(BaseActivationOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteL2_NORMALIZATIONOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteL2_NORMALIZATIONOp, attr_dict)
        assert self.check_required(), 'LiteL2_NORMALIZATIONOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteL2_NORMALIZATIONOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.l2_normalize(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LpNormalization', 'version': 1}


class LiteLEAKY_RELUOp(BaseReluOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 0.01}},
                2: {'alpha': {'type': AttrType.FLOAT, 'default': 0.01}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteLEAKY_RELUOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteLEAKY_RELUOp, attr_dict)
        assert self.check_required(), 'LiteLEAKY_RELUOp is missing a required parameter.'
        self.activations = 'LEAKYRELU'
        self.negative_slope = self.alpha

    def infer_shape(self):
        super(LiteLEAKY_RELUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = self.cal_activation(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LeakyRelu', 'version': 6}


class LiteLESSOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteLESSOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteLESSOp, attr_dict)
        assert self.check_required(), 'LiteLESSOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteLESSOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.less(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Less', 'version': 9}


class LiteLESS_EQUALOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteLESS_EQUALOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteLESS_EQUALOp, attr_dict)
        assert self.check_required(), 'LiteLESS_EQUALOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteLESS_EQUALOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.less_equal(*inputs)
        self.set_out_tensor(out_tensor)


class LiteLOCAL_RESPONSE_NORMALIZATIONOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'required': True},
                    'beta': {'type': AttrType.FLOAT, 'required': True},
                    'bias': {'type': AttrType.FLOAT, 'required': True},
                    'radius': {'type': AttrType.INT, 'required': True}
                    },
                2: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteLOCAL_RESPONSE_NORMALIZATIONOp,
              self).__init__(graph, attr_dict)
        self.update_attributes(LiteLOCAL_RESPONSE_NORMALIZATIONOp, attr_dict)
        assert self.check_required(
        ), 'LiteLOCAL_RESPONSE_NORMALIZATIONOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteLOCAL_RESPONSE_NORMALIZATIONOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.local_response_normalization(
            inputs[0], self.radius, self.bias, self.alpha, self.beta).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LRN', 'version': 1}


class LiteLOGICAL_ANDOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteLOGICAL_ANDOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteLOGICAL_ANDOp, attr_dict)
        assert self.check_required(), 'LiteLOGICAL_ANDOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteLOGICAL_ANDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.logical_and(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'And', 'version': 7}


class LiteLOGICAL_NOTOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteLOGICAL_NOTOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteLOGICAL_NOTOp, attr_dict)
        assert self.check_required(), 'LiteLOGICAL_NOTOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteLOGICAL_NOTOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.logical_not(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Not', 'version': 1}


class LiteLOGICAL_OROp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteLOGICAL_OROp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteLOGICAL_OROp, attr_dict)
        assert self.check_required(), 'LiteLOGICAL_OROp is missing a required parameter.'

    def infer_shape(self):
        super(LiteLOGICAL_OROp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.logical_or(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Or', 'version': 7}


class LiteLOGISTICOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteLOGISTICOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteLOGISTICOp, attr_dict)
        assert self.check_required(), 'LiteLOGISTICOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteLOGISTICOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if np.issubdtype(inputs[0].dtype, np.integer):
            inp = inputs[0].astype(np.float32)
        else:
            inp = inputs[0]
        out_tensor = tf.sigmoid(inp).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sigmoid', 'version': 6}


class LiteLOG_SOFTMAXOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteLOG_SOFTMAXOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteLOG_SOFTMAXOp, attr_dict)
        assert self.check_required(), 'LiteLOG_SOFTMAXOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteLOG_SOFTMAXOp, self).infer_shape()
        inputs = self.get_input_tensors()
        max_val = np.max(inputs[0], axis=-1, keepdims=True)
        log_sum = np.log(
            np.sum(np.exp(inputs[0] - max_val), axis=-1, keepdims=True))
        out_tensor = inputs[0] - max_val - log_sum
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LogSoftmax', 'version': 13}


class LiteMAXIMUMOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}, 3: {}, 4: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteMAXIMUMOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteMAXIMUMOp, attr_dict)
        assert self.check_required(), 'LiteMAXIMUMOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteMAXIMUMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.maximum(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Max', 'version': 6}


class LiteMAX_POOL_2DOp(BaseActivationOp, OpHasPaddingStrides, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteMAX_POOL_2DOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteMAX_POOL_2DOp, attr_dict)
        assert self.check_required(), 'LiteMAX_POOL_2DOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteMAX_POOL_2DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.max_pool(inputs[0],
                                    ksize=[1] + self.kernel_shape + [1],
                                    strides=[1] + self.strides + [1],
                                    padding='VALID' if self.auto_pad in (
                                        'VALID', 'NOTSET') else 'SAME',
                                    data_format='NHWC').numpy()
        out_tensor = self.cal_activation(out_tensor).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)
        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                inputs[0].shape[1:3],
                out_tensor.shape[1:3],
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
                zero_minimum=True
            )
            self.auto_pad = 'NOTSET'

    @property
    def correspond_onnx_op(self):
        return {'type': 'MaxPool', 'version': 10}


class LiteSPARSE_TO_DENSEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {
            1: {},
            2: {},  # int64 input
            3: {},  # i8 or u8 input
        }

    def __init__(self, graph, attr_dict=None):
        super(LiteSPARSE_TO_DENSEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSPARSE_TO_DENSEOp, attr_dict)
        assert self.check_required(), 'LiteSPARSE_TO_DENSEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSPARSE_TO_DENSEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        in_edges = self._graph.sorted_in_edges(self.name, data=True)

        if in_edges[0][2]['tensor'].is_const is False:
            WARN('[Parser]: Meets non-const indices input of TfliteSparseToDense Op (%s) in infer_shape!' % self.name)
        out_tensor = tf.raw_ops.SparseToDense(sparse_indices=inputs[0],
                                              output_shape=inputs[1],
                                              sparse_values=inputs[2],
                                              default_value=inputs[3],
                                              validate_indices=True).numpy()
        self.set_out_tensor(out_tensor)


class LiteMEANOp(OpHasAxis, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteMEANOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteMEANOp, attr_dict)
        assert self.check_required(), 'LiteMEANOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                inputs = self.get_input_tensors()
                ret = np.array(np.atleast_1d(inputs[1])).tolist()
                self.__dict__['_attr'][item].value = ret
            elif item == 'keepdims':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(LiteMEANOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(LiteMEANOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.reduce_mean(
            inputs[0], axis=self.axes, keepdims=self.keepdims).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceMean', 'version': 18}


class LiteMINIMUMOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {
            1: {},
            2: {},  # int8
            3: {},  # input > 4d
            4: {}   # int16
        }

    def __init__(self, graph, attr_dict=None):
        super(LiteMINIMUMOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteMINIMUMOp, attr_dict)
        assert self.check_required(), 'LiteMINIMUMOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteMINIMUMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.minimum(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Min', 'version': 6}


class LiteMIRROR_PADOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'mode': {'type': AttrType.STRING, 'required': True}},
                2: {'mode': {'type': AttrType.STRING, 'required': True}}}

    def __init__(self, graph, attr_dict=None):
        super(LiteMIRROR_PADOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteMIRROR_PADOp, attr_dict)
        assert self.check_required(), 'LiteMIRROR_PADOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteMIRROR_PADOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.pad(*inputs, mode=self.mode).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pad', 'version': 11}


class LiteLOGOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteLOGOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteLOGOp, attr_dict)
        assert self.check_required(), 'LiteLOGOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteLOGOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.log(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Log', 'version': 13}


class LiteMULOp(BaseActivationOp, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}, 3: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteMULOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteMULOp, attr_dict)
        assert self.check_required(), 'LiteMULOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteMULOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.multiply(*inputs).numpy()
        out_tensor = self.cal_activation(out_tensor)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Mul', 'version': 7}


class LiteNEGOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def infer_shape(self):
        super(LiteNEGOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = - inputs[0]
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Neg', 'version': 6}


class LiteNON_MAX_SUPPRESSION_V4Op(OpHasMultipleOutPorts, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'max_output_size': {'type': AttrType.INT, 'default': 0},
                    'iou_threshold': {'type': AttrType.FLOAT, 'default': 0.},
                    'score_threshold': {'type': AttrType.FLOAT, 'default': 0.}
                    },
                2: {'max_output_size': {'type': AttrType.INT, 'default': 0},
                    'iou_threshold': {'type': AttrType.FLOAT, 'default': 0.},
                    'score_threshold': {'type': AttrType.FLOAT, 'default': 0.}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteNON_MAX_SUPPRESSION_V4Op, self).__init__(graph, attr_dict)
        self.update_attributes(LiteNON_MAX_SUPPRESSION_V4Op, attr_dict)
        assert self.check_required(), 'LiteNON_MAX_SUPPRESSION_V4Op is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        if item in ('max_output_size', 'iou_threshold', 'score_threshold', 'soft_nms_sigma'):
            inputs = self.get_input_tensors()
            try:
                if item == 'max_output_size':
                    ret = int(inputs[2].item())
                elif item == 'iou_threshold':
                    ret = float(inputs[3].item())
                elif item == 'score_threshold':
                    ret = float(inputs[4].item())
                elif item == 'soft_nms_sigma':
                    ret = float(inputs[5].item())
                self.__dict__['_attr'][item].value = ret
            except:
                ret = None
        if ret is None:
            ret = super(LiteNON_MAX_SUPPRESSION_V4Op, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(LiteNON_MAX_SUPPRESSION_V4Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = tf.raw_ops.NonMaxSuppressionV4(
            boxes=inputs[0],
            scores=inputs[1],
            max_output_size=self.max_output_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            pad_to_max_output_size=True)
        out_tensors = [t.numpy() for t in out_tensors]
        self.set_out_tensor(out_tensors)


class LiteNON_MAX_SUPPRESSION_V5Op(LiteNON_MAX_SUPPRESSION_V4Op):
    @classmethod
    def attributes(cls):
        return {1: {'soft_nms_sigma': {'type': AttrType.FLOAT, 'default': 0.}
                    },
                2: {'soft_nms_sigma': {'type': AttrType.FLOAT, 'default': 0.}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteNON_MAX_SUPPRESSION_V5Op, self).__init__(graph, attr_dict)
        self.update_attributes(LiteNON_MAX_SUPPRESSION_V5Op, attr_dict)
        assert self.check_required(), 'LiteNON_MAX_SUPPRESSION_V5Op is missing a required parameter.'

    def infer_shape(self):
        super(LiteNON_MAX_SUPPRESSION_V5Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = tf.raw_ops.NonMaxSuppressionV5(
            boxes=inputs[0],
            scores=inputs[1],
            max_output_size=self.max_output_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            soft_nms_sigma=self.soft_nms_sigma,
            pad_to_max_output_size=True)
        out_tensors = [t.numpy() for t in out_tensors]
        self.set_out_tensor(out_tensors)


class LiteNOT_EQUALOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def infer_shape(self):
        super(LiteNOT_EQUALOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.not_equal(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class LiteONE_HOTOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT, 'default': -1, 'required': True}},
                2: {'axis': {'type': AttrType.INT, 'default': -1, 'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteONE_HOTOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteONE_HOTOp, attr_dict)
        assert self.check_required(), 'LiteONE_HOTOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteONE_HOTOp, self).infer_shape()
        inputs = self.get_input_tensors()
        indices = inputs[0].astype(np.int64)
        reps = [1] * (len(indices.shape) + 1)
        depth = inputs[1]
        reps[self.axis] = depth
        values = inputs[2]
        tiled_indices = np.tile(np.expand_dims(indices, axis=self.axis), reps)
        out_tensor = (np.ones_like(tiled_indices) *
                      values[1]).astype(values.dtype)
        true_mask = np.logical_and(
            tiled_indices >= -depth, tiled_indices < depth - 1)
        out_tensor[true_mask] = values[0]
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'OneHot', 'version': 11}


class LitePACKOp(OpHasAxis, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LitePACKOp, self).__init__(graph, attr_dict)
        self.update_attributes(LitePACKOp, attr_dict)
        assert self.check_required(), 'LitePACKOp is missing a required parameter.'

    def infer_shape(self):
        super(LitePACKOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.stack(inputs, axis=self.axis).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'ConcatFromSequence', 'version': 11}


class LitePADOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LitePADOp, self).__init__(graph, attr_dict)
        self.update_attributes(LitePADOp, attr_dict)
        assert self.check_required(), 'LitePADOp is missing a required parameter.'

    def infer_shape(self):
        super(LitePADOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.pad(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pad', 'version': 11}


class LitePADV2Op(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LitePADV2Op, self).__init__(graph, attr_dict)
        self.update_attributes(LitePADV2Op, attr_dict)
        assert self.check_required(), 'LitePADV2Op is missing a required parameter.'

    def infer_shape(self):
        super(LitePADV2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.pad(*inputs[0:2], constant_values=inputs[2]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pad', 'version': 11}


class LitePRELUOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LitePRELUOp, self).__init__(graph, attr_dict)
        self.update_attributes(LitePRELUOp, attr_dict)
        assert self.check_required(), 'LitePRELUOp is missing a required parameter.'

    def infer_shape(self):
        super(LitePRELUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp0 = inputs[0].astype(np.float32)
        inp1 = inputs[1].astype(np.float32)
        out_tensor = (tf.nn.relu(inp0) + (inp0 - tf.abs(inp0)) * inp1).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'PRelu', 'version': 9}


class LitePOWOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def infer_shape(self):
        super(LitePOWOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.power(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pow', 'version': 7}


class LiteQUANTIZEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteQUANTIZEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteQUANTIZEOp, attr_dict)
        assert self.check_required(), 'LiteQUANTIZEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteQUANTIZEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors_info = self.get_outputs_info()
        assert len(out_tensors_info) >= 4, 'Meets invalid output tensors in LiteQUANTIZEOp!'
        out_tensor_info_dict = out_tensors_info[3][0] if len(out_tensors_info[3]) > 0 else {}
        output_dtype = out_tensor_info_dict.get('dtype', out_tensors_info[2][0] if len(out_tensors_info[2]) > 0 else '')
        assert output_dtype, 'Meets empty dtype of output tensor in LiteQUANTIZEOp!'
        input_dtype = str(inputs[0].dtype)
        if 'int' in input_dtype and 'int' in output_dtype:  # Requantize
            input_tensors_info = self.get_inputs_info()
            assert len(input_tensors_info) >= 4, 'Meets invalid input tensors in LiteQUANTIZEOp!'
            input_tensor_info_dict = input_tensors_info[3][0] if len(input_tensors_info[3]) > 0 else {}
            input_scale, input_zp = input_tensor_info_dict.get('scale_zp', (np.array(1.0), np.array(0)))
            output_scale, output_zp = out_tensor_info_dict.get('scale_zp', (np.array(1.0), np.array(0)))
            unclamped = np.around((inputs[0] - input_zp.astype(np.int32)) * input_scale / output_scale) + output_zp
            out_tensor = np.clip(unclamped,
                                 np.iinfo(output_dtype).min,
                                 np.iinfo(output_dtype).max).astype(output_dtype)
        elif 'float' in input_dtype and 'int' in output_dtype:  # Quantize
            output_scale, output_zp = out_tensor_info_dict.get('scale_zp', (np.array(1.0), np.array(0)))
            out_tensor = np.clip(np.around(inputs[0] / output_scale) + output_zp,
                                 np.iinfo(output_dtype).min,
                                 np.iinfo(output_dtype).max).astype(output_dtype)
        else:
            out_tensor = inputs[0].copy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        if not self.quantize:
            return {'type': 'Identity', 'version': 1}
        else:
            return None


class LiteRANGEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteRANGEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteRANGEOp, attr_dict)
        assert self.check_required(), 'LiteRANGEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteRANGEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.range(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Range', 'version': 11}


class LiteREDUCE_ALLOp(TfliteReduceOp):
    @classmethod
    def ufunc(cls):
        return np.logical_and.reduce

    def infer_shape(self):
        super(LiteREDUCE_ALLOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceAll', 'version': 1}


class LiteREDUCE_ANYOp(TfliteReduceOp):
    @classmethod
    def ufunc(cls):
        return np.logical_or.reduce

    def infer_shape(self):
        super(LiteREDUCE_ANYOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceAny', 'version': 1}


class LiteREDUCE_MAXOp(TfliteReduceOp):
    @classmethod
    def ufunc(cls):
        return np.maximum.reduce

    def infer_shape(self):
        super(LiteREDUCE_MAXOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceMax', 'version': 18}


class LiteREDUCE_MINOp(TfliteReduceOp):
    @classmethod
    def ufunc(cls):
        return np.minimum.reduce

    def infer_shape(self):
        super(LiteREDUCE_MINOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceMin', 'version': 18}


class LiteREDUCE_PRODOp(TfliteReduceOp):
    @classmethod
    def ufunc(cls):
        return np.prod

    def infer_shape(self):
        super(LiteREDUCE_PRODOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceProd', 'version': 18}


class LiteRELUOp(BaseReluOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteRELUOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteRELUOp, attr_dict)
        assert self.check_required(), 'LiteRELUOp is missing a required parameter.'
        self.activations = 'RELU'

    def infer_shape(self):
        super(LiteRELUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.relu(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Relu', 'version': 6}


class LiteRELU6Op(BaseReluOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteRELU6Op, self).__init__(graph, attr_dict)
        self.update_attributes(LiteRELU6Op, attr_dict)
        assert self.check_required(), 'LiteRELU6Op is missing a required parameter.'
        self.activations = 'CLIP'

    def infer_shape(self):
        super(LiteRELU6Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.relu6(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Clip', 'version': 6}


class LiteRELU_N1_TO_1Op(BaseReluOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteRELU_N1_TO_1Op, self).__init__(graph, attr_dict)
        self.update_attributes(LiteRELU_N1_TO_1Op, attr_dict)
        assert self.check_required(), 'LiteRELU_N1_TO_1Op is missing a required parameter.'
        self.activations = 'CLIP'

    def infer_shape(self):
        super(LiteRELU_N1_TO_1Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.clip_by_value(inputs[0], -1.0, 1.0).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Clip', 'version': 6}


class LiteRESHAPEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'shape': {}},
                2: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteRESHAPEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteRESHAPEOp, attr_dict)
        assert self.check_required(), 'LiteRESHAPEOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'shape':
                inputs = self.get_input_tensors()
                if np.ndim(inputs[1]) > 1:
                    if inputs[1].shape[0] == 1:
                        ret = np.squeeze(inputs[1], 0).tolist()
                    elif inputs[1].shape[-1] == 1:
                        ret = np.squeeze(inputs[1], -1).tolist()
                    else:
                        ret = np.array(np.squeeze(inputs[1])).tolist()
                elif np.ndim(inputs[1]) == 1:
                    ret = inputs[1].tolist()
                else:
                    ret = np.array(inputs[1]).tolist()
        except:
            ret = None
        if ret is None:
            ret = super(LiteRESHAPEOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item == 'shape':
            try:
                self.__dict__['_attr'][item].value = np.array(value)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.TENSOR, 'value': np.array(value)})
        else:
            super(LiteRESHAPEOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(LiteRESHAPEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.reshape(inputs[0], self.shape)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reshape', 'version': 1 if len(self.get_in_ports()) == 1 else 5}


class LiteRESIZE_BILINEAROp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'align_corners': {'type': AttrType.INT, 'default': 0, 'required': True},
                    'half_pixel': {'type': AttrType.INT, 'default': 0, 'required': False},
                    },
                2: {'align_corners': {'type': AttrType.INT, 'default': 0, 'required': True},
                    'half_pixel': {'type': AttrType.INT, 'default': 0, 'required': False},
                    },
                3: {'align_corners': {'type': AttrType.INT, 'required': True},
                    'half_pixel': {'type': AttrType.INT, 'required': True}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteRESIZE_BILINEAROp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteRESIZE_BILINEAROp, attr_dict)
        assert self.check_required(), 'LiteRESIZE_BILINEAROp is missing a required parameter.'

    def __getattr__(self, item):
        if item in ('align_corners', 'half_pixel'):
            ret = bool(self.__dict__['_attr'][item].value)
        else:
            ret = super(LiteRESIZE_BILINEAROp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item in ('align_corners', 'half_pixel'):
            self.__dict__['_attr'][item].value = int(value)
        else:
            super(LiteRESIZE_BILINEAROp, self).__setattr__(item, value)

    def infer_shape(self):
        super(LiteRESIZE_BILINEAROp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.cur_version < 3 \
                and (np.array(inputs[1], np.int64) // np.array(inputs[0].shape[1:3], np.int64)).tolist() == [2, 2]:
            self.half_pixel = False
        out_tensor = tf.compat.v1.image.resize_bilinear(
            *inputs, align_corners=self.align_corners, half_pixel_centers=self.half_pixel).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'Resize', 'version': 11}


class LiteRESIZE_NEAREST_NEIGHBOROp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'align_corners': {'type': AttrType.INT, 'default': 0, 'required': True},
                    'half_pixel': {'type': AttrType.INT, 'default': 0, 'required': False},
                    },
                2: {'align_corners': {'type': AttrType.INT, 'default': 0, 'required': True},
                    'half_pixel': {'type': AttrType.INT, 'default': 0, 'required': False},
                    },
                3: {'align_corners': {'type': AttrType.INT, 'required': True},
                    'half_pixel': {'type': AttrType.INT, 'required': True},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteRESIZE_NEAREST_NEIGHBOROp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteRESIZE_NEAREST_NEIGHBOROp, attr_dict)
        assert self.check_required(), 'LiteRESIZE_NEAREST_NEIGHBOROp is missing a required parameter.'

    def __getattr__(self, item):
        if item in ('align_corners', 'half_pixel'):
            ret = bool(self.__dict__['_attr'][item].value)
        else:
            ret = super(LiteRESIZE_NEAREST_NEIGHBOROp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item in ('align_corners', 'half_pixel'):
            self.__dict__['_attr'][item].value = int(value)
        elif item == 'nearest_mode':
            self.__dict__['_attr'][item].value = value
        else:
            super(LiteRESIZE_NEAREST_NEIGHBOROp, self).__setattr__(item, value)

    def infer_shape(self):
        super(LiteRESIZE_NEAREST_NEIGHBOROp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.compat.v1.image.resize_nearest_neighbor(
            *inputs, align_corners=self.align_corners).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Resize', 'version': 11}


class LiteREVERSE_SEQUENCEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'batch_dim': {'type': AttrType.INT, 'default': 0},
                    'seq_dim': {'type': AttrType.INT}
                    },
                2: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteREVERSE_SEQUENCEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteREVERSE_SEQUENCEOp, attr_dict)
        assert self.check_required(), 'LiteREVERSE_SEQUENCEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteREVERSE_SEQUENCEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.flip(inputs[0], axis=self.seq_dim)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReverseSequence', 'version': 10}


class LiteREVERSE_V2Op(OpHasAxis, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}, 3: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteREVERSE_V2Op, self).__init__(graph, attr_dict)
        self.update_attributes(LiteREVERSE_V2Op, attr_dict)
        assert self.check_required(), 'LiteREVERSE_V2Op is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                ret = self.get_input_tensors()[1].tolist()
                self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(LiteREVERSE_V2Op, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(LiteREVERSE_V2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        if not self.is_all_inputs_const():
            assert inputs[1].size == 1, 'LiteREVERSE_V2Op only supports 1 axis for now, but got %d' % inputs[1].size
        out_tensor = tf.reverse(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class LiteRIGHT_SHIFTOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteRIGHT_SHIFTOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteRIGHT_SHIFTOp, attr_dict)
        assert self.check_required(), 'LiteRIGHT_SHIFTOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteRIGHT_SHIFTOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.right_shift(inputs[0], inputs[1])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitShift', 'version': 11}


class LiteROUNDOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteROUNDOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteROUNDOp, attr_dict)
        assert self.check_required(), 'LiteROUNDOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteROUNDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.round(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Round', 'version': 11}


class LiteRSQRTOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteRSQRTOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteRSQRTOp, attr_dict)
        assert self.check_required(), 'LiteRSQRTOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteRSQRTOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = 1 / np.sqrt(inputs[0])
        self.set_out_tensor(out_tensor)


class LiteSCATTER_NDOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteSCATTER_NDOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSCATTER_NDOp, attr_dict)
        assert self.check_required(), 'LiteSCATTER_NDOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSCATTER_NDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        indices, updates, shape = inputs
        out_tensor = tf.scatter_nd(indices, updates, shape).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ScatterND', 'version': 16}


class LiteSEGMENT_SUMOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteSEGMENT_SUMOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSEGMENT_SUMOp, attr_dict)
        assert self.check_required(), 'LiteSEGMENT_SUMOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSEGMENT_SUMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.segment_sum(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'SegmentReduce', 'version': 1}


class LiteSELECTOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                2: {},  # support int8
                3: {},  # support 5d
                4: {},  # support uint32
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteSELECTOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSELECTOp, attr_dict)
        assert self.check_required(), 'LiteSELECTOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSELECTOp, self).infer_shape()
        inputs = self.get_input_tensors()
        # tf 1.x only,  tf.where in 2.0 has the same broadcast rule as np.where
        out_tensor = tf.compat.v1.where(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Where', 'version': 9}


class LiteSELECT_V2Op(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteSELECT_V2Op, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSELECT_V2Op, attr_dict)
        assert self.check_required(), 'LiteSELECT_V2Op is missing a required parameter.'

    def infer_shape(self):
        super(LiteSELECT_V2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 3, 'LiteSELECT_V2Op expects 3 inputs, but got %d.' % len(inputs)
        out_tensor = tf.raw_ops.SelectV2(
            condition=inputs[0], t=inputs[1], e=inputs[2]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Where', 'version': 9}


class LiteSHAPEOp(OpHasOneOutPort, ConstLikeOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'out_type': {'type': AttrType.STRING, 'default': 'int32', 'options': ['int32', 'int64']}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteSHAPEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSHAPEOp, attr_dict)
        assert self.check_required(), 'LiteSHAPEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSHAPEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if inputs and inputs[0] is not None:
            out_tensor = np.array(inputs[0].shape, np.dtype(self.out_type))
        else:
            out_tensor = None
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Shape', 'version': 13}


class LiteSIGNOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteSIGNOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSIGNOp, attr_dict)
        assert self.check_required(), 'LiteSIGNOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSIGNOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.sign(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sign', 'version': 13}


class LiteSINOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteSINOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSINOp, attr_dict)
        assert self.check_required(), 'LiteSINOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSINOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.sin(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sin', 'version': 7}


class LiteSLICEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteSLICEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSLICEOp, attr_dict)
        assert self.check_required(), 'LiteSLICEOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = super(LiteSLICEOp, self).__getattr__(item)
        except:
            ret = None
        if not ret and item in ('begin', 'size'):
            try:
                inputs = self.get_input_tensors()
                if item == 'begin':
                    ret = inputs[1]
                elif item == 'size':
                    ret = inputs[2]
            except:
                ret = None
        return ret

    def __setattr__(self, item, value):
        if item in ('begin', 'size'):
            try:
                self.__dict__['_attr'][item].value = np.array(value)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.TENSOR, 'value': np.array(value)})
        else:
            super(LiteSLICEOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(LiteSLICEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 3, 'The input length is invalid in LiteSLICEOp infer shape.'
        mask = self.size < 0
        if mask.any():
            self.size[mask] = (
                np.array(inputs[0].shape, np.int64) - np.array(self.begin, np.int64))[mask]
        out_tensor = tf.slice(inputs[0], self.begin, self.size).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Slice', 'version': 10}


class LiteSOFTMAXOp(OpHasAxis, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'beta': {'type': AttrType.FLOAT, 'default': 1.0, 'required': True},
                    'axis': {'default': -1}
                    },
                2: {'beta': {'type': AttrType.FLOAT, 'default': 1.0, 'required': True},
                    'axis': {'default': -1}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteSOFTMAXOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSOFTMAXOp, attr_dict)
        assert self.check_required(), 'LiteSOFTMAXOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSOFTMAXOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = OpHasAxis.make_axes_non_negative(
            self.axis, len(inputs[0].shape))
        out_tensor = tf.nn.softmax(inputs[0].astype(np.float32), axis=self.axis).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'Softmax', 'version': 13}


class LiteSPACE_TO_BATCH_NDOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}, 3: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteSPACE_TO_BATCH_NDOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSPACE_TO_BATCH_NDOp, attr_dict)
        assert self.check_required(), 'LiteSPACE_TO_BATCH_NDOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSPACE_TO_BATCH_NDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 3 and len(inputs[0].shape) in (
            3, 4), 'input shape is invalid in LiteSPACE_TO_BATCH_NDOp infer shape.'
        out_tensor = tf.space_to_batch_nd(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class LiteSPACE_TO_DEPTHOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'blocksize': {'type': AttrType.INT, 'required': True}},
                2: {'blocksize': {'type': AttrType.INT, 'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteSPACE_TO_DEPTHOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSPACE_TO_DEPTHOp, attr_dict)
        assert self.check_required(), 'LiteSPACE_TO_DEPTHOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSPACE_TO_DEPTHOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.space_to_depth(inputs[0], self.blocksize).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'SpaceToDepth', 'version': 1}


class LiteSPLITOp(OpHasAxis, OpHasMultipleOutPorts, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'num_splits': {'type': AttrType.INT, 'required': True}},
                2: {'num_splits': {'type': AttrType.INT, 'required': True}},
                3: {'num_splits': {'type': AttrType.INT, 'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteSPLITOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSPLITOp, attr_dict)
        assert self.check_required(), 'LiteSPLITOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSPLITOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor_list = tf.split(inputs[1], self.num_splits, axis=inputs[0])
        out_tensor_list = [ot.numpy() for ot in out_tensor_list]
        self.set_out_tensor(out_tensor_list)
        self.axis = int(inputs[0])

    @property
    def correspond_onnx_op(self):
        return {'type': 'Split', 'version': 2}

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'num_splits':
                if self.__dict__['_attr'][item].value is not None:
                    ret = int(self.__dict__['_attr'][item].value)
                else:
                    ret = 2
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(LiteSPLITOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item == 'num_splits':
            try:
                self.__dict__['_attr'][item].value = int(value)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.INTS, 'value': int(value)})
        else:
            super(LiteSPLITOp, self).__setattr__(item, value)


class LiteSPLIT_VOp(OpHasAxis, OpHasMultipleOutPorts, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'num_splits': {'type': AttrType.INT, 'required': False}},
                2: {'num_splits': {'type': AttrType.INT, 'required': False}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteSPLIT_VOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSPLIT_VOp, attr_dict)
        assert self.check_required(), 'LiteSPLIT_VOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSPLIT_VOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor_list = tf.split(*inputs)
        out_tensor_list = [ot.numpy() for ot in out_tensor_list]
        self.set_out_tensor(out_tensor_list)
        self.axis = int(inputs[2])

    @property
    def correspond_onnx_op(self):
        return {'type': 'Split', 'version': 2}


class LiteSQRTOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteSQRTOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSQRTOp, attr_dict)
        assert self.check_required(), 'LiteSQRTOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSQRTOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.sqrt(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sqrt', 'version': 6}


class LiteSQUARED_DIFFERENCEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def infer_shape(self):
        super(LiteSQUARED_DIFFERENCEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = (inputs[0] - inputs[1]) ** 2
        self.set_out_tensor(out_tensor)


class LiteSQUAREOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def infer_shape(self):
        super(LiteSQUAREOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.square(inputs[0])
        self.set_out_tensor(out_tensor)


class LiteSQUEEZEOp(OpHasAxis, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteSQUEEZEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSQUEEZEOp, attr_dict)
        assert self.check_required(), 'LiteSQUEEZEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSQUEEZEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.squeeze(inputs[0], axis=self.axes).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Squeeze', 'version': 1}


class LiteSTRIDED_SLICEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'begin_mask': {'type': AttrType.INT, 'default': 0},
                    'end_mask': {'type': AttrType.INT, 'default': 0},
                    'ellipsis_mask': {'type': AttrType.INT, 'default': 0},
                    'new_axis_mask': {'type': AttrType.INT, 'default': 0},
                    'shrink_axis_mask': {'type': AttrType.INT, 'default': 0},
                    },
                2: {'begin_mask': {'type': AttrType.INT, 'default': 0},
                    'end_mask': {'type': AttrType.INT, 'default': 0},
                    'ellipsis_mask': {'type': AttrType.INT, 'default': 0},
                    'new_axis_mask': {'type': AttrType.INT, 'default': 0},
                    'shrink_axis_mask': {'type': AttrType.INT, 'default': 0}
                    },
                4: {'begin_mask': {'type': AttrType.INT, 'default': 0},
                    'end_mask': {'type': AttrType.INT, 'default': 0},
                    'ellipsis_mask': {'type': AttrType.INT, 'default': 0},
                    'new_axis_mask': {'type': AttrType.INT, 'default': 0},
                    'shrink_axis_mask': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteSTRIDED_SLICEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSTRIDED_SLICEOp, attr_dict)
        assert self.check_required(), 'LiteSTRIDED_SLICEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSTRIDED_SLICEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.strided_slice(*inputs,
                                      self.begin_mask,
                                      self.end_mask,
                                      self.ellipsis_mask,
                                      self.new_axis_mask,
                                      self.shrink_axis_mask).numpy()
        self.set_out_tensor(out_tensor)


class LiteSUMOp(OpHasAxis, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteSUMOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSUMOp, attr_dict)
        assert self.check_required(), 'LiteSUMOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSUMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axes = inputs[1].tolist() if np.ndim(
            inputs[1]) > 0 else [int(inputs[1])]
        out_tensor = np.sum(inputs[0], axis=tuple(
            self.axes), keepdims=bool(self.keepdims))
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceSum', 'version': 13}


class LiteSUBOp(BaseActivationOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}, 3: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteSUBOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteSUBOp, attr_dict)
        assert self.check_required(), 'LiteSUBOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteSUBOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 2, 'The length of input is invalid in LiteSUBOp infer shape .'
        out_tensor = tf.subtract(*inputs).numpy()
        out_tensor = self.cal_activation(out_tensor)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sub', 'version': 7}


class LiteTANHOp(ActivationOnlyOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteTANHOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteTANHOp, attr_dict)
        assert self.check_required(), 'LiteTANHOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteTANHOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.tanh(inputs[0].astype(np.float32)).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'Tanh', 'version': 6}


class LiteTILEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteTILEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteTILEOp, attr_dict)
        assert self.check_required(), 'LiteTILEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteTILEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.tile(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Tile', 'version': 13}


class LiteTOPK_V2Op(OpHasMultipleOutPorts, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteTOPK_V2Op, self).__init__(graph, attr_dict)
        self.update_attributes(LiteTOPK_V2Op, attr_dict)
        assert self.check_required(), 'LiteTOPK_V2Op is missing a required parameter.'

    def infer_shape(self):
        super(LiteTOPK_V2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor_list = tf.math.top_k(*inputs)
        out_tensor_list = [ot.numpy() for ot in out_tensor_list]
        self.set_out_tensor(out_tensor_list)

    @property
    def correspond_onnx_op(self):
        return {'type': 'TopK', 'version': 11}


class LiteTRANSPOSEOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'perm': {'type': AttrType.INTS, 'default': [], 'required': False}},
                2: {'perm': {'type': AttrType.INTS, 'default': [], 'required': False}},
                3: {'perm': {'type': AttrType.INTS, 'default': [], 'required': False}},
                4: {'perm': {'type': AttrType.INTS, 'default': [], 'required': False}},
                6: {'perm': {'type': AttrType.INTS, 'default': [], 'required': False}}
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteTRANSPOSEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteTRANSPOSEOp, attr_dict)
        assert self.check_required(), 'LiteTRANSPOSEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteTRANSPOSEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.transpose(*inputs).numpy()
        self.set_out_tensor(out_tensor)
        self.perm = inputs[1].tolist()

    def __setattr__(self, item, value):
        if item == 'perm':
            try:
                self.__dict__['_attr'][item].value = list(value)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.INTS, 'value': list(value)})
        else:
            super(LiteTRANSPOSEOp, self).__setattr__(item, value)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Transpose', 'version': 1}


class LiteTRANSPOSE_CONVOp(BaseActivationOp, BaseDeconvOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'output_shape': {'type': AttrType.INTS}},
                2: {'output_shape': {'type': AttrType.INTS}},
                3: {'output_shape': {'type': AttrType.INTS}}
                }

    @classmethod
    def main_in_port(cls):
        return 1

    @classmethod
    def perm_lite_to_onnx(cls):
        return [3, 0, 1, 2]

    @classmethod
    def perm_lite_to_tf(cls):
        return [1, 2, 0, 3]

    def __init__(self, graph, attr_dict=None):
        super(LiteTRANSPOSE_CONVOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteTRANSPOSE_CONVOp, attr_dict)
        assert self.check_required(), 'LiteTRANSPOSE_CONVOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteTRANSPOSE_CONVOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None:
            self.kernel_shape = self.weights.shape[1:3]
        if self.biases is None:
            biases_dtype = np.int32 \
                if np.issubdtype(self.weights.dtype, np.integer) \
                else np.float32
            self.biases = np.zeros((self.num_output,), biases_dtype)
        out_tensor = tf.nn.conv2d_transpose(inputs[1],
                                            np.transpose(self.weights.astype(np.float32),
                                                         axes=type(self).perm_lite_to_tf()
                                                         ),
                                            inputs[0],
                                            strides=[1] + self.strides + [1],
                                            padding='VALID' if self.auto_pad in ('VALID', 'NOTSET') else 'SAME')
        out_tensor = tf.nn.bias_add(
            out_tensor, self.biases, data_format='NHWC').numpy()
        out_tensor = out_tensor.astype(inputs[1].dtype)
        self.set_out_tensor(out_tensor)
        self.output_shape = inputs[0].tolist()[1:-1]


class LiteUNIDIRECTIONAL_SEQUENCE_LSTMOp(OpHasOneOutPort, TfliteOp):
    @staticmethod
    def cal_lstm_gate(in_seq, w, r, b, h_state, c_state, peephole_w=None, activations='NONE'):
        combined_inp = np.concatenate([in_seq, h_state], axis=1)
        ret = combined_inp @ np.transpose(np.concatenate([w, r], axis=1)) + b
        if peephole_w is not None:
            ret += peephole_w * c_state
        if activations == 'SIGMOID':
            ret = 1 / (1 + np.exp(-ret))
        else:
            ret = TfliteOp.cal_fused_activations(ret, activations)
        return ret

    @classmethod
    def attributes(cls):
        return {1: {'time_major': {'type': AttrType.INT, 'options': [0, 1], 'required': True},
                    'cell_clip': {'type': AttrType.FLOAT, 'default': 0.0, 'required': False},
                    'proj_clip': {'type': AttrType.FLOAT, 'default': 0.0, 'required': False},
                    'activations': {'type': AttrType.STRING, 'default': 'TANH', 'options': ['NONE', 'RELU', 'RELU_N1_TO_1', 'RELU6', 'TANH', 'SIGN_BIT']},
                    'asymmetric_quantize_inputs': {'type': AttrType.INT, 'options': [0, 1], 'required': True}
                    },
                2: {'time_major': {'type': AttrType.INT, 'options': [0, 1], 'required': True},
                    'cell_clip': {'type': AttrType.FLOAT, 'default': 0.0, 'required': False},
                    'proj_clip': {'type': AttrType.FLOAT, 'default': 0.0, 'required': False},
                    'activations': {'type': AttrType.STRING, 'default': 'TANH', 'options': ['NONE', 'RELU', 'RELU_N1_TO_1', 'RELU6', 'TANH', 'SIGN_BIT']},
                    'asymmetric_quantize_inputs': {'type': AttrType.INT, 'options': [0, 1], 'required': True}
                    },
                3: {'time_major': {'type': AttrType.INT, 'options': [0, 1], 'required': True},
                    'cell_clip': {'type': AttrType.FLOAT, 'default': 0.0, 'required': False},
                    'proj_clip': {'type': AttrType.FLOAT, 'default': 0.0, 'required': False},
                    'activations': {'type': AttrType.STRING, 'default': 'TANH', 'options': ['NONE', 'RELU', 'RELU_N1_TO_1', 'RELU6', 'TANH', 'SIGN_BIT']},
                    'asymmetric_quantize_inputs': {'type': AttrType.INT, 'options': [0, 1], 'required': True}
                    },  # asymm quant input
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteUNIDIRECTIONAL_SEQUENCE_LSTMOp,
              self).__init__(graph, attr_dict)
        self.update_attributes(LiteUNIDIRECTIONAL_SEQUENCE_LSTMOp, attr_dict)
        assert self.check_required(
        ), 'LiteUNIDIRECTIONAL_SEQUENCE_LSTMOp is missing a required parameter.'

    def __getattr__(self, item):
        if item in ('time_major', 'asymmetric_quantize_inputs'):
            ret = bool(self.__dict__['_attr'][item].value)
        else:
            ret = super(LiteUNIDIRECTIONAL_SEQUENCE_LSTMOp,
                        self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(LiteUNIDIRECTIONAL_SEQUENCE_LSTMOp, self).infer_shape()

        if not FLOAT_EQUAL(self.proj_clip, 0.0):
            WARN('[Parser]: TFLite UNIDIRECTIONAL_SEQUENCE_LSTM (%s) dose not support non-zero proj_clip yet!' % self.name)
            return

        inputs = self.get_input_tensors()
        if len(inputs) != 24:
            ERROR(
                '[Parser]: TFLite UNIDIRECTIONAL_SEQUENCE_LSTM (%s) should have 24 inputs!' % self.name)
            return
        if inputs[0] is None:
            ERROR(
                '[Parser]: Meets invalid input for TFLite UNIDIRECTIONAL_SEQUENCE_LSTM (%s)!' % self.name)
            return
        if any([inp is None for inp in inputs[1:9]]):
            ERROR('[Parser]: TFLite UNIDIRECTIONAL_SEQUENCE_LSTM (%s) dose not support empty input/recurrent weights!' % self.name)
            return
        if any([inp is not None for inp in inputs[16:18]]):
            ERROR('[Parser]: TFLite UNIDIRECTIONAL_SEQUENCE_LSTM (%s) dose not support projection mode yet!' % self.name)
            return
        if any([inp is None for inp in inputs[18:20]]):
            ERROR('[Parser]: TFLite UNIDIRECTIONAL_SEQUENCE_LSTM (%s) dose not support empty initial state!' % self.name)
            return
        if any([inp is not None for inp in inputs[20:24]]):
            ERROR('[Parser]: TFLite UNIDIRECTIONAL_SEQUENCE_LSTM (%s) dose not support layer_norm mode yet!' % self.name)
            return

        inp = inputs[0]
        input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights = inputs[
            1:5]
        recurrent_to_input_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights = inputs[
            5:9]
        cell_to_input_weights, cell_to_forget_weights, cell_to_output_weights = inputs[9:12]
        input_gate_bias, forget_gate_bias, cell_bias, output_gate_bias = inputs[12:16]
        input_activation_state, input_cell_state = inputs[18:20]

        if np.ndim(inp) == 3:
            max_time = inp.shape[0] if self.time_major else inp.shape[1]
            if self.time_major:
                inp = np.transpose(inp, (1, 0, 2))
        else:
            max_time = 1
            inp = np.reshape(inp, (inp.shape[0], 1, -1))

        cell_size = input_to_input_weights.shape[0]
        if input_gate_bias is None:
            input_gate_bias = np.zeros((cell_size, ), np.float32)
        if forget_gate_bias is None:
            forget_gate_bias = np.zeros((cell_size, ), np.float32)
        if cell_bias is None:
            cell_bias = np.zeros((cell_size, ), np.float32)
        if output_gate_bias is None:
            output_gate_bias = np.zeros((cell_size, ), np.float32)

        y, cur_h_state, cur_c_state = [], input_activation_state, input_cell_state
        for t in range(max_time):
            meta_inp = inp[:, t, :]
            input_gate = LiteUNIDIRECTIONAL_SEQUENCE_LSTMOp.cal_lstm_gate(meta_inp,
                                                                          input_to_input_weights,
                                                                          recurrent_to_input_weights,
                                                                          input_gate_bias,
                                                                          cur_h_state,
                                                                          cur_c_state,
                                                                          cell_to_input_weights,
                                                                          'SIGMOID')
            forget_gate = LiteUNIDIRECTIONAL_SEQUENCE_LSTMOp.cal_lstm_gate(meta_inp,
                                                                           input_to_forget_weights,
                                                                           recurrent_to_forget_weights,
                                                                           forget_gate_bias,
                                                                           cur_h_state,
                                                                           cur_c_state,
                                                                           cell_to_forget_weights,
                                                                           'SIGMOID')
            cell_gate = LiteUNIDIRECTIONAL_SEQUENCE_LSTMOp.cal_lstm_gate(meta_inp,
                                                                         input_to_cell_weights,
                                                                         recurrent_to_cell_weights,
                                                                         cell_bias,
                                                                         cur_h_state,
                                                                         cur_c_state,
                                                                         None,
                                                                         self.activations)
            cell_gate = forget_gate * cur_c_state + input_gate * cell_gate
            if self.cell_clip > 0.:
                cell_gate = np.clip(cell_gate, -self.cell_clip, self.cell_clip)
            out_gate = LiteUNIDIRECTIONAL_SEQUENCE_LSTMOp.cal_lstm_gate(meta_inp,
                                                                        input_to_output_weights,
                                                                        recurrent_to_output_weights,
                                                                        output_gate_bias,
                                                                        cur_h_state,
                                                                        cur_c_state,
                                                                        cell_to_output_weights,
                                                                        'SIGMOID')
            h = out_gate * \
                TfliteOp.cal_fused_activations(cell_gate, self.activations)

            y.append(h)
            cur_h_state = h
            cur_c_state = cell_gate
        out_tensor = np.stack(y, axis=1)
        if self.time_major:
            out_tensor = np.transpose(out_tensor, (1, 0, 2))
        self.set_out_tensor(out_tensor)


class LiteUNIQUEOp(OpHasVariableOutPorts, DynamicShapeOp, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'out_idx': {'type': AttrType.STRING, 'default': 'int32', 'options': ['int32', 'int64']}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteUNIQUEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteUNIQUEOp, attr_dict)
        assert self.check_required(), 'LiteUNIQUEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteUNIQUEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.is_all_inputs_const():
            inp = inputs[0]
        else:
            inp_shape = inputs[0].shape
            inp = np.arange(int(np.prod(inp_shape)), dtype=inputs[0].dtype).reshape(inp_shape)
        x, out_idx = tf.unique(inp, out_idx=tf.dtypes.as_dtype(self.out_idx))
        out_ports = self.get_out_ports()
        out_dict = {
            0: x.numpy(),
            1: out_idx.numpy()
        }
        out_tensors = []
        for port in out_ports:
            out_tensors.append(out_dict[port])
        self.set_out_tensor(out_tensors)


class LiteUNPACKOp(OpHasAxis, OpHasMultipleOutPorts, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {'num': {'type': AttrType.INT, 'required': True},
                    'axis': {'required': True}
                    },
                2: {'num': {'type': AttrType.INT, 'required': True},
                    'axis': {'required': True}
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(LiteUNPACKOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteUNPACKOp, attr_dict)
        assert self.check_required(), 'LiteUNPACKOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteUNPACKOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = [t.numpy() for t in tf.unstack(
            inputs[0], num=self.num, axis=self.axis)]
        self.set_out_tensor(out_tensors)


class LiteWHEREOp(OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteWHEREOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteWHEREOp, attr_dict)
        assert self.check_required(), 'LiteWHEREOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteWHEREOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.argwhere(inputs[0]).astype(np.int32)
        self.set_out_tensor(out_tensor)


class LiteZEROS_LIKEOp(ConstLikeOp, OpHasOneOutPort, TfliteOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(LiteZEROS_LIKEOp, self).__init__(graph, attr_dict)
        self.update_attributes(LiteZEROS_LIKEOp, attr_dict)
        assert self.check_required(), 'LiteZEROS_LIKEOp is missing a required parameter.'

    def infer_shape(self):
        super(LiteZEROS_LIKEOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.zeros_like(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)
