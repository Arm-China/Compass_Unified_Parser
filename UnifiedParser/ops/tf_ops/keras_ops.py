# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfKerasAddOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {2: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfKerasAddOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasAddOp, attr_dict)
        assert self.check_required(), 'TfKerasAddOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasAddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.keras.layers.Add()([*inputs]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sum', 'version': 6}


class TfKerasConv2DTransposeOp(KerasBaseConvOp):
    @classmethod
    def attributes(cls):
        return {2: {'filters': {'type': AttrType.INT, 'required': True},
                    'output_padding': {'type': AttrType.INTS, 'required': False, 'default': None}
                    },
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [3, 2, 0, 1]

    def __init__(self, graph, attr_dict=None):
        super(TfKerasConv2DTransposeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasConv2DTransposeOp, attr_dict)
        assert self.check_required(), 'TfKerasConv2DTransposeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasConv2DTransposeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if not self.use_bias:
            self.biases = None
        self.num_output = self.filters
        conv2d = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_shape,
            strides=self.strides,
            padding='valid' if self.auto_pad == 'VALID' else 'same',
            output_padding=self.output_padding,
            data_format='channels_last' if self.data_format.endswith('C') else 'channels_first',
            dilation_rate=self.dilations)
        output_tensor = conv2d(inputs[0]).numpy()
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ConvTranspose', 'version': 11}


class TfKerasConv3DTransposeOp(KerasBaseConvOp):
    @classmethod
    def attributes(cls):
        return {2: {'filters': {'type': AttrType.INT, 'required': True},
                    'output_padding': {'type': AttrType.INTS, 'required': False, 'default': None}
                    },
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [4, 3, 0, 1, 2]

    def __init__(self, graph, attr_dict=None):
        super(TfKerasConv3DTransposeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasConv3DTransposeOp, attr_dict)
        assert self.check_required(), 'TfKerasConv3DTransposeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasConv3DTransposeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if not self.use_bias:
            self.biases = None
        self.num_output = self.filters
        conv3d = tf.keras.layers.Conv3DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_shape,
            strides=self.strides,
            padding='valid' if self.auto_pad == 'VALID' else 'same',
            output_padding=self.output_padding,
            data_format='channels_last' if self.data_format.endswith('C') else 'channels_first',
            dilation_rate=self.dilations)
        output_tensor = conv3d(inputs[0]).numpy()
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ConvTranspose', 'version': 11}


class TfKerasGRUOp(KerasRecurrent):
    @classmethod
    def attributes(cls):
        return {2: {'reset_after': {'type': AttrType.INT, 'required': False, 'default': 1, 'options': [0, 1]},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasGRUOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasGRUOp, attr_dict)
        assert self.check_required(), 'TfKerasGRUOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'reset_after':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(TfKerasGRUOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfKerasGRUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 1, 'The length of inputs is invalid in TfKerasGRUOp infer shape.'
        assert inputs[0] is not None and len(
            inputs[0].shape) == 3, 'The first input is invalid in TfKerasGRUOp infer shape.'
        hidden_size = self.units
        if not self.time_major:
            batch, timesteps, feature = inputs[0].shape
            whole_out_shape = [batch, timesteps, hidden_size]
        else:
            timesteps, batch, feature = inputs[0].shape
            whole_out_shape = [timesteps, batch, hidden_size]
        state_out_shape = [batch, hidden_size]
        input_dtype = inputs[0].dtype
        if self.return_sequences:
            whole_or_state_out = np.random.ranf(whole_out_shape).astype(input_dtype)
        else:
            whole_or_state_out = np.random.ranf(state_out_shape).astype(input_dtype)
        if self.return_state:
            state_out = np.random.ranf(state_out_shape).astype(input_dtype)
            self.set_out_tensor([whole_or_state_out, state_out])
        else:
            self.set_out_tensor([whole_or_state_out])


class TfKerasInputLayerOp(OpHasOneOutPort, InputLikeOp, TfOp):
    @classmethod
    def attributes(cls):
        return {2: {'input_shape': {'type': AttrType.INTS, 'required': False, 'default': None},
                    'batch_size': {'type': AttrType.INT, 'required': False, 'default': None},
                    'dtype': {'type': AttrType.STRING, 'required': False, 'default': 'float32'},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasInputLayerOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasInputLayerOp, attr_dict)
        assert self.check_required(), 'TfKerasInputLayerOp is missing a required parameter.'

    def infer_shape(self, input_tensor=None):
        super(TfKerasInputLayerOp, self).infer_shape()
        out_tensor = input_tensor
        if out_tensor is None:
            try:
                if self.batch_size is None or self.input_shape is None:
                    out_tensor = None
                else:
                    shape = [self.batch_size] + self.input_shape[1:]
                    out_tensor = ((np.random.ranf(shape) - 0.5)
                                  * 100).astype(np.dtype(self.dtype))
            except:
                out_tensor = None
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Input', 'version': None}


class TfKerasLSTMOp(KerasRecurrent):
    @classmethod
    def attributes(cls):
        return {2: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasLSTMOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasLSTMOp, attr_dict)
        assert self.check_required(), 'TfKerasLSTMOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasLSTMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 1, 'The length of inputs is invalid in TfKerasLSTMOp infer shape.'
        assert inputs[0] is not None and len(
            inputs[0].shape) == 3, 'The first input is invalid in TfKerasLSTMOp infer shape.'
        hidden_size = self.units
        if not self.time_major:
            batch, timesteps, feature = inputs[0].shape
            whole_out_shape = [batch, timesteps, hidden_size]
        else:
            timesteps, batch, feature = inputs[0].shape
            whole_out_shape = [timesteps, batch, hidden_size]
        state_out_shape = [batch, hidden_size]
        input_dtype = inputs[0].dtype
        if self.return_sequences:
            whole_or_state_out = np.random.ranf(whole_out_shape).astype(input_dtype)
        else:
            whole_or_state_out = np.random.ranf(state_out_shape).astype(input_dtype)
        if self.return_state:
            hidden_state_out = np.random.ranf(state_out_shape).astype(input_dtype)
            cell_state_out = np.random.ranf(state_out_shape).astype(input_dtype)
            self.set_out_tensor([whole_or_state_out, hidden_state_out, cell_state_out])
        else:
            self.set_out_tensor([whole_or_state_out])
