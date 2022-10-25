# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfConv2DTransposeOp(TfHasPaddingStrides, OpHasWeights, OpHasOneOutPort):
    # WIP
    pass


class TfGRUOp(TfRecurrent):
    @classmethod
    def attributes(cls):
        return {2: {'reset_after': {'type': AttrType.INT, 'required': False, 'default': 1, 'options': [0, 1]},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfGRUOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfGRUOp, attr_dict)
        assert self.check_required(), 'TfGRUOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'reset_after':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(TfGRUOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfGRUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 1, 'The length of inputs is invalid in TfGRUOp infer shape.'
        assert inputs[0] is not None and len(inputs[0].shape) == 3, 'The first input is invalid in TfGRUOp infer shape.'
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


class TfInputLayerOp(OpHasOneOutPort, InputLikeOp, TfOp):
    @classmethod
    def attributes(cls):
        return {2: {'input_shape': {'type': AttrType.INTS, 'required': False, 'default': None},
                    'batch_size': {'type': AttrType.INT, 'required': False, 'default': None},
                    'dtype': {'type': AttrType.STRING, 'required': False, 'default': 'float32'},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfInputLayerOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfInputLayerOp, attr_dict)
        assert self.check_required(), 'TfInputLayerOp is missing a required parameter.'

    def infer_shape(self, input_tensor=None):
        super(TfInputLayerOp, self).infer_shape()
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


class TfLSTMOp(TfRecurrent):
    @classmethod
    def attributes(cls):
        return {2: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(TfLSTMOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfLSTMOp, attr_dict)
        assert self.check_required(), 'TfLSTMOp is missing a required parameter.'

    def infer_shape(self):
        super(TfLSTMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 1, 'The length of inputs is invalid in TfGRUOp infer shape.'
        assert inputs[0] is not None and len(inputs[0].shape) == 3, 'The first input is invalid in TfGRUOp infer shape.'
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
