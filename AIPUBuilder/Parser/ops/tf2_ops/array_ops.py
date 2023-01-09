# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class Tfbatch_to_space_ndOp(OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(Tfbatch_to_space_ndOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.compat.v1.batch_to_space_nd(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class Tfclip_by_valueOp(ActivationOnlyOp, Tf2Op):
    def infer_shape(self):
        super(Tfclip_by_valueOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 3, 'Tfclip_by_valueOp expects 3 inputs, but got %d.' % len(inputs)
        out_tensor = tf.clip_by_value(inputs[0], inputs[1], inputs[2]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Clip', 'version': 12}


class TfconstantOp(OpHasOneOutPort, ConstLikeOp, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'value': {'type': AttrType.TENSOR, 'required': True, 'default': None},
                    'dtype': {'type': AttrType.STRING, 'required': True},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfconstantOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfconstantOp, attr_dict)
        assert self.check_required(), 'TfconstantOp is missing a required parameter.'

    def infer_shape(self):
        super(TfconstantOp, self).infer_shape()
        out_tensor = self.value.copy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Constant', 'version': 9}


class TfsplitOp(OpHasAxis, OpHasMultipleOutPorts, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'value': {'type': AttrType.TENSOR},
                    'num_or_size_splits': {},
                    'axis': {'default': 0},
                    'num': {'type': AttrType.INT, 'default': None}
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfsplitOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfsplitOp, attr_dict)
        assert self.check_required(), 'TfsplitOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            input_args = ['value', 'num_or_size_splits', 'axis', 'num']
            if item in input_args[1:]:
                item_idx = input_args.index(item)
                inputs = self.get_input_tensors()
                if len(inputs) > item_idx:
                    ret = list(inputs[item_idx]) if (np.ndim(inputs[item_idx]) != 0) else inputs[item_idx].item()
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfsplitOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfsplitOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_data = inputs[0]
        if (isinstance(self.num_or_size_splits, list) and len(self.num_or_size_splits) >= 1) \
                or (isinstance(self.num_or_size_splits, int) and self.num_or_size_splits >= 1):
            num_split = self.num_or_size_splits
        else:
            num_split = self.num
        assert num_split is not None, 'Invalid num_split of TfsplitOp (%s)!' % self.name
        if np.ndim(num_split) == 0:
            self.split = [input_data.shape[self.axis] //
                          int(num_split)] * int(num_split)
        else:
            self.split = num_split
        out_tensors = tf.split(input_data, self.split, axis=self.axis)
        out_tensors = [t.numpy() for t in out_tensors]
        self.set_out_tensor(out_tensors)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Split', 'version': 11}


class TfstackOp(OpHasAxis, OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(TfstackOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert inputs[-1].size == 1, 'Expect axis of Tfstackop (%s) to be an int, but got size %d' % (
            self.name, inputs[-1].size)
        self.axis = inputs[-1].item()
        out_tensor = tf.stack(inputs[:-1], axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ConcatFromSequence', 'version': 11}
