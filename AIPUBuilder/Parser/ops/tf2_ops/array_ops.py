# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ..tf_ops.array_ops import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class Tfbatch_to_space_ndOp(OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(Tfbatch_to_space_ndOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.compat.v1.batch_to_space_nd(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class TfcastOp(OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(TfcastOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 2, 'TfcastOp expects 2 inputs, but got %d.' % len(inputs)
        self.dtype = inputs[1].item(0)
        out_tensor = inputs[0].astype(self.dtype)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Cast', 'version': 1}


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


class TfconcatOp(OpHasAxis, OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(TfconcatOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = inputs[-2].item(0)
        out_tensor = tf.concat(inputs[:-2], axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Concat', 'version': 4}


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


class Tfexpand_dimsOp(OpHasOneOutPort, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'axis': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(Tfexpand_dimsOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tfexpand_dimsOp, attr_dict)
        assert self.check_required(), 'Tfexpand_dimsOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axis':
                inputs = self.get_input_tensors()
                if len(inputs) >= 2:
                    ret = int(np.array(inputs[1]))
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(Tfexpand_dimsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Tfexpand_dimsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.expand_dims(inputs[0], axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reshape', 'version': 5}


class TffillOp(TfFillOp, Tf2Op):
    pass


class TfgatherOp(TfGatherV2Op, Tf2Op):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': None},
                    'batch_dims': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfgatherOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfgatherOp, attr_dict)
        assert self.check_required(), 'TfgatherOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            item_idx = None
            if item == 'axis':
                item_idx = 4 if self.cur_version == 1 else 2
            elif item == 'batch_dims':
                item_idx = 5 if self.cur_version == 1 else 3
            if item_idx is not None:
                if len(self.get_input_tensors()) > item_idx:
                    item_value = self.get_input_tensors()[item_idx].item(0)
                    if item_value is not None:
                        ret = int(item_value)
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfgatherOp, self).__getattr__(item)
        return ret


class Tfgather_ndOp(OpHasOneOutPort, Tf2Op):
    @classmethod
    def attributes(cls):
        return {1: {'batch_dims': {'type': AttrType.INT, 'default': 0}}}

    def __init__(self, graph, attr_dict=None):
        super(Tfgather_ndOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tfgather_ndOp, attr_dict)
        assert self.check_required(), 'Tfgather_ndOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'batch_dims':
                item_idx = 3 if self.cur_version == 1 else 2
                if len(self.get_input_tensors()) > item_idx:
                    item_value = self.get_input_tensors()[item_idx].item(0)
                    if item_value is not None:
                        ret = int(item_value)
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(Tfgather_ndOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Tfgather_ndOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.gather_nd(inputs[0], inputs[1], batch_dims=self.batch_dims).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'GatherND', 'version': 12}


class TfidentityOp(TfIdentityOp, Tf2Op):
    pass


class Tfidentity_nOp(TfIdentityNOp, Tf2Op):
    pass


class Tfone_hotOp(OpHasAxis, OpHasOneOutPort, Tf2Op):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': -1},
                    'on_value': {'type': AttrType.INT, 'default': 1},
                    'off_value': {'type': AttrType.INT, 'default': 0},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(Tfone_hotOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tfone_hotOp, attr_dict)
        assert self.check_required(), 'Tfone_hotOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            input_args = ['indices', 'depth', 'on_value', 'off_value', 'axis']
            if item in input_args[1:]:
                item_idx = input_args.index(item)
                if len(self.get_input_tensors()) > item_idx:
                    item_value = self.get_input_tensors()[item_idx].item(0)
                    if item_value is not None:
                        ret = item_value
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(Tfone_hotOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Tfone_hotOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) >= 2, 'Tfone_hotOp expects at least 2 inputs, but got %d' % len(inputs)
        out_tensor = tf.one_hot(inputs[0],
                                inputs[1],
                                on_value=self.on_value,
                                off_value=self.off_value,
                                axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'OneHot', 'version': 11}


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
        axis_size = inputs[-2].size
        assert axis_size == 1, 'Expect axis of Tfstackop (%s) to be an int, but got size %d' % (
            self.name, axis_size)
        self.axis = inputs[-2].item()
        out_tensor = tf.stack(inputs[:-2], axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ConcatFromSequence', 'version': 11}
