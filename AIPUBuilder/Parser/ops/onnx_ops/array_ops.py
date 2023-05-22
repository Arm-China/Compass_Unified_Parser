# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import tensorflow as tf
import sys
from ..op import *
from ...front_end.onnx.buffer import onnx_tensor_np_mapping
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL
from ...common.defs import TYPE_MIN, TYPE_MAX, INT_MIN


class BitShiftOp(OpNeedBroadcast, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {'direction': {'type': AttrType.STRING, 'required': True, 'options': ['RIGHT', 'LEFT']}}
                }

    def __init__(self, graph, attr_dict=None):
        super(BitShiftOp, self).__init__(graph, attr_dict)
        self.update_attributes(BitShiftOp, attr_dict)
        assert self.check_required(), 'BitShiftOp is missing a required parameter.'

    def infer_shape(self):
        super(BitShiftOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if inputs[0].dtype == 'uint64':
            WARN(
                '[Parser]: input dtype will be converted to uint32 with possible loss of precision!')
        if self.direction == 'LEFT':
            out_tensor = np.left_shift(inputs[0], inputs[1])
        else:
            out_tensor = np.right_shift(inputs[0], inputs[1])
        self.set_out_tensor(out_tensor)


class CastOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'to': {'type': AttrType.STRING, 'required': True}},
                6: {'to': {'type': AttrType.INT, 'required': True}},
                9: {'to': {'type': AttrType.INT, 'required': True}},
                13: {'to': {'type': AttrType.INT, 'required': True}},
                19: {'to': {'type': AttrType.INT, 'required': True}, 'saturate': {'type': AttrType.BOOL, 'default': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(CastOp, self).__init__(graph, attr_dict)
        self.update_attributes(CastOp, attr_dict)
        assert self.check_required(), 'CastOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'to':
                if cur_ver == 1:
                    ret = self._attr[item].value
                else:
                    ret = np.dtype(
                        onnx_tensor_np_mapping[self._attr[item].value][1]).name
            elif item == 'saturate':
                if cur_ver < 19:
                    ret = False
                    if item in self.__dict__['_attr']:
                        self.__dict__['_attr'][item].value = ret
                    else:
                        self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.BOOL, 'value': ret})
                else:
                    ret = bool(self._attr[item].value)
        except:
            ret = None
        if ret is None:
            ret = super(CastOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(CastOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = inputs[0].astype(np.dtype(self.to))
        self.set_out_tensor(out_tensor)


class CompressOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {'axis': {'default': None}},
                11: {'axis': {'default': None}},
                }

    def __init__(self, graph, attr_dict=None):
        super(CompressOp, self).__init__(graph, attr_dict)
        self.update_attributes(CompressOp, attr_dict)
        assert self.check_required(), 'CompressOp is missing a required parameter.'

    def infer_shape(self):
        super(CompressOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.compress(inputs[1], inputs[0], axis=self.axis)
        self.set_out_tensor(out_tensor)


class ConcatOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'dafault': 1, 'required': False}},
                4: {'axis': {'required': True}},
                11: {'axis': {'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ConcatOp, self).__init__(graph, attr_dict)
        self.update_attributes(ConcatOp, attr_dict)
        assert self.check_required(), 'ConcatOp is missing a required parameter.'

    def infer_shape(self):
        super(ConcatOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.concatenate([*inputs], axis=self.axis)
        self.set_out_tensor(out_tensor)


class ConstantOp(OpHasOneOutPort, ConstLikeOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'value': {'type': AttrType.TENSOR, 'required': True}},
                9: {'value': {'type': AttrType.TENSOR, 'required': True}},
                11: {'sparse_value': {'type': AttrType.SPARSE_TENSOR},
                     'value': {'type': AttrType.TENSOR}},
                12: {
                    'sparse_value': {'type': AttrType.SPARSE_TENSOR},
                    'value': {'type': AttrType.TENSOR},
                    'value_float': {'type': AttrType.FLOAT},
                    'value_floats': {'type': AttrType.FLOATS},
                    'value_int': {'type': AttrType.INT},
                    'value_ints': {'type': AttrType.INTS},
                    'value_string': {'type': AttrType.STRING},
                    'value_strings': {'type': AttrType.STRINGS}},
                13: {
                    'sparse_value': {'type': AttrType.SPARSE_TENSOR},
                    'value': {'type': AttrType.TENSOR},
                    'value_float': {'type': AttrType.FLOAT},
                    'value_floats': {'type': AttrType.FLOATS},
                    'value_int': {'type': AttrType.INT},
                    'value_ints': {'type': AttrType.INTS},
                    'value_string': {'type': AttrType.STRING},
                    'value_strings': {'type': AttrType.STRINGS}},
                }

    def __init__(self, graph, attr_dict=None):
        super(ConstantOp, self).__init__(graph, attr_dict)
        self.update_attributes(ConstantOp, attr_dict)
        assert self.check_required(), 'ConstantOp is missing a required parameter.'

    def infer_shape(self):
        super(ConstantOp, self).infer_shape()
        out_tensor = self.value.copy()
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        try:
            cur_ver = self.__dict__['_attr']['cur_version'].value
            if item == 'value':
                if cur_ver in (1, 9):
                    return self._attr['value'].value
                elif cur_ver in (11, 12, 13):
                    values = np.asarray([self._attr[k].value for k, _ in type(
                        self).attributes()[cur_ver].items() if self._attr[k].value is not None])
                    assert len(
                        values) == 1, 'The length of value is invalid in ConstantOp.'
                    return values[0]
                else:
                    ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                          (cur_ver, type(self).__name__))
        except:
            ret = None
        if ret is None:
            ret = super(ConstantOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item == 'value':
            cur_ver = self.__dict__['_attr']['cur_version'].value
            if cur_ver in (1, 9, 11, 12, 13):
                self._attr['value'].value = value
            else:
                ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                      (cur_ver, type(self).__name__))
        else:
            super(ConstantOp, self).__setattr__(item, value)


class ConstantOfShapeOp(OpHasOneOutPort, ConstLikeOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {'value': {'type': AttrType.TENSOR, 'required': False}}}

    def __init__(self, graph, attr_dict=None):
        super(ConstantOfShapeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ConstantOfShapeOp, attr_dict)
        assert self.check_required(), 'ConstantOfShapeOp is missing a required parameter.'

    def infer_shape(self):
        super(ConstantOfShapeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs) >= 1 and inputs[0] is not None:
            inp = inputs[0].astype(np.int64)
            out_tensor = np.ndarray(inp.tolist())
            if self.value is not None:
                out_tensor.fill(self.value[0])
                out_tensor = out_tensor.astype(self.value.dtype)
            else:
                out_tensor.fill(0)
                out_tensor = out_tensor.astype(np.float32)
        else:
            out_tensor = None
        self.set_out_tensor(out_tensor)


class DepthToSpaceOp(LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'blocksize': {'type': AttrType.INT, 'required': True}
                    },
                11: {'blocksize': {'type': AttrType.INT, 'required': True},
                     'mode': {'type': AttrType.STRING, 'default': 'DCR', 'options': ['DCR', 'CRD']}
                     },
                13: {'blocksize': {'type': AttrType.INT, 'required': True},
                     'mode': {'type': AttrType.STRING, 'default': 'DCR', 'options': ['DCR', 'CRD']}
                     }
                }

    def __init__(self, graph, attr_dict=None):
        super(DepthToSpaceOp, self).__init__(graph, attr_dict)
        self.update_attributes(DepthToSpaceOp, attr_dict)
        assert self.check_required(), 'DepthToSpaceOp is missing a required parameter.'

    def infer_shape(self):
        super(DepthToSpaceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.data_format == 'NHWC':
            if self.mode == 'DCR':
                n, h, w, c = inputs[0].shape
                inp = np.reshape(
                    inputs[0], (n, h, w, self.blocksize, self.blocksize, c // self.blocksize ** 2))
                out = np.transpose(inp, (0, 3, 4, 1, 5, 2))
                out_tensor = np.reshape(
                    out, (n, h * self.blocksize, w * self.blocksize, c // self.blocksize ** 2))
            else:
                n, h, w, c = inputs[0].shape
                inp = np.reshape(
                    inputs[0], (n, h, w, c // self.blocksize ** 2, self.blocksize, self.blocksize))
                out = np.transpose(inp, (0, 1, 4, 2, 5, 3))
                out_tensor = np.reshape(
                    out, (n, h * self.blocksize, w * self.blocksize, c // self.blocksize ** 2))
        else:
            if self.mode == 'DCR':
                n, c, h, w = inputs[0].shape
                inp = np.reshape(
                    inputs[0], (n, self.blocksize, self.blocksize, c // self.blocksize ** 2, h, w))
                out = np.transpose(inp, (0, 3, 4, 1, 5, 2))
                out_tensor = np.reshape(
                    out, (n, c // self.blocksize ** 2, h * self.blocksize, w * self.blocksize))
            else:
                n, c, h, w = inputs[0].shape
                inp = np.reshape(
                    inputs[0], (n, c // self.blocksize ** 2, self.blocksize, self.blocksize, h, w))
                out = np.transpose(inp, (0, 1, 4, 2, 5, 3))
                out_tensor = np.reshape(
                    out, (n, c // self.blocksize ** 2, h * self.blocksize, w * self.blocksize))
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'mode':
                if cur_ver < 11:
                    ret = 'DCR'
                else:
                    ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        if ret is None:
            ret = super(DepthToSpaceOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item == 'mode':
            try:
                self.__dict__['_attr'][item].value = str(value)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.INTS, 'value': str(value)})
        else:
            super(DepthToSpaceOp, self).__setattr__(item, value)


class ExpandOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {8: {},
                13: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(ExpandOp, self).__init__(graph, attr_dict)
        self.update_attributes(ExpandOp, attr_dict)
        assert self.check_required(), 'ExpandOp is missing a required parameter.'

    def infer_shape(self):
        super(ExpandOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = inputs[0] * np.ones(inputs[1].tolist(), inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class EyeLikeOp(OpHasOneOutPort, ConstLikeOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {'dtype': {'type': AttrType.STRING},
                    'k': {'type': AttrType.INT, 'default': 0}},
                }

    def __init__(self, graph, attr_dict=None):
        super(EyeLikeOp, self).__init__(graph, attr_dict)
        self.update_attributes(EyeLikeOp, attr_dict)
        assert self.check_required(), 'EyeLikeOp is missing a required parameter.'

    def infer_shape(self):
        super(EyeLikeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if inputs and inputs[0] is not None:
            out_tensor = np.eye(inputs[0].shape[0], inputs[0].shape[1], k=self.k,
                                dtype=np.dtype(self.dtype) if self.dtype is not None else inputs[0].dtype)
        else:
            out_tensor = None
        self.set_out_tensor(out_tensor)


class FlattenOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 1}},
                9: {'axis': {'default': 1}},
                11: {'axis': {'default': 1}},
                13: {'axis': {'default': 1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(FlattenOp, self).__init__(graph, attr_dict)
        self.update_attributes(FlattenOp, attr_dict)
        assert self.check_required(), 'FlattenOp is missing a required parameter.'

    def infer_shape(self):
        super(FlattenOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = int(OpHasAxis.make_axes_non_negative(
            self.axis, len(inputs[0].shape)))
        if np.ndim(inputs[0]) in (0, 1):
            inp = np.expand_dims(inputs[0], axis=0)
            inp = np.reshape(inp, newshape=(1, inp.size))
        else:
            inp = inputs[0]
        new_shape = (int(np.prod(inp.shape[0:self.axis])),
                     int(np.prod(inp.shape[self.axis:])))
        out_tensor = np.reshape(inp, newshape=new_shape)
        self.set_out_tensor(out_tensor)


class GatherOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 0}},
                11: {'axis': {'default': 0}},
                13: {'axis': {'default': 0}},
                }

    def __init__(self, graph, attr_dict=None):
        super(GatherOp, self).__init__(graph, attr_dict)
        self.update_attributes(GatherOp, attr_dict)
        assert self.check_required(), 'GatherOp is missing a required parameter.'

    def infer_shape(self):
        super(GatherOp, self).infer_shape()
        inputs = self.get_input_tensors()
        indices = inputs[1]
        try:
            out_tensor = np.take(inputs[0], np.array(
                indices, np.int32), axis=self.axis)
        except:
            if self.is_all_inputs_const():
                ERROR('[Parser]: Meets invalid indices input of Gather Op (%s) in infer_shape!' % self.name)
            indices = np.zeros_like(indices)
            out_tensor = np.take(inputs[0], indices, axis=self.axis)
        self.set_out_tensor(out_tensor)


class GatherElementsOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {'axis': {'default': 0}},
                13: {'axis': {'default': 0}},
                }

    @staticmethod
    def make_indices_non_negative(indices, ref_shape):
        '''If indice is negative, return non-negative indices.'''
        mask = indices < 0
        indices[mask] = (indices + ref_shape)[mask]
        return indices

    def __init__(self, graph, attr_dict=None):
        super(GatherElementsOp, self).__init__(graph, attr_dict)
        self.update_attributes(GatherElementsOp, attr_dict)
        assert self.check_required(), 'GatherElementsOp is missing a required parameter.'

    def infer_shape(self):
        super(GatherElementsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        indices = inputs[1]
        indices = GatherElementsOp.make_indices_non_negative(
            indices, inputs[0].shape[self.axis])
        torch_input = torch.from_numpy(inputs[0])
        torch_indices = torch.from_numpy(np.array(indices, np.int64))
        out_tensor = torch.gather(torch_input, self.axis, torch_indices)
        out_tensor = out_tensor.numpy()
        self.set_out_tensor(out_tensor)


class GatherNDOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {},
                12: {'batch_dims': {'default': 0}},
                13: {'batch_dims': {'default': 0}},
                }

    def __init__(self, graph, attr_dict=None):
        super(GatherNDOp, self).__init__(graph, attr_dict)
        self.update_attributes(GatherNDOp, attr_dict)
        assert self.check_required(), 'GatherNDOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'batch_dims':
                if cur_ver >= 12:
                    ret = self.__dict__['_attr'][item].value
                else:
                    ret = 0
        except:
            ret = None
        if ret is None:
            ret = super(GatherNDOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(GatherNDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        in_rank = len(inputs[0].shape)
        assert inputs[1].shape[-1] <= in_rank, 'The shape of input is invalid in GatherNDOp.'
        batch_dims_shape = []
        batch_dims_size = 1
        for i in range(self.batch_dims):
            batch_dims_shape.append(inputs[1].shape[i])
            batch_dims_size *= inputs[1].shape[i]
        output_shape = batch_dims_shape + list(inputs[1].shape)[self.batch_dims:-1] \
            if (inputs[1].shape[-1] == in_rank - self.batch_dims) \
            else batch_dims_shape + list(inputs[1].shape)[self.batch_dims:-1] + list(inputs[0].shape)[self.batch_dims + inputs[1].shape[-1]:]

        out_data = []
        reshaped_indices = inputs[1].reshape(
            batch_dims_size, -1, inputs[1].shape[-1])
        reshaped_data = inputs[0].reshape(
            (batch_dims_size,) + inputs[0].shape[self.batch_dims:])
        for batch_dim in range(reshaped_indices.shape[0]):
            for outer_dim in range(reshaped_indices.shape[1]):
                gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
                try:
                    out_data.append(reshaped_data[(batch_dim,) + gather_index])
                except:
                    gather_index = tuple(np.zeros_like(reshaped_indices[batch_dim][outer_dim]))
                    out_data.append(reshaped_data[(batch_dim,) + gather_index])
        out_tensor = np.asarray(
            out_data, dtype=inputs[0].dtype).reshape(output_shape)
        self.set_out_tensor(out_tensor)


class IdentityOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                13: {},
                14: {},
                16: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(IdentityOp, self).__init__(graph, attr_dict)
        self.update_attributes(IdentityOp, attr_dict)
        assert self.check_required(), 'IdentityOp is missing a required parameter.'

    def infer_shape(self):
        super(IdentityOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = inputs[0].copy()
        self.set_out_tensor(out_tensor)


class OneHotOp(OpHasOneOutPort, OpHasAxis, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            9: {'axis': {'type': AttrType.INT, 'default': -1},
                },
            11: {'axis': {'type': AttrType.INT, 'default': -1}
                 }
        }

    def __init__(self, graph, attr_dict=None):
        super(OneHotOp, self).__init__(graph, attr_dict)
        self.update_attributes(OneHotOp, attr_dict)
        assert self.check_required(), 'OneHotOp is missing a required parameter.'

    def infer_shape(self):
        super(OneHotOp, self).infer_shape()
        inputs = self.get_input_tensors()
        indices = inputs[0].astype(np.int64)
        depth, values = inputs[1:]
        depth = int(depth.item()) if isinstance(
            depth, np.ndarray) else int(depth)

        reps = [1] * (len(indices.shape) + 1)
        reps[self.axis] = depth

        tiled_indices = np.tile(np.expand_dims(indices, axis=self.axis), reps)
        out_tensor = (np.ones_like(tiled_indices) *
                      values[1]).astype(values.dtype)
        off_mask = np.logical_and(
            tiled_indices >= -depth, tiled_indices < depth - 1)
        out_tensor[off_mask] = values[0]
        self.set_out_tensor(out_tensor)


class PadOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'mode': {'type': AttrType.STRING, 'default': 'constant', 'options': ['constant', 'reflect', 'edge', 'symmetric']},
                    'paddings': {'type': AttrType.INTS, 'required': True},
                    'value': {'type': AttrType.FLOAT, 'default': 0.0}
                    },
                2: {'mode': {'type': AttrType.STRING, 'default': 'constant', 'options': ['constant', 'reflect', 'edge', 'symmetric']},
                    'pads': {'type': AttrType.INTS, 'required': True},
                    'value': {'type': AttrType.FLOAT, 'default': 0.0}
                    },
                11: {'mode': {'type': AttrType.STRING, 'default': 'constant', 'options': ['constant', 'reflect', 'edge', 'symmetric']}},
                13: {'mode': {'type': AttrType.STRING, 'default': 'constant', 'options': ['constant', 'reflect', 'edge', 'symmetric']}},
                18: {'mode': {'type': AttrType.STRING, 'default': 'constant', 'options': ['constant', 'reflect', 'edge', 'symmetric']}},
                }

    def __init__(self, graph, attr_dict=None):
        super(PadOp, self).__init__(graph, attr_dict)
        self.update_attributes(PadOp, attr_dict)
        assert self.check_required(), 'PadOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'pads':
                if cur_ver == 1:
                    ret = self.__dict__['_attr']['paddings'].value
                elif cur_ver >= 11:
                    inputs = self.get_input_tensors()
                    ret = inputs[1].flatten().tolist()
                    if cur_ver >= 18 and len(inputs) > 3 and inputs[3] is not None and len(inputs[3]) > 0:
                        axes = inputs[3].tolist()
                        input_length = len(inputs[0].shape)
                        new_pads = np.zeros([input_length * 2], np.int64)
                        positive_axes = [(axis + input_length) if axis < 0 else axis for axis in axes]
                        complete_idx = positive_axes + [(axis + input_length) for axis in positive_axes]
                        np.put(new_pads, complete_idx, np.array(ret[:(len(axes) * 2)]))
                        ret = new_pads.tolist()
            elif item == 'value':
                if cur_ver <= 2:
                    ret = self.__dict__['_attr'][item].value
                elif cur_ver >= 11:
                    inputs = self.get_input_tensors()
                    ret = 0 if len(inputs) <= 2 or (
                        len(inputs) > 2 and inputs[2] is None) else np.asscalar(inputs[2].flatten())
        except:
            ret = None
        if ret is None:
            ret = super(PadOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(PadOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version

        if cur_ver == 1:
            pads = self.paddings
            const_value = self.value
        elif cur_ver == 2:
            pads = self.pads
            const_value = self.value
        else:
            pads = self.pads
            const_value = 0 if len(inputs) <= 2 or (
                len(inputs) > 2 and inputs[2] is None) else np.asscalar(inputs[2].flatten())
        negative_pads = [pad if pad < 0 else None for pad in pads]
        if all(pad is None for pad in negative_pads):
            sliced_input = inputs[0]
        else:
            input_length = len(inputs[0].shape)
            slice_obj = [slice(-begin if begin is not None else None, end, 1)
                         for begin, end in zip(negative_pads[:input_length], negative_pads[input_length:])]
            sliced_input = inputs[0][tuple(slice_obj)]
            pads = [pad if pad >= 0 else 0 for pad in pads]
        if self.mode in ('reflect', 'symmetric'):
            pads = np.reshape(np.array(pads, np.int32), (2, -1))
            pads = np.transpose(pads)
            out_tensor = np.pad(sliced_input, pads, mode=self.mode)
        else:
            torch_input = torch.from_numpy(sliced_input)
            out_tensor = torch.nn.functional.pad(torch_input,
                                                 OpHasPaddingStrides.onnx_to_torch(
                                                     pads),
                                                 mode=self.mode if self.mode in (
                                                     'constant', 'reflect') else 'replicate',
                                                 value=const_value).numpy()
        self.set_out_tensor(out_tensor)

    def is_fusable(self):
        pads = np.reshape(np.array(self.pads, np.int64), (2, -1))
        non_space_pads = pads[:, 0:2] if self.data_format == 'NCHW' else np.concatenate(
            [pads[:, 0:1], pads[:, -1:]], axis=1)
        return self.mode == 'constant' and np.all(non_space_pads == 0)

    def space_pads(self):
        pads = np.reshape(np.array(self.pads, np.int64), (2, -1))
        space = pads[:, 2:] if self.data_format == 'NCHW' else pads[:, 1:-1]
        return space.flatten().tolist()


class RangeOp(OpHasOneOutPort, ConstLikeOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {'start': {'default': None},
                     'limit': {'default': None},
                     'delta': {'default': None}}}

    def __init__(self, graph, attr_dict=None):
        super(RangeOp, self).__init__(graph, attr_dict)
        self.update_attributes(RangeOp, attr_dict)
        assert self.check_required(), 'RangeOp is missing a required parameter.'

    def infer_shape(self):
        super(RangeOp, self).infer_shape()
        if any(val is None for val in [self.start, self.limit, self.delta]):
            out_tensor = None
        else:
            out_tensor = np.arange(self.start, self.limit, self.delta)
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'start':
                in_edges = self._graph.sorted_in_edges(self.name, data=True)
                ret = in_edges[0][2]['tensor'].value
            elif item == 'limit':
                in_edges = self._graph.sorted_in_edges(self.name, data=True)
                ret = in_edges[1][2]['tensor'].value
            elif item == 'delta':
                in_edges = self._graph.sorted_in_edges(self.name, data=True)
                ret = in_edges[2][2]['tensor'].value
        except:
            ret = None
        if ret is None:
            ret = super(RangeOp, self).__getattr__(item)
        return ret


class ReshapeOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'shape': {'type': AttrType.INTS, 'required': True, 'default': None},
                    'consumed_inputs': {'type': AttrType.INTS}},
                5: {},
                13: {},
                14: {'allowzero': {'type': AttrType.INT, 'default': 0}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReshapeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReshapeOp, attr_dict)
        assert self.check_required(), 'ReshapeOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            cur_ver = self.__dict__['_attr']['cur_version'].value
            if item == 'shape':
                if cur_ver == 14 and bool(self.allowzero):
                    WARN('[Parser]: Reshape-%d(%s) with allowzero is not supported!' %
                         (cur_ver, self.name))
                else:
                    if cur_ver == 1:
                        shape = self._attr[item].value
                    else:
                        in_edges = self._graph.sorted_in_edges(
                            self.name, data=True)
                        try:
                            shape = np.array(
                                in_edges[1][2]['tensor'].value, np.int32).tolist()
                        except:
                            ERROR(
                                '[Parser]: Meets exception when obtaining shape of Reshape(%s) for %s!' % self.name)
                            shape = None
                    try:
                        if np.ndim(shape) == 0:
                            shape = [shape]
                        if 0 in shape:
                            for idx, val in enumerate(shape):
                                if val == 0:
                                    shape[idx] = self.get_input_tensors()[
                                        0].shape[idx]
                    except:
                        ERROR(
                            '[Parser]: Meets exception when obtaining shape of Reshape(%s) for %s!' % self.name)
                        shape = None
                    if shape is not None:
                        self.__dict__['_attr'][item] = Attribute(
                            item, {'type': AttrType.INTS, 'value': shape})
                    ret = shape
        except:
            ret = None
        if ret is None:
            ret = super(ReshapeOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ReshapeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.reshape(inputs[0], self.shape)
        self.set_out_tensor(out_tensor)


class ReverseSequenceOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {10: {'batch_axis': {'type': AttrType.INT, 'default': 1, 'options': [0, 1]},
                     'time_axis': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}
                     }
                }

    def __init__(self, graph, attr_dict=None):
        super(ReverseSequenceOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReverseSequenceOp, attr_dict)
        assert self.check_required(), 'ReverseSequenceOp is missing a required parameter.'

    def infer_shape(self):
        super(ReverseSequenceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.flip(inputs[0], axis=self.time_axis)
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'batch_axis':
                attr = self.__dict__['_attr'][item].value
                ret = int(attr.value) if attr is not None else 1
            elif item == 'time_axis':
                attr = self.__dict__['_attr'][item].value
                return int(attr.value) if attr is not None else 0
        except:
            ret = None
        if ret is None:
            ret = super(ReverseSequenceOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item == 'batch_axis':
            try:
                self.__dict__['_attr'][item].value = int(value)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.INTS, 'value': int(value)})
        elif item == 'time_axis':
            try:
                self.__dict__['_attr'][item].value = int(value)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.INTS, 'value': int(value)})
        else:
            super(ReverseSequenceOp, self).__setattr__(item, value)


class ScatterNDOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {},
                13: {},
                16: {'reduction': {'type': AttrType.STRING, 'options': ['none', 'mul', 'add'], 'default': 'none'}},
                18: {'reduction': {'type': AttrType.STRING, 'options': ['none', 'mul', 'add', 'max', 'min'], 'default': 'none'}},
                }

    def __init__(self, graph, attr_dict=None):
        super(ScatterNDOp, self).__init__(graph, attr_dict)
        self.update_attributes(ScatterNDOp, attr_dict)
        assert self.check_required(), 'ScatterNDOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            cur_ver = self.__dict__['_attr']['cur_version'].value
            if item == 'reduction':
                if cur_ver < 16:
                    ret = 'none'
                    self.__dict__['_attr'][item] = Attribute(
                        item, {'type': AttrType.STRING, 'value': str(ret)})
                else:
                    ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        if ret is None:
            ret = super(ScatterNDOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item == 'reduction':
            try:
                assert value in [
                    'none', 'mul', 'add', 'max', 'min'], 'ScatterNDOp __setattr__ has an invalid value.'
                cur_ver = self.__dict__['_attr']['cur_version'].value
                if cur_ver < 16:
                    assert value == 'none', 'ScatterNDOp __setattr__ has an invalid value.'
                    self.__dict__['_attr'][item] = Attribute(
                        item, {'type': AttrType.STRING, 'value': str(value)})
                else:
                    self.__dict__['_attr'][item].value = str(value)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.STRING, 'value': 'none'})
        else:
            super(ScatterNDOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(ScatterNDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        data, indices, updates = inputs
        out_tensor = np.copy(data)
        update_indices = indices.shape[:-1]
        for idx in np.ndindex(update_indices):
            index = tuple(indices[idx])
            if self.reduction == 'mul':
                out_tensor[index] *= updates[idx]
            elif self.reduction == 'add':
                out_tensor[index] += updates[idx]
            elif self.reduction == 'max':
                out_tensor[index] = np.maximum(out_tensor[index], updates[idx])
            elif self.reduction == 'min':
                out_tensor[index] = np.minimum(out_tensor[index], updates[idx])
            else:
                out_tensor[index] = updates[idx]
        self.set_out_tensor(out_tensor)


class ScatterOp(OpHasOneOutPort, OpHasAxis, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {'axis': {'type': AttrType.INT, 'default': 0}},
                }

    def __init__(self, graph, attr_dict=None):
        super(ScatterOp, self).__init__(graph, attr_dict)
        self.update_attributes(ScatterOp, attr_dict)
        assert self.check_required(), 'ScatterOp is missing a required parameter.'

    def infer_shape(self):
        super(ScatterOp, self).infer_shape()
        inputs = self.get_input_tensors()
        data, indices, updates = inputs
        data_torch = torch.from_numpy(data)
        indices = GatherElementsOp.make_indices_non_negative(
            indices, inputs[0].shape[self.axis])
        index_torch = torch.from_numpy(np.array(indices).astype(np.int64))
        update_torch = torch.from_numpy(updates)
        out_tensor = torch.Tensor.scatter_(
            data_torch, src=update_torch, dim=self.axis, index=index_torch).numpy()
        self.set_out_tensor(out_tensor)


class ScatterElementsOp(OpHasOneOutPort, OpHasAxis, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {'axis': {'type': AttrType.INT, 'default': 0}},
                13: {'axis': {'type': AttrType.INT, 'default': 0}},
                16: {'axis': {'type': AttrType.INT, 'default': 0},
                     'reduction': {'type': AttrType.STRING, 'options': ['none', 'mul', 'add'], 'default': 'none'}},
                18: {'axis': {'type': AttrType.INT, 'default': 0},
                     'reduction': {'type': AttrType.STRING, 'options': ['none', 'mul', 'add', 'max', 'min'], 'default': 'none'}},
                }

    def __init__(self, graph, attr_dict=None):
        super(ScatterElementsOp, self).__init__(graph, attr_dict)
        self.update_attributes(ScatterElementsOp, attr_dict)
        assert self.check_required(), 'ScatterElementsOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            cur_ver = self.__dict__['_attr']['cur_version'].value
            if item == 'reduction':
                if cur_ver < 16:
                    ret = 'none'
                    self.__dict__['_attr'][item] = Attribute(
                        item, {'type': AttrType.STRING, 'value': str(ret)})
                else:
                    ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        if ret is None:
            ret = super(ScatterElementsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ScatterElementsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        data, indices, updates = inputs
        data_torch = torch.from_numpy(data)
        indices = GatherElementsOp.make_indices_non_negative(
            indices, inputs[0].shape[self.axis])
        index_torch = torch.from_numpy(np.array(indices).astype(np.int64))
        update_torch = torch.from_numpy(updates)
        if self.reduction == 'none':
            out_tensor = torch.Tensor.scatter_(
                data_torch, src=update_torch, dim=self.axis, index=index_torch).numpy()
        else:
            onnx_reduction_map = {'mul': 'prod', 'add': 'sum', 'max': 'amax', 'min': 'amin'}
            assert self.reduction in onnx_reduction_map, 'Meets invalid reduction %s in infer_shape of ScatterElementsOp!' % self.reduction
            out_tensor = torch.Tensor.scatter_reduce(
                data_torch, src=update_torch, dim=self.axis, index=index_torch, reduce=onnx_reduction_map[self.reduction]).numpy()
        self.set_out_tensor(out_tensor)


class ShapeOp(OpHasOneOutPort, ConstLikeOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                13: {},
                15: {'end': {'type': AttrType.INT, 'required': False, 'default': None},
                     'start': {'type': AttrType.INT, 'required': False, 'default': 0}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ShapeOp, self).__init__(graph, attr_dict)
        self.update_attributes(ShapeOp, attr_dict)
        assert self.check_required(), 'ShapeOp is missing a required parameter.'

    def infer_shape(self):
        super(ShapeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if inputs and inputs[0] is not None:
            shape = inputs[0].shape
            if self.cur_version >= 15:
                rank = len(shape)
                true_end = rank
                true_start = 0
                need_slicing = False
                if self.end is not None:
                    true_end = (self.end + rank) if self.end < 0 else self.end
                    true_end = 0 if true_end < 0 else (
                        rank if true_end > rank else true_end)
                    need_slicing = True
                if self.start != 0:
                    true_start = (
                        self.start + rank) if self.start < 0 else self.start
                    true_start = 0 if true_start < 0 else (
                        rank - 1 if true_start >= rank else true_start)
                    need_slicing = True
                if need_slicing:
                    shape = shape[true_start: true_end] if (
                        true_end - true_start) >= 0 else []
            out_tensor = np.array(shape, np.int32)
        else:
            out_tensor = None
        self.set_out_tensor(out_tensor)


class SizeOp(OpHasOneOutPort, ConstLikeOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}, 13: {}}

    def __init__(self, graph, attr_dict=None):
        super(SizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(SizeOp, attr_dict)
        assert self.check_required(), 'SizeOp is missing a required parameter.'

    def infer_shape(self):
        super(SizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0].size).astype(np.int32)
        self.set_out_tensor(out_tensor)


class SliceOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axes': {'type': AttrType.INTS, 'required': False},
                    'ends': {'type': AttrType.INTS, 'required': True},
                    'starts': {'type': AttrType.INTS, 'required': True}
                    },
                10: {},
                11: {},
                13: {}
                }

    @staticmethod
    def cal_sliced(starts, ends, input_shape):
        assert (len(starts) == len(ends)) and (len(starts) == len(input_shape)), \
            'input_shape is invalid in SliceOp.'
        starts_np, ends_np, in_shape_np = [np.array(s, np.int64) for s in [
            starts, ends, input_shape]]
        end_neg_mask = ends_np < 0
        ends_np[end_neg_mask] = (ends_np + in_shape_np)[end_neg_mask]
        ends_np = in_shape_np - ends_np
        ends_np[ends_np < 0] = 0
        return starts_np.tolist() + ends_np.tolist()

    @staticmethod
    def trim_start_end(dim, start, end, step=1):
        def clamp(val, min, max):
            return min if (val < min) else (max if (val > max) else val)
        assert step != 0, 'SliceOp step is zero in trim_start_end.'
        if start < 0:
            start += dim
        if step < 0:
            start = clamp(start, 0, dim - 1)
        else:
            start = clamp(start, 0, dim)
        if step < 0:
            if -dim <= end < 0:
                end += dim
            end = clamp(end, -1, dim)
            if end == -1:
                end = INT_MIN
        else:
            if end < 0:
                end += dim
            end = clamp(end, 0, dim)
        return start, end

    def __init__(self, graph, attr_dict=None):
        super(SliceOp, self).__init__(graph, attr_dict)
        self.update_attributes(SliceOp, attr_dict)
        assert self.check_required(), 'SliceOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = super(SliceOp, self).__getattr__(item)
        except:
            ret = None
        if not ret and item in ('starts', 'ends', 'axes', 'steps'):
            cur_ver = self.__dict__['_attr']['cur_version'].value
            try:
                if item == 'axes':
                    if cur_ver == 1:
                        ret = list(
                            range(len(self.__dict__['_attr']['starts'].value)))
                    else:
                        inputs = self.get_input_tensors()
                        ret = np.array(inputs[3]).tolist() if len(
                            inputs) > 3 else list(range(len(inputs[0].shape)))
                elif item in ('starts', 'ends'):
                    if cur_ver == 1:
                        ret = self.__dict__['_attr'][item].value
                    else:
                        inputs = self.get_input_tensors()
                        if item == 'starts':
                            ret = np.array(inputs[1]).tolist()
                        else:
                            ret = np.array(inputs[2]).tolist()
                elif item == 'steps':
                    inputs = self.get_input_tensors()
                    if cur_ver == 1:
                        ret = [1] * len(inputs[0].shape)
                    else:
                        ret = np.array(inputs[4]).tolist() if len(
                            inputs) > 4 else [1] * len(inputs[0].shape)
            except:
                ret = None
        return ret

    def __setattr__(self, item, value):
        if item in ('starts', 'ends', 'axes', 'steps'):
            try:
                self.__dict__['_attr'][item].value = list(value)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.INTS, 'value': list(value)})
        else:
            super(SliceOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(SliceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver not in list(SliceOp.attributes().keys()):
            ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                  (cur_ver, type(self).__name__))

        if inputs[0] is None or inputs[0].size == 0:
            ERROR(
                '[Parser]: Meets invalid input tensor in Slice(%s) infer_shape!' % self.name)
            return

        rank = len(inputs[0].shape)
        if cur_ver == 1:
            assert len(inputs) == 1, \
                ('[Parser]: Only support one input for Onnx Slice (%s) of ver %d!' % (
                    self.name, cur_ver))
            assert len(self.starts) == len(self.ends), \
                ('[Parser]: Length of starts should be equal to length of ends for Onnx Slice (%s) of ver %d!' % (
                    self.name, cur_ver))
            if self.axes is None:
                self.axes = list(range(len(self.starts)))
            else:
                assert len(self.axes) == len(self.starts), \
                    ('[Parser]: Length of starts should be equal to length of axes for Onnx Slice (%s) of ver %d!' % (
                        self.name, cur_ver))
                self.axes = OpHasAxis.make_axes_non_negative(
                    self.axes, len(inputs[0].shape))
        else:
            assert 3 <= len(inputs) <= 5, \
                ('[Parser]: Only support 3/4/5 inputs for Onnx Slice (%s) of ver %d!' %
                 (self.name, cur_ver))
            if self.axes is None:
                self.axes = list(range(len(self.starts)))
            else:
                self.axes = OpHasAxis.make_axes_non_negative(
                    self.axes, len(inputs[0].shape))
            if self.starts and len(self.ends) < len(self.starts):
                self.ends = self.ends + [sys.maxsize] * \
                    (len(self.starts) - len(self.ends))

        if self.axes != sorted(self.axes):
            indices = [self.axes.index(x) for x in sorted(self.axes)]
            self.axes = sorted(self.axes)
            self.starts = np.take(
                np.array(self.starts, np.int64), indices).tolist()
            self.ends = np.take(
                np.array(self.ends, np.int64), indices).tolist()
            self.steps = np.take(
                np.array(self.steps, np.int64), indices).tolist()
        axes = list(range(rank))
        starts = [0] * rank
        ends = list(inputs[0].shape)
        steps = [1] * rank
        j = 0
        for i, d in enumerate(inputs[0].shape):
            if j < len(self.axes) and i == self.axes[j]:
                meta_step = 1 if cur_ver == 1 else self.steps[j]
                start, end = SliceOp.trim_start_end(
                    d, self.starts[j], self.ends[j], meta_step)
                starts[i], ends[i], steps[i] = start, end, meta_step
                j += 1
        self.axes, self.starts, self.ends, self.steps = axes, starts, ends, steps
        obj = tuple(slice(s, None if (p < 0 and e < 0) else e, p)
                    for s, e, p in zip(self.starts, self.ends, self.steps))
        out_tensor = inputs[0][obj]
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        from ...front_end.onnx.passes.common_passes import insert_constant
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            if cur_ver == 1:
                insert_constant(self._graph, self.name + '_starts',
                                np.array(self.starts, np.int32), self.name, in_port=1)
                insert_constant(self._graph, self.name + '_ends',
                                np.array(self.ends, np.int32), self.name, in_port=2)
            self.cur_version = max_ver

        in_edges = self._graph.sorted_in_edges(self.name, keys=True, data=True)
        if len(in_edges) >= 2:
            src, _, k, in_attr = in_edges[1]
            if in_attr.get('tensor', None) is None \
                    or in_attr['tensor'].value is None \
                    or np.array(in_attr['tensor'].value).tolist() != self.starts:
                self._graph.remove_edge(src, self.name, key=k)
                insert_constant(self._graph, self.name + '_starts',
                                np.array(self.starts, np.int32), self.name, in_port=1)

        if len(in_edges) >= 3:
            src, _, k, in_attr = in_edges[2]
            if in_attr.get('tensor', None) is None \
                    or in_attr['tensor'].value is None \
                    or np.array(in_attr['tensor'].value).tolist() != self.ends:
                self._graph.remove_edge(src, self.name, key=k)
                insert_constant(self._graph, self.name + '_ends',
                                np.array(self.ends, np.int32), self.name, in_port=2)

        if len(in_edges) < 4:
            insert_constant(self._graph, self.name + '_axes',
                            np.array(self.axes, np.int32), self.name, in_port=3)
        else:
            src, _, k, in_attr = in_edges[3]
            if in_attr.get('tensor', None) is None \
                    or in_attr['tensor'].value is None \
                    or np.array(in_attr['tensor'].value).tolist() != self.axes:
                self._graph.remove_edge(src, self.name, key=k)
                insert_constant(self._graph, self.name + '_axes',
                                np.array(self.axes, np.int32), self.name, in_port=3)

        if len(in_edges) < 5:
            insert_constant(self._graph, self.name + '_steps',
                            np.array(self.steps, np.int32), self.name, in_port=4)
        else:
            src, _, k, in_attr = in_edges[4]
            if in_attr.get('tensor', None) is None \
                    or in_attr['tensor'].value is None \
                    or np.array(in_attr['tensor'].value).tolist() != self.steps:
                self._graph.remove_edge(src, self.name, key=k)
                insert_constant(self._graph, self.name + '_steps',
                                np.array(self.steps, np.int32), self.name, in_port=4)


class SpaceToDepthOp(LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'blocksize': {'type': AttrType.INT, 'required': True}},
                13: {'blocksize': {'type': AttrType.INT, 'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(SpaceToDepthOp, self).__init__(graph, attr_dict)
        self.update_attributes(SpaceToDepthOp, attr_dict)
        assert self.check_required(), 'SpaceToDepthOp is missing a required parameter.'

    def infer_shape(self):
        super(SpaceToDepthOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.blocksize == 1:
            out_tensor = inputs[0]
        else:
            if self.data_format == 'NHWC':
                out_tensor = tf.nn.space_to_depth(
                    inputs[0], self.blocksize).numpy()
            else:
                torch_input = torch.from_numpy(inputs[0])
                n, c, h, w = torch_input.size()
                block_size = self.blocksize
                out_tensor = torch_input.view(
                    n, c, h // block_size, block_size, w // block_size, block_size)
                out_tensor = out_tensor.permute(0, 3, 5, 1, 2, 4).contiguous()
                out_tensor = out_tensor.view(
                    n, c * (block_size ** 2), h // block_size, w // block_size)
                out_tensor = out_tensor.numpy()
        self.set_out_tensor(out_tensor)


class SplitOp(OpHasAxis, OpHasMultipleOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT},
                    'split': {'type': AttrType.INTS, 'required': False}
                    },
                2: {'axis': {'type': AttrType.INT, 'default': 0},
                    'split': {'type': AttrType.INTS, 'required': False}
                    },
                11: {'axis': {'type': AttrType.INT, 'default': 0},  # Accepted range is [-rank, rank-1] where r = rank(input).
                     # >= 0
                     'split': {'type': AttrType.INTS, 'required': False}
                     },
                13: {'axis': {'type': AttrType.INT, 'default': 0},
                     },
                18: {'axis': {'type': AttrType.INT, 'default': 0},
                     'num_outputs': {'type': AttrType.INT, 'default': None},
                     }}

    def __init__(self, graph, attr_dict=None):
        super(SplitOp, self).__init__(graph, attr_dict)
        self.update_attributes(SplitOp, attr_dict)
        assert self.check_required(), 'SplitOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'split':
                try:
                    ret = self.__dict__['_attr'][item].value
                except:
                    pass
                if ret is None:
                    inputs = self.get_input_tensors()
                    if len(inputs) == 2:
                        ret = np.array(inputs[1]).tolist()
                    else:
                        split_size = inputs[0].shape[self.axis]
                        if self.cur_version >= 18:
                            size = int(np.ceil(split_size / self.num_outputs))
                            remainder = split_size % size
                            ret = [size] * self.num_outputs
                            if remainder != 0:
                                ret[-1] = remainder
                        else:
                            ret = [split_size //
                                   len(self.get_out_ports())] * len(self.get_out_ports())
                    if self.cur_version < 13:
                        self.__dict__['_attr'][item].value = ret
                    else:
                        self.__dict__['_attr'][item] = Attribute(
                            item, {'type': AttrType.INTS, 'value': list(ret)})
        except:
            ret = None
        if ret is None:
            ret = super(SplitOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(SplitOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver == 1:
            assert len(inputs) in (
                1, 2), 'The length of input is invalid in SplitOp.'
        elif cur_ver >= 18:
            assert (len(inputs) == 2 and self.num_outputs is None) or (len(inputs) == 1 and self.num_outputs is not None), \
                'Either the second input split or attribute num_outputs should be specified in SplitOp.'
        else:
            assert len(
                inputs) >= 1, 'The length of input is invalid in SplitOp.'
        out_tensors = tf.split(inputs[0],
                               np.array(self.split, np.int64),
                               axis=self.axis)
        self.set_out_tensor([o.numpy() for o in out_tensors])


class SqueezeOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                11: {},
                13: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(SqueezeOp, self).__init__(graph, attr_dict)
        self.update_attributes(SqueezeOp, attr_dict)
        assert self.check_required(), 'SqueezeOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = super(SqueezeOp, self).__getattr__(item)
        except:
            ret = None
        if ret is None:
            cur_ver = self.__dict__['_attr']['cur_version'].value
            if item == 'axes':
                if cur_ver >= 13:
                    inputs = self.get_input_tensors()
                    if len(inputs) == 1:
                        ret = None
                        if item not in self.__dict__['_attr']:
                            self.__dict__['_attr'][item] \
                                = Attribute(item, {'type': AttrType.INTS, 'value': ret})
                    elif len(inputs) == 2:
                        ret = np.array(inputs[1], np.int64).tolist()
                        self.__dict__['_attr'][item] \
                            = Attribute(item, {'type': AttrType.INTS, 'value': ret})
                    else:
                        ERROR(
                            '[Parser]: Invalid inputs number of Squeeze(%s)!' % self.name)
        return ret

    def infer_shape(self):
        super(SqueezeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.squeeze(np.array(inputs[0]), axis=tuple(
            self.axes) if self.axes else None)
        self.set_out_tensor(out_tensor)


class TileOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                6: {},
                13: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(TileOp, self).__init__(graph, attr_dict)
        self.update_attributes(TileOp, attr_dict)
        assert self.check_required(), 'TileOp is missing a required parameter.'

    def infer_shape(self):
        super(TileOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver == 1:
            assert len(inputs) == 3, 'The length of input is invalid in TileOp.'
            out_tensor = None
        else:
            assert len(inputs) == 2, 'The length of input is invalid in TileOp.'
            out_tensor = np.tile(*inputs)
        self.set_out_tensor(out_tensor)


class TransposeOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'perm': {'type': AttrType.INTS, 'default': [], 'required': False}},
                13: {'perm': {'type': AttrType.INTS, 'default': [], 'required': False}},
                }

    def __init__(self, graph, attr_dict=None):
        super(TransposeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TransposeOp, attr_dict)
        assert self.check_required(), 'TransposeOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'perm':
                ret = self.__dict__['_attr'][item].value
                if not ret:
                    inputs = self.get_input_tensors()
                    ret = list(range(0, len(inputs[0].shape)))[::-1]
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TransposeOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TransposeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.transpose(
            inputs[0], axes=self.perm if self.perm else None)
        self.set_out_tensor(out_tensor)


class UniqueOp(OpHasVariableOutPorts, OpHasAxis, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {'axis': {'required': False},
                     'sorted': {'type': AttrType.INT, 'default': 1, 'required': False, 'options': [0, 1]}
                     }}

    def __init__(self, graph, attr_dict=None):
        super(UniqueOp, self).__init__(graph, attr_dict)
        self.update_attributes(UniqueOp, attr_dict)
        assert self.check_required(), 'UniqueOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'sorted':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(UniqueOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(UniqueOp, self).infer_shape()
        inputs = self.get_input_tensors()
        y, indices, inverse_indices, counts = np.unique(
            inputs[0], True, True, True, axis=self.axis)
        if self.sorted is False:
            argsorted_indices = np.argsort(indices)
            inverse_indices_map = {i: si for i, si in zip(
                argsorted_indices, np.arange(len(argsorted_indices)))}
            indices = indices[argsorted_indices]
            y = np.take(inputs[0], indices, axis=self.axis)
            inverse_indices = np.asarray(
                [inverse_indices_map[i] for i in inverse_indices], dtype=np.int64)
            counts = counts[argsorted_indices]
        out_ports = self.get_out_ports()
        if out_ports == [0]:
            out_tensors = [y]
        elif out_ports == [0, 1]:
            out_tensors = [y, indices]
        elif out_ports == [0, 2]:
            out_tensors = [y, inverse_indices]
        elif out_ports == [0, 3]:
            out_tensors = [y, counts]
        elif out_ports == [0, 1, 2]:
            out_tensors = [y, indices, inverse_indices]
        elif out_ports == [0, 2, 3]:
            out_tensors = [y, inverse_indices, counts]
        elif out_ports == [0, 1, 3]:
            out_tensors = [y, indices, counts]
        else:
            out_tensors = [y, indices, inverse_indices, counts]
        self.set_out_tensor(out_tensors)


class UnsqueezeOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axes': {'required': True}},    # non-negative
                11: {'axes': {'required': True}},
                13: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(UnsqueezeOp, self).__init__(graph, attr_dict)
        self.update_attributes(UnsqueezeOp, attr_dict)
        assert self.check_required(), 'UnsqueezeOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'axes':
                if cur_ver <= 11:
                    ret = self.__dict__['_attr'][item].value
                else:
                    inputs = self.get_input_tensors()
                    ret = np.array(inputs[1]).tolist()
        except:
            ret = None
        if ret is None:
            ret = super(UnsqueezeOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item == 'axes':
            try:
                self.__dict__['_attr'][item].value = list(value)
            except:
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.INTS, 'value': list(value)})
        else:
            super(UnsqueezeOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(UnsqueezeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.expand_dims(np.array(inputs[0]), axis=self.axes)
        self.set_out_tensor(out_tensor)


class WhereOp(OpNeedBroadcast, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {9: {},
                16: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(WhereOp, self).__init__(graph, attr_dict)
        self.update_attributes(WhereOp, attr_dict)
        assert self.check_required(), 'WhereOp is missing a required parameter.'

    def infer_shape(self):
        super(WhereOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs) == len(self.broad_casted):
            inputs = self.broad_casted
        out_tensor = tf.where(*inputs).numpy()   # tf 1.x only
        self.set_out_tensor(out_tensor)
