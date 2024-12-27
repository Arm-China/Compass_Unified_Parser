# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import tensorflow as tf
import math
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL
from ...common.defs import TYPE_MIN, TYPE_MAX, INT_MIN, INT_MAX


class TfBatchToSpaceNDOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfBatchToSpaceNDOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfBatchToSpaceNDOp, attr_dict)
        assert self.check_required(), 'TfBatchToSpaceNDOp is missing a required parameter.'

    def infer_shape(self):
        super(TfBatchToSpaceNDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.compat.v1.batch_to_space_nd(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class TfCastOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'DstT': {'type': AttrType.STRING, 'required': True}},
                }

    def __init__(self, graph, attr_dict=None):
        super(TfCastOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfCastOp, attr_dict)
        assert self.check_required(), 'TfCastOp is missing a required parameter.'

    def infer_shape(self):
        super(TfCastOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = inputs[0].astype(np.dtype(self.DstT))
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Cast', 'version': 19}


class TfConcatV2Op(OpHasAxis, OpHasOneOutPort, TfHasN):
    @classmethod
    def attributes(cls):
        return {1: {'N': {'required': True},
                    'Tidx': {'default': 'int32'}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfConcatV2Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfConcatV2Op, attr_dict)
        assert self.check_required(), 'TfConcatV2Op is missing a required parameter.'

    def infer_shape(self):
        super(TfConcatV2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs) != self.N + 1:
            raise Exception(
                'TfConcat Op (%s) inputs number does not equal to N + 1!' % self.name)
        out_tensor = tf.concat(inputs[:-1], axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        if item in ('axis',):
            try:
                inputs = self.get_input_tensors()
                ret = int(inputs[-1])
                self.__dict__['_attr'][item] = Attribute(
                    item, {'type': AttrType.INT, 'value': int(ret)})
            except:
                ret = None
        if ret is None:
            try:
                ret = super(TfConcatV2Op, self).__getattr__(item)
            except:
                ret = None
        return ret

    @property
    def correspond_onnx_op(self):
        return {'type': 'Concat', 'version': 4}


class TfConstOp(OpHasOneOutPort, ConstLikeOp, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'value': {'type': AttrType.TENSOR, 'required': True, 'default': None},
                    'dtype': {'type': AttrType.STRING, 'required': True},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfConstOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfConstOp, attr_dict)
        assert self.check_required(), 'TfConstOp is missing a required parameter.'

    def infer_shape(self):
        super(TfConstOp, self).infer_shape()
        self.set_out_tensor(self.value)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Constant', 'version': 9}


class TfExpandDimsOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'Tdim': {'type': AttrType.STRING, 'default': 'int32', 'options': ['int32', 'int64']},
                    'axis': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfExpandDimsOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfExpandDimsOp, attr_dict)
        assert self.check_required(), 'TfExpandDimsOp is missing a required parameter.'

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
            ret = super(TfExpandDimsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfExpandDimsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.expand_dims(inputs[0], self.axis).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reshape', 'version': 5}


class TfFakeQuantWithMinMaxVarsOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'num_bits': {'type': AttrType.INT, 'default': 8},
                    'narrow_range': {'type': AttrType.INT, 'default': 0},
                    },
                }

    @staticmethod
    def nudge(min_val, max_val, quant_min, quant_max):
        quant_min_float = float(quant_min)
        quant_max_float = float(quant_max)
        nudged_scale = (max_val - min_val) / \
            (quant_max_float - quant_min_float)
        zero_point_from_min = quant_min_float - min_val / nudged_scale
        if zero_point_from_min < quant_min_float:
            nudged_zero_point = np.uint16(quant_min)
        elif zero_point_from_min > quant_max_float:
            nudged_zero_point = np.uint16(quant_max)
        else:
            nudged_zero_point = np.uint16(np.round(zero_point_from_min))
        nudged_min = (quant_min_float - nudged_zero_point) * nudged_scale
        nudged_max = (quant_max_float - nudged_zero_point) * nudged_scale
        return nudged_min, nudged_max, nudged_scale

    @staticmethod
    def cal_output(inp, nudged_min, nudged_max, nudged_scale):
        clamped = np.clip(inp, nudged_min, nudged_max)
        clamped_shifted = clamped - nudged_min
        out_tensor = np.floor(
            clamped_shifted / nudged_scale + 0.5) * nudged_scale + nudged_min
        return out_tensor.astype(inp.dtype)

    def __init__(self, graph, attr_dict=None):
        super(TfFakeQuantWithMinMaxVarsOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfFakeQuantWithMinMaxVarsOp, attr_dict)
        assert self.check_required(), 'TfFakeQuantWithMinMaxVarsOp is missing a required parameter.'

    def infer_shape(self):
        super(TfFakeQuantWithMinMaxVarsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        quant_min = 1 if bool(self.narrow_range) else 0
        quant_max = 2 ** self.num_bits - 1
        nudged_min, nudged_max, nudged_scale \
            = TfFakeQuantWithMinMaxVarsOp.nudge(inputs[1], inputs[2], quant_min, quant_max)
        out_tensor = TfFakeQuantWithMinMaxVarsOp.cal_output(
            inputs[0], nudged_min, nudged_max, nudged_scale)
        self.set_out_tensor(out_tensor)


class TfFakeQuantWithMinMaxVarsPerChannelOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'num_bits': {'type': AttrType.INT, 'default': 8},
                    'narrow_range': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfFakeQuantWithMinMaxVarsPerChannelOp,
              self).__init__(graph, attr_dict)
        self.update_attributes(
            TfFakeQuantWithMinMaxVarsPerChannelOp, attr_dict)
        assert self.check_required(
        ), 'TfFakeQuantWithMinMaxVarsPerChannelOp is missing a required parameter.'

    def infer_shape(self):
        super(TfFakeQuantWithMinMaxVarsPerChannelOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) >= 3, 'TfFakeQuantWithMinMaxVarsPerChannelOp expects 3 inputs, but got %d!' % len(inputs)
        out_tensor = tf.quantization.fake_quant_with_min_max_vars_per_channel(
            inputs[0], inputs[1], inputs[2], self.num_bits, bool(self.narrow_range)).numpy()
        self.set_out_tensor(out_tensor)


class TfFillOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfFillOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfFillOp, attr_dict)
        assert self.check_required(), 'TfFillOp is missing a required parameter.'

    def infer_shape(self):
        super(TfFillOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.fill(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Fill', 'version': 1}


class TfGatherOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'validate_indices': {'type': AttrType.INT, 'default': 1}}}

    def __init__(self, graph, attr_dict=None):
        super(TfGatherOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfGatherOp, attr_dict)
        assert self.check_required(), 'TfGatherOp is missing a required parameter.'

    def infer_shape(self):
        super(TfGatherOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.Gather(params=inputs[0],
                                       indices=inputs[1],
                                       validate_indices=bool(
                                           self.validate_indices)
                                       ).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Gather', 'version': 11}


class TfGatherV2Op(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 0},
                    'batch_dims': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfGatherV2Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfGatherV2Op, attr_dict)
        assert self.check_required(), 'TfGatherV2Op is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axis':
                if len(self.get_input_tensors()) >= 3:
                    np_axis = self.get_input_tensors()[2]
                    if np_axis is not None and np_axis.size == 1:
                        ret = int(np_axis)
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfGatherV2Op, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfGatherV2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.GatherV2(params=inputs[0],
                                         indices=inputs[1],
                                         axis=self.axis,
                                         batch_dims=self.batch_dims).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        if self.batch_dims == 0:
            return {'type': 'Gather', 'version': 11}
        else:
            return {'type': 'BatchGather', 'version': 1}


class TfGatherNdOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1:
                {'batch_dims': {'default': 0}}}

    def __init__(self, graph, attr_dict=None):
        super(TfGatherNdOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfGatherNdOp, attr_dict)
        assert self.check_required(), 'TfGatherNdOp is missing a required parameter.'

    def infer_shape(self):
        super(TfGatherNdOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.gather_nd(*inputs, batch_dims=self.batch_dims).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'GatherND', 'version': 11}


class TfIdentityOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfIdentityOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfIdentityOp, attr_dict)
        assert self.check_required(), 'TfIdentityOp is missing a required parameter.'

    def infer_shape(self):
        super(TfIdentityOp, self).infer_shape()
        self.set_out_tensor(self.get_input_tensors()[0])

    @property
    def correspond_onnx_op(self):
        return {'type': 'Identity', 'version': 1}


class TfIdentityNOp(OpHasVariableOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfIdentityNOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfIdentityNOp, attr_dict)
        assert self.check_required()

    def infer_shape(self):
        super(TfIdentityNOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_ports = self.get_out_ports()
        out_tensor_list = []
        for i in out_ports:
            out_tensor_list.append(inputs[i].copy())
        self.set_out_tensor(out_tensor_list)


class TfListDiffOp(OpHasVariableOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'out_idx': {'type': AttrType.STRING, 'default': 'int32'}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfListDiffOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfListDiffOp, attr_dict)
        assert self.check_required(), 'TfListDiffOp is missing a required parameter.'

    def infer_shape(self):
        super(TfListDiffOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = tf.raw_ops.ListDiff(
            x=inputs[0],
            y=inputs[1],
            out_idx=self.out_idx
        )
        out_tensors = list(map(out_tensors.__getitem__, self.get_out_ports()))
        out_tensors = [out.numpy() for out in out_tensors]
        self.set_out_tensor(out_tensors)


class TfMirrorPadOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'mode': {'type': AttrType.STRING, 'required': 'true', 'options': ['SYMMETRIC', 'REFLECT']}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfMirrorPadOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfMirrorPadOp, attr_dict)
        assert self.check_required(), 'TfMirrorPadOp is missing a required parameter.'

    def infer_shape(self):
        super(TfMirrorPadOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.MirrorPad(input=inputs[0],
                                          paddings=inputs[1],
                                          mode=self.mode).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pad', 'version': 11}


class TfOneHotOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': -1},
                    'on_value': {'type': AttrType.TENSOR, 'default': 1},
                    'off_value': {'type': AttrType.TENSOR, 'default': 0},
                    'TI': {'type': AttrType.STRING, 'default': None},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfOneHotOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfOneHotOp, attr_dict)
        assert self.check_required(), 'TfOneHotOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'on_value':
                inputs = self.get_input_tensors()
                if len(inputs) >= 2:
                    ret = inputs[2].astype(self.TI).item(0) if self.TI else inputs[2].item(0)
                    self.__dict__['_attr'][item].value = ret

            elif item == 'off_value':
                inputs = self.get_input_tensors()
                if len(inputs) >= 3:
                    ret = inputs[3].astype(self.TI).item(0) if self.TI else inputs[3].item(0)
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfOneHotOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfOneHotOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 4, 'TfOneHotOp expects 4 inputs, but got %d.' % len(inputs)
        out_tensor = tf.one_hot(inputs[0],
                                inputs[1],
                                on_value=self.on_value,
                                off_value=self.off_value,
                                axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)


class TfPadOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'Tpaddings': {'type': AttrType.STRING, 'default': 'int32', 'options': ['int32', 'int64']}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfPadOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfPadOp, attr_dict)
        assert self.check_required(), 'TfPadOp is missing a required parameter.'

    def infer_shape(self):
        super(TfPadOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.pad(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pad', 'version': 11}


class TfPadV2Op(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfPadV2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.PadV2(input=inputs[0],
                                      paddings=inputs[1],
                                      constant_values=inputs[2]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pad', 'version': 11}


class TfPlaceholderOp(OpHasOneOutPort, InputLikeOp, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'shape': {'type': AttrType.INTS, 'required': False, 'default': []},
                    'dtype': {'type': AttrType.STRING, 'required': True},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfPlaceholderOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfPlaceholderOp, attr_dict)
        assert self.check_required(), 'TfPlaceholderOp is missing a required parameter.'

    def infer_shape(self, input_tensor=None):
        super(TfPlaceholderOp, self).infer_shape()
        try:
            out_tensor = self._graph._attr['input_tensors'][self.name].value
        except:
            out_tensor = None
        if out_tensor is None:
            try:
                out_tensor = ((np.random.ranf(self.shape) - 0.5)
                              * 100).astype(np.dtype(self.dtype))
            except:
                out_tensor = None
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Input', 'version': 1}


class TfPlaceholderWithDefaultOp(OpHasOneOutPort, ConstLikeOp, TfOp):
    def infer_shape(self):
        super(TfPlaceholderWithDefaultOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = inputs[0]
        self.set_out_tensor(out_tensor)


class TfPackOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': 0}}}

    def __init__(self, graph, attr_dict=None):
        super(TfPackOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfPackOp, attr_dict)
        assert self.check_required()

    def infer_shape(self):
        super(TfPackOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.stack(inputs, axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ConcatFromSequence', 'version': 11}


class TfReverseV2Op(OpHasAxis, OpHasOneOutPort, TfOp):
    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                ret = self.get_input_tensors()[1].tolist()
                self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfReverseV2Op, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfReverseV2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.reverse(inputs[0], axis=inputs[1]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReverseSequence', 'version': 10}


class TfReshapeOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'Tshape': {'type': AttrType.STRING, 'default': 'int32', 'options': ['int32', 'int64']}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfReshapeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfReshapeOp, attr_dict)
        assert self.check_required(), 'TfReshapeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfReshapeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.reshape(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reshape', 'version': 5}


class TfReverseSequenceOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'batch_dim': {'type': AttrType.INT, 'default': 0},
                    'seq_dim': {'type': AttrType.INT, 'required': True},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfReverseSequenceOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfReverseSequenceOp, attr_dict)
        assert self.check_required(), 'TfReverseSequenceOp is missing a required parameter.'

    def infer_shape(self):
        super(TfReverseSequenceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.reverse_sequence(*inputs,
                                         seq_axis=self.seq_dim,
                                         batch_axis=self.batch_dim).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReverseSequence', 'version': 10}


class TfRollOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfRollOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.roll(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Roll', 'version': 1}


class TfScatterNdOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfScatterNdOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfScatterNdOp, attr_dict)
        assert self.check_required(), 'TfScatterNdOp is missing a required parameter.'

    def infer_shape(self):
        super(TfScatterNdOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.scatter_nd(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ScatterND', 'version': 16}


class TfSelectOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSelectOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.Select(condition=inputs[0],
                                       x=inputs[1],
                                       y=inputs[2]
                                       ).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Where', 'version': 9}


class TfSelectV2Op(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSelectV2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 3 or (len(inputs) == 1 and self._graph.sorted_in_edges(self._attr['name'].value, data=True)[
            0][2]['tensor'].is_const), 'TfSelectV2Op expects 3 inputs, but got %d.' % len(inputs)
        out_tensor = tf.raw_ops.SelectV2(
            condition=inputs[0], t=inputs[1], e=inputs[2]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Where', 'version': 9}


class TfShapeOp(OpHasOneOutPort, ConstLikeOp, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'out_type': {'type': AttrType.STRING, 'default': 'int32', 'options': ['int32', 'int64']}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfShapeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfShapeOp, attr_dict)
        assert self.check_required(), 'TfShapeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfShapeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if inputs and inputs[0] is not None:
            out_tensor = np.array(inputs[0].shape, np.dtype(self.out_type))
        else:
            out_tensor = None
        self.set_out_tensor(out_tensor)


class TfSizeOp(OpHasOneOutPort, ConstLikeOp, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'out_type': {'type': AttrType.STRING, 'default': 'int32'}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfSizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSizeOp, attr_dict)
        assert self.check_required(), 'TfSizeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfSizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if inputs and inputs[0] is not None:
            out_tensor = np.array(inputs[0].size, np.dtype(self.out_type))
        else:
            out_tensor = None
        self.set_out_tensor(out_tensor)


class TfSliceOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSliceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.slice(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Slice', 'version': 10}


class TfSpaceToBatchNDOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfSpaceToBatchNDOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSpaceToBatchNDOp, attr_dict)
        assert self.check_required(), 'TfSpaceToBatchNDOp is missing a required parameter.'

    def infer_shape(self):
        super(TfSpaceToBatchNDOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 3, 'The length of inputs and the shape of inputs are invalid in TfSpaceToBatchNDOp.'
        out_tensor = tf.space_to_batch_nd(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class TfSplitOp(OpHasAxis, OpHasMultipleOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'num_split': {'type': AttrType.INT, 'required': True},
                    'axis': {'default': 0}
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfSplitOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSplitOp, attr_dict)
        assert self.check_required(), 'TfSplitOp is missing a required parameter.'

    def infer_shape(self):
        super(TfSplitOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) >= 2, 'TfSplitOp expects 2 inputs, but got %d.' % len(inputs)
        self.axis = int(inputs[0])
        input_data = inputs[1]
        num_split = self.num_split
        assert num_split is not None, 'Invalid num_split of TfSplitOp(%s)!' % self.name
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


class TfSplitVOp(OpHasAxis, OpHasMultipleOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'num_split': {'type': AttrType.INT, 'required': True},
                    'split': {'type': AttrType.INTS, 'required': False},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfSplitVOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSplitVOp, attr_dict)
        assert self.check_required(), 'TfSplitVOp is missing a required parameter.'

    def infer_shape(self):
        super(TfSplitVOp, self).infer_shape()
        inputs = self.get_input_tensors()
        self.axis = int(inputs[2])
        self.split = inputs[1].tolist()
        if -1 in self.split:
            non_neg_sum = np.sum([s for s in self.split if s != -1])
            index = self.split.index(-1)
            self.split[index] = inputs[0].shape[self.axis] - non_neg_sum
        out_tensors = tf.split(inputs[0], self.split, axis=self.axis)
        out_tensors = [t.numpy() for t in out_tensors]
        self.set_out_tensor(out_tensors)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Split', 'version': 11}


class TfSqueezeOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'squeeze_dims': {'type': AttrType.INTS, 'required': False, 'default': []},
                    'axes': {'default': []}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfSqueezeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSqueezeOp, attr_dict)
        assert self.check_required(), 'TfSqueezeOp is missing a required parameter.'
        if self.squeeze_dims and not self.axes:
            self.axes = copy.deepcopy(self.squeeze_dims)

    def infer_shape(self):
        super(TfSqueezeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.squeeze(inputs[0], axis=self.axes).numpy()
        self.set_out_tensor(out_tensor)


class TfStridedSliceOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'Index': {'type': AttrType.STRING, 'default': 'int32', 'options': ['int32', 'int64']},
                    'begin_mask': {'type': AttrType.INT, 'default': 0},
                    'end_mask': {'type': AttrType.INT, 'default': 0},
                    'ellipsis_mask': {'type': AttrType.INT, 'default': 0},
                    'new_axis_mask': {'type': AttrType.INT, 'default': 0},
                    'shrink_axis_mask': {'type': AttrType.INT, 'default': 0},
                    }
                }

    @staticmethod
    def set_attr_remove_mask_tf(shape, begin, end, strides, shrink_axis_mask, new_axis_mask, ellipsis_mask, begin_mask, end_mask):
        def convert_negative_indices(indices, shape):
            for ind, value in enumerate(indices):
                if value < 0:
                    indices[ind] += shape[ind]

        reshape_dim1 = None
        splits_dim = None
        split_axis = []
        reshape_dim2 = None
        shrink_seen = False
        begin_mask = (~begin_mask) & ((1 << len(begin)) - 1)
        end_mask = (~end_mask) & ((1 << len(begin)) - 1)
        convert_negative_indices(begin, shape)
        convert_negative_indices(end, shape)
        if new_axis_mask != 0:
            new_axises = []
            for i in range(0, len(begin)):
                if ((new_axis_mask >> i) & 1) == 0 or i > len(begin):
                    continue
                if ((ellipsis_mask >> i) & 1) == 1:
                    new_axis_mask &= ~(1 << i)
                    continue
                new_axises.append(i)

            for axis in new_axises:
                shrink_axis_mask &= ~(1 << axis)
                new_axis_mask &= ~(1 << axis)
                ellipsis_mask &= ~(1 << axis)
                if len(begin) > axis:
                    begin[axis] = 0
                    end[axis] = 1
                    strides[axis] = 1
            reshape_dim1 = np.expand_dims(
                np.ones(shape), axis=new_axises).shape
            shape = reshape_dim1

        shape_idx = 0
        begin_id_tmp = []
        end_id_tmp = []
        stride_tmp = []
        out_shape = []

        dims = len(shape)
        strides_size = len(begin)
        for idx in range(dims):
            if shape_idx == dims:
                break
            if idx < strides_size:
                if (shrink_axis_mask >> idx) & 1:
                    shrink_seen = True

                if (new_axis_mask >> idx) & 1:
                    ERROR(
                        '[Parser]: new_axis_mask should be processed to 0 (%s) in convert_strided_slice!' % strided_slice)
                if (ellipsis_mask >> idx) & 1:
                    ellipsis_size = min(
                        dims - strides_size + 1, dims)
                    for _ in range(0, ellipsis_size):
                        left = 0
                        right = shape[shape_idx]
                        step = 1
                        if step < 0 and right < 0:
                            right = INT_MIN
                        begin_id_tmp.append(left)
                        end_id_tmp.append(right)
                        stride_tmp.append(step)
                        out_shape.append(shape[shape_idx])
                        shape_idx = shape_idx + 1
                    continue

            step = strides[idx] if idx < strides_size else 1
            def_beg = 0 if step > 0 else (shape[shape_idx] - 1)
            def_end = shape[shape_idx] if step > 0 else -1
            left = begin[idx] if idx < strides_size and (begin_mask >> idx) & 1 and begin[idx] in range(
                0, shape[shape_idx] + 1) else def_beg
            right = end[idx] if idx < strides_size and (end_mask >> idx) & 1 and end[idx] in range(
                -1, shape[shape_idx] + 1) else def_end
            shape_idx = shape_idx + 1
            if step < 0 and right < 0:
                right = INT_MIN
            if (right - left) / step < 0:
                step = -step
            begin_id_tmp.append(left)
            end_id_tmp.append(right)
            stride_tmp.append(step)
            out_num = (right - left) // step
            if (right - left) % step == 0:
                out_shape.append(out_num)
            else:
                out_shape.append(out_num + 1)
        begin = begin_id_tmp
        end = end_id_tmp
        strides = stride_tmp
        if shrink_seen:
            # need insert split and reshape
            remove_axises = []
            dims = len(begin)
            for i in reversed(range(0, len(begin))):
                if ((shrink_axis_mask >> i) & 1) == 0 or i > len(begin):
                    continue
                if ((ellipsis_mask >> i) & 1) == 1:
                    shrink_axis_mask &= ~(1 << i)
                    continue
                remove_axises.append(i)

            splits_len = 0
            for axis in remove_axises:
                splits_len = out_shape[axis]
                if splits_len != 1:
                    splits_dim = [1, splits_len - 1]
                    split_axis = axis
                reshape_dim2 = out_shape[:axis] + out_shape[axis + 1:]
                out_shape = reshape_dim2
        return np.array(begin), np.array(end), np.array(strides), out_shape, reshape_dim1, split_axis, splits_dim, reshape_dim2

    def __init__(self, graph, attr_dict=None):
        super(TfStridedSliceOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfStridedSliceOp, attr_dict)
        assert self.check_required(), 'TfStridedSliceOp is missing a required parameter.'

    def infer_shape(self):
        super(TfStridedSliceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.strided_slice(*inputs,
                                      begin_mask=self.begin_mask,
                                      end_mask=self.end_mask,
                                      ellipsis_mask=self.ellipsis_mask,
                                      new_axis_mask=self.new_axis_mask,
                                      shrink_axis_mask=self.shrink_axis_mask).numpy()
        self.set_out_tensor(out_tensor)


class TfStopGradientOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfStopGradientOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.StopGradient(input=inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class TfTensorScatterAddOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfTensorScatterAddOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfTensorScatterAddOp, attr_dict)
        assert self.check_required(), 'TfTensorScatterAddOp is missing a required parameter.'

    def infer_shape(self):
        super(TfTensorScatterAddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.TensorScatterAdd(
            tensor=inputs[0], indices=inputs[1], updates=inputs[2]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ScatterND', 'version': 16}


class TfTensorScatterUpdateOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfTensorScatterUpdateOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfTensorScatterUpdateOp, attr_dict)
        assert self.check_required(), 'TfTensorScatterUpdateOp is missing a required parameter.'

    def infer_shape(self):
        super(TfTensorScatterUpdateOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.TensorScatterUpdate(
            tensor=inputs[0], indices=inputs[1], updates=inputs[2]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ScatterND', 'version': 16}


class TfTileOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfTileOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfTileOp, attr_dict)
        assert self.check_required(), 'TfTileOp is missing a required parameter.'

    def infer_shape(self):
        super(TfTileOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 2, 'TfTileOp expects two inputs, but got %d.' % len(inputs)
        out_tensor = tf.tile(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Tile', 'version': 6}


class TfTransposeOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfTransposeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfTransposeOp, attr_dict)
        assert self.check_required(), 'TfTransposeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfTransposeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.transpose(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Transpose', 'version': 1}


class TfUnpackOp(OpHasAxis, OpHasMultipleOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'num': {'type': AttrType.INT, 'required': True},
                    'axis': {'required': True}
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfUnpackOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfUnpackOp, attr_dict)
        assert self.check_required(), 'TfUnpackOp is missing a required parameter.'

    def infer_shape(self):
        super(TfUnpackOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = [t.numpy() for t in tf.unstack(
            inputs[0], num=self.num, axis=self.axis)]
        self.set_out_tensor(out_tensors)


class TfUniqueOp(OpHasMultipleOutPorts, DynamicShapeOp, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'out_idx': {'type': AttrType.STRING, 'default': 'int32', 'options': ['int32', 'int64']}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfUniqueOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfUniqueOp, attr_dict)
        assert self.check_required(), 'TfUniqueOp is missing a required parameter.'

    def infer_shape(self):
        super(TfUniqueOp, self).infer_shape()
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


class TfUniqueWithCountsOp(OpHasMultipleOutPorts, DynamicShapeOp, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'out_idx': {'type': AttrType.STRING, 'default': 'int32', 'options': ['int32', 'int64']}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfUniqueWithCountsOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfUniqueWithCountsOp, attr_dict)
        assert self.check_required(), 'TfUniqueWithCountsOp is missing a required parameter.'

    def infer_shape(self):
        super(TfUniqueWithCountsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.is_all_inputs_const():
            inp = inputs[0]
        else:
            inp_shape = inputs[0].shape
            inp = np.arange(int(np.prod(inp_shape)), dtype=inputs[0].dtype).reshape(inp_shape)
        x, out_idx, counts = tf.unique_with_counts(inp, out_idx=tf.dtypes.as_dtype(self.out_idx))
        out_ports = self.get_out_ports()
        out_dict = {
            0: x.numpy(),
            1: out_idx.numpy(),
            2: counts.numpy()
        }
        out_tensors = []
        for port in out_ports:
            out_tensors.append(out_dict[port])
        self.set_out_tensor(out_tensors)


class TfWhereOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfWhereOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.where(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        if len(self.get_input_tensors()) == 3:
            return {'type': 'Where', 'version': 9}
        else:
            return None


class TfZerosLikeOp(ConstLikeOp, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'dtype': {'default': None,
                              'options': [None,
                                          'float16', 'float32', 'float64',
                                          'int8', 'uint8', 'int16', 'uint16',
                                          'int32', 'int64',
                                          'complex64', 'complex128',
                                          'bool', 'string']
                              }
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfZerosLikeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfZerosLikeOp, attr_dict)
        assert self.check_required(), 'TfZerosLikeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfZerosLikeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        dtype = tf.dtypes.as_dtype(self.dtype) if self.dtype else None
        out_tensor = tf.zeros_like(inputs[0], dtype=dtype).numpy()
        self.set_out_tensor(out_tensor)
