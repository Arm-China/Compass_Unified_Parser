# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ..tf_ops.math_ops import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfabsOp(TfAbsOp, Tf2Op):
    pass


class TfacosOp(TfAcosOp, Tf2Op):
    pass


class TfacoshOp(TfAcoshOp, Tf2Op):
    pass


class TfaddOp(TfAddOp, Tf2Op):
    pass


class Tfadd_nOp(TfAddNOp, Tf2Op):
    def infer_shape(self):
        super(TfAddNOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.add_n(inputs[:-1]).numpy()
        self.set_out_tensor(out_tensor)


class TfargmaxOp(OpHasAxis, OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(TfargmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 3, 'TfargmaxOp expects 3 inputs, but got %d.' % len(inputs)
        self.axis = inputs[1].item(0)
        output_type = inputs[2].item(0)
        out_tensor = tf.argmax(inputs[0],
                               axis=self.axis,
                               output_type=tf.int32
                               if output_type == 'int32'
                               else tf.int64).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ArgMax', 'version': 13}


class TfargminOp(OpHasAxis, OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(TfargminOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 3, 'TfargminOp expects 3 inputs, but got %d.' % len(inputs)
        self.axis = inputs[1].item(0)
        output_type = inputs[2].item(0)
        out_tensor = tf.argmin(inputs[0],
                               axis=self.axis,
                               output_type=tf.int32
                               if output_type == 'int32'
                               else tf.int64).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ArgMin', 'version': 13}


class TfasinOp(TfAsinOp, Tf2Op):
    pass


class TfasinhOp(TfAsinhOp, Tf2Op):
    pass


class TfatanOp(TfAtanOp, Tf2Op):
    pass


class TfatanhOp(TfAtanhOp, Tf2Op):
    pass


class TfceilOp(TfCeilOp, Tf2Op):
    pass


class TfcosOp(TfCosOp, Tf2Op):
    pass


class TfcoshOp(TfCoshOp, Tf2Op):
    pass


class TfcumprodOp(OpHasOneOutPort, OpHasAxis, Tf2Op):
    def infer_shape(self):
        super(TfcumprodOp, self).infer_shape()
        inputs = self.get_input_tensors()
        for idx in range(1, len(inputs)):
            assert inputs[idx].size == 1, 'Expect inputs at index %d of TfcumprodOp (%s) to be scalar, but got size %d' % (
                idx, self.name, inputs[idx].size)
        self.axis = inputs[1].item()
        self.exclusive, self.reverse = [int(inp.item()) for inp in inputs[2:4]]
        out_tensor = tf.math.cumprod(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'CumProd', 'version': 1}


class TfcumsumOp(OpHasOneOutPort, OpHasAxis, Tf2Op):
    def infer_shape(self):
        super(TfcumsumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        for idx in range(1, len(inputs)):
            assert inputs[idx].size == 1, 'Expect inputs at index %d of TfcumsumOp (%s) to be scalar, but got size %d' % (
                idx, self.name, inputs[idx].size)
        self.axis = inputs[1].item()
        self.exclusive, self.reverse = [int(inp.item()) for inp in inputs[2:4]]
        out_tensor = tf.math.cumsum(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'CumSum', 'version': 14}


class TfequalOp(TfEqualOp, Tf2Op):
    pass


class TferfOp(TfErfOp, Tf2Op):
    pass


class TfexpOp(TfExpOp, Tf2Op):
    pass


class TffloorOp(TfFloorOp, Tf2Op):
    pass


class TffloormodOp(TfFloorModOp, Tf2Op):
    pass


class TfgreaterOp(TfGreaterOp, Tf2Op):
    pass


class Tfgreater_equalOp(TfGreaterEqualOp, Tf2Op):
    pass


class Tfin_top_kOp(OpHasOneOutPort, Tf2Op):
    @classmethod
    def attributes(cls):
        return {1: {'k': {'type': AttrType.INT}}
                }

    def __init__(self, graph, attr_dict=None):
        super(Tfin_top_kOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tfin_top_kOp, attr_dict)
        assert self.check_required(), 'Tfin_top_kOp is missing a required parameter.'

    def infer_shape(self):
        super(Tfin_top_kOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 3, 'Tfin_top_kOp expects at least 3 inputs, but got %d' % len(inputs)
        if self.cur_version == 1:
            predictions, targets = inputs[0:2]
        else:
            targets, predictions = inputs[0:2]
        self.k = inputs[2].item(0)
        out_tensor = tf.math.in_top_k(targets, predictions, k=self.k).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'InTopK', 'version': 1}


class Tfl2_normalizeOp(OpHasAxis, OpHasOneOutPort, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-12},
                    'dim': {'default': None},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(Tfl2_normalizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tfl2_normalizeOp, attr_dict)
        assert self.check_required(), 'Tfl2_normalizeOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            input_args = ['x', 'axes', 'epsilon', 'name', 'dim']
            if item != 'name' and item in input_args[1:]:
                inputs = self.get_input_tensors()
                item_idx = input_args.index(item)
                if len(inputs) > item_idx:
                    ret = inputs[item_idx].item() if inputs[item_idx].size == 1 else list(inputs[item_idx])
                    if item in ('axes', 'dim') and ret is not None:
                        ret = ret if isinstance(ret, list) else [ret]
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(Tfl2_normalizeOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Tfl2_normalizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = self.dim
        out_tensor = tf.math.l2_normalize(inputs[0], self.axes, self.epsilon).numpy()
        self.set_out_tensor(out_tensor)


class TflessOp(TfLessOp, Tf2Op):
    pass


class Tfless_equalOp(TfLessEqualOp, Tf2Op):
    pass


class TflogOp(TfLogOp, Tf2Op):
    pass


class Tflogical_andOp(TfLogicalAndOp, Tf2Op):
    pass


class Tflogical_notOp(TfLogicalNotOp, Tf2Op):
    pass


class Tflogical_orOp(TfLogicalOrOp, Tf2Op):
    pass


class TfmatmulOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {2: {'transpose_a': {'type': AttrType.BOOL, 'default': False},
                    'transpose_b': {'type': AttrType.BOOL, 'default': False},
                    'adjoint_a': {'type': AttrType.BOOL, 'default': False},
                    'adjoint_b': {'type': AttrType.BOOL, 'default': False},
                    'output_type': {'type': AttrType.STRING, 'default': None},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfmatmulOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfmatmulOp, attr_dict)
        assert self.check_required(), 'TfmatmulOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            input_args = ['a', 'b', 'transpose_a', 'transpose_b', 'adjoint_a', 'adjoint_b',
                          'a_is_sparse', 'b_is_sparse', 'output_type']
            if item in input_args[2:]:
                inputs = self.get_input_tensors()
                item_idx = input_args.index(item)
                if len(inputs) > item_idx and inputs[item_idx].size == 1:
                    ret = inputs[item_idx].item()
            if ret is not None:
                self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfmatmulOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfmatmulOp, self).infer_shape()
        inputs = self.get_input_tensors()
        adjoint_a, adjoint_b = self.adjoint_a, self.adjoint_b
        if 'complex' in str(inputs[0].dtype) and (adjoint_a or adjoint_b):
            WARN('[Parser]: Complex dtype is not supported so adjoint_a/b in TfmatmulOp (%s) will be treated as transpose_a/b!' % self.name)
        out_tensor = tf.linalg.matmul(inputs[0],
                                      inputs[1],
                                      transpose_a=self.transpose_a,
                                      transpose_b=self.transpose_b,
                                      adjoint_a=adjoint_a,
                                      adjoint_b=adjoint_b,
                                      output_type=self.output_type).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'MatMul', 'version': 9}


class TfmaximumOp(TfMaximumOp, Tf2Op):
    pass


class TfminimumOp(TfMinimumOp, Tf2Op):
    pass


class TfmultiplyOp(TfMulOp, Tf2Op):
    pass


class TfnegativeOp(TfNegOp, Tf2Op):
    pass


class TfnormOp(OpHasAxis, OpHasOneOutPort, Tf2Op):
    @classmethod
    def attributes(cls):
        return {1: {'ord': {'default': 'euclidean'},
                    'keepdims': {'default': 0},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfnormOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfnormOp, attr_dict)
        assert self.check_required(), 'TfnormOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            input_args = ['tensor', 'ord', 'axes', 'keepdims']
            if item in input_args[1:]:
                inputs = self.get_input_tensors()
                item_idx = input_args.index(item)
                if len(inputs) > item_idx:
                    ret = inputs[item_idx].item() if inputs[item_idx].size == 1 else list(inputs[item_idx])
                    if item == 'keepdims':
                        ret = 1 if ret else 0
                    if item == 'axes' and ret is not None:
                        # By default axes value's type is int64, so here need convert axes to int type
                        # because tf runtime will raise error if axes is not int type.
                        ret = [int(axis) for axis in ret] if isinstance(ret, list) else [int(ret)]
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfnormOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfnormOp, self).infer_shape()
        inputs = self.get_input_tensors()
        norm_axis = None
        if self.axes is not None:
            norm_axis = tuple(self.axes) if len(self.axes) > 1 else self.axes[0]
        out_tensor = tf.linalg.norm(inputs[0],
                                    ord=self.ord,
                                    axis=norm_axis,
                                    keepdims=bool(self.keepdims)).numpy()
        self.set_out_tensor(out_tensor)


class TfnormalizeOp(OpHasAxis, OpHasMultipleOutPorts, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'ord': {'default': 'euclidean'},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfnormalizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfnormalizeOp, attr_dict)
        assert self.check_required(), 'TfnormalizeOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            input_args = ['tensor', 'ord', 'axes']
            if item in input_args[1:]:
                inputs = self.get_input_tensors()
                item_idx = input_args.index(item)
                if len(inputs) > item_idx:
                    ret = inputs[item_idx].item() if inputs[item_idx].size == 1 else list(inputs[item_idx])
                    if item == 'axes' and ret is not None:
                        ret = [int(axis) for axis in ret] if isinstance(ret, list) else [int(ret)]
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfnormalizeOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfnormalizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        norm_axis = None
        if self.axes is not None:
            norm_axis = tuple(self.axes) if len(self.axes) > 1 else self.axes[0]
        out_tensors = tf.linalg.normalize(inputs[0],
                                          ord=self.ord,
                                          axis=norm_axis)
        out_tensors = [out.numpy() for out in out_tensors]
        self.set_out_tensor(out_tensors)


class Tfnot_equalOp(TfNotEqualOp, Tf2Op):
    pass


class TfpowOp(TfPowOp, Tf2Op):
    pass


class TfreciprocalOp(TfReciprocalOp, Tf2Op):
    pass


class Tfreduce_allOp(Tf2ReduceOp):
    @classmethod
    def ufunc(cls):
        return tf.math.reduce_all

    def infer_shape(self):
        super(Tfreduce_allOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceAll', 'version': 1}


class Tfreduce_anyOp(Tf2ReduceOp):
    @classmethod
    def ufunc(cls):
        return tf.math.reduce_any

    def infer_shape(self):
        super(Tfreduce_anyOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceAny', 'version': 1}


class Tfreduce_logsumexpOp(Tf2ReduceOp):
    @classmethod
    def ufunc(cls):
        return tf.math.reduce_logsumexp

    def infer_shape(self):
        super(Tfreduce_logsumexpOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceLogSumExp', 'version': 13}


class Tfreduce_maxOp(Tf2ReduceOp):
    @classmethod
    def ufunc(cls):
        return tf.math.reduce_max

    def infer_shape(self):
        super(Tfreduce_maxOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceMax', 'version': 13}


class Tfreduce_meanOp(Tf2ReduceOp):
    @classmethod
    def ufunc(cls):
        return tf.math.reduce_mean

    def infer_shape(self):
        super(Tfreduce_meanOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceMean', 'version': 13}


class Tfreduce_minOp(Tf2ReduceOp):
    @classmethod
    def ufunc(cls):
        return tf.math.reduce_min

    def infer_shape(self):
        super(Tfreduce_minOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceMin', 'version': 13}


class Tfreduce_prodOp(Tf2ReduceOp):
    @classmethod
    def ufunc(cls):
        return tf.math.reduce_prod

    def infer_shape(self):
        super(Tfreduce_prodOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceProd', 'version': 13}


class Tfreduce_sumOp(Tf2ReduceOp):
    @classmethod
    def ufunc(cls):
        return tf.math.reduce_sum

    def infer_shape(self):
        super(Tfreduce_sumOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceSum', 'version': 13}


class Tfreduce_varianceOp(Tf2ReduceOp):
    @classmethod
    def ufunc(cls):
        return tf.math.reduce_variance

    def infer_shape(self):
        super(Tfreduce_varianceOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceVariance', 'version': 1}


class TfroundOp(TfRoundOp, Tf2Op):
    pass


class TfrsqrtOp(TfRsqrtOp, Tf2Op):
    pass


class Tfsegment_sumOp(TfSegmentSumOp, Tf2Op):
    pass


class TfsignOp(TfSignOp, Tf2Op):
    pass


class TfsigmoidOp(TfSigmoidOp, Tf2Op):
    pass


class TfsinOp(TfSinOp, Tf2Op):
    pass


class TfsinhOp(TfSinhOp, Tf2Op):
    pass


class TfsoftplusOp(TfSoftplusOp, Tf2Op):
    pass


class TfsqrtOp(TfSqrtOp, Tf2Op):
    pass


class TfsubtractOp(TfSubOp, Tf2Op):
    pass


class TftanOp(TfTanOp, Tf2Op):
    pass


class TftanhOp(TfTanhOp, Tf2Op):
    pass
