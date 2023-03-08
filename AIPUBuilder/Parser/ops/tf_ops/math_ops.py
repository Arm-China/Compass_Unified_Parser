# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfAbsOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfAbsOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfAbsOp, attr_dict)
        assert self.check_required(), 'TfAbsOp is missing a required parameter.'

    def infer_shape(self):
        super(TfAbsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.abs(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Abs', 'version': 6}


class TfAcosOp(LayoutUnawareOp, OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfAcosOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arccos(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Acos', 'version': 7}


class TfAcoshOp(LayoutUnawareOp, OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfAcoshOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arccosh(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Acosh', 'version': 9}


class TfAllOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 0},
                    'axes': {'default': None}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfAllOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfAllOp, attr_dict)
        assert self.check_required(), 'TfAllOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                inputs = self.get_input_tensors()
                if len(inputs) == 2:
                    ret = [inputs[1].item()] if inputs[1].size == 1 else list(
                        inputs[1])
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfAllOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfAllOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.All(
            input=inputs[0], axis=self.axes, keep_dims=bool(self.keepdims)).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceAll', 'version': 1}


class TfAnyOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 0},
                    'axes': {'default': None}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfAnyOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfAnyOp, attr_dict)
        assert self.check_required(), 'TfAnyOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                inputs = self.get_input_tensors()
                if len(inputs) == 2:
                    ret = [inputs[1].item()] if inputs[1].size == 1 else list(
                        inputs[1])
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfAnyOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfAnyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.Any(
            input=inputs[0], axis=self.axes, keep_dims=bool(self.keepdims)).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceAny', 'version': 1}


class TfAsinOp(LayoutUnawareOp, OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfAsinOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arcsin(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Asin', 'version': 7}


class TfAsinhOp(LayoutUnawareOp, OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfAsinhOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.arcsinh(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Asinh', 'version': 9}


class TfAddOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfAddOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfAddOp, attr_dict)
        assert self.check_required(), 'TfAddOp is missing a required parameter.'

    def infer_shape(self):
        super(TfAddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.add(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Add', 'version': 7}


class TfAddNOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfAddNOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfAddNOp, attr_dict)
        assert self.check_required(), 'TfAddNOp is missing a required parameter.'

    def infer_shape(self):
        super(TfAddNOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.add_n([*inputs]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sum', 'version': 6}


class TfAddV2Op(TfAddOp):
    pass


class TfArgMaxOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'output_type': {'type': AttrType.STRING,
                                    'options': ['int32', 'int64'],
                                    'default': 'int64'},
                    'axis': {'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfArgMaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfArgMaxOp, attr_dict)
        assert self.check_required(), 'TfArgMaxOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axis':
                inputs = self.get_input_tensors()
                if len(inputs) == 2:
                    ret = int(np.array(inputs[1]))
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfArgMaxOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfArgMaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.argmax(inputs[0],
                               axis=self.axis,
                               output_type=tf.int32
                               if self.output_type == 'int32'
                               else tf.int64).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ArgMax', 'version': 13}


class TfArgMinOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'output_type': {'type': AttrType.STRING,
                                    'options': ['int32', 'int64'],
                                    'default': 'int64'},
                    'axis': {'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfArgMinOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfArgMinOp, attr_dict)
        assert self.check_required(), 'TfArgMinOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axis':
                inputs = self.get_input_tensors()
                if len(inputs) == 2:
                    ret = int(np.array(inputs[1]))
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfArgMinOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfArgMinOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.argmin(inputs[0],
                               axis=self.axis,
                               output_type=tf.int32
                               if self.output_type == 'int32'
                               else tf.int64).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ArgMin', 'version': 13}


class TfAtanOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfAtanOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.atan(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Atan', 'version': 7}


class TfAtanhOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfAtanhOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.atanh(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Atanh', 'version': 9}


class TfBatchMatMulV2Op(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'adj_x': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                    'adj_y': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfBatchMatMulV2Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfBatchMatMulV2Op, attr_dict)
        assert self.check_required(), 'TfBatchMatMulV2Op is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item in ('adj_x', 'adj_y'):
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(TfBatchMatMulV2Op, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfBatchMatMulV2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        if 'complex' in str(inputs[0].dtype) and (self.adj_x or self.adj_y):
            WARN('[Parser]: Complex dtype is not supported so adj_x/y in Op (%s) means transpose only!' % self.name)
        out_tensor = tf.raw_ops.BatchMatMulV2(x=inputs[0],
                                              y=inputs[1],
                                              adj_x=self.adj_x,
                                              adj_y=self.adj_y).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'MatMul', 'version': 9}


class TfBatchMatMulOp(TfBatchMatMulV2Op):
    pass


class TfCeilOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfCeilOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.ceil(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Ceil', 'version': 13}


class TfClipByValueOp(ActivationOnlyOp, TfOp):
    def infer_shape(self):
        super(TfClipByValueOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 3, 'TfClipByValueOp expects 3 inputs, but got %d.' % len(inputs)
        out_tensor = np.clip(inputs[0], inputs[1], inputs[2])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Clip', 'version': 12}


class TfBitwiseAndOp(OpHasOneOutPort, TfOp):

    def infer_shape(self):
        super(TfBitwiseAndOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.bitwise_and(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitwiseAnd', 'version': 18}


class TfBitwiseOrOp(OpHasOneOutPort, TfOp):

    def infer_shape(self):
        super(TfBitwiseOrOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.bitwise_or(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitwiseOr', 'version': 18}


class TfBitwiseXorOp(OpHasOneOutPort, TfOp):

    def infer_shape(self):
        super(TfBitwiseXorOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.bitwise_xor(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitwiseXor', 'version': 18}


class TfCosOp(LayoutUnawareOp, OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfCosOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.cos(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Cos', 'version': 7}


class TfCumprodOp(OpHasOneOutPort, OpHasAxis, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'exclusive': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                    'reverse': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfCumprodOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfCumprodOp, attr_dict)
        assert self.check_required(), 'TfCumprodOp is missing a required parameter.'

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
            ret = super(TfCumprodOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item in ('exclusive', 'reverse'):
            self.__dict__['_attr'][item].value = int(value)
        else:
            super(TfCumprodOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(TfCumprodOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.cumprod(inputs[0], axis=self.axis, exclusive=bool(
            self.exclusive), reverse=bool(self.reverse)).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'CumProd', 'version': 1}


class TfCumsumOp(OpHasOneOutPort, OpHasAxis, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'exclusive': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                    'reverse': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}, }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfCumsumOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfCumsumOp, attr_dict)
        assert self.check_required(), 'TfCumsumOp is missing a required parameter.'

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
            ret = super(TfCumsumOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item in ('exclusive', 'reverse'):
            self.__dict__['_attr'][item].value = int(value)
        else:
            super(TfCumsumOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(TfCumsumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.cumsum(inputs[0], axis=self.axis, exclusive=bool(
            self.exclusive), reverse=bool(self.reverse)).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'CumSum', 'version': 14}


class TfCoshOp(LayoutUnawareOp, OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfCoshOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.cosh(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Cosh', 'version': 9}


class TfDivNoNanOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfDivNoNanOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.divide_no_nan(inputs[0], inputs[1]).numpy()
        self.set_out_tensor(out_tensor)


class TfEqualOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfEqualOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.equal(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Equal', 'version': 11}


class TfErfOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfErfOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.erf(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Erf', 'version': 13}


class TfExpOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfExpOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.exp(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Exp', 'version': 13}


class TfFloorOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfFloorOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.floor(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Floor', 'version': 13}


class TfFloorDivOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfFloorDivOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.floordiv(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Div', 'version': 13}


class TfFloorModOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfFloorModOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.floormod(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Mod', 'version': 13}


class TfGreaterOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfGreaterOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.greater(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Greater', 'version': 9}


class TfGreaterEqualOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfGreaterEqualOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.greater_equal(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'GreaterOrEqual', 'version': 12}


class TfInTopKOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'k': {'type': AttrType.INT, 'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfInTopKOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfInTopKOp, attr_dict)
        assert self.check_required(), 'TfInTopKOp is missing a required parameter.'

    def infer_shape(self):
        super(TfInTopKOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 2, 'TfInTopKOp expects 2 inputs, but got %d.' % len(inputs)
        out_tensor = tf.raw_ops.InTopK(
            predictions=inputs[0], targets=inputs[1], k=self.k).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'InTopK', 'version': 1}


class TfInTopKV2Op(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'k': {'type': AttrType.INT, 'required': False}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfInTopKV2Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfInTopKV2Op, attr_dict)
        assert self.check_required(), 'TfInTopKV2Op is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'k':
                inputs = self.get_input_tensors()
                if len(inputs) == 3 and inputs[2] is not None:
                    ret = int(inputs[2].item()) if isinstance(
                        inputs[2], np.ndarray) else int(inputs[2])
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfInTopKV2Op, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfInTopKV2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 3, 'TfInTopKV2Op expects 3 inputs, but got %d.' % len(inputs)
        out_tensor = tf.raw_ops.InTopKV2(
            predictions=inputs[0], targets=inputs[1], k=self.k).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'InTopK', 'version': 1}


class TfInvertPermutationOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfInvertPermutationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 1, 'TfInvertPermutationOp expects 1 inputs, but got %d.' % len(inputs)
        try:
            # This function will raise exception if input is duplicated or larger than input size.
            out_tensor = tf.math.invert_permutation(inputs[0]).numpy()
        except:
            out_tensor = inputs[0]
        self.set_out_tensor(out_tensor)


class TfIsFiniteOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfIsFiniteOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.is_finite(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class TfLessOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfLessOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.less(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Less', 'version': 9}


class TfLessEqualOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfLessEqualOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.less_equal(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LessOrEqual', 'version': 12}


class TfLogOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfLogOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.log(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Log', 'version': 13}


class TfLogicalAndOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfLogicalAndOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.logical_and(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'And', 'version': 7}


class TfLogicalNotOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfLogicalNotOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.logical_not(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Not', 'version': 1}


class TfLogicalOrOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfLogicalOrOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.logical_or(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Or', 'version': 7}


class TfMatMulOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'transpose_a': {'type': AttrType.INT, 'default': 0},
                    'transpose_b': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfMatMulOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfMatMulOp, attr_dict)
        assert self.check_required(), 'TfMatMulOp is missing a required parameter.'

    def infer_shape(self):
        super(TfMatMulOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.linalg.matmul(
            *(inputs[0:2]), transpose_a=self.transpose_a, transpose_b=self.transpose_b).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'MatMul', 'version': 9}


class TfMaxOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 0},
                    'axes': {'default': None}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfMaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfMaxOp, attr_dict)
        assert self.check_required(), 'TfMaxOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                inputs = self.get_input_tensors()
                if len(inputs) == 2:
                    ret = [inputs[1].item()] if inputs[1].size == 1 else list(
                        inputs[1])
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfMaxOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfMaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.Max(
            input=inputs[0], axis=self.axes, keep_dims=bool(self.keepdims)).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceMax', 'version': 11}


class TfMaximumOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfMaximumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.maximum(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Max', 'version': 6}


class TfMeanOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 0},
                    'axes': {'default': None}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfMeanOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfMeanOp, attr_dict)
        assert self.check_required(), 'TfMeanOp is missing a required parameter.'

    def infer_shape(self):
        super(TfMeanOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs) > 1:
            self.axes = inputs[1].tolist() if np.ndim(
                inputs[1]) != 0 else [int(inputs[1])]
        if not self.axes:
            self.axes = list(range(len(inputs[0])))
        out_tensor = tf.math.reduce_mean(
            inputs[0], axis=self.axes, keepdims=self.keepdims).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceMean', 'version': 11}


class TfMinOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 0},
                    'axes': {'default': None}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfMinOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfMinOp, attr_dict)
        assert self.check_required(), 'TfMinOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                inputs = self.get_input_tensors()
                if len(inputs) == 2:
                    ret = [inputs[1].item()] if inputs[1].size == 1 else list(
                        inputs[1])
                    self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfMinOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfMinOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.Min(
            input=inputs[0], axis=self.axes, keep_dims=bool(self.keepdims)).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceMin', 'version': 11}


class TfMinimumOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfMinimumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.minimum(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Min', 'version': 6}


class TfMulOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfMulOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.multiply(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Mul', 'version': 7}


class TfNegOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def infer_shape(self):
        super(TfNegOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = - inputs[0]
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Neg', 'version': 6}


class TfNotEqualOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfNotEqualOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.not_equal(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class TfPowOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfPowOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfPowOp, attr_dict)
        assert self.check_required(), 'TfPowOp is missing a required parameter.'

    def infer_shape(self):
        super(TfPowOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.pow(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pow', 'version': 7}


class TfProdOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'axes': {'default': None},
                    'keepdims': {'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfProdOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfProdOp, attr_dict)
        assert self.check_required(), 'TfProdOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                if len(self.get_input_tensors()) >= 2:
                    np_axes = self.get_input_tensors()[1]
                    if np_axes is not None and np_axes.size > 0:
                        ret = np_axes.tolist()
                        if isinstance(ret, int):
                            ret = [ret]
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfProdOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfProdOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.reduce_prod(
            inputs[0], self.axes, keepdims=bool(self.keepdims)).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceProd', 'version': 13}


class TfRangeOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfRangeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfRangeOp, attr_dict)
        assert self.check_required(), 'TfRangeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfRangeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.range(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class TfRealDivOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfRealDivOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.realdiv(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Div', 'version': 7}


class TfReciprocalOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfReciprocalOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.reciprocal(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reciprocal', 'version': 13}


class TfRoundOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfRoundOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.round(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Round', 'version': 11}


class TfRsqrtOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfRsqrtOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.rsqrt(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class TfSegmentSumOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSegmentSumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.segment_sum(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'SegmentReduce', 'version': 1}


class TfSignOp(LayoutUnawareOp, OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSignOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.sign(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sign', 'version': 9}


class TfSigmoidOp(ActivationOnlyOp, TfOp):
    def infer_shape(self):
        super(TfSigmoidOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.sigmoid(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sigmoid', 'version': 6}


class TfSinOp(LayoutUnawareOp, OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSinOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.sin(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sin', 'version': 7}


class TfSinhOp(LayoutUnawareOp, OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSinhOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.sinh(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sinh', 'version': 9}


class TfSoftsignOp(LayoutUnawareOp, ActivationOnlyOp, TfOp):
    def infer_shape(self):
        super(TfSoftsignOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0])
        out_tensor = out_tensor / (1 + tf.abs(out_tensor))
        self.set_out_tensor(out_tensor.numpy())

    @property
    def correspond_onnx_op(self):
        return {'type': 'Softsign', 'version': 1}


class TfSoftplusOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSoftplusOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.softplus(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Softplus', 'version': 1}


class TfSparseSegmentMeanOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfSparseSegmentMeanOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSparseSegmentMeanOp, attr_dict)
        assert self.check_required(), 'TfSparseSegmentMeanOp is missing a required parameter.'

    def infer_shape(self):
        super(TfSparseSegmentMeanOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.SparseSegmentMean(data=inputs[0], indices=inputs[1], segment_ids=inputs[2]).numpy()
        self.set_out_tensor(out_tensor)


class TfSparseSegmentSqrtNOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfSparseSegmentSqrtNOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSparseSegmentSqrtNOp, attr_dict)
        assert self.check_required(), 'TfSparseSegmentSqrtNOp is missing a required parameter.'

    def infer_shape(self):
        super(TfSparseSegmentSqrtNOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.SparseSegmentSqrtN(data=inputs[0], indices=inputs[1], segment_ids=inputs[2]).numpy()
        self.set_out_tensor(out_tensor)


class TfSparseSegmentSumOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfSparseSegmentSumOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSparseSegmentSumOp, attr_dict)
        assert self.check_required(), 'TfSparseSegmentSumOp is missing a required parameter.'

    def infer_shape(self):
        super(TfSparseSegmentSumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.SparseSegmentSum(data=inputs[0], indices=inputs[1], segment_ids=inputs[2]).numpy()
        self.set_out_tensor(out_tensor)


class TfSqrtOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSqrtOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.sqrt(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sqrt', 'version': 6}


class TfSquareOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSquareOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.square(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class TfSquaredDifferenceOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSquaredDifferenceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.squared_difference(inputs[0], inputs[1]).numpy()
        self.set_out_tensor(out_tensor)


class TfSubOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfSubOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.subtract(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sub', 'version': 7}


class TfSumOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 0},
                    'axes': {'default': None}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfSumOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSumOp, attr_dict)
        assert self.check_required(), 'TfSumOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                if len(self.get_input_tensors()) >= 2:
                    np_axes = self.get_input_tensors()[1]
                    if np_axes is not None and np_axes.size > 0:
                        ret = np_axes.tolist()
                        if isinstance(ret, int):
                            ret = [ret]
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(TfSumOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfSumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.Sum(
            input=inputs[0], axis=self.axes, keep_dims=self.keepdims).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceSum', 'version': 11}


class TfTanOp(LayoutUnawareOp, OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfTanOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.tan(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Tan', 'version': 7}


class TfTanhOp(ActivationOnlyOp, TfOp):
    def infer_shape(self):
        super(TfTanhOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.tanh(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Tanh', 'version': 6}


class TfTopKV2Op(OpHasVariableOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'sorted': {'type': AttrType.INT, 'default': 1, 'options': [0, 1]}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfTopKV2Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfTopKV2Op, attr_dict)
        assert self.check_required(), 'TfTopKV2Op is missing a required parameter.'

    def infer_shape(self):
        super(TfTopKV2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        values, indices = tf.math.top_k(*inputs, sorted=bool(self.sorted))
        out_ports = self.get_out_ports()
        if out_ports == [0]:
            out_tensors = [values.numpy()]
        elif out_ports == [1]:
            out_tensors = [indices.numpy()]
        elif out_ports == [0, 1]:
            out_tensors = [values.numpy(), indices.numpy()]
        else:
            ERROR('[Parser]: Meets invalid out_ports of TfTopKV2Op(%s)!' % self.name)
            out_tensors = []
        self.set_out_tensor(out_tensors)

    @property
    def correspond_onnx_op(self):
        return {'type': 'TopK', 'version': 11}


class TfUniqueOp(OpHasVariableOutPorts, TfOp):
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
        x, out_idx = tf.unique(
            inputs[0], out_idx=tf.dtypes.as_dtype(self.out_idx))
        out_ports = self.get_out_ports()
        if out_ports == [0]:
            out_tensors = [x.numpy()]
        elif out_ports == [1]:
            out_tensors = [out_idx.numpy()]
        elif out_ports == [0, 1]:
            out_tensors = [x.numpy(), out_idx.numpy()]
        else:
            ERROR('[Parser]: Meets invalid out_ports of TfUniqueOp(%s)!' % self.name)
            out_tensors = []
        self.set_out_tensor(out_tensors)
