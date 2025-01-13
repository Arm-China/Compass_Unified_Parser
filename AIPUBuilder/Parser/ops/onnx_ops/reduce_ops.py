# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from ..op import *


class ReduceL1Op(OnnxReduceOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                11: {},
                13: {},
                18: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceL1Op, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceL1Op, attr_dict)
        assert self.check_required(), 'ReduceL1Op is missing a required parameter.'

    @classmethod
    def ufunc(cls):
        return lambda x, y, z: np.sum(np.abs(x), axis=y, keepdims=z)

    def infer_shape(self):
        super(ReduceL1Op, self).infer_shape()


class ReduceL2Op(OnnxReduceOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                11: {},
                13: {},
                18: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceL2Op, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceL2Op, attr_dict)
        assert self.check_required(), 'ReduceL2Op is missing a required parameter.'

    @classmethod
    def ufunc(cls):
        return lambda x, y, z: np.sqrt(np.sum(np.square(x), axis=y, keepdims=z))

    def infer_shape(self):
        super(ReduceL2Op, self).infer_shape()


class ReduceLogSumOp(OnnxReduceOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                11: {},
                13: {},
                18: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceLogSumOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceLogSumOp, attr_dict)
        assert self.check_required(), 'ReduceLogSumOp is missing a required parameter.'

    @classmethod
    def ufunc(cls):
        return lambda x, y, z: np.log(np.sum(x, axis=y, keepdims=z))

    def infer_shape(self):
        super(ReduceLogSumOp, self).infer_shape()


class ReduceLogSumExpOp(OnnxReduceOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                11: {},
                13: {},
                18: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceLogSumExpOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceLogSumExpOp, attr_dict)
        assert self.check_required(), 'ReduceLogSumExpOp is missing a required parameter.'

    @classmethod
    def ufunc(cls):
        return lambda x, y, z: np.log(np.sum(np.exp(x), axis=y, keepdims=z))

    def infer_shape(self):
        super(ReduceLogSumExpOp, self).infer_shape()


class ReduceMaxOp(OnnxReduceOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                11: {},
                12: {},
                13: {},
                18: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceMaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceMaxOp, attr_dict)
        assert self.check_required(), 'ReduceMaxOp is missing a required parameter.'

    @classmethod
    def ufunc(cls):
        return lambda x, y, z: np.maximum.reduce(x, axis=y, keepdims=z)

    def infer_shape(self):
        super(ReduceMaxOp, self).infer_shape()


class ReduceMeanOp(OnnxReduceOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                # Accepted axes range is [-r, r-1] where r = rank(data)
                11: {},
                13: {},
                18: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceMeanOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceMeanOp, attr_dict)
        assert self.check_required(), 'ReduceMeanOp is missing a required parameter.'

    @classmethod
    def ufunc(cls):
        return lambda x, y, z: np.mean(x, axis=y, keepdims=z)

    def infer_shape(self):
        super(ReduceMeanOp, self).infer_shape()


class ReduceMinOp(OnnxReduceOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                11: {},
                12: {},
                13: {},
                18: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceMinOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceMinOp, attr_dict)
        assert self.check_required(), 'ReduceMinOp is missing a required parameter.'

    @classmethod
    def ufunc(cls):
        return lambda x, y, z: np.minimum.reduce(x, axis=y, keepdims=z)

    def infer_shape(self):
        super(ReduceMinOp, self).infer_shape()


class ReduceProdOp(OnnxReduceOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                11: {},
                13: {},
                18: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceProdOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceProdOp, attr_dict)
        assert self.check_required(), 'ReduceProdOp is missing a required parameter.'

    @classmethod
    def ufunc(cls):
        return lambda x, y, z: np.prod(x, axis=y, keepdims=z)

    def infer_shape(self):
        super(ReduceProdOp, self).infer_shape()


class ReduceSumOp(OnnxReduceOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                11: {},
                13: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceSumOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceSumOp, attr_dict)
        assert self.check_required(), 'ReduceSumOp is missing a required parameter.'

    @classmethod
    def ufunc(cls):
        return lambda x, y, z: np.sum(x, axis=y, keepdims=z)

    def infer_shape(self):
        super(ReduceSumOp, self).infer_shape()


class ReduceSumSquareOp(OnnxReduceOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                11: {},
                13: {},
                18: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceSumSquareOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceSumSquareOp, attr_dict)
        assert self.check_required(), 'ReduceSumSquareOp is missing a required parameter.'

    @classmethod
    def ufunc(cls):
        return lambda x, y, z: np.sum(np.square(x), axis=y, keepdims=z)

    def infer_shape(self):
        super(ReduceSumSquareOp, self).infer_shape()
