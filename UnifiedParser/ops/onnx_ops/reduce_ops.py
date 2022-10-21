# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from ..op import *


class ReduceL1Op(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 1}},
                11: {'keepdims': {'default': 1}},
                13: {'keepdims': {'default': 1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceL1Op, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceL1Op, attr_dict)
        assert self.check_required(), 'ReduceL1Op is missing a required parameter.'

    def infer_shape(self):
        super(ReduceL1Op, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = np.sum(np.abs(inputs[0]), axis=tuple(
            self.axes), keepdims=bool(self.keepdims))
        self.set_out_tensor(out_tensor)


class ReduceL2Op(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 1}},
                11: {'keepdims': {'default': 1}},
                13: {'keepdims': {'default': 1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceL2Op, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceL2Op, attr_dict)
        assert self.check_required(), 'ReduceL2Op is missing a required parameter.'

    def infer_shape(self):
        super(ReduceL2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = np.sqrt(
            np.sum(np.square(inputs[0]), axis=tuple(self.axes), keepdims=bool(self.keepdims)))
        self.set_out_tensor(out_tensor)


class ReduceLogSumOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'type': AttrType.INT, 'default': 1}},
                11: {'keepdims': {'type': AttrType.INT, 'default': 1}},
                13: {'keepdims': {'type': AttrType.INT, 'default': 1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceLogSumOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceLogSumOp, attr_dict)
        assert self.check_required(), 'ReduceLogSumOp is missing a required parameter.'

    def infer_shape(self):
        super(ReduceLogSumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = np.log(np.sum(inputs[0], axis=tuple(self.axes), keepdims=self.keepdims))
        self.set_out_tensor(out_tensor)


class ReduceLogSumExpOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'type': AttrType.INT, 'default': 1}},
                11: {'keepdims': {'type': AttrType.INT, 'default': 1}},
                13: {'keepdims': {'type': AttrType.INT, 'default': 1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceLogSumExpOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceLogSumExpOp, attr_dict)
        assert self.check_required(), 'ReduceLogSumExpOp is missing a required parameter.'

    def infer_shape(self):
        super(ReduceLogSumExpOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = np.log(np.sum(np.exp(inputs[0]), axis=tuple(self.axes), keepdims=self.keepdims))
        self.set_out_tensor(out_tensor)


class ReduceMaxOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 1}},
                11: {'keepdims': {'default': 1}},
                12: {'keepdims': {'default': 1}},
                13: {'keepdims': {'default': 1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceMaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceMaxOp, attr_dict)
        assert self.check_required(), 'ReduceMaxOp is missing a required parameter.'

    def infer_shape(self):
        super(ReduceMaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = np.maximum.reduce(
            inputs[0], axis=tuple(self.axes), keepdims=self.keepdims)
        self.set_out_tensor(out_tensor)


class ReduceMeanOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                # Accepted axes range is [-r, r-1] where r = rank(data)
                11: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceMeanOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceMeanOp, attr_dict)
        assert self.check_required(), 'ReduceMeanOp is missing a required parameter.'

    def infer_shape(self):
        super(ReduceMeanOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = np.mean(inputs[0], axis=tuple(
            self.axes), keepdims=self.keepdims)
        self.set_out_tensor(out_tensor)


class ReduceMinOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 1}},
                11: {'keepdims': {'default': 1}},
                12: {'keepdims': {'default': 1}},
                13: {'keepdims': {'default': 1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceMinOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceMinOp, attr_dict)
        assert self.check_required(), 'ReduceMinOp is missing a required parameter.'

    def infer_shape(self):
        super(ReduceMinOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = np.minimum.reduce(
            inputs[0], axis=tuple(self.axes), keepdims=self.keepdims)
        self.set_out_tensor(out_tensor)


class ReduceProdOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 1}},
                11: {'keepdims': {'default': 1}},
                13: {'keepdims': {'default': 1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceProdOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceProdOp, attr_dict)
        assert self.check_required(), 'ReduceProdOp is missing a required parameter.'

    def infer_shape(self):
        super(ReduceProdOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = np.prod(inputs[0], axis=tuple(
            self.axes), keepdims=self.keepdims)
        self.set_out_tensor(out_tensor)


class ReduceSumOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {},
                11: {},
                13: {'keepdims': {'default': 1},
                     'noop_with_empty_axes': {'type': AttrType.INT, 'default': 0}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceSumOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceSumOp, attr_dict)
        assert self.check_required(), 'ReduceSumOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            cur_ver = self.__dict__['_attr']['cur_version'].value
            if item == 'noop_with_empty_axes':
                if cur_ver < 13:
                    ret = False
                else:
                    ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(OpHasAxis, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ReduceSumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver >= 13 and len(inputs) == 1 and self.noop_with_empty_axes:
            self.keepdims = True
            out_tensor = inputs[0]
        else:
            if len(inputs) == 2:
                self.axes = inputs[1].tolist() if np.ndim(
                    inputs[1]) != 0 else [int(inputs[1])]
            if self.axes is None:
                self.axes = list(range(len(inputs[0].shape)))
            out_tensor = np.sum(inputs[0], axis=tuple(
                self.axes), keepdims=self.keepdims)
        self.set_out_tensor(out_tensor)


class ReduceSumSquareOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'type': AttrType.INT, 'default': 1}},
                11: {'keepdims': {'type': AttrType.INT, 'default': 1}},
                13: {'keepdims': {'type': AttrType.INT, 'default': 1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceSumSquareOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceSumSquareOp, attr_dict)
        assert self.check_required(), 'ReduceSumSquareOp is missing a required parameter.'

    def infer_shape(self):
        super(ReduceSumSquareOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = np.sum(np.square(inputs[0]), axis=tuple(
            self.axes), keepdims=self.keepdims)
        self.set_out_tensor(out_tensor)
