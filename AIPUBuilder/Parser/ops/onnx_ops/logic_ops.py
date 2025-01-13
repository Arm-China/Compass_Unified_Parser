# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class AndOp(OpNeedBroadcast, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0}
                    },
                7: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(AndOp, self).__init__(graph, attr_dict)
        self.update_attributes(AndOp, attr_dict)
        assert self.check_required(), 'AndOp is missing a required parameter.'

    def infer_shape(self):
        super(AndOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver == 1:
            if self.broadcast:
                second_input = OpHasAxis.broadcast_to(
                    inputs[1], inputs[0].shape, int(self.axis))
            else:
                assert list(inputs[0].shape) == list(
                    inputs[1].shape), 'The lengths of input0 and input1 are not the same in AndOp infer shape.'
                second_input = inputs[1]
            out_tensor = np.logical_and(inputs[0], second_input)
        else:
            out_tensor = np.logical_and(*inputs)
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'broadcast':
                if cur_ver == 1:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                          (cur_ver, type(self).__name__))
        except:
            ret = None
        if ret is None:
            ret = super(AndOp, self).__getattr__(item)
        return ret


class EqualOp(OpNeedBroadcast, OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0}
                    },
                7: {},
                11: {},
                13: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(EqualOp, self).__init__(graph, attr_dict)
        self.update_attributes(EqualOp, attr_dict)
        assert self.check_required(), 'EqualOp is missing a required parameter.'

    def infer_shape(self):
        super(EqualOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver == 1:
            if self.broadcast:
                second_input = OpHasAxis.broadcast_to(
                    inputs[1], inputs[0].shape, int(self.axis))
            else:
                assert list(inputs[0].shape) == list(
                    inputs[1].shape), 'The lengths of input0 and input1 are not the same in AndOp infer shape.'
                second_input = inputs[1]
            out_tensor = np.equal(inputs[0], second_input)
        else:
            out_tensor = np.equal(*inputs)
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'broadcast':
                if cur_ver == 1:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                          (cur_ver, type(self).__name__))
        except:
            ret = None
        if ret is None:
            ret = super(EqualOp, self).__getattr__(item)
        return ret


class GreaterOp(OpNeedBroadcast, OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0}
                    },
                7: {},
                9: {},
                13: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(GreaterOp, self).__init__(graph, attr_dict)
        self.update_attributes(GreaterOp, attr_dict)
        assert self.check_required(), 'GreaterOp is missing a required parameter.'

    def infer_shape(self):
        super(GreaterOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver == 1:
            if self.broadcast:
                second_input = OpHasAxis.broadcast_to(
                    inputs[1], inputs[0].shape, int(self.axis))
            else:
                assert list(inputs[0].shape) == list(
                    inputs[1].shape), 'The lengths of input0 and input1 are not the same in AndOp infer shape.'
                second_input = inputs[1]
            out_tensor = np.greater(inputs[0], second_input)
        else:
            out_tensor = np.greater(*inputs)
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'broadcast':
                if cur_ver == 1:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                          (cur_ver, type(self).__name__))
        except:
            ret = None
        if ret is None:
            ret = super(GreaterOp, self).__getattr__(item)
        return ret


class GreaterOrEqualOp(OpNeedBroadcast, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {12: {},
                16: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(GreaterOrEqualOp, self).__init__(graph, attr_dict)
        self.update_attributes(GreaterOrEqualOp, attr_dict)
        assert self.check_required(), 'GreaterOrEqualOp is missing a required parameter.'

    def infer_shape(self):
        super(GreaterOrEqualOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.greater_equal(*inputs)
        self.set_out_tensor(out_tensor)


class LessOp(OpNeedBroadcast, OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0}
                    },
                7: {},
                9: {},
                13: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(LessOp, self).__init__(graph, attr_dict)
        self.update_attributes(LessOp, attr_dict)
        assert self.check_required(), 'LessOp is missing a required parameter.'

    def infer_shape(self):
        super(LessOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver == 1:
            if self.broadcast:
                second_input = OpHasAxis.broadcast_to(
                    inputs[1], inputs[0].shape, int(self.axis))
            else:
                assert list(inputs[0].shape) == list(
                    inputs[1].shape), 'The lengths of input0 and input1 are not the same in AndOp infer shape.'
                second_input = inputs[1]
            out_tensor = np.less(inputs[0], second_input)
        else:
            out_tensor = np.less(*inputs)
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'broadcast':
                if cur_ver == 1:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                          (cur_ver, type(self).__name__))
        except:
            ret = None
        if ret is None:
            ret = super(LessOp, self).__getattr__(item)
        return ret


class LessOrEqualOp(OpNeedBroadcast, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {12: {},
                16: {}
                }

    def __init__(self, graph, attr_dict=None):
        super(LessOrEqualOp, self).__init__(graph, attr_dict)
        self.update_attributes(LessOrEqualOp, attr_dict)
        assert self.check_required(), 'LessOrEqualOp is missing a required parameter.'

    def infer_shape(self):
        super(LessOrEqualOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.less_equal(*inputs)
        self.set_out_tensor(out_tensor)


class NotOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}, }

    def __init__(self, graph, attr_dict=None):
        super(NotOp, self).__init__(graph, attr_dict)
        self.update_attributes(NotOp, attr_dict)
        assert self.check_required(), 'NotOp is missing a required parameter.'

    def infer_shape(self):
        super(NotOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.logical_not(inputs[0])
        self.set_out_tensor(out_tensor)


class OrOp(OpNeedBroadcast, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0}
                    },
                7: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(OrOp, self).__init__(graph, attr_dict)
        self.update_attributes(OrOp, attr_dict)
        assert self.check_required(), 'OrOp is missing a required parameter.'

    def infer_shape(self):
        super(OrOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver == 1:
            if self.broadcast:
                second_input = OpHasAxis.broadcast_to(
                    inputs[1], inputs[0].shape, int(self.axis))
            else:
                assert list(inputs[0].shape) == list(
                    inputs[1].shape), 'The lengths of input0 and input1 are not the same in AndOp infer shape.'
                second_input = inputs[1]
            out_tensor = np.logical_or(inputs[0], second_input)
        else:
            out_tensor = np.logical_or(*inputs)
        self.set_out_tensor(out_tensor)

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'broadcast':
                if cur_ver == 1:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ERROR('[Parser]: Unsupported op version [%s] for %s!' %
                          (cur_ver, type(self).__name__))
        except:
            ret = None
        if ret is None:
            ret = super(OrOp, self).__getattr__(item)
        return ret


class XorOp(OpNeedBroadcast, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'type': AttrType.INT},
                    'broadcast': {'type': AttrType.INT, 'default': 0}
                    },
                7: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(XorOp, self).__init__(graph, attr_dict)
        self.update_attributes(XorOp, attr_dict)
        assert self.check_required(), 'XorOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        cur_ver = self.__dict__['_attr']['cur_version'].value
        try:
            if item == 'broadcast':
                if cur_ver == 1:
                    ret = bool(self.__dict__['_attr'][item].value)
                else:
                    ret = None
        except:
            ret = None
        if ret is None:
            ret = super(XorOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(XorOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cur_ver = self.cur_version
        if cur_ver == 1:
            if self.broadcast:
                second_input = OpHasAxis.broadcast_to(
                    inputs[1], inputs[0].shape, int(self.axis))
            else:
                assert list(inputs[0].shape) == list(
                    inputs[1].shape), 'The lengths of input0 and input1 are not the same in AndOp infer shape.'
                second_input = inputs[1]
            out_tensor = np.logical_xor(inputs[0], second_input)
        else:
            out_tensor = np.logical_xor(*inputs)
        self.set_out_tensor(out_tensor)
