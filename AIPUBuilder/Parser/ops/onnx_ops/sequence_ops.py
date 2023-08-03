# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from ..op import *


class ConcatFromSequenceOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {'axis': {'required': True},
                     'new_axis': {'type': AttrType.INT, 'default': 0}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ConcatFromSequenceOp, self).__init__(graph, attr_dict)
        self.update_attributes(ConcatFromSequenceOp, attr_dict)
        assert self.check_required(), 'ConcatFromSequenceOp is missing a required parameter.'

    def infer_shape(self):
        super(ConcatFromSequenceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.new_axis == 0:
            out_tensor = np.concatenate(inputs, axis=self.axis)
        else:
            out_tensor = np.stack(inputs, self.axis)
        self.set_out_tensor(out_tensor)


class SequenceAtOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {}}

    def __init__(self, graph, attr_dict=None):
        super(SequenceAtOp, self).__init__(graph, attr_dict)
        self.update_attributes(SequenceAtOp, attr_dict)
        assert self.check_required(), 'SequenceAtOp is missing a required parameter.'

    def infer_shape(self):
        super(SequenceAtOp, self).infer_shape()


class SequenceConstructOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {11: {}}

    def __init__(self, graph, attr_dict=None):
        super(SequenceConstructOp, self).__init__(graph, attr_dict)
        self.update_attributes(SequenceConstructOp, attr_dict)
        assert self.check_required(), 'SequenceConstructOp is missing a required parameter.'

    def infer_shape(self):
        super(SequenceConstructOp, self).infer_shape()
