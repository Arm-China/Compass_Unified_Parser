"""
//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
"""

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
