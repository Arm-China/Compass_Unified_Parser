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

import tensorflow.compat.v1 as tf
from ..op import *
from ...common.errors import *


class TfLeftShiftOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfLeftShiftOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.left_shift(*inputs).eval()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitShift', 'version': 11}


class TfRightShiftOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfRightShiftOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.right_shift(*inputs).eval()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitShift', 'version': 11}
