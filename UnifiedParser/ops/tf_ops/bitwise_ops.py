# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


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
