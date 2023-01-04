# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfaddOp(OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(TfaddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.add(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Add', 'version': 7}


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


class TfminimumOp(OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(TfminimumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.minimum(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Min', 'version': 6}
