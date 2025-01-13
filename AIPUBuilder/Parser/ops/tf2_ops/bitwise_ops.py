# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class Tfbitwise_andOp(OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(Tfbitwise_andOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.bitwise_and(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitwiseAnd', 'version': 18}


class Tfbitwise_orOp(OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(Tfbitwise_orOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.bitwise_or(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitwiseOr', 'version': 18}


class Tfbitwise_xorOp(OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(Tfbitwise_xorOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.bitwise_xor(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitwiseXor', 'version': 18}


class Tfleft_shiftOp(OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(Tfleft_shiftOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.left_shift(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitShift', 'version': 11}


class Tfright_shiftOp(OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(Tfright_shiftOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.bitwise.right_shift(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BitShift', 'version': 11}
