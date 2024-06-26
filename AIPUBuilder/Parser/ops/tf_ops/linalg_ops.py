# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfMatrixBandPartOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfMatrixBandPartOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.MatrixBandPart(input=inputs[0],
                                               num_lower=inputs[1],
                                               num_upper=inputs[2]
                                               ).numpy()
        self.set_out_tensor(out_tensor)
