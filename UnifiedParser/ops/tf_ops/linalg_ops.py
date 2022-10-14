# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow.compat.v1 as tf
from ..op import *
from ...common.errors import *


class TfMatrixBandPartOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfMatrixBandPartOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.MatrixBandPart(input=inputs[0],
                                               num_lower=inputs[1],
                                               num_upper=inputs[2]
                                               ).eval()
        self.set_out_tensor(out_tensor)
