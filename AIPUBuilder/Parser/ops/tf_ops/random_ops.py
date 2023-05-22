# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfRandomUniformOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'seed': {'type': AttrType.INT, 'default': 0},
                    'seed2': {'type': AttrType.INT, 'default': 0},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfRandomUniformOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfRandomUniformOp, attr_dict)
        assert self.check_required(), 'TfRandomUniformOp is missing a required parameter.'

    def infer_shape(self):
        super(TfRandomUniformOp, self).infer_shape()
        # It will generate non-deterministic random numbers if both seed and seeds are 0.
        if self.seed == 0 and self.seed2 == 0:
            WARN('[Parser]: TfRandomUniformOp does not support both seed and seeds to be 0!')
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.RandomUniform(shape=inputs[0],
                                              dtype=self.dtype,
                                              seed=self.seed,
                                              seed2=self.seed2).numpy()
        self.set_out_tensor(out_tensor)
