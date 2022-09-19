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
                                              seed2=self.seed2).eval()
        self.set_out_tensor(out_tensor)
