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


class TfDenseToDenseSetOperationOp(OpHasMultipleOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'validate_indices':  {'type': AttrType.INT, 'default': 1},
                    'set_operation': {'type': AttrType.STRING, 'required': True}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfDenseToDenseSetOperationOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfDenseToDenseSetOperationOp, attr_dict)
        assert self.check_required(), 'TfDenseToDenseSetOperationOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'validate_indices':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(TfDenseToDenseSetOperationOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfDenseToDenseSetOperationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        indices, values, shape = tf.raw_ops.DenseToDenseSetOperation(set1=inputs[0],
                                                                     set2=inputs[1],
                                                                     set_operation=self.set_operation,
                                                                     validate_indices=self.validate_indices
                                                                     ).eval()
        out_tensors = [indices.eval(), values.eval(), shape.eval()]
        self.set_out_tensor(out_tensors)


class TfSparseToDenseOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'validate_indices': {'type': AttrType.INT, 'default': 1}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfSparseToDenseOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSparseToDenseOp, attr_dict)
        assert self.check_required(), 'TfSparseToDenseOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'validate_indices':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(TfSparseToDenseOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfSparseToDenseOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.SparseToDense(sparse_indices=inputs[0],
                                              output_shape=inputs[1],
                                              sparse_values=inputs[2],
                                              default_value=inputs[3],
                                              validate_indices=self.validate_indices).eval()
        self.set_out_tensor(out_tensor)
