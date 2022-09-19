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


class TfTensorArrayV3Op(OpHasVariableOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'element_shape': {'type': AttrType.INTS, 'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfTensorArrayV3Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfTensorArrayV3Op, attr_dict)
        assert self.check_required(), 'TfTensorArrayV3Op is missing a required parameter.'

    def infer_shape(self):
        super(TfTensorArrayV3Op, self).infer_shape()
        inputs = self.get_input_tensors()


class TfTensorArrayGatherV3Op(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfTensorArrayGatherV3Op, self).infer_shape()


class TfTensorArrayReadV3Op(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfTensorArrayReadV3Op, self).infer_shape()
        inputs = self.get_input_tensors()


class TfTensorArrayScatterV3Op(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfTensorArrayScatterV3Op, self).infer_shape()
        inputs = self.get_input_tensors()


class TfTensorArraySizeV3Op(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfTensorArraySizeV3Op, self).infer_shape()


class TfTensorArrayWriteV3Op(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfTensorArrayWriteV3Op, self).infer_shape()
