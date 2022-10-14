# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


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
