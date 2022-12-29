# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfTensorArrayV3Op(OpHasVariableOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'element_shape': {'type': AttrType.INTS, 'required': True},
                    'dynamic_size': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                    'clear_after_read': {'type': AttrType.INT, 'default': 1, 'options': [0, 1]},
                    'identical_element_shapes': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfTensorArrayV3Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfTensorArrayV3Op, attr_dict)
        assert self.check_required(), 'TfTensorArrayV3Op is missing a required parameter.'

    def infer_shape(self):
        super(TfTensorArrayV3Op, self).infer_shape()
        inputs = self.get_input_tensors()
        handle, flow = tf.raw_ops.TensorArrayV3(
            size=inputs[0],
            dtype=self.dtype,
            element_shape=self.element_shape,
            dynamic_size=self.dynamic_size,
            clear_after_read=self.clear_after_read,
            identical_element_shapes=self.identical_element_shapes)
        out_ports = self.get_out_ports()
        if out_ports == [0]:
            out_tensors = [handle]
        elif out_ports == [1]:
            out_tensors = [flow.numpy()]
        else:
            out_tensors = [handle, flow.numpy()]
        self.set_out_tensor(out_tensors)


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
