# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfTensorArrayV3Op(OpHasVariableOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'element_shape': {'type': AttrType.INTS, 'default': None},
                    'dynamic_size': {'type': AttrType.BOOL, 'default': False},
                    'clear_after_read': {'type': AttrType.BOOL, 'default': True},
                    'identical_element_shapes': {'type': AttrType.BOOL, 'default': False},
                    'tensor_array_name': {'type': AttrType.STRING, 'default': ''}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfTensorArrayV3Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfTensorArrayV3Op, attr_dict)
        assert self.check_required(), 'TfTensorArrayV3Op is missing a required parameter.'

    def infer_shape(self):
        super(TfTensorArrayV3Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = tf.raw_ops.TensorArrayV3(size=inputs[0],
                                               dtype=self.dtype,
                                               element_shape=self.element_shape,
                                               dynamic_size=self.dynamic_size,
                                               clear_after_read=self.clear_after_read,
                                               identical_element_shapes=self.identical_element_shapes,
                                               tensor_array_name=self.tensor_array_name,
                                               )
        out_tensors = [o if i == 0 else o.numpy() for i, o in enumerate(out_tensors)]
        out_tensors = list(map(out_tensors.__getitem__, self.get_out_ports()))
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
