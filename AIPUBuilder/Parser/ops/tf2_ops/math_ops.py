# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfaddOp(OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(TfaddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.add(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Add', 'version': 7}


class TfconstantOp(OpHasOneOutPort, ConstLikeOp, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'value': {'type': AttrType.TENSOR, 'required': True, 'default': None},
                    'dtype': {'type': AttrType.STRING, 'required': True},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfconstantOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfconstantOp, attr_dict)
        assert self.check_required(), 'TfconstantOp is missing a required parameter.'

    def infer_shape(self):
        super(TfconstantOp, self).infer_shape()
        out_tensor = self.value.copy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Constant', 'version': 9}
