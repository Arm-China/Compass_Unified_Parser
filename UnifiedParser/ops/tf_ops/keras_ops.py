# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ...common.errors import *


class TfInputLayerOp(OpHasOneOutPort, InputLikeOp, TfOp):
    @classmethod
    def attributes(cls):
        return {2: {'input_shape': {'type': AttrType.INTS, 'required': False, 'default': None},
                    'batch_size': {'type': AttrType.INT, 'required': False, 'default': None},
                    'dtype': {'type': AttrType.STRING, 'required': False, 'default': 'float32'},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfInputLayerOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfInputLayerOp, attr_dict)
        assert self.check_required(), 'TfInputLayerOp is missing a required parameter.'

    def infer_shape(self, input_tensor=None):
        super(TfInputLayerOp, self).infer_shape()
        try:
            out_tensor = self._graph._attr['input_tensors'][self.name].value
        except:
            try:
                if self.batch_size is None or self.input_shape is None:
                    out_tensor = None
                else:
                    shape = [self.batch_size] + self.input_shape[1:]
                    out_tensor = ((np.random.ranf(shape) - 0.5)
                                  * 100).astype(np.dtype(self.dtype))
            except:
                out_tensor = None
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Input', 'version': None}
