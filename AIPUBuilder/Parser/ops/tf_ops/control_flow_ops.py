# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfAssertOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'summarize': {'type': AttrType.INT, },
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfAssertOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfAssertOp, attr_dict)
        assert self.check_required(), 'TfAssertOp is missing a required parameter.'

    def infer_shape(self):
        super(TfAssertOp, self).infer_shape()
        inputs = self.get_input_tensors()


class TfEnterOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'parallel_iterations': {'type': AttrType.INT, 'default': 10},
                    'frame_name': {'type': AttrType.STRING, 'default': ''},
                    'is_constant': {'type': AttrType.INT, 'default': 0},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfEnterOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfEnterOp, attr_dict)
        assert self.check_required(), 'TfEnterOp is missing a required parameter.'

    def infer_shape(self):
        super(TfEnterOp, self).infer_shape()
        inputs = self.get_input_tensors()


class TfExitOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfExitOp, self).infer_shape()
        inputs = self.get_input_tensors()


class TfMergeOp(OpHasVariableOutPorts, TfHasN):
    @classmethod
    def attributes(cls):
        return {1: {'N': {'default': 1},
                    'value_index': {'type': AttrType.INT, 'required': False, 'default': None}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfMergeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfMergeOp, attr_dict)
        assert self.check_required(), 'TfMergeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfMergeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array([])
        for index, inp in enumerate(inputs):
            if inp is not None and inp.size != 0:
                out_tensor = inp
                self.value_index = index
                break
        self.set_out_tensor([out_tensor])


class TfNextIterationOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfNextIterationOp, self).infer_shape()
        inputs = self.get_input_tensors()


class TfLoopCondOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfLoopCondOp, self).infer_shape()
        inputs = self.get_input_tensors()


class TfStatelessIfOp(OpHasVariableOutPorts, TfOp):
    def infer_shape(self):
        super(TfStatelessIfOp, self).infer_shape()
        inputs = self.get_input_tensors()


class TfSwitchOp(OpHasVariableOutPorts, TfOp):
    def infer_shape(self):
        super(TfSwitchOp, self).infer_shape()
        inputs = self.get_input_tensors()
        pred = inputs[1].item()
        from tensorflow.python.ops import control_flow_ops
        output_false, output_true = control_flow_ops.switch(inputs[0], inputs[1], dtype=self.dtype)
        try:
            output_false = output_false.numpy()
        except:
            output_false = np.empty(0, dtype=self.dtype) if pred else inputs[0].astype(self.dtype)
        try:
            output_true = output_true.numpy()
        except:
            output_true = inputs[0].astype(self.dtype) if pred else np.empty(0, dtype=self.dtype)
        out_ports = self.get_out_ports()
        if out_ports == [0]:
            self.set_out_tensor([output_false])
        elif out_ports == [1]:
            self.set_out_tensor([output_true])
        else:
            self.set_out_tensor([output_false, output_true])
