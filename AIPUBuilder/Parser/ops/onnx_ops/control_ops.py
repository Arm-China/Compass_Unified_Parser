# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from ..op import *


class IfOp(OpHasSubGraph, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'else_branch': {'type': AttrType.GRAPH, 'required': True},
                    'then_branch': {'type': AttrType.GRAPH, 'required': True},
                    },
                11: {'else_branch': {'type': AttrType.GRAPH, 'required': True},
                     'then_branch': {'type': AttrType.GRAPH, 'required': True},
                     },
                13: {'else_branch': {'type': AttrType.GRAPH, 'required': True},
                     'then_branch': {'type': AttrType.GRAPH, 'required': True},
                     }
                }

    def __init__(self, graph, attr_dict=None):
        super(IfOp, self).__init__(graph, attr_dict)
        self.update_attributes(IfOp, attr_dict)
        assert self.check_required(), 'IfOp is missing a required parameter.'

    def infer_shape(self):
        super(IfOp, self).infer_shape()
        inputs = self.get_input_tensors()
        in_ports = self.then_branch._attr['root_in_ports'] if inputs[0] else self.else_branch._attr['root_in_ports']
        self.set_out_tensor(inputs[in_ports[0]: in_ports[-1] + 1])


class LoopOp(OpHasSubGraph, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'body': {'type': AttrType.GRAPH, 'required': True}},
                11: {'body': {'type': AttrType.GRAPH, 'required': True}},
                13: {'body': {'type': AttrType.GRAPH, 'required': True}},
                }

    def __init__(self, graph, attr_dict=None):
        super(LoopOp, self).__init__(graph, attr_dict)
        self.update_attributes(LoopOp, attr_dict)
        assert self.check_required(), 'LoopOp is missing a required parameter.'

    def infer_shape(self):
        super(LoopOp, self).infer_shape()
        inputs = self.get_input_tensors()
        pass
