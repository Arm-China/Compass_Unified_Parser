# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.Parser.plugin_loader import register_plugin, PluginType


@register_plugin(PluginType.Parser, '0.1')
class TfPreprocess(ParserOp):
    op_type = 'TfPreprocess'
    priority = 50
    input_nodes = ['INPUT']  # input node names to insert preprocess
    input_shapes = [[1, 4, 200, 200]]  # new input shapes after adding preprocess

    def __init__(self, fw, attrs):
        super(TfPreprocess, self).__init__(fw, attrs)

    def infer_shape(self, input_tensors, *args):
        assert len(input_tensors) == 1, 'Meet invalid input_tensors: expect length=1, but got %d' % len(input_tensors)

        inp = input_tensors[0]
        self.constants['bias'] = np.zeros(inp.shape[-1], dtype=inp.dtype)
        self.params['axis'] = 3
        res = np.sum(inp, self.params['axis'], keepdims=True)
        return [res]
