# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.Parser.plugin_loader import register_plugin, PluginType


@register_plugin(PluginType.Parser, '0.1')
class TestPreprocess(ParserOp):
    op_type = 'Preprocess'
    priority = 50
    input_nodes = ['ONNX_INPUT0', 'ONNX_INPUT1']  # input node names to insert preprocess
    input_shapes = [[1, 300, 2], [1, 400, 300, 2]]  # new input shapes after adding preprocess

    def __init__(self, fw, attrs):
        super(TestPreprocess, self).__init__(fw, attrs)
        self.constants['bias'] = np.zeros([2], dtype=np.float32)
        self.params['axis'] = 1
