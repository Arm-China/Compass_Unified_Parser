# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.Parser.plugin_loader import register_plugin, PluginType


@register_plugin(PluginType.Parser, '0.1')
class TestPreprocessNeg(ParserOp):
    op_type = 'PreprocessNeg'
    priority = 49  # priority is lower than the one in aipubt_parser_plugin_onnx_preprocess_no_infer.py
    input_nodes = ['ONNX_INPUT0']  # also appears in aipubt_parser_plugin_onnx_preprocess_no_infer.py
    input_shapes = [[1, 150, 3]]  # different input shape

    def __init__(self, fw, attrs):
        super(TestPreprocessNeg, self).__init__(fw, attrs)
        self.constants['bias'] = np.zeros([1], dtype=np.float32)
        self.params['axis'] = 2
