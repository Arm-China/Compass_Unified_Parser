# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.Parser.plugin_loader import register_plugin, PluginType


@register_plugin(PluginType.Parser, '0.1')
class InvertOp(ParserOp):
    op_type = 'Invert'

    def __init__(self, fw, attrs):
        super(InvertOp, self).__init__(fw, attrs)
        self.params['for_test'] = attrs.get('k', 0)

    def infer_shape(self, input_tensors, *args):
        self.params['input_dim'] = len(input_tensors[0].shape)
        out_tensors = input_tensors[0]
        return out_tensors
