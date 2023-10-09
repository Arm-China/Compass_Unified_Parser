# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.Parser.plugin_loader import register_plugin, PluginType


@register_plugin(PluginType.Parser, '0.1')
class IsNaNOp(ParserOp):
    alias_op_type = 'NewIsNaN'
    op_type = 'IsNaN'

    def __init__(self, fw, attrs):
        super(IsNaNOp, self).__init__(fw, attrs)
        self.params['for_test'] = attrs.get('k', 0)

    def infer_shape(self, input_tensors, *args):
        out_tensors = input_tensors[0].astype(np.uint8)
        return out_tensors
