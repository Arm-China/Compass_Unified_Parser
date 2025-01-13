# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.Parser.plugin_loader import register_plugin, PluginType


@register_plugin(PluginType.Parser, '0.1')
class PowOp(ParserOp):
    op_type = 'Pow'

    def __init__(self, fw, attrs):
        super(PowOp, self).__init__(fw, attrs)
        self.params['for_test'] = 1

    def infer_shape(self, input_tensors, *args):
        out_tensors = np.power(input_tensors[0], input_tensors[1])
        self.constants['exponent'] = input_tensors[1]
        self.constants['exponent_1'] = input_tensors[1]
        self.remove_inputs_at(1)
        return out_tensors
