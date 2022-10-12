import numpy as np
from UnifiedParser.plugin_op import ParserOp
from UnifiedParser.plugin_loader import register_plugin, PluginType


@register_plugin(PluginType.Parser, '0.1')
class IsNaNOp(ParserOp):
    op_type = 'IsNaN'

    def __init__(self, fw, attrs):
        super(IsNaNOp, self).__init__(fw, attrs)
        self.params['for_test'] = attrs.get('k', 0)

    def infer_shape(self, input_tensors, *args):
        out_tensors = input_tensors[0].astype(np.uint8)
        return out_tensors
