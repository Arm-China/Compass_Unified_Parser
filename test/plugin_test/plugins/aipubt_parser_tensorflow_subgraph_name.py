import numpy as np
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.Parser.plugin_loader import register_plugin, PluginType


@register_plugin(PluginType.Parser, '0.1')
class SubGraphTest_NameMatch(ParserOp):
    op_type = 'MySubGraph'
    priority = 50
    # [div -> mul -> sub] -> add
    start_nodes = ['div']
    end_nodes = ['sub']

    def __init__(self, fw, attrs):
        super(SubGraphTest_NameMatch, self).__init__(fw, attrs)
        self.framework = fw.upper()

    def infer_shape(self, input_tensors, *args):
        '''Note that input_tensors is nested input.
        '''
        assert len(input_tensors) == 2, 'Meet invalid input_tensors: expect length=2, but got %d' % len(input_tensors)

        div_inputs = input_tensors[0]
        sub_inputs = input_tensors[1]
        inp = div_inputs[0]
        sub_y = sub_inputs[0]
        self.constants['div_y'] = div_inputs[1]
        self.constants['bias'] = np.array(-1 * sub_y)
        self.remove_inputs_at(tensor_index=1)
        self.remove_inputs_at(tensor_index=0, node_index=1)
        # Raise error for below: Infer of Node(sub_plugin) meets issues: Expect tensor_index < inputs length (1), but got 1
        # self.remove_inputs_at(tensor_index=1, node_index=1)
        res = np.subtract(inp, sub_y)
        return [res]
