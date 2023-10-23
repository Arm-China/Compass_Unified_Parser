import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class adaptivepool_model(nn.Module):
    def __init__(self, output_size, method):
        super(adaptivepool_model, self).__init__()
        if method == 'AVG':
            self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=output_size)
        else:
            self.adaptive_pool = nn.AdaptiveMaxPool2d(output_size=output_size)

    def forward(self, x):
        return torch.flatten(self.adaptive_pool(x), 1, -1)


def create_avgpool_model(model_path, x, output_size, method):
    try:
        model = adaptivepool_model(output_size, method)
        # model_scripted = torch.jit.script(model)  # output_size cannot be None or include None
        inputs = (torch.from_numpy(x), )
        model_scripted = torch.jit.trace(model, inputs)  # output_size can be None or have None
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


# TODO: Add return_indices arg for AdaptiveMaxPool2d after op is supported.
TEST_NAME = 'adaptive_pool'

for input_shape in ([2, 51, 52], [2, 6, 51, 52]):
    x_data = np.random.ranf(input_shape).astype(np.float32)
    feed_dict = {'x': x_data}
    for idx, output_size in enumerate([[None, 21], [7, None], 9, [51, 52], [1, 2]]):
        for method in ('AVG', 'MAX'):
            model_path = '-'.join([TEST_NAME, str(len(input_shape)), str(idx), method]) + '.pt'
            # prepare model and input datas
            create_avgpool_model(model_path, x_data, output_size, method)
            # FIXME: Enable verify after opt supports this op
            if output_size in ([51, 52], [1, 2]):  # convert to other pool type
                expected_keywords = []
                unexpected_keywords = ['layer_type=AdaptivePool']
                verify = True
            else:
                expected_keywords = ['layer_type=AdaptivePool']
                unexpected_keywords = []
                verify = False
            exit_status = run_parser(model_path, feed_dict, verify=verify,
                                     expected_keywords=expected_keywords+['method='+method],
                                     unexpected_keywords=unexpected_keywords)
            assert exit_status
