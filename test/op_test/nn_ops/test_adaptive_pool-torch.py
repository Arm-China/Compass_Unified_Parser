import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class adaptivepool_model(nn.Module):
    def __init__(self, output_size, method, dim):
        super(adaptivepool_model, self).__init__()
        if method == 'AVG':
            if dim == 1:
                self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=output_size)
            elif dim == 2:
                self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=output_size)
            else:
                self.adaptive_pool = nn.AdaptiveAvgPool3d(output_size=output_size)
        else:
            if dim == 1:
                self.adaptive_pool = nn.AdaptiveMaxPool2d(output_size=output_size)
            elif dim == 2:
                self.adaptive_pool = nn.AdaptiveMaxPool2d(output_size=output_size)
            else:
                self.adaptive_pool = nn.AdaptiveMaxPool3d(output_size=output_size)

    def forward(self, x):
        # return torch.flatten(self.adaptive_pool(x), 1, -1)
        return self.adaptive_pool(x)


def create_avgpool_model(model_path, x, output_size, method, dim):
    try:
        model = adaptivepool_model(output_size, method, dim)
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
            create_avgpool_model(model_path, x_data, output_size, method, 2)
            exit_status = run_parser(model_path, feed_dict, verify=True,
                                     expected_keywords=['method='+method])
            assert exit_status

input_shapes = [[2, 51, 52], [2, 6, 51, 52], [2, 4, 30, 31, 32]]
output_sizes = [(21), (51, 52), (41, 42, 43)]
for input_shape, output_size in zip(input_shapes, output_sizes):
    x_data = np.random.ranf(input_shape).astype(np.float32)
    feed_dict = {'x': x_data}
    for method in ('MAX', 'AVG', ):
        model_path = '-'.join([TEST_NAME, str(len(input_shape)), method]) + '.pt'
        # prepare model and input datas
        create_avgpool_model(model_path, x_data, output_size, method, len(input_shape)-2)
        # opt only support input [N,H,W,C] or [H,W,C]
        verify = False if len(input_shape) != 4 else True
        exit_status = run_parser(model_path, feed_dict, verify=verify,
                                 expected_keywords=['method='+method])
        assert exit_status
