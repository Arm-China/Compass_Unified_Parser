import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class avgpool_model(nn.Module):
    def __init__(self, dim=2, count_include_pad=False):
        super(avgpool_model, self).__init__()
        if dim == 1:
            self.avgpool = nn.AvgPool1d(8, stride=11, padding=2, ceil_mode=True,
                                        count_include_pad=count_include_pad)
        elif dim == 2:
            self.avgpool = nn.AvgPool2d((7, 8), stride=(10, 11), padding=(2, 2),
                                        ceil_mode=True, count_include_pad=count_include_pad)
        else:
            self.avgpool = nn.AvgPool3d((7, 8, 4), stride=(10, 11, 3), padding=(2, 2, 2),
                                        ceil_mode=True, count_include_pad=count_include_pad)

    def forward(self, x):
        return self.avgpool(x)


def create_avgpool_model(model_path, dim, count_include_pad):
    try:
        model = avgpool_model(dim, count_include_pad)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'avgpool'

for dim in (1, 2, 3):
    for count_include_pad in (False, True):
        model_path = '-'.join([TEST_NAME, str(dim), str(count_include_pad)]) + '.pt'
        # prepare model and input datas
        input_shape = [2, 6, 52] if dim == 1 else ([2, 6, 51, 52] if dim == 2 else [2, 6, 51, 52, 45])
        x_data = np.random.ranf(input_shape).astype(np.float32)
        feed_dict = {'x': x_data}
        create_avgpool_model(model_path, dim, count_include_pad)
        exit_status = run_parser(model_path, feed_dict, verify=True)
        assert exit_status
