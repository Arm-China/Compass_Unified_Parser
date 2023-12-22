# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class lppool_model(nn.Module):
    def __init__(self, dim, norm_type, ceil_mode):
        super(lppool_model, self).__init__()
        if dim == 1:
            self.lppool = nn.LPPool1d(norm_type, 8, stride=11, ceil_mode=ceil_mode)
        else:  # dim == 2
            self.lppool = nn.LPPool2d(norm_type, (7, 8), stride=(10, 11), ceil_mode=ceil_mode)

    def forward(self, x):
        return self.lppool(x)


def create_lppool_model(model_path, dim, norm_type, ceil_mode):
    try:
        model = lppool_model(dim, norm_type, ceil_mode)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'lppool'

for dim in (1, 2, ):
    input_shape = [2, 6, 52] if dim == 1 else [2, 6, 51, 52]
    x_data = np.random.ranf(input_shape).astype(np.float32) - 0.5
    feed_dict = {'x': x_data}
    # np.save('input', feed_dict)
    for norm_type in (1, 2):
        for ceil_mode in (False, True):
            model_path = '-'.join([TEST_NAME, str(dim), str(norm_type), str(ceil_mode)]) + '.pt'
            create_lppool_model(model_path, dim, norm_type, ceil_mode)
            exit_status = run_parser(model_path, feed_dict, verify=True,
                                     unexpected_keywords=['layer_type=Abs', 'layer_type=Pad'])
            assert exit_status
