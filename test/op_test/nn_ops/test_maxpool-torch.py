# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class maxpool_model(nn.Module):
    def __init__(self, dim=2, return_indices=False):
        super(maxpool_model, self).__init__()
        if dim == 1:
            self.maxpool = nn.MaxPool1d(8, stride=11, padding=2, ceil_mode=True, return_indices=return_indices)
        elif dim == 2:
            self.maxpool = nn.MaxPool2d((7, 8), stride=(10, 11), padding=(
                2, 2), ceil_mode=True, return_indices=return_indices)
        else:
            self.maxpool = nn.MaxPool3d((7, 8, 4), stride=(10, 11, 3), padding=(2, 2, 2),
                                        ceil_mode=True, return_indices=return_indices)

    def forward(self, x):
        return self.maxpool(x)


def create_maxpool_model(model_path, dim, return_indices):
    try:
        model = maxpool_model(dim, return_indices)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'maxpool'

for dim in (1, 2, 3):
    for return_indices in (False, True):
        if return_indices and dim != 2:
            # TODO: Currently only supports MaxPool2d with reduce_indices=True
            continue
        model_path = '-'.join([TEST_NAME, str(dim), str(return_indices)]) + '.pt'
        # prepare model and input datas
        input_shape = [2, 6, 52] if dim == 1 else ([2, 6, 51, 52] if dim == 2 else [2, 6, 51, 52, 45])
        x_data = np.random.ranf(input_shape).astype(np.float32)
        feed_dict = {'x': x_data}
        create_maxpool_model(model_path, dim, return_indices)
        exit_status = run_parser(model_path, feed_dict, verify=True)
        assert exit_status
