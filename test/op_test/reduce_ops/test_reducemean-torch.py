# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class mean_model(nn.Module):
    def __init__(self, dim, dtype, keepdim):
        super(mean_model, self).__init__()
        self.dim = dim
        self.dtype = dtype
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is None:
            return torch.mean(x, dtype=self.dtype)
        return torch.mean(x, self.dim, self.keepdim, dtype=self.dtype)


def create_mean_model(model_path, dim=None, dtype=None, keepdim=False):
    try:
        model = mean_model(dim, dtype, keepdim)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'mean'
input_shape = [2, 6, 51, 52]
x_data = np.random.ranf(input_shape).astype(np.float32) * 1000
feed_dict = {'x': x_data}

for dim in (1, [0, 3]):
    for dtype in (None, torch.float32):
        for keepdim in (False, True):
            model_path = '-'.join([TEST_NAME, str(dtype), str(keepdim)]) + '.pt'
            create_mean_model(model_path, dim, dtype, keepdim)
            exit_status = run_parser(model_path, feed_dict, verify=True)
            assert exit_status

# test reduce all
model_path = TEST_NAME + '.pt'
create_mean_model(model_path)
exit_status = run_parser(model_path, feed_dict, verify=True)
assert exit_status
