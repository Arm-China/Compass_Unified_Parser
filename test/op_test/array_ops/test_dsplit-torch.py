# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import numpy as np
from utils.run import run_parser


class dsplit_model_int(torch.nn.Module):
    def __init__(self, split_size):
        super().__init__()
        self.size = split_size

    def forward(self, x):
        out = torch.dsplit(x, sections=self.size)
        return out


class dsplit_model_tuple(torch.nn.Module):
    def __init__(self, split_size):
        super().__init__()
        self.size = split_size

    def forward(self, x):
        out = torch.dsplit(x, indices=self.size)
        return out


def create_dsplit_model(model_path, split_size):
    try:
        if isinstance(split_size, int):
            model = dsplit_model_int(split_size)
        else:
            model = dsplit_model_tuple(split_size)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'dsplit'
input_shapes = [[2, 3, 40]]

for size in [5, (2, 3, 4)]:
    input_data = np.random.ranf(input_shapes[0]).astype(np.float32)
    feed_dict = {'x': input_data}
    model_path = '-'.join([TEST_NAME, str(size)]) + '.pt'
    # prepare model and input datas
    create_dsplit_model(model_path, size)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
