# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class round_model(nn.Module):
    def __init__(self, decimal=0):
        super(round_model, self).__init__()
        self.decimal = decimal

    def forward(self, x):
        return torch.round(x, decimals=self.decimal)


def create_round_model(model_path, decimal):
    try:
        model = round_model(decimal)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'round'
input_shapes = [[2, 3, 4, 5, 6], [3], [2, 3, 4]]
decimals = [3, -4, 0]

for x_shape in input_shapes:
    x_data = np.random.ranf(x_shape).astype(np.float32) * 2000
    feed_dict = {'x': x_data}
    # np.save('input', feed_dict)
    for decimal in decimals:
        model_path = '-'.join([TEST_NAME, str(len(x_shape)), str(decimal)]) + '.pt'
        create_round_model(model_path, decimal)
        exit_status = run_parser(model_path, feed_dict)
        assert exit_status
