# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class gelu_model(nn.Module):
    def __init__(self, approximate):
        super(gelu_model, self).__init__()
        self.gelu = nn.GELU(approximate)

    def forward(self, x):
        return self.gelu(x) * 100


def create_gelu_model(model_path, approximate):
    try:
        model = gelu_model(approximate)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'gelu'

for input_shape in ([2, 75, 121], ):
    # prepare model and input datas
    x_data = np.random.randint(-1000, 3000, input_shape) / 1000
    feed_dict = {'x': x_data}
    np.save('input', feed_dict)
    for approximate in ('none', 'tanh'):
        model_path = '-'.join([TEST_NAME, str(len(input_shape)), approximate]) + '.pt'
        create_gelu_model(model_path, approximate)
        exit_status = run_parser(model_path, feed_dict, unexpected_keywords=['layer_type=Eltwise', 'float64'],
                                 expected_keywords=['layer_type=Activation', 'method=GELU', 'approximate=' + approximate.upper()])
        assert exit_status
