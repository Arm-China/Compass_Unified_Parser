# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import numpy as np
from utils.run import run_parser


class flatten_model(torch.nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super(flatten_model, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, self.start_dim, self.end_dim)


def create_flatten_model(model_path, start_dim=0, end_dim=-1):
    try:
        model = flatten_model(start_dim, end_dim)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'flatten'

for input_shape in ([], [3], [3, 4], [2, 5, 3]):
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.pt'
    # prepare model and input datas
    create_flatten_model(model_path)
    input_data = np.random.ranf(input_shape).astype(np.float32)
    feed_dict = {'x': input_data}
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status

for input_shape in ([3, 2, 4, 5], ):
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.pt'
    # prepare model and input datas
    create_flatten_model(model_path, 1, -2)
    input_data = np.random.ranf(input_shape).astype(np.float32)
    feed_dict = {'x': input_data}
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
