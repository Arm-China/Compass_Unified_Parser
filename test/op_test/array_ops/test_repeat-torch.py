# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import torch
import numpy as np
from utils.run import run_parser


class repeat_model(torch.nn.Module):
    def __init__(self, repeats, dim):
        super(repeat_model, self).__init__()
        self.repeats = repeats
        self.dim = dim

    def forward(self, x):
        return torch.repeat_interleave(x, self.repeats, self.dim)


def create_repeat_model(model_path, repeats, dim):
    try:
        model = repeat_model(repeats, dim)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'repeat'

for input_shape in ([2, 3, 5], [4, 3], ):
    input_data = np.random.randint(-100, 100, input_shape)
    feed_dict = {'x': input_data}
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.pt'
    # test repeats
    for repeats in (torch.tensor([3, 2, 3]), 2):
        create_repeat_model(model_path, repeats, dim=1)
        exit_status = run_parser(model_path, feed_dict)
        assert exit_status
    # test dim=None
    create_repeat_model(model_path, 3, dim=None)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
