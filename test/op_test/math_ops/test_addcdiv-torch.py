# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import numpy as np
import pytest
import os
from utils.run import run_parser


class addcdiv_model_default(torch.nn.Module):
    def forward(self, x, y, z):
        out = torch.addcdiv(x, y, z)
        return out


class addcdiv_model_value(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, y, z):
        out = torch.addcdiv(x, y, z, value=self.value)
        return out


def create_addcdiv_model(model_path, value):
    try:
        if value == 'default':
            model = addcdiv_model_default()
        else:
            model = addcdiv_model_value(value)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'addcdiv'


@pytest.mark.parametrize('input_shapes',
                         [
                             [[], [], []],
                             [[2, 3, 4], [4], [2, 1, 1]]
                         ])
@pytest.mark.parametrize('value',
                         [
                             'default',
                             2.1,
                             -3.2
                         ])
def test_addcdiv(input_shapes, value):
    feed_dict = {}
    for i, shape in enumerate(input_shapes):
        rand_v = np.random.ranf(shape).astype(np.float32)
        feed_dict[f'input_{i}'] = rand_v
    model_path = '-'.join([TEST_NAME, str(value)]) + '.pt'
    create_addcdiv_model(model_path, value)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status


if __name__ == '__main__':
    pytest.main([__file__])
