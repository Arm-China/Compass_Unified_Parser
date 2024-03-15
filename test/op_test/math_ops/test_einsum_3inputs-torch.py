# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import numpy as np
from utils.run import run_parser


class einsum_model(torch.nn.Module):
    def __init__(self, equation):
        super(einsum_model, self).__init__()
        self.equation = equation

    def forward(self, x1, x2, x3):
        y = torch.einsum(self.equation, x1, x2, x3)
        return y


def create_einsum_model(model_path, equation):
    try:
        model = einsum_model(equation)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'einsum'
model_path = TEST_NAME + '.pt'
shape_dict = {'i': 10, 'j': 12, 'k': 14, 'l': 15, 'm': 9, 'n': 8,
              'a': 4, 'b': 5}

for equation in (
        'ijk,ijl,ijk->ijkl', ):
    inputs = equation.split('->')[0].split(',')
    feed_dict = {}
    for idx, inp in enumerate(inputs):
        input_shape = [shape_dict[s] for s in inp if s in shape_dict]
        data = np.random.ranf(input_shape) * (idx + 1)
        feed_dict.update({('x' + str(idx)): data})
    # np.save('input', feed_dict)

    create_einsum_model(model_path, equation)
    exit_status = run_parser(model_path, feed_dict)
    # TODO: Enable checking after einsum with 3 inputs is supported
    # assert exit_status
