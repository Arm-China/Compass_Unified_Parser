# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import numpy as np
from utils.run import run_parser


class einsum_model(torch.nn.Module):
    def __init__(self, equation):
        super(einsum_model, self).__init__()
        self.equation = equation

    def forward(self, x1, x2):
        y = torch.einsum(self.equation, x1, x2)
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
        'nijkl,imkln->ikjnm', 'ijknl,niljm->nikjm',
        'ijkl,imkl->ikjm', 'ijkl,iljm->ikjm',
        'mij,jk->mik', 'ij,mkj->mik', 'mij,mjk->mik', 'mij,mkj->mik',
        'l i j, l j k -> l i k',
        'l m n i k, l m n j k -> l m n i j',
        'l m n i j, l m n j k -> l m n i k',
        'ij,jk->ik', 'ij,kj->ik',
        'ijkl,ilm->ijkm', '...i,ji->...ji', 'ijkl,ijkm->ilm',):
    if '...' in equation:
        new_equation = equation.replace('...', 'ab')
    else:
        new_equation = equation
    input1, input2 = new_equation.split('->')[0].split(',')
    input1_shape = [shape_dict[s] for s in input1 if s in shape_dict]
    input2_shape = [shape_dict[s] for s in input2 if s in shape_dict]
    x1_data = np.random.ranf(input1_shape)
    x2_data = np.random.ranf(input2_shape) * 100
    feed_dict = {'x1': x1_data, 'x2': x2_data}
    np.save('input', feed_dict)

    create_einsum_model(model_path, equation)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
