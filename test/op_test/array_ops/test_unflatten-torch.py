# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import torch
import numpy as np
from utils.run import run_parser


class unflatten_model(torch.nn.Module):
    def __init__(self, dim, sizes):
        super(unflatten_model, self).__init__()
        self.dim = dim
        self.sizes = sizes

    def forward(self, x):
        return torch.unflatten(x, self.dim, self.sizes)


def create_unflatten_model(model_path, dim, sizes):
    try:
        model = unflatten_model(dim, sizes)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'unflatten'
input_shapes = [[6], [6, 4], [3, 15, 3]]
sizes = [[2, 3], [2, -1], [-1, 3]]
for input_shape, size in zip(input_shapes, sizes):
    input_data = np.random.ranf(input_shape).astype(np.float32)
    feed_dict = {'x': input_data}
    for dim in (0, 1, -1):
        if dim > 0 and len(input_shape) >= dim:
            continue
        model_path = '-'.join([TEST_NAME, str(len(input_shape)), str(dim)]) + '.pt'
        # prepare model and input datas
        create_unflatten_model(model_path, dim, size)
        exit_status = run_parser(model_path, feed_dict)
        assert exit_status
