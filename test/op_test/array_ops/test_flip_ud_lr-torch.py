# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import numpy as np
from utils.run import run_parser


class fliplr_model(torch.nn.Module):
    def forward(self, x):
        out = torch.fliplr(x)
        return out


class flipud_model(torch.nn.Module):
    def forward(self, x):
        out = torch.flipud(x)
        return out


def create_flip_model(model_path, name):
    try:
        if name == 'fliplr':
            model = fliplr_model()
        else:
            model = flipud_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAMES = ['fliplr', 'flipud']
input_shapes = [[2, 3, 4]]

for t_name in TEST_NAMES:
    input_data = np.random.ranf(input_shapes[0]).astype(np.float32)
    feed_dict = {'x': input_data}
    model_path = '-'.join([t_name, str(len(input_shapes[0]))]) + '.pt'
    # prepare model and input datas
    create_flip_model(model_path, t_name)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
