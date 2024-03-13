# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import numpy as np
from utils.run import run_parser


class trunc_model(torch.nn.Module):
    def __init__(self):
        super(trunc_model, self).__init__()

    def forward(self, x):
        return torch.flatten(torch.trunc(x), 1, -1)


def create_trunc_model(model_path):
    try:
        model = trunc_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'trunc'

for input_shape in ([3, 4, 5, 6], [2, 5, 3]):
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.pt'
    # prepare model and input datas
    create_trunc_model(model_path)
    input_data = np.random.ranf(input_shape).astype(np.float32) * 100
    feed_dict = {'x': input_data}
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
