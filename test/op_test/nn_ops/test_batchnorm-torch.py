# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class batchnorm_model(nn.Module):
    def __init__(self):
        super(batchnorm_model, self).__init__()
        self.batchnorm = torch.nn.BatchNorm1d(16)

    def forward(self, x):
        return self.batchnorm(x)


def create_batchnorm_model(model_path):
    try:
        model = batchnorm_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'batchnorm'
model_path = TEST_NAME + '.pt'

for input_shape in ([2, 16], [3, 16, 2], ):
    # prepare model and input datas
    x_data = np.random.ranf(input_shape).astype(np.float32)
    feed_dict = {'x': x_data}
    np.save('input', feed_dict)
    create_batchnorm_model(model_path)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
