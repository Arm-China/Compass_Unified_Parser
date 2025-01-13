# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class conv_model(nn.Module):
    def __init__(self, padding):
        super(conv_model, self).__init__()
        self.conv = nn.Conv3d(16, 33, (3, 4, 5), stride=(1, 1, 1), padding=padding, dilation=(3, 1, 1))

    def forward(self, x):
        return self.conv(x)


def create_conv3d_model(model_path, padding):
    try:
        model = conv_model(padding)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'conv3d'
model_path = TEST_NAME + '.pt'

# prepare model and input datas
x_data = np.random.ranf([20, 16, 50, 70, 100]).astype(np.float32)
feed_dict = {'x': x_data}
for padding in (['same', 'valid', (4, 2, 2)]):
    create_conv3d_model(model_path, padding)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
