# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import numpy as np
from utils.run import run_parser


class channel_shuffle_model(torch.nn.Module):
    def __init__(self, groups):
        super(channel_shuffle_model, self).__init__()
        self.channel_shuffle = torch.nn.ChannelShuffle(groups)

    def forward(self, x):
        y = self.channel_shuffle(x)
        return torch.flatten(y, 1, -1)


def create_channel_shuffle_model(model_path, groups):
    try:
        model = channel_shuffle_model(groups)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'channel_shuffle'
input_shape = [3, 14, 4, 5]
x_data = np.random.ranf(input_shape).astype(np.float32)
feed_dict = {'x': x_data}

for groups in (1, 2, 7):
    model_path = '-'.join([TEST_NAME, str(groups)]) + '.pt'
    create_channel_shuffle_model(model_path, groups)
    unexpected_keywords = []
    if groups == 1:
        unexpected_keywords = ['ChannelShuffle']
    exit_status = run_parser(model_path, feed_dict, unexpected_keywords=unexpected_keywords)
    assert exit_status
