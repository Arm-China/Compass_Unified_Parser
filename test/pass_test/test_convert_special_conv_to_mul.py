# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class conv_model(nn.Module):
    def __init__(self, dim, set_bias, groups):
        super(conv_model, self).__init__()
        self.set_bias = set_bias
        self.groups = groups
        if dim == 1:
            self.func = torch.nn.functional.conv1d
        elif dim == 2:
            self.func = torch.nn.functional.conv2d
        else:
            self.func = torch.nn.functional.conv3d

    def forward(self, x, weight, bias):
        if self.set_bias:
            return self.func(x, weight, bias, groups=self.groups)
        return self.func(x, weight, groups=self.groups)


def create_conv2d_model(model_path, dim, set_bias, groups):
    try:
        model = conv_model(dim, set_bias, groups)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'conv'
input_shapes = [[16, 16, 50, 60], [16, 16, 50, 60], [1, 16, 50, 60], [1, 16, 50, 60],
                [16, 16, 50], [16, 16, 50], [1, 16, 50], [1, 16, 50],
                [16, 16, 10, 12, 15], [16, 16, 10, 12, 15], [1, 16, 10, 12, 15], [1, 16, 10, 12, 15]]
weights_shapes = [[16, 1, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 1, 1, 1],
                  [16, 16, 1], [16, 1, 1], [16, 16, 1], [16, 1, 1],
                  [16, 16, 1, 1, 1], [16, 1, 1, 1, 1], [16, 16, 1, 1, 1], [16, 1, 1, 1, 1]]

feed_dict = {}
for input_shape, weight_shape in zip(input_shapes, weights_shapes):
    # prepare model and input datas
    feed_dict.clear()
    x_data = np.random.ranf(input_shape).astype(np.float32)
    w_data = np.random.ranf(weight_shape).astype(np.float32)
    b_data = np.random.ranf(weight_shape[0]).astype(np.float32)
    feed_dict = {'x': x_data, 'w': w_data, 'b': b_data}
    dim = len(input_shape) - 2
    groups = input_shape[1] // weight_shape[1]
    for set_bias in (False, True):
        model_path = '-'.join([TEST_NAME, str(dim), str(groups), str(set_bias)]) + '.pt'
        create_conv2d_model(model_path, dim, set_bias, groups)
        exit_status = run_parser(model_path, feed_dict)
        assert exit_status
