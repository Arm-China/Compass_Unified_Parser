# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import numpy as np
import torch
import torchvision

from utils.run import run_parser


class deform_conv_model(torch.nn.Module):
    def __init__(self, weight_shape):
        super().__init__()
        self.weight = torch.randn(weight_shape, dtype=torch.float32)
        self.bias = torch.randn(weight_shape[0], dtype=torch.float32)

    def forward(self, x, offset, mask):
        y = torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, stride=2, padding=1, mask=mask)
        return y


def create_deform_conv_model(model_path, inputs, weight_shape):
    try:
        model = deform_conv_model(weight_shape)
        # model_scripted = torch.jit.script(model)
        # model_scripted.save(model_path)
        traced = torch.jit.trace(model, inputs)
        traced.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'deform_conv'
input_shapes = [[2, 6, 51, 52], ]
output_height, output_width = 23, 24
kernel_height, kernel_width = 8, 7
output_channels = 12

# prepare model and input datas
for input_shape in input_shapes:
    x_data = np.random.ranf(input_shape).astype(np.float32)
    for group, offset_group in zip([1, 1, 3, 2], [1, 2, 6, 1]):
        offset_shape = [input_shape[0], int(2*offset_group*kernel_height*kernel_width), output_height, output_width]
        offset = np.random.randint(-4, 10, offset_shape).astype(np.float32)
        mask_shape = [input_shape[0], int(offset_group*kernel_height*kernel_width), output_height, output_width]
        mask = np.random.randint(-3, 3, mask_shape).astype(np.float32)
        feed_dict = {'x': x_data, 'offset': offset, 'mask': mask}
        inputs = ()
        for data in feed_dict.values():
            inputs += (torch.tensor(data), )
        assert input_shape[1] % group == 0, 'input shape %s is invalid' % str(input_shape)
        model_path = '-'.join([TEST_NAME, str(group), str(offset_group)]) + '.pt'
        weight_shape = [output_channels, int(input_shape[1]//group), kernel_height, kernel_width]
        create_deform_conv_model(model_path, inputs, weight_shape)
        # create_deform_conv_model(model_path)
        exit_status = run_parser(model_path, feed_dict)
        assert exit_status
