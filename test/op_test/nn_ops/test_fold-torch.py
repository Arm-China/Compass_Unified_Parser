# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class fold_model(nn.Module):
    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        super(fold_model, self).__init__()
        self.fold = nn.Fold(output_size=output_size, kernel_size=kernel_size,
                            dilation=dilation, padding=padding, stride=stride)

    def forward(self, x):
        return self.fold(x)


def create_fold_model(model_path, output_size, kernel_size, dilation=1, padding=0, stride=1):
    try:
        model = fold_model(output_size, kernel_size, dilation=1, padding=0, stride=1)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'fold'

for input_shape in ([2, 75, 121], [75, 121]):
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.pt'
    # prepare model and input datas
    x_data = np.random.ranf(input_shape).astype(np.float32)
    feed_dict = {'x': x_data}
    create_fold_model(model_path, [15, 15], [5, 5])
    # FIXME: Enable verify after opt supports Col2Im op
    exit_status = run_parser(model_path, feed_dict, verify=False)
    assert exit_status
