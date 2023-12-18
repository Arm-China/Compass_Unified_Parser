# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import torch
import numpy as np
from utils.run import run_parser


class logical_model(torch.nn.Module):
    def __init__(self):
        super(logical_model, self).__init__()

    def forward(self, x1, x2):
        and_out = torch.logical_and(x1, x2)
        return torch.logical_not(and_out)


def create_logical_model(model_path):
    try:
        model = logical_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'logical'

for input_shape in ([], [3, 4], ):
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.pt'
    # prepare model and input datas
    create_logical_model(model_path)
    input1_data = np.random.randint(-100, 100, input_shape)
    input2_data = np.random.randint(-100, 100, input_shape)
    for dtype in ('float32', 'int8', 'int32'):
        input1_data = input1_data.astype(dtype)
        input2_data = input2_data.astype(dtype)
        feed_dict = {'x1': input1_data, 'x2': input2_data}
        exit_status = run_parser(model_path, feed_dict, unexpected_keywords=['layer_type=Cast'])
        assert exit_status
