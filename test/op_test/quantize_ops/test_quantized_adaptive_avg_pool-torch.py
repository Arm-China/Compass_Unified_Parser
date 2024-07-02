# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class quantize_adaptive_avg_pool_model(nn.Module):
    def __init__(self, output_size):
        super(quantize_adaptive_avg_pool_model, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        q_input = torch.quantize_per_tensor(x, scale=0.5, zero_point=12, dtype=torch.qint32)
        return torch.ao.nn.quantized.functional.adaptive_avg_pool2d(q_input, output_size=self.output_size)


def create_quantize_adaptive_avg_pool_model(model_path, output_size):
    try:
        model = quantize_adaptive_avg_pool_model(output_size)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'quantize_adaptive_avg_pool'
output_size = [34, 41]

for idx, input_shape in enumerate([[3, 16, 20, 30], [3, 15, 45, 46]]):
    model_path = '-'.join([TEST_NAME, str(idx)]) + '.pt'
    # prepare model and input datas
    x_data = np.random.randint(-64, 64, input_shape).astype(np.float32)
    feed_dict = {'x': x_data}
    np.save('input', feed_dict)
    create_quantize_adaptive_avg_pool_model(model_path, output_size)
    # need qtlib to generate ir before opt forward
    exit_status = run_parser(model_path, feed_dict, verify=False,
                             expected_keywords=['quantize_zp_type=int32'],
                             unexpected_keywords=['quantize_zp_type=int8', 'layer_type=Cast'])
    assert exit_status
