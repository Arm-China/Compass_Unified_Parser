# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class quantize_bn_model(nn.Module):
    def __init__(self, num_features):
        super(quantize_bn_model, self).__init__()
        self.quantize_bn = nn.quantized.BatchNorm2d(num_features)
        # weights of quantized batchnorm cannot be quantized type(the following line causes error):
        # self.quantize_bn.weight = torch.quantize_per_tensor((torch.randn(num_features)), 0.5, 0, torch.quint8)
        self.quantize_bn.weight = nn.Parameter(torch.randn(num_features))
        self.quantize_bn.bias = nn.Parameter(torch.randn(num_features))
        self.quantize_bn.scale = torch.tensor(0.5, dtype=torch.float32)
        self.quantize_bn.zero_point = torch.tensor(10, dtype=torch.int8)

    def forward(self, x):
        q_input = torch.quantize_per_tensor(x, scale=0.5, zero_point=12, dtype=torch.quint8)
        return self.quantize_bn(q_input)  # output dtype is quint8(although zp dtype is int8)


def create_quantize_bn_model(model_path, dim):
    try:
        model = quantize_bn_model(dim)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'quantize_bn'

for idx, input_shape in enumerate([[3, 16, 20, 30], [3, 15, 45, 46]]):
    model_path = '-'.join([TEST_NAME, str(idx)]) + '.pt'
    # prepare model and input datas
    x_data = np.random.randint(-64, 64, input_shape).astype(np.float32)
    feed_dict = {'x': x_data}
    np.save('input', feed_dict)
    create_quantize_bn_model(model_path, input_shape[1])
    # need qtlib to generate ir before opt forward
    exit_status = run_parser(model_path, feed_dict, verify=False,
                             expected_keywords=['quantize_zp_type=uint8'], unexpected_keywords=['quantize_zp_type=int8'])
    assert exit_status
