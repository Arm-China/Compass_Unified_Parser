# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class qdeconv_model(nn.Module):
    def __init__(self):
        super(qdeconv_model, self).__init__()
        self.qdeconv = torch.ao.nn.quantized.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        weight = torch.quantize_per_tensor(torch.randn(16, 33, 3, 5), 0.5, 0, torch.qint8)
        bias = torch.arange(33).to(torch.float) - 16
        self.qdeconv.set_weight_bias(weight, bias)

    def forward(self, x):
        q_input = torch.quantize_per_tensor(x, scale=0.5, zero_point=128, dtype=torch.quint8)
        return self.qdeconv(q_input)


def create_qdeconv_model(model_path):
    try:
        model = qdeconv_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'qdeconv'

if str(torch.onnx.producer_version) >= '2.1.0':  # quantized conv_transpose cannot be converted until torch 2.1
    for idx, input_shape in enumerate([[2, 16, 50, 60], [3, 16, 20, 30], ]):
        model_path = '-'.join([TEST_NAME, str(idx)]) + '.pt'
        # prepare model and input datas
        x_data = np.random.ranf(input_shape).astype(np.float32)
        feed_dict = {'x': x_data}
        # np.save('input', feed_dict)
        create_qdeconv_model(model_path)
        # need qtlib to generate ir before opt forward
        exit_status = run_parser(model_path, feed_dict, verify=False)
        assert exit_status
