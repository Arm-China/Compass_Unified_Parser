# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import torch
import numpy as np
from utils.run import run_parser


class matmul_model(torch.nn.Module):
    def __init__(self):
        super(matmul_model, self).__init__()
        self.other0 = torch.randn([256, 32000])
        self.other1 = torch.randn([256, 32000])

    def forward(self, x):
        input0, input1 = torch.split(x, [2, 1], dim=0)
        out0 = torch.matmul(input0, self.other0)
        out1 = torch.matmul(input1, self.other1)
        out2 = torch.matmul(input0, self.other1)
        out3 = torch.matmul(input1, self.other0)
        return (out0 * out1) + (out2 * out3)


def create_matmul_model(model_path):
    try:
        model = matmul_model()
        model.eval()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'matmul'
input_data = np.random.randint(-10, 20, [3, 256])
feed_dict = {'input': input_data.astype(np.float32)}
np.save('input', feed_dict)
# np.save('int_input', {'input': input_data.astype(np.int8)})

model_path = TEST_NAME + '.pt'
# prepare model and input datas
create_matmul_model(model_path)
exit_status = run_parser(model_path, feed_dict,
                         expected_keywords=['layer_type=FullyConnected'],
                         unexpected_keywords=['layer_type=MatMul'])
assert exit_status
