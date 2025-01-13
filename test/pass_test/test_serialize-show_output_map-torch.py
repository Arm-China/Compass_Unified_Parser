# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.
import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class argmax_model(nn.Module):
    def __init__(self):
        super(argmax_model, self).__init__()

    def forward(self, x1, x2):
        out = torch.argmax(x1) + torch.argmax(x2, dim=-1)
        return out


def create_argmax_model(model_path):
    try:
        model = argmax_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'argmax'
x1_shape = [2, 3, 4, 5, 6]
x2_shape = [6]

# prepare model and input datas
x1_data = np.random.ranf(x1_shape).astype(np.float32)
x2_data = np.random.ranf(x2_shape).astype(np.float32)
feed_dict = {'x1': x1_data, 'x2': x2_data}
# np.save('input', feed_dict)
model_path = TEST_NAME + '.pt'
create_argmax_model(model_path)
exit_status = run_parser(model_path, feed_dict, output_names=['4'],
                         expected_logs=['Output 4 from cfg is shown as tensor /Add_post_reshape_0_0 in IR'])
assert exit_status
