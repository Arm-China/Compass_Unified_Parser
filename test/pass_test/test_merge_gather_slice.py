# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import numpy as np
from utils.run import run_parser


class gather_slice_model(torch.nn.Module):
    def __init__(self, axis):
        super(gather_slice_model, self).__init__()
        self.axis = axis

    def forward(self, x1, x2):
        offset = x2[0, :]
        if self.axis == 0:
            y = x1[offset: offset + 1]
        elif self.axis == 1:
            y = x1[:, offset: offset + 1]
        else:  # self.axis == -1
            y = x1[..., offset: offset + 1]
        return y


def create_gather_slice_model(model_path, inputs, axis):
    try:
        model = gather_slice_model(axis)
        model_traced = torch.jit.trace(model, inputs)
        model_traced.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'gather_slice'

for input_shape, indice_shape in zip([[10, 40, 50]], [[1, 1]], ):
    # prepare model and input datas
    input_data = np.random.ranf(input_shape).astype(np.float32)
    indice = np.random.randint(0, 10, indice_shape)
    inputs = (torch.tensor(input_data), torch.tensor(indice), )
    for axis in (0, 1, -1, ):
        model_path = '-'.join([TEST_NAME, str(len(input_shape)), str(axis)]) + '.pt'
        create_gather_slice_model(model_path, inputs, axis)
        feed_dict = {'x1': input_data, 'x2': indice}
        exit_status = run_parser(model_path, feed_dict)
        assert exit_status
