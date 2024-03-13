# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import numpy as np
from utils.run import run_parser


class meshgrid_model(torch.nn.Module):
    def __init__(self, indexing):
        super(meshgrid_model, self).__init__()
        self.indexing = indexing

    def forward(self, x1, x2):
        y = torch.meshgrid(x1, x2, indexing=self.indexing)
        return torch.concat(y, dim=-1)
        # return y  # the output node will be SequenceConstruct


def create_meshgrid_model(model_path, indexing):
    try:
        model = meshgrid_model(indexing)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'meshgrid'
feed_dict = {}
# opt only supports 2 inputs for now
for idx, input_shape in enumerate([[4], [], ]):  # [[4]], [[4], [5]], [[4], [5], [3]]
    input_data = np.random.ranf(input_shape).astype(np.float32)
    feed_dict.update({'x' + str(idx): input_data})

for indexing in ('ij', 'xy', ):
    model_path = '-'.join([TEST_NAME, indexing]) + '.pt'
    # prepare model and input datas
    create_meshgrid_model(model_path, indexing)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
