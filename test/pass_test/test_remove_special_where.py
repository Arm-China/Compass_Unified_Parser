# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import torch
import numpy as np
from utils.run import run_parser


class where_model(torch.nn.Module):
    def __init__(self, cond_is_true, cond_shape):
        super(where_model, self).__init__()
        if cond_is_true:
            self.condition = torch.tensor(True).expand(cond_shape)
        else:
            self.condition = torch.tensor(False).expand(cond_shape)

    def forward(self, x1, x2):
        return torch.where(self.condition, x1, x2)


def create_where_model(model_path, cond_is_true, cond_shape):
    try:
        model = where_model(cond_is_true, cond_shape)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'where'
input1_shapes = [[20, 20, 30], [1, 2, 3], [10, 20]]
input2_shapes = [[20, 20, 30], [1], [1, 20]]

# Generate input data
feed_dict = dict()

for idx, (input1_shape, input2_shape) in enumerate(zip(input1_shapes, input2_shapes)):
    feed_dict['x1'] = (np.random.ranf(input1_shape) * 100).astype(np.float32)
    feed_dict['x2'] = (np.random.ranf(input2_shape) * 10).astype(np.float32)
    for cond_is_true in (True, False):
        for cond_shape in ([], [1], [1, 1] + input1_shape, [5] + input1_shape):
            model_name = '-'.join([TEST_NAME, str(idx), str(cond_is_true), str(len(cond_shape))])
            model_path = model_name + '.pt'
            # Create model
            create_where_model(model_path, cond_is_true, cond_shape)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(model_path, feed_dict, unexpected_keywords=['layer_type=Where'])
            assert exit_status
