# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import torch
import numpy as np
from utils.run import run_parser


class transformer_model(torch.nn.Module):
    def __init__(self, activation, batch_first, norm_first, bias):
        super(transformer_model, self).__init__()
        # dropout won't work if model.eval() is called(inference mode).
        self.transformer = torch.nn.Transformer(nhead=1, num_encoder_layers=1,
                                                activation=activation, batch_first=batch_first,
                                                norm_first=norm_first, bias=bias, dropout=1.)

    def forward(self, src, tgt):
        return self.transformer(src, tgt)


def create_transformer_model(model_path, inputs, activation, batch_first, norm_first, bias):
    try:
        model = transformer_model(activation, batch_first, norm_first, bias)
        model.eval()
        model_traced = torch.jit.trace(model, inputs)
        model_traced.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'transformer'
src_shape = [10, 32, 512]
target_shape = [20, 32, 512]

for batch_first in (True, False):
    src_data = np.random.ranf(src_shape).astype(np.float32)
    target_data = np.random.ranf(target_shape).astype(np.float32)
    if batch_first:
        src_data = np.transpose(src_data, [1, 0, 2]) * 10
        target_data = np.transpose(target_data, [1, 0, 2]) * 20
    feed_dict = {'src': src_data, 'tgt': target_data}
    np.save('input', feed_dict)
    inputs = ()
    for data in feed_dict.values():
        inputs += (torch.tensor(data), )
    for activation in ('relu', 'gelu'):
        for norm_first in (True, False):
            for bias in (False, True):
                model_path = '-'.join([TEST_NAME, activation, str(batch_first), str(norm_first), str(bias)]) + '.pt'
                # prepare model and input datas
                create_transformer_model(model_path, inputs, activation, batch_first, norm_first, bias)
                exit_status = run_parser(model_path, feed_dict)
                assert exit_status
