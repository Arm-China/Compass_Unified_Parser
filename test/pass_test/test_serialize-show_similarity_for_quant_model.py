# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import os
import torch
import torch.nn as nn
import numpy as np

from utils.run import generate_ir


class conv_model(torch.nn.Module):
    def __init__(self):
        super(conv_model, self).__init__()
        self.qconv = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        weight = torch.quantize_per_tensor(torch.randn(33, 16, 3, 5), 0.5, 0, torch.qint8)
        bias = torch.arange(33).to(torch.float) - 16
        self.qconv.set_weight_bias(weight, bias)

    def forward(self, x):
        q_input = torch.quantize_per_tensor(x, scale=0.5, zero_point=128, dtype=torch.quint8)
        return self.qconv(q_input)


def create_torch_model(model_path):
    try:
        model = conv_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'quant_similarity'
input_shape = [3, 16, 20, 30]
# Generate input data
feed_dict = dict()
feed_dict['X'] = (np.random.ranf(input_shape)).astype(np.float32)
np.save('input', feed_dict)

output_dir = './output_dir'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_name = TEST_NAME
cfg_path = model_name + '.cfg'
model_path = model_name + '.pt'
# Create model
create_torch_model(model_path)
# Generate cfg
cfg_content = '''[Common]
model_type = torch
model_name = similarity
input_model = {0}
input = X
input_shape = {1}
output_dir = ./output_dir
similarity_input_npy = input.npy
'''.format(model_path, str(input_shape))
with open(cfg_path, 'w') as txt_file:
    txt_file.write(cfg_content)
# Run tests with parser and compare result with runtime
exit_status = generate_ir(cfg_path, verbose=True)
# Need to run in compiled AIPUBuilder package env or put compiled folders/files under AIPUBuilder folder
# assert exit_status
