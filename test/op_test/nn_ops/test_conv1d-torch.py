import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class conv_model(nn.Module):
    def __init__(self, padding):
        super(conv_model, self).__init__()
        self.conv = nn.Conv1d(16, 33, 5, padding=padding, dilation=3)

    def forward(self, x):
        return self.conv(x)


def create_conv1d_model(model_path, padding):
    try:
        model = conv_model(padding)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'conv1d'
model_path = TEST_NAME + '.pt'

# prepare model and input datas
x_data = np.random.ranf([20, 16, 50]).astype(np.float32)
feed_dict = {'x': x_data}
for padding in (['same', 'valid', 4]):
    create_conv1d_model(model_path, padding)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
