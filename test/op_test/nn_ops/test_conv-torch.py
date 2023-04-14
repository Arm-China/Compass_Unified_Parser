import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class conv_model(nn.Module):
    def __init__(self):
        super(conv_model, self).__init__()
        self.conv = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))

    def forward(self, x):
        return self.conv(x)


def create_conv2d_model(model_path):
    try:
        model = conv_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'conv'
model_path = TEST_NAME + '.pt'

# prepare model and input datas
create_conv2d_model(model_path)
x_data = np.random.ranf([20, 16, 50, 100]).astype(np.float32)
feed_dict = {'x': x_data}
exit_status = run_parser(model_path, feed_dict)
assert exit_status
