import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class maxpool_model(nn.Module):
    def __init__(self):
        super(maxpool_model, self).__init__()
        self.maxpool = nn.MaxPool2d((3, 2), stride=(2, 1), return_indices=True)

    def forward(self, x):
        return self.maxpool(x)


def create_maxpool2d_model(model_path):
    try:
        model = maxpool_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'maxpool2d'
model_path = TEST_NAME + '.pt'

# prepare model and input datas
x_data = np.random.ranf([20, 16, 50, 32]).astype(np.float32)
feed_dict = {'x': x_data}
create_maxpool2d_model(model_path)
exit_status = run_parser(model_path, feed_dict, verify=True)
assert exit_status
