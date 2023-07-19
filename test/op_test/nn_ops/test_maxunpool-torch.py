import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class maxunpool_model(nn.Module):
    def forward(self, x, indices):
        return torch.nn.functional.max_unpool2d(x, indices, [2, 2], output_size=[20, 16, 49, 61])


def create_maxunpool2d_model(model_path):
    try:
        model = maxunpool_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'maxunpool2d'
model_path = TEST_NAME + '.pt'

# prepare model and input datas
x_data = np.random.ranf([20, 16, 24, 31]).astype(np.float32)
indices_data = np.tile(np.arange(31), [20, 16, 24, 1]).astype(np.int64)
feed_dict = {'x': x_data, 'indices': indices_data}
create_maxunpool2d_model(model_path)
exit_status = run_parser(model_path, feed_dict, verify=True)
assert exit_status
