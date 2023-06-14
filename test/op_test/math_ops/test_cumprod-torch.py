import torch
import numpy as np
from utils.run import run_parser


class cumprod_model(torch.nn.Module):
    def __init__(self, dim, dtype):
        super(cumprod_model, self).__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, x):
        y = torch.cumprod(x, self.dim, dtype=self.dtype)
        return y


def create_cumprod_model(model_path, dim, dtype=None):
    try:
        model = cumprod_model(dim, dtype)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'cumprod'
input_shape = [3, 4, 5]
x_data = np.random.ranf(input_shape).astype(np.float32)
feed_dict = {'x': x_data}

for dim in (0, 1, -1):
    for dtype in (None, torch.int32, torch.uint8):
        model_path = '-'.join([TEST_NAME, str(dim), str(dtype)]) + '.pt'
        create_cumprod_model(model_path, dim, dtype)
        exit_status = run_parser(model_path, feed_dict)
        assert exit_status
