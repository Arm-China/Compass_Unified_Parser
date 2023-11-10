import torch
import numpy as np
from utils.run import run_parser


class hsplit_model_int(torch.nn.Module):
    def __init__(self, split_size):
        super().__init__()
        self.size = split_size

    def forward(self, x):
        out = torch.hsplit(x, sections=self.size)
        return out


class hsplit_model_tuple(torch.nn.Module):
    def __init__(self, split_size):
        super().__init__()
        self.size = split_size

    def forward(self, x):
        out = torch.hsplit(x, indices=self.size)
        return out


def create_hsplit_model(model_path, split_size):
    try:
        if isinstance(split_size, int):
            model = hsplit_model_int(split_size)
        else:
            model = hsplit_model_tuple(split_size)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'hsplit'
input_shapes = [[40], [2, 40]]

for size in [5, (2, 3, 4)]:
    for inp_shape in input_shapes:
        input_data = np.random.ranf(inp_shape).astype(np.float32)
        feed_dict = {'x': input_data}
        model_path = '-'.join([TEST_NAME, str(len(inp_shape)), str(size)]) + '.pt'
        # prepare model and input datas
        create_hsplit_model(model_path, size)
        exit_status = run_parser(model_path, feed_dict)
        assert exit_status
