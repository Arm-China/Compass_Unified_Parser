import torch
import numpy as np
from utils.run import run_parser


class fill_model(torch.nn.Module):
    def __init__(self):
        super(fill_model, self).__init__()

    def forward(self, x):
        index = torch.tensor([0, 2, 1], dtype=torch.int64)
        fill_value = torch.tensor(-1, dtype=torch.float32)
        return x.index_fill_(-2, index, fill_value)


def create_index_fill_model(model_path):
    try:
        model = fill_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'index_fill'
x_data = np.random.randint(0, 1, [7, 8]).astype(np.int64)
feed_dict = {'x': x_data}

model_path = '-'.join([TEST_NAME]) + '.pt'
create_index_fill_model(model_path)
exit_status = run_parser(model_path, feed_dict)
assert exit_status
