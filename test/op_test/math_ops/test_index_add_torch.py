import torch
import numpy as np
from utils.run import run_parser


class add_model(torch.nn.Module):
    def __init__(self, add_alpha=1):
        super(add_model, self).__init__()
        self.add_alpha = add_alpha

    def forward(self, x):
        index = torch.tensor([0, 2, 1, 3, 4, 5, 6])
        t = torch.arange(start=0, end=336, step=1,
                         dtype=torch.float).resize(7, 6, 8)
        return x.index_add_(-3, index, t, alpha=self.add_alpha)


def create_index_add_model(model_path, alpha):
    try:
        model = add_model(alpha)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'index_add'

x_data = np.random.randint(0, 1, [10, 6, 8]).astype(np.float32)
feed_dict = {'x': x_data}

for alpha in (1, -1):
    model_path = '-'.join([TEST_NAME, str(alpha)]) + '.pt'
    create_index_add_model(model_path, alpha)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
