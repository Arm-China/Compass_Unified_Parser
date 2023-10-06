import torch
import numpy as np
from utils.run import run_parser


class put_model(torch.nn.Module):
    def __init__(self, accumulate=False):
        super(put_model, self).__init__()
        self.accumulate = accumulate

    def forward(self, x):
        index = torch.tensor([1, 0, 3, 2, 4, 6, 5, ])
        t = torch.arange(start=0, end=48, step=1,
                         dtype=torch.float).resize(6, 8)

        return x.index_put_([index], t, accumulate=self.accumulate)


def create_index_put_model(model_path, accumulate):
    try:
        model = put_model(accumulate)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'index_put'

x_data = np.random.randint(0, 1, [10, 6, 8]).astype(np.float32)
feed_dict = {'x': x_data}

for accumulate in (False, True):
    model_path = '-'.join([TEST_NAME, str(accumulate)]) + '.pt'
    create_index_put_model(model_path, accumulate)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
