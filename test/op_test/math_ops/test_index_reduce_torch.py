import torch
import numpy as np
from utils.run import run_parser


class reduce_model(torch.nn.Module):
    def __init__(self, reduce='prod', include_self=False, index=torch.tensor([0, 1])):
        super(reduce_model, self).__init__()
        self.reduce = reduce
        self.include_self = include_self
        self.index = index

    def forward(self, x):
        index = self.index
        t = torch.arange(start=0, end=672, step=1,
                         dtype=torch.float).resize(2, 6, 7, 8)

        return x.index_reduce_(0, index, t, self.reduce, include_self=self.include_self)


def create_index_reduce_model(model_path, reduce, include_self, index):
    try:
        model = reduce_model(reduce, include_self, index)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'index_reduce'

x_data = np.random.randint(1, 2, [5, 6, 7, 8]).astype(np.float32)
feed_dict = {'x': x_data}
index_data = (torch.tensor([0, 0]), torch.tensor([0, 1]))
for index in index_data:
    for include_self in (False, True):
        for reduce in ('prod', 'mean',):  # 'amax''amin'
            if reduce == 'mean':
                continue
            model_path = '-'.join([TEST_NAME, str(reduce),
                                   str(include_self)]) + '.pt'
            create_index_reduce_model(model_path, reduce, include_self, index)
            exit_status = run_parser(model_path, feed_dict)
            assert exit_status
