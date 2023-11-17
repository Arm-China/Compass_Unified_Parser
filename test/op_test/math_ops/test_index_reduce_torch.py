import torch
import numpy as np
from utils.run import run_parser


class reduce_model(torch.nn.Module):
    def __init__(self, reduce, include_self):
        super(reduce_model, self).__init__()
        self.reduce = reduce
        self.include_self = include_self

    def forward(self, x, t, index):
        return x.index_reduce_(-3, index, t, self.reduce, include_self=self.include_self)


def create_index_reduce_model(model_path, reduce, include_self):
    try:
        model = reduce_model(reduce, include_self)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'index_reduce'

x_data = np.random.ranf([10, 6, 8]).astype(np.float32)

index_data = (np.array([0, 0, 1, 2]), np.array([0, 2, 1, 3]))
t = torch.arange(start=0, end=192, step=1,
                 dtype=torch.float).resize(4, 6, 8).numpy()
feed_dict = {'x': x_data, 't': t}


for index in index_data:
    feed_dict.update({'index': index})
    for include_self in (False, True):
        for reduce in ('prod', 'mean', 'amin', 'amin', ):
            model_path = '-'.join([TEST_NAME, str(reduce),
                                   str(include_self)]) + '.pt'
            create_index_reduce_model(
                model_path, reduce, include_self)
            exit_status = run_parser(model_path, feed_dict)
            assert exit_status
