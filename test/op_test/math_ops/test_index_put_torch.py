import torch
import numpy as np
from utils.run import run_parser


class put_model(torch.nn.Module):
    def __init__(self, accumulate=False, index_len=1, update_rank=0):
        super(put_model, self).__init__()
        self.accumulate = accumulate
        if update_rank == 0:
            self.update = torch.randn([]) * 20
        if index_len == 1:
            self.index = (torch.tensor([6, 2, 5, 5, ]), )
            if update_rank == 1:
                self.update = torch.randn([8]) * 20
            elif update_rank == 2:
                self.update = torch.randn([4, 8]) * 30
        else:
            self.index = (torch.tensor([[1, 0, 3, 0, 2, 3, 4, 4]]),
                          torch.tensor([2]),)
            if update_rank == 1:
                self.update = torch.randn([1]) * 20
            elif update_rank == 2:
                self.update = torch.randn([8]) * 20

    def forward(self, x):
        return x.index_put_(self.index, self.update, accumulate=self.accumulate)


def create_index_put_model(model_path, accumulate, index_len, update_rank):
    try:
        model = put_model(accumulate, index_len, update_rank)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'index_put'

x_data = np.random.ranf([7, 8]).astype(np.float32)
feed_dict = {'x': x_data}

for accumulate in (False, True):
    for index_len in (2, 1):
        for update_rank in (2, 0, 1):
            model_path = '-'.join([TEST_NAME, str(accumulate), str(index_len), str(update_rank)]) + '.pt'
            create_index_put_model(model_path, accumulate, index_len, update_rank)
            exit_status = run_parser(model_path, feed_dict)
            assert exit_status
