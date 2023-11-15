import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class gru_model(nn.Module):
    def __init__(self):
        super(gru_model, self).__init__()
        self.rnn = nn.GRU(10, 20, 2, bias=True, bidirectional=True)

    def forward(self, x1, x2):
        return self.rnn(x1, x2)


def create_torch_model(model_path):
    try:
        model = gru_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'gru'
model_path = TEST_NAME + '.pt'

# prepare model and input datas
create_torch_model(model_path)
x1_data = np.random.ranf([5, 3, 10]).astype(np.float32)
x2_data = np.random.ranf([4, 3, 20]).astype(np.float32)
feed_dict = {'x1': x1_data, 'x2': x2_data}
exit_status = run_parser(model_path, feed_dict)
assert exit_status
