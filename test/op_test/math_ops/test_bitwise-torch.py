import torch
import numpy as np
from utils.run import run_parser


class bitwise_model(torch.nn.Module):
    def __init__(self):
        super(bitwise_model, self).__init__()

    def forward(self, x, y):
        x_sub = x - (x * y)
        y_add = y + 3 * x
        return torch.bitwise_xor(x_sub, y_add)


def create_bitwise_model(model_path):
    try:
        model = bitwise_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'bitwise_xor'

x_data = np.random.randint(-10, 20, [2, 16, 5]).astype(np.int32)
y_data = np.random.randint(-20, 30, [1, 16, 5]).astype(np.int32)
feed_dict = {'x': x_data, 'y': y_data}

model_path = TEST_NAME + '.pt'
create_bitwise_model(model_path)
exit_status = run_parser(model_path, feed_dict, verify=True)
assert exit_status
