import torch
import numpy as np
from utils.run import run_parser
from typing import Tuple
from torch import Tensor


class test_tuple_input_model(torch.nn.Module):
    def forward(self, x, x_groups1: Tuple[Tensor, Tensor], x_groups2: Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]):
        add_1 = x + (x_groups1[0] * x_groups1[1])
        add_2 = add_1 + x_groups2[0]
        out = add_2 + (x_groups2[1][0] * x_groups2[1][1] * x_groups2[1][2])
        return out


def create_test_tuple_input_model(model_path):
    try:
        model = test_tuple_input_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


model_path = 'tuple_input.pt'

# prepare model and input datas
create_test_tuple_input_model(model_path)
feed_dict = {}
for input_idx in range(7):
    input_name = 'input_' + str(input_idx)
    x_data = np.random.randint(-10, 20, [2, 16, 5]).astype(np.float32)
    feed_dict[input_name] = x_data
exit_status = run_parser(model_path, feed_dict)
assert exit_status
