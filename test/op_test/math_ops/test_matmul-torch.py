import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class matmul_model(nn.Module):
    def __init__(self):
        super(matmul_model, self).__init__()

    def forward(self, x1, x2):
        return torch.matmul(x1, x2)


def create_matmul_model(model_path):
    try:
        model = matmul_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'matmul'
x1_shapes = [[2, 3, 4, 5, 6], [3], [2, 3, 4], [2, 3, 4, 5], [2, 3, 4, 5, 6]]
x2_shapes = [[6], [2, 3, 4], [1, 2, 4, 5], [3, 5, 6], [1, 6, 4]]

for x1_shape, x2_shape in zip(x1_shapes, x2_shapes):
    # prepare model and input datas
    x1_data = np.random.ranf(x1_shape).astype(np.float32)
    x2_data = np.random.ranf(x2_shape).astype(np.float32)
    feed_dict = {'x1': x1_data, 'x2': x2_data}
    # np.save('input', feed_dict)
    model_path = '-'.join([TEST_NAME, str(len(x1_shape)), str(len(x2_shape))]) + '.pt'
    create_matmul_model(model_path)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
