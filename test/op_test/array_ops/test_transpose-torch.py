import torch
import numpy as np
from utils.run import run_parser


class transpose_model(torch.nn.Module):
    def __init__(self, dim0, dim1):
        super(transpose_model, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


def create_transpose_model(model_path, dim0=0, dim1=1):
    try:
        model = transpose_model(dim0, dim1)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'transpose'
input_shapes = [[], [2], [8, 7], [8, 6, 5], [2, 3, 4, 5]]
dims = [[0, 0], [0, 0], [1, 0], [0, 2], [3, 2]]

for input_shape, dim in zip(input_shapes, dims):
    input_data = np.random.randint(-10, 20, input_shape).astype(np.float32)
    feed_dict = {'x': input_data}
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.pt'
    # prepare model and input datas
    create_transpose_model(model_path, dim[0], dim[1])
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
