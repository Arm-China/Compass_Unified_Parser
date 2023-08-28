import torch
import numpy as np
from utils.run import run_parser


class select_scatter_model(torch.nn.Module):
    def __init__(self, dim, index):
        super(select_scatter_model, self).__init__()
        self.dim = dim
        self.index = index

    def forward(self, x, src):
        return x.select_scatter(src, self.dim, self.index)


def create_select_scatter_model(model_path, dim, index):
    try:
        model = select_scatter_model(dim, index)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'select_scatter'
input_shapes = [[6], [8, 7], [8, 8, 5], ]
src_shapes = [[], [7], [8, 5], ]

for input_shape, src_shape in zip(input_shapes, src_shapes):
    input_data = np.random.randint(-10, 20, input_shape).astype(np.int32)
    src = np.random.randint(-5, 15, src_shape).astype(np.float32)
    feed_dict = {'x': input_data, 'src': src}
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.pt'
    # prepare model and input datas
    if len(input_shape) <= 2:
        create_select_scatter_model(model_path, dim=0, index=4)
    else:
        create_select_scatter_model(model_path, dim=1, index=2)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
