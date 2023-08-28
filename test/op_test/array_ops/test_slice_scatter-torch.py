import torch
import numpy as np
from utils.run import run_parser


class slice_scatter_model(torch.nn.Module):
    def __init__(self, dim=0, start=None, end=None, step=1):
        super(slice_scatter_model, self).__init__()
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

    def forward(self, x, src):
        if self.start is None and self.end is None and self.step == 1:
            return x.slice_scatter(src, self.dim)
        return x.slice_scatter(src, self.dim, self.start, self.end, self.step)


def create_slice_scatter_model(model_path, dim=0, start=None, end=None, step=1):
    try:
        model = slice_scatter_model(dim, start, end, step)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'slice_scatter'
input_shapes = [[8, 7], [8, 8, 5], [7]]
src_shapes = [[2, 7], [8, 2, 5], [1]]

for input_shape, src_shape in zip(input_shapes, src_shapes):
    input_data = np.random.randint(-10, 20, input_shape).astype(np.int32)
    src = np.random.randint(-5, 15, src_shape).astype(np.float32)
    feed_dict = {'x': input_data, 'src': src}
    model_path = '-'.join([TEST_NAME, str(len(input_shape))]) + '.pt'
    # prepare model and input datas
    if len(input_shape) <= 2:
        create_slice_scatter_model(model_path, start=6)
    else:
        create_slice_scatter_model(model_path, dim=1, start=2, end=6, step=2)
    exit_status = run_parser(model_path, feed_dict)
    assert exit_status
