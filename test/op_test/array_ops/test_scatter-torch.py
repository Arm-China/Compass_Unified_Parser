import torch
import numpy as np
from utils.run import run_parser


class scatter_model(torch.nn.Module):
    def __init__(self, reduce):
        super(scatter_model, self).__init__()
        self.reduce = reduce

    def forward(self, x, src):
        index = torch.tensor([[0, 1, 2, 0]])
        if self.reduce is not None:
            return x.scatter(0, index, src, reduce=self.reduce)
        return x.scatter(0, index, src)


def create_scatter_model(model_path, reduce=None):
    try:
        model = scatter_model(reduce)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'scatter'
input_shapes = [[3, 5], ]

for input_shape in input_shapes:
    for src_dtype in (np.float32, ):  # np.int32 raises error: RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype
        src = np.random.randint(-10, 20, [2, 5]).astype(src_dtype)
        for reduce in (None, 'add', 'multiply'):
            model_path = '-'.join([TEST_NAME, str(len(input_shape)), np.dtype(src_dtype).name, str(reduce)]) + '.pt'
            # prepare model and input datas
            create_scatter_model(model_path, reduce)
            input_data = np.random.ranf(input_shape).astype(np.float32)
            feed_dict = {'x': input_data, 'src': src}
            exit_status = run_parser(model_path, feed_dict)
            assert exit_status
