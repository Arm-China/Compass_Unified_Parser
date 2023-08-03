import torch
import numpy as np
from utils.run import run_parser


class chunk_model(torch.nn.Module):
    def __init__(self, chunks, dim=0):
        super(chunk_model, self).__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, x):
        y = torch.chunk(x, self.chunks, self.dim)
        return y[1] * 10.1


def create_chunk_model(model_path, chunks=1, dim=0):
    try:
        model = chunk_model(chunks, dim)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'chunk'

for input_shape in ([3, 2, 4, 5], ):
    for dim in (-2, 3):
        model_path = '-'.join([TEST_NAME, str(len(input_shape)), str(dim)]) + '.pt'
        # prepare model and input datas
        create_chunk_model(model_path, 2, dim)
        input_data = np.random.ranf(input_shape).astype(np.float32)
        feed_dict = {'x': input_data}
        exit_status = run_parser(model_path, feed_dict)
        assert exit_status
