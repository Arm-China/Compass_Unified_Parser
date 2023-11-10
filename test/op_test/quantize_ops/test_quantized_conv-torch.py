import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class qconv_model(nn.Module):
    def __init__(self, dim):
        super(qconv_model, self).__init__()
        if dim == 1:
            self.qconv = nn.quantized.Conv1d(17, 35, 3, stride=2, padding=4)
            weight = torch.quantize_per_tensor(torch.randn(35, 17, 3), 0.5, 0, torch.qint8)
            bias = torch.arange(35).to(torch.float) - 16
        elif dim == 2:
            self.qconv = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
            weight = torch.quantize_per_tensor(torch.randn(33, 16, 3, 5), 0.5, 0, torch.qint8)
            bias = torch.arange(33).to(torch.float) - 16
        else:  # dim == 3
            self.qconv = nn.quantized.Conv3d(15, 32, (3, 4, 5), stride=(2, 1, 2), padding=(0, 4, 2))
            weight = torch.quantize_per_tensor(torch.randn(32, 15, 3, 4, 5), 0.5, 0, torch.qint8)
            bias = torch.arange(32).to(torch.float) - 16
        self.qconv.set_weight_bias(weight, bias)

    def forward(self, x):
        q_input = torch.quantize_per_tensor(x, scale=0.5, zero_point=128, dtype=torch.quint8)
        return self.qconv(q_input)


def create_qconv_model(model_path, dim):
    try:
        model = qconv_model(dim)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'qconv'

for idx, input_shape in enumerate([[2, 17, 60], [3, 16, 20, 30], [3, 15, 45, 46, 47]]):
    model_path = '-'.join([TEST_NAME, str(idx)]) + '.pt'
    # prepare model and input datas
    x_data = np.random.ranf(input_shape).astype(np.float32)
    feed_dict = {'x': x_data}
    np.save('input', feed_dict)
    create_qconv_model(model_path, len(input_shape) - 2)
    # need qtlib to generate ir before opt forward
    exit_status = run_parser(model_path, feed_dict, verify=False)
    assert exit_status
