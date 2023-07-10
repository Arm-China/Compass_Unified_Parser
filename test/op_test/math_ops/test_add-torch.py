import torch
import numpy as np
from utils.run import run_parser


class add_model(torch.nn.Module):
    def __init__(self, add_alpha=1):
        super(add_model, self).__init__()
        self.add_alpha = add_alpha

    def forward(self, x, y):
        x_sub = torch.sub(y, x, alpha=3)
        return torch.add(x_sub, y, alpha=self.add_alpha)


def create_add_model(model_path, alpha):
    try:
        model = add_model(alpha)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'add'

x_data = np.random.randint(-10, 20, [2, 16, 5]).astype(np.int32)
y_data = np.random.randint(-20, 30, [1, 16, 5]).astype(np.int32)
feed_dict = {'x': x_data, 'y': y_data}

for alpha in (1, 5, -4):
    model_path = '-'.join([TEST_NAME, str(alpha)]) + '.pt'
    create_add_model(model_path, alpha)
    exit_status = run_parser(model_path, feed_dict, expected_keywords=['layer_top_type=[int32]'])
    assert exit_status
