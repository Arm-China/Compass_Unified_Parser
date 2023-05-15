import torch
import numpy as np
from utils.run import run_parser


class add_model(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y, alpha=1)


def create_add_model(model_path):
    try:
        model = add_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'add'
model_path = TEST_NAME + '.pt'

# prepare model and input datas
create_add_model(model_path)
x_data = np.random.randint(-10, 20, [2, 16, 5]).astype(np.int32)
y_data = np.random.randint(-20, 30, [1, 16, 5]).astype(np.int32)
feed_dict = {'x': x_data, 'y': y_data}
exit_status = run_parser(model_path, feed_dict, expected_keywords=['layer_top_type=[int32]'])
assert exit_status
