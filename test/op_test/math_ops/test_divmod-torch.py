import torch
import numpy as np
from utils.run import run_parser


class divmod_model(torch.nn.Module):
    def __init__(self):
        super(divmod_model, self).__init__()

    def forward(self, x):
        other = torch.tensor(4, dtype=torch.int64)
        y0 = torch.div(x, other, rounding_mode='trunc')
        remainder = x - y0 * other
        return y0, remainder


def create_divmod_model(model_path):
    try:
        model = divmod_model()
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'divmod'

# x_data = np.random.randint(-300, 5000, [10]).astype(np.int64)
x_data = np.array(list(range(-5, 5)))
feed_dict = {'x': x_data}
np.save('input', feed_dict)

model_path = TEST_NAME + '.pt'
create_divmod_model(model_path)
exit_status = run_parser(model_path, feed_dict, expected_keywords=['layer_type=DivMod'])
assert exit_status
