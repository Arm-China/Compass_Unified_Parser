import numpy as np
import torch
import torchvision

from utils.run import run_parser


class deform_conv_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn([10, 6, 7, 8], dtype=torch.float32)

    def forward(self, x, offset, mask):
        y = torchvision.ops.deform_conv2d(x, offset, self.weight, mask=mask)
        return y


def create_deform_conv_model(model_path, inputs):
    try:
        model = deform_conv_model()
        # model_scripted = torch.jit.script(model)
        # model_scripted.save(model_path)
        traced = torch.jit.trace(model, inputs)
        traced.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'deform_conv'
model_path = TEST_NAME + '.pt'

# prepare model and input datas
x_data = np.random.ranf([2, 6, 51, 52]).astype(np.float32)
offset = np.random.randint(-4, 10, [2, 112, 45, 45]).astype(np.float32)
mask = np.random.randint(-3, 3, [2, 56, 45, 45]).astype(np.float32)
feed_dict = {'x': x_data, 'offset': offset, 'mask': mask}
inputs = ()
for data in feed_dict.values():
    inputs += (torch.tensor(data), )
create_deform_conv_model(model_path, inputs)
# create_deform_conv_model(model_path)
exit_status = run_parser(model_path, feed_dict)
assert exit_status
