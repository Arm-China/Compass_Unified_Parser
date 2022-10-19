import os
import torch
import torch.nn as nn
import numpy as np
from utils.run import generate_ir
from utils.compare import compare_data_dict
from utils.forward import opt_forward


class gru_model(nn.Module):
    def __init__(self):
        super(gru_model, self).__init__()
        self.rnn = nn.GRU(10, 20, 2)

    def forward(self, x1, x2):
        return self.rnn(x1, x2)


def create_torch_model(model_path, cfg_path):
    model = gru_model()
    model_scripted = torch.jit.script(model)
    model_scripted.save(model_path)

    output_dir = './output_dir'
    cfg_content = '''
[Common]
model_type = torch
model_name = gru
detection_postprocess = 
input_model = gru.pt
input = x1, x2
input_shape = [5,3,10], [2,3,20]
output_dir = ./output_dir
    '''

    with open(cfg_path, "w") as txt_file:
        txt_file.write(cfg_content)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_ir_without_postfix = os.path.join(output_dir, 'gru')
    return output_ir_without_postfix


TEST_NAME = 'gru'
model_path = TEST_NAME + '.pt'
cfg_path = TEST_NAME + '.cfg'
try:
    # prepare model and input datas
    output_ir_without_postfix = create_torch_model(model_path, cfg_path)
    x1_data = np.random.ranf([5, 3, 10]).astype(np.float32)
    x2_data = np.random.ranf([2, 3, 20]).astype(np.float32)
    feed_dict = {'x1': x1_data, 'x2': x2_data}
    # run tests with parser
    exit_status = generate_ir(cfg_path)
    assert exit_status
    # torch forward
    model = torch.jit.load(model_path)
    out_tensors = model(torch.tensor(x1_data), torch.tensor(x2_data))
    torch_outputs_dict = {}
    for idx, out_tensor in enumerate(out_tensors):
        out_name = str(idx)
        torch_outputs_dict[out_name] = out_tensor.detach().numpy()
    # opt forward
    opt_outputs_dict = opt_forward(output_ir_without_postfix + '.txt',
                                   output_ir_without_postfix + '.bin',
                                   feed_dict)
    # compare results
    same_outputs = compare_data_dict(torch_outputs_dict, opt_outputs_dict)
    if same_outputs:
        print('Passed: torch forward and opt forward get same result!')
    else:
        assert same_outputs, 'Failed: torch forward and opt forward get different result!'
except Exception as e:
    print('Fail to run tests because %s' % str(e))
