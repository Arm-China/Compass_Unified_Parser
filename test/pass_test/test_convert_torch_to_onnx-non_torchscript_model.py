import torch
import torchvision
import numpy as np
from utils.run import run_parser
from typing import Tuple


def create_test_non_torchscript_model(model_path):
    try:
        model = torchvision.models.alexnet()
        torch.save(model, model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


model_path = 'non_torchscript.pt'

# prepare model and input datas
create_test_non_torchscript_model(model_path)
feed_dict = {}
feed_dict['x'] = np.random.randint(-10, 20, [1, 3, 224, 224]).astype(np.float32)
exit_status = run_parser(model_path, feed_dict)
assert exit_status
