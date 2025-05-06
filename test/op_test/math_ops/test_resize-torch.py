# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import torch
import numpy as np
from utils.run import run_parser


class resize_model(torch.nn.Module):
    def __init__(self, scale, mode, align_corners, recompute_scale_factor, antialias):
        super(resize_model, self).__init__()
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def forward(self, x):
        y = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode,
                                            align_corners=self.align_corners,
                                            recompute_scale_factor=self.recompute_scale_factor,
                                            antialias=self.antialias)
        return y


def create_resize_model(model_path, inp, scale, mode, align_corners, recompute_scale_factor=True, antialias=True):
    try:
        model = resize_model(scale, mode, align_corners, recompute_scale_factor, antialias)
        # model_scripted = torch.jit.script(model)
        # model_scripted.save(model_path)
        model_traced = torch.jit.trace(model, inp)
        model_traced.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'resize'

input_shape = [3, 4, 5, 6]
# x_data = np.random.randint(-30, 500, input_shape).astype(np.float32)
x_data = np.reshape(np.arange(np.prod(input_shape)), input_shape).astype(np.float32)
feed_dict = {'x': x_data}
np.save('input', feed_dict)

for idx, scale in enumerate([[0.8, 0.5], [1.3, 2.6]]):
    for mode in ('nearest', 'bilinear', ):
        for align_corners in (True, False, None):
            # align_corners option can only be set with the interpolating modes:
            # linear | bilinear | bicubic | trilinear
            if mode == 'nearest':
                if align_corners in (True, False):
                    continue
            else:
                if align_corners is None:
                    continue
            for recompute_scale_factor in (True, False):
                for antialias in (False, ):  # FIXME: The output from onnx and pytorch is different when antialias=True
                    model_path = '-'.join([TEST_NAME, str(idx), mode, str(align_corners),
                                           str(recompute_scale_factor), str(antialias)]) + '.pt'
                    create_resize_model(model_path, torch.tensor(x_data), scale, mode,
                                        align_corners, recompute_scale_factor, antialias)
                    exit_status = run_parser(model_path, feed_dict, expected_keywords=['layer_type=Resize'])
                    assert exit_status
