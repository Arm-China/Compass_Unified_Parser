# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


# you should install AIPUBuilder first
from ...Optimizer.tools.optimizer_forward import OptForward


def run_ir_forward(txt_path, bin_path, input_data, forward_type='float'):
    # init a forward instance
    opt_forward = OptForward(txt_path, bin_path)

    # input_data should be a dict and data order should be same with input tensors
    try:
        outputs = opt_forward.forward(input_data, transfer_to_float=(forward_type == 'float'))
    except Exception as e:
        print(e)
        raise RuntimeError("Error happened during IR forward...")
    return outputs
