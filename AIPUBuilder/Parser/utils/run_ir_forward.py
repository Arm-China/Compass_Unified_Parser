# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


# you should install AIPUBuilder first
from ...Optimizer.tools.optimizer_forward import OptForward


def run_ir_forward(txt_path, bin_path, input_data, forward_type='float',
                   transfer_to_float=None, input_is_quantized=False):
    # init a forward instance
    opt_forward = OptForward(txt_path, bin_path)

    if transfer_to_float is None:
        transfer_to_float = (forward_type == 'float')

    # input_data should be a dict and data order should be same with input tensors
    try:
        if input_is_quantized:
            outputs = opt_forward.forward_with_quantized_data(input_data, transfer_to_float=transfer_to_float)
        else:
            outputs = opt_forward.forward(input_data, transfer_to_float=transfer_to_float)
    except Exception as e:
        print(e)
        raise RuntimeError('Error happened during IR forward...')
    return outputs
