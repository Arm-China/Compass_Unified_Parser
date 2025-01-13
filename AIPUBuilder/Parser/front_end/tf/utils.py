# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


def trim_tensor_name(tensor_name):
    ''' The value of "input" and "output" could be node name or tensor name.
    Convert tensor to node if tensor is provided.
    '''
    return tensor_name.rsplit(':', 1)[0] if ':' in tensor_name else tensor_name
