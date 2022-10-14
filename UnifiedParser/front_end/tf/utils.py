# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def trim_tensor_name(tensor_name):
    ''' The value of "input" and "output" could be node name or tensor name.
    Convert tensor to node if tensor is provided.
    '''
    return tensor_name.rsplit(':', 1)[0] if ':' in tensor_name else tensor_name
