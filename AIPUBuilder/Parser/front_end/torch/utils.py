# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch


def get_tuple_from_tensor_type(torch_type, tensor_list, start_index=0):
    '''Get torch tensors tuple basing on torch tensor type, tensor_list and start_index.
    Return a tensor tuple and next index.
    '''
    tensors = ()
    index = start_index
    if isinstance(torch_type, torch._C.TupleType):
        nested_tensors = ()
        for nested_type in torch_type.elements():
            out_tensors, index = get_tuple_from_tensor_type(nested_type, tensor_list, index)
            nested_tensors += out_tensors
        tensors += (nested_tensors, )
    elif isinstance(torch_type, torch._C.TensorType):
        assert len(tensor_list) > index, 'Meets invalid tensors in get_tuple_from_tensor_type!'
        tensors += (tensor_list[index], )
        index += 1
    else:  # self(the class of the model)
        pass
    return tensors, index
