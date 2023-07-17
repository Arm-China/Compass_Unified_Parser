# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.onnx.symbolic_helper as helper
import functools


def quantize_helper(
    g,
    tensor,
    scale,
    zero_point,
    axis=None,
):
    if (
        axis is not None
        and not _is_none(axis)
        and GLOBALS.export_onnx_opset_version < 13
    ):
        _onnx_opset_unsupported_detailed(
            'QuantizeLinear',
            GLOBALS.export_onnx_opset_version,
            13,
            'Attribute axis is not supported.',
        )

    assert tensor is not None
    if tensor.type().scalarType() is None:
        tensor = g.op('Cast', tensor,
                      to_i=torch._C._onnx.TensorProtoDataType.FLOAT)

    assert scale is not None
    if scale.type().scalarType() != 'Float':
        scale = g.op(
            'Cast', scale, to_i=torch._C._onnx.TensorProtoDataType.FLOAT)

    assert zero_point is not None
    if zero_point.type().scalarType() not in ('Byte', 'Char'):
        zero_point = g.op('Cast', zero_point,
                          to_i=torch._C._onnx.TensorProtoDataType.UINT8)
    output = g.op(
        'QuantizeLinear',
        tensor,
        scale,
        zero_point,
        axis_i=helper._get_const(axis, 'i', 'axis'),
    )
    args = [output, scale, zero_point]
    if axis is not None and not _is_none(axis):
        args.append(axis)
    return g.op('prim::TupleConstruct', *args)


def quantize_helper_multi(
    g,
    tensors,
    scale,
    zero_point,
    axis=None,
):
    if (
        axis is not None
        and not _is_none(axis)
        and GLOBALS.export_onnx_opset_version < 13
    ):
        _onnx_opset_unsupported_detailed(
            'QuantizeLinear',
            GLOBALS.export_onnx_opset_version,
            13,
            'Attribute axis is not supported.',
        )

    assert scale is not None
    if scale.type().scalarType() != 'Float':
        scale = g.op(
            'Cast', scale, to_i=torch._C._onnx.TensorProtoDataType.FLOAT)

    assert zero_point is not None
    if zero_point.type().scalarType() not in ('Byte', 'Char'):
        zero_point = g.op('Cast', zero_point,
                          to_i=torch._C._onnx.TensorProtoDataType.UINT8)

    res = []
    for tensor in tensors:
        assert tensor is not None
        if tensor.type().scalarType() is None:
            tensor = g.op('Cast', tensor,
                          to_i=torch._C._onnx.TensorProtoDataType.FLOAT)

        args = []
        output = g.op(
            'QuantizeLinear',
            tensor,
            scale,
            zero_point,
            axis_i=helper._get_const(axis, 'i', 'axis'),
        )
        args.append(output)
        args.append(scale)
        args.append(zero_point)
        if axis is not None and not _is_none(axis):
            args.append(axis)
        res.append(g.op('prim::TupleConstruct', *args))

    return res


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
