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


def quantized_args(*arg_q_descriptors, scale=None, zero_point=None):
    def decorator(fn):
        fn._scale = scale
        fn._zero_point = zero_point

        @functools.wraps(fn)
        def wrapper(g, *args, **kwargs):
            _scale = fn._scale
            if _scale is not None:
                _scale = g.op('Constant', value_t=torch.tensor(_scale))
            _zero_point = fn._zero_point
            if _zero_point is not None:
                _zero_point = g.op(
                    'Constant', value_t=torch.tensor(_zero_point))

            arg_q_descriptors_extended = arg_q_descriptors + (False,) * (
                len(args) - len(arg_q_descriptors)
            )
            descriptor_args = tuple(zip(arg_q_descriptors_extended, args))
            if not any(
                (descriptor and arg.node().kind() == 'prim::TupleConstruct')
                for descriptor, arg in descriptor_args
            ):
                return fn(g, *args, **kwargs)

            dequantized_args = []
            for descriptor, arg in descriptor_args:
                if descriptor:
                    dequantized_arg, scale, zero_point, _ = helper.dequantize_helper(
                        g, arg)
                    dequantized_args.append(dequantized_arg)
                    if _scale is None:
                        _scale = scale
                    if _zero_point is None:
                        _zero_point = zero_point
                else:
                    dequantized_args.append(arg)
            output = fn(g, *dequantized_args, **kwargs)

            if isinstance(output, list):
                return quantize_helper_multi(g, output, _scale, _zero_point)
            else:
                return quantize_helper(g, output, _scale, _zero_point)

        return wrapper

    return decorator