# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import os
import numpy as np
import math
import itertools
import onnx
import torch
import torch.nn as nn
import torch.onnx.symbolic_helper as helper
from torch.onnx import symbolic_opset9 as opset9
from multiprocessing import Process
from .utils import get_tuple_from_tensor_type, quantized_args, quantize_helper, quantize_helper_multi, \
    get_onnx_pool_output_shape, get_torch_pool_output_shape
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL
from ...common.utils import get_version
from ...common.defs import FLOAT_EQUAL


# global variance
ONNX_OPSET_VERSION = 9
CUSTOM_OPSET_18 = 'opset_18::'
CUSTOM_OPSET_19 = 'opset_19::'


@helper.parse_args('v')
@quantized_args(True)
def convert_acosh(g, x):
    return g.op('Acosh', x)


@helper.parse_args('v')
@quantized_args(True)
def convert_asinh(g, x):
    return g.op('Asinh', x)


@helper.parse_args('v')
@quantized_args(True)
def convert_atanh(g, x):
    return g.op('Atanh', x)


def convert_adaptive_pool(g, x, output_size, dim, method):
    assert dim in (
        1, 2, 3), 'Meets invalid dim (%s) in convert_adaptive_pool!' % str(dim)
    assert method in (
        'AVG', 'MAX'), 'Meets invalid method (%s) in convert_adaptive_pool!' % str(method)
    input_shape = helper._get_tensor_sizes(x)
    need_reshape = (len(input_shape) == 3)
    if need_reshape:
        x = helper._reshape_helper(g, x, [1] + input_shape)  # from CHW to NCHW
    if helper._is_packed_list(output_size):  # None in output_size
        # None in output_size is handled by torch/nn/functional.py(function _list_with_default)
        # and then output_size becomes a packed list with const ints inside.
        output_size = [helper._parse_arg(size_val, 'i')
                       for size_val in helper._unpack_list(output_size)]
    else:
        size = helper._maybe_get_const(output_size, 'is')
        if helper._is_value(size):  # output_size is scalar
            output_size = dim * [helper._parse_arg(output_size, 'i')]
        else:  # output_size is a tuple of ints
            output_size = size
    if all(s == 1 for s in output_size):
        output = g.op('GlobalAveragePool', x) if method == 'AVG' else g.op(
            'GlobalMaxPool', x)
    elif all((dim % s == 0) for dim, s in zip(input_shape[-2:], output_size)):
        kernel_shape = [int(dim / s)
                        for dim, s in zip(input_shape[-2:], output_size)]
        if method == 'AVG':
            output = g.op('AveragePool', x,
                          kernel_shape_i=kernel_shape, strides_i=kernel_shape)
        else:
            output = g.op('MaxPool', x, outputs=1,
                          kernel_shape_i=kernel_shape, strides_i=kernel_shape)
    else:
        output_type = x.type().with_sizes(
            ([1] if need_reshape else []) + input_shape[:-2] + output_size)
        output = g.op('custom::AdaptivePool', x, output_size_i=output_size,
                      method_s=method).setType(output_type)
    if need_reshape:
        output_shape = input_shape[0:1] + output_size
        output = helper._reshape_helper(
            g, output, output_shape)  # from NCHW to CHW
    # aten::adaptive_max_poolNd always return 2 outputs but only the first one is used when return_indices=False.
    return output if method == 'AVG' else (output, None)


@quantized_args(True, False, False)
def convert_adaptive_avg_pool2d(g, x, output_size):
    return convert_adaptive_pool(g, x, output_size, 2, 'AVG')


@quantized_args(True, False, False)
def convert_adaptive_max_pool2d(g, x, output_size):
    return convert_adaptive_pool(g, x, output_size, 2, 'MAX')


def convert_addbmm(g, x, batch1, batch2, beta, alpha):
    scalar_type = x.type().scalarType()
    batch_mul = g.op('MatMul', batch1, batch2)
    if ONNX_OPSET_VERSION >= 13:
        axes_input = g.op('Constant', value_t=torch.tensor([0], dtype=torch.int64))
        reduce_batch_mul = g.op('ReduceSum', batch_mul, axes_input, keepdims_i=0)
    else:
        reduce_batch_mul = g.op('ReduceSum', axes_i=[0], keepdims_i=0)
    mul_a = g.op('Mul', reduce_batch_mul, g.op('Cast', alpha, to_i=helper.cast_pytorch_to_onnx[scalar_type]))
    mul_b = g.op('Mul', x, g.op('Cast', beta, to_i=helper.cast_pytorch_to_onnx[scalar_type]))
    return g.op('Add', mul_a, mul_b)


def convert_add_sub(g, input, other, alpha, op_type):
    if alpha and not FLOAT_EQUAL(helper._maybe_get_const(alpha, 'f'), 1):
        other = g.op('Mul', other, alpha)
    return g.op(op_type, input, other)


@helper.parse_args('v', 'v', 'v')
def convert_add(g, input, other, alpha=None):
    return convert_add_sub(g, input, other, alpha, 'Add')


@helper.parse_args('v', 'v', 'v', 'f')
def convert_addcdiv(g, input, tensor1, tensor2, value=1.0):
    value_tens = g.op('Constant', value_t=torch.tensor(value))
    return opset9.add(g, input, opset9.mul(g, opset9.div(g, tensor1, tensor2), value_tens))


@helper.parse_args('v', 'v', 'v', 'f')
def convert_addcmul(g, input, tensor1, tensor2, value=1.0):
    value_tens = g.op('Constant', value_t=torch.tensor(value))
    return opset9.add(g, input, opset9.mul(g, opset9.mul(g, tensor1, tensor2), value_tens))


@helper.parse_args('v', 'v', 'v', 'f', 'f')
def convert_addmv(g, input, tensor1, tensor2, beta=1.0, alpha=1.0):
    value_beta = g.op('Constant', value_t=torch.tensor(beta))
    value_alpha = g.op('Constant', value_t=torch.tensor(alpha))
    return opset9.add(g, opset9.mul(g, input, value_beta), opset9.mul(g, opset9.matmul(g, tensor1, tensor2), value_alpha))


@helper.parse_args('v', 'v', 'v', 'f', 'f')
def convert_addr(g, input, tensor1, tensor2, beta=1.0, alpha=1.0):
    value_beta = g.op('Constant', value_t=torch.tensor(beta))
    value_alpha = g.op('Constant', value_t=torch.tensor(alpha))
    t1_shape = helper._get_tensor_sizes(tensor1)
    t2_shape = helper._get_tensor_sizes(tensor2)
    m1 = helper._reshape_helper(g, tensor1, list(t1_shape) + [1, ])
    m2 = helper._reshape_helper(g, tensor2, [1, ] + list(t2_shape))
    return opset9.add(g, opset9.mul(g, input, value_beta), opset9.mul(g, opset9.matmul(g, m1, m2), value_alpha))


@helper.parse_args('v', 'v', 'v')
def convert_rsub(g, input, other, alpha=None):
    return convert_add_sub(g, other, input, alpha, 'Sub')


@helper.parse_args('v', 'v', 'v')
def convert_sub(g, input, other, alpha=None):
    return convert_add_sub(g, input, other, alpha, 'Sub')


def convert_argmax_argmin(g, input, dim, keepdim, op_type):
    if helper._is_none(dim):
        flatten = helper._reshape_helper(g, input, [-1])
        output = g.op(op_type, flatten, axis_i=0, keepdims_i=False)
        if keepdim:
            input_shape = helper._get_tensor_sizes(input)
            output_shape = np.ones_like(input_shape)
            output = helper._reshape_helper(g, output, output_shape)
    else:
        dim = helper._parse_arg(dim, 'i')
        output = g.op(op_type, input, axis_i=dim, keepdims_i=keepdim)
    return output


@helper.parse_args('v', 'v', 'i')
def convert_argmax(g, input, dim=None, keepdim=False):
    return convert_argmax_argmin(g, input, dim, keepdim, 'ArgMax')


@helper.parse_args('v', 'v', 'i')
def convert_argmin(g, input, dim=None, keepdim=False):
    return convert_argmax_argmin(g, input, dim, keepdim, 'ArgMin')


def convert_bitshift(g, input, other, direction):
    input_dtype = input.type().dtype()
    onnx_op = 'opset_11::BitShift' if input_dtype.is_signed else 'BitShift'
    return g.op(onnx_op, input, other, direction_s=direction)


def convert_iand(g, input1, input2):
    return convert_bitwise_and(g, input1, input2)


@helper.parse_args('v', 'v')
def convert_bitshift_left(g, input, other):
    return convert_bitshift(g, input, other, 'LEFT')


@helper.parse_args('v', 'v')
def convert_bitshift_right(g, input, other):
    return convert_bitshift(g, input, other, 'RIGHT')


@helper.parse_args('v', 'v')
def convert_bitwise_and(g, input, other):
    return g.op(CUSTOM_OPSET_18 + 'BitwiseAnd', input, other)


@helper.parse_args('v')
def convert_bitwise_not(g, input):
    return g.op(CUSTOM_OPSET_18 + 'BitwiseNot', input)


@helper.parse_args('v', 'v')
def convert_bitwise_or(g, input, other):
    return g.op(CUSTOM_OPSET_18 + 'BitwiseOr', input, other)


@helper.parse_args('v', 'v')
def convert_bitwise_xor(g, input, other):
    return g.op(CUSTOM_OPSET_18 + 'BitwiseXor', input, other)


@helper.parse_args('v', 'i')
@quantized_args(True, False)
def convert_channel_shuffle(g, x, groups):
    return g.op('custom::ChannelShuffle', x, group_i=groups).setType(x.type())


@helper.parse_args('v', 'i', 'i')
@quantized_args(True, False, False)
def convert_chunk(g, input, chunks, dim):
    from torch.onnx.symbolic_opset13 import split
    input_shape = helper._get_tensor_sizes(input)
    if input_shape is None or None in input_shape:
        try:
            from torch.onnx.symbolic_opset11 import Prim
            return Prim.ConstantChunk(g, input, chunks, dim)
        except ImportError:  # torch 2.1
            from torch.onnx.symbolic_opset11 import prim_constant_chunk
            return prim_constant_chunk(g, input, chunks, dim)
    dim_size = input_shape[dim]
    chunk_size = (dim_size + chunks - 1) // chunks
    split_sizes = [chunk_size] * (dim_size // chunk_size)
    leftover = dim_size % chunk_size
    if leftover:
        split_sizes.append(leftover)
    splits = g.op('Constant',
                  value_t=torch.tensor(split_sizes, dtype=torch.long))
    split_outs = split(g, input, splits, dim, len(split_sizes))
    return g.op('prim::ListConstruct', *split_outs)


@quantized_args(True, False, False)
def convert_constant_chunk(g, input, chunks, dim):
    try:
        from torch.onnx.symbolic_opset11 import Prim
        return Prim.ConstantChunk(g, input, chunks, dim)
    except ImportError:  # torch 2.1
        from torch.onnx.symbolic_opset11 import prim_constant_chunk
        return prim_constant_chunk(g, input, chunks, dim)


@helper.parse_args('v', 'is', 'is', 'is', 'is', 'is')
@quantized_args(True)
def convert_col2im(g, input, output_size, kernel_size, dilation, padding, stride):
    input_shape = helper._get_tensor_sizes(input)
    need_reshape = False
    if len(input_shape) == 2:
        need_reshape = True
        # reshape unbatched to batch=1
        input = helper._reshape_helper(g, input, [1] + input_shape)
    output_size = g.op('Constant', value_t=torch.tensor(output_size, dtype=torch.int64))
    kernel_size = g.op('Constant', value_t=torch.tensor(kernel_size, dtype=torch.int64))
    pads = np.tile(padding, 2)
    out = g.op(CUSTOM_OPSET_18 + 'Col2Im', input, output_size, kernel_size,
               dilations_i=dilation, pads_i=pads, strides_i=stride)
    if need_reshape:
        # reshape back to unbatched
        out = helper._squeeze_helper(g, out, [0])
    return out


@helper.parse_args('v', 'v', 'v', 'is', 'v', 'is', 'i')
def convert_conv(g, input, weight, bias, stride, padding, dilation, groups):
    # Support padding as string. Refer to https://github.com/pytorch/pytorch/pull/89107
    ret = None
    weight_shape = helper._get_tensor_sizes(weight)
    try:
        kernel_shape = weight_shape[2:]
    except:
        kernel_shape = None
    if kernel_shape is None or None in kernel_shape:
        ERROR('[Parser]: Meets invalid kernel shape of Conv op in convert_conv!')
        return ret

    args = [input, weight]
    need_separate_add = False
    if not helper._is_none(bias):
        if helper._get_tensor_rank(bias) == 1:
            args.append(bias)
        else:
            need_separate_add = True

    kwargs = {'kernel_shape_i': kernel_shape, 'strides_i': stride,
              'dilations_i': dilation, 'group_i': groups}

    str_padding = helper._parse_arg(padding, 's')
    if str_padding in ('valid', 'same'):
        auto_pad = 'VALID' if str_padding == 'valid' else 'SAME_UPPER'
        kwargs.update({'auto_pad_s': auto_pad})
    else:
        padding = helper._parse_arg(padding, 'is')
        padding = padding + padding
        kwargs.update({'pads_i': padding})

    conv = g.op('Conv', *args, **kwargs)

    if need_separate_add:
        return g.op('Add', conv, bias)
    return conv


@helper.parse_args('v')
@quantized_args(True)
def convert_cosh(g, x):
    return g.op('Cosh', x)


@helper.parse_args('v', 'i', 'i')
def convert_cumprod(g, x, dim, dtype):
    if dtype is not None:
        x = g.op('Cast', x, to_i=helper.scalar_type_to_onnx[dtype])
    return g.op('custom::CumProd', x, axis_i=dim).setType(x.type())


@helper.parse_args('v', 'v', 'v', 'v', 'v', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'b')
@helper.quantized_args(True, True, True, True, True, False, False, False, False, False, False, False, False, False)
def convert_deform_conv(g, input, weight, offset, mask, bias,
                        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                        n_weight_grps, n_offset_grps, use_mask):
    pads = [pad_h, pad_w] * 2
    weight_shape = helper._get_tensor_sizes(weight)
    kernel_shape = weight_shape[2:]
    if not use_mask:
        mask = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
    return g.op(CUSTOM_OPSET_19 + 'DeformConv', input, weight, offset, bias, mask,
                dilations_i=[dilation_h, dilation_w], kernel_shape_i=kernel_shape,
                pads_i=pads, strides_i=[stride_h, stride_w],
                group_i=n_weight_grps, offset_group_i=n_offset_grps)


@helper.parse_args('s', 'v', 's', 'v')
def convert_dict_construct(g, key_0, value_0, key_1=None, value_1=None):
    keys = ', '.join([key_0] + ([] if key_1 is None else [key_1]))
    WARN(
        '[Parser]: prim::DictConstruct is unsupported and is changed to return a tensor or a list of tensors(key(s): %s) instead!' % keys)
    if value_1 is not None:
        g.registerOutput(value_0)  # value_0 is the first output of graph
        # value_1 is the second output of graph if this node is output
        return g.op('Identity', value_1)
    return g.op('Identity', value_0)


@helper.parse_args('v', 'v', 'v')
@helper.quantized_args(True, False, False)
def convert_div(g, input, other, *args):
    from torch.onnx.symbolic_opset10 import div
    return div(g, input, other, *args)


@helper.parse_args('v', 'v')
def convert_dsplit(g, input, split_size_or_sizes):
    input_rank = helper._get_tensor_rank(input)
    assert input_rank > 2, 'input dim should > 2 in dsplit.'
    return convert_tensor_split(g, input, split_size_or_sizes, 2)


@helper.parse_args('v')
def convert_erfc(g, input):
    return opset9.sub(g, g.op('Constant', value_t=torch.tensor(1.)), opset9.erf(g, input))


def convert_quantized_add_relu(g, x, y, op_scale, op_zero_point):
    x, _, _, _ = helper.dequantize_helper(g, x)
    y, _, _, _ = helper.dequantize_helper(g, y)

    output = opset9.add(g, x, y)
    output = opset9.relu(g, output)

    return helper.quantize_helper(g, output, op_scale, op_zero_point)


@helper.parse_args('v', 'i', 'i')
@helper.quantized_args(True, False, False)
def convert_flatten(g, input, start_dim, end_dim):
    input_rank = helper._get_tensor_rank(input)
    assert input_rank is not None, 'Meets unknown rank in convert_flatten!'
    if input_rank == 0:
        return helper._reshape_helper(g, input, [1])
    if input_rank == 1:
        return g.op('Identity', input)
    start_dim = (start_dim + input_rank) if start_dim < 0 else start_dim
    end_dim = (end_dim + input_rank) if end_dim < 0 else end_dim
    return helper._flatten_helper(g, input, start_dim, end_dim, input_rank)


@helper.parse_args('v')
def convert_fliplr(g, input):
    return helper._slice_helper(
        g,
        input,
        axes=[1],
        starts=[-1],
        ends=[-torch.onnx._constants.INT64_MAX],
        steps=[-1],
    )


@helper.parse_args('v')
def convert_flipud(g, input):
    return helper._slice_helper(
        g,
        input,
        axes=[0],
        starts=[-1],
        ends=[-torch.onnx._constants.INT64_MAX],
        steps=[-1],
    )


@helper.parse_args('v', 'v', 'v', 'v', 'v', 'v')
def convert_gru_cell(g, input, hidden, w_ih, w_hh, b_ih, b_hh):
    from torch.onnx.symbolic_opset9 import _generic_rnn

    input = helper._unsqueeze_helper(g, input, [0])
    hidden = helper._unsqueeze_helper(g, hidden, [0])
    if helper._is_tensor(b_ih):
        weight = (w_ih, w_hh, b_ih, b_hh)
        has_biases = True
    else:
        weight = (w_ih, w_hh)
        has_biases = False
    _, h_out = _generic_rnn(g, 'GRU', input, hidden, weight,
                            has_biases, num_layers=1, dropout=False,
                            train=False, bidirectional=False,
                            batch_first=False)
    return helper._squeeze_helper(g, h_out, [0])


@helper.parse_args('v', 'f')
@quantized_args(True, False)
def convert_hardshrink(g, input, lambd):
    # The output shape issue for scalar input is fixed in torch 2.0.1.
    # Refer to https://github.com/pytorch/pytorch/pull/79695
    # torch converts hard/soft shrink to some logical ops, which is not needed;
    # convert to onnx Shrink directly.
    return g.op('Shrink', input, lambd_f=lambd)


@helper.parse_args('v', 'v')
def convert_hsplit(g, input, split_size_or_sizes):
    input_rank = helper._get_tensor_rank(input)
    assert input_rank > 0, 'input dim should > 0 in hsplit.'
    if input_rank == 1:
        return convert_tensor_split(g, input, split_size_or_sizes, 0)
    else:
        return convert_tensor_split(g, input, split_size_or_sizes, 1)


@helper.parse_args('v', 'f')
@quantized_args(True, False)
def convert_softshrink(g, input, lambd):
    return g.op('Shrink', input, lambd_f=lambd, bias_f=lambd)


def convert_to_bool(g, input):
    input_dtype_str = input.type().scalarType()
    if input_dtype_str != 'Bool':
        input = g.op(
            'Cast', input, to_i=torch._C._onnx.TensorProtoDataType.BOOL)
    return input


@helper.parse_args('v', 'v')
def convert_logaddexp(g, input, other):
    return opset9.log(g, opset9.add(g, opset9.exp(g, input), opset9.exp(g, other)))


def convert_logical(g, input, other=None, op_type=''):
    assert len(op_type) > 0, 'Meets empty op_type in convert_logical!'
    if other is None:
        return g.op(op_type, convert_to_bool(g, input))
    return g.op(op_type, convert_to_bool(g, input), convert_to_bool(g, other))


@helper.parse_args('v')
def convert_logical_not(g, input):
    return convert_logical(g, input, op_type='Not')


@helper.parse_args('v', 'v')
def convert_equal(g, input, other):
    '''torch equal op is different with logical equal op. It returns scalar
    True if two tensors have the same size and elements, False otherwise.
    '''
    input_shape = helper._get_tensor_sizes(input)
    other_shape = helper._get_tensor_sizes(other)
    if input_shape != other_shape:
        return g.op('Constant', value_t=torch.tensor(False))
    equal = convert_logical(g, input, other, 'Equal')
    not_equal = g.op('Not', equal)
    reduce_sum = g.op('ReduceSum', not_equal, keepdims_i=0)
    return g.op('Not', reduce_sum)


def convert_avg_pool(g, input, kernel_size, strides, paddings, ceil_mode, count_include_pad, divisor_override, dim):
    assert isinstance(dim, int) and dim in [
        1, 2, 3], 'Meets invalid dim in convert_avg_pool!'
    if not ceil_mode:
        try:
            from torch.onnx.symbolic_opset11 import avg_pool1d, avg_pool2d, avg_pool3d
            avg_pool_func = avg_pool1d if dim == 1 else (
                avg_pool2d if dim == 2 else avg_pool3d)
        except ImportError:  # torch 2.1
            from torch.onnx.symbolic_opset10 import _avg_pool
            if dim == 1:
                avg_pool_func = _avg_pool('avg_pool1d', 1)
            elif dim == 2:
                avg_pool_func = _avg_pool('avg_pool2d', 2)
            else:  # dim == 3
                avg_pool_func = _avg_pool('avg_pool3d', 3)
        return avg_pool_func(g, input, kernel_size, strides, paddings, ceil_mode, count_include_pad, divisor_override)

    input_shape = helper._get_tensor_sizes(input)
    spatial_input_shape = input_shape[2:]
    if not strides:
        strides = kernel_size

    torch_spatial_output_shapes = []
    for input_size, kernel, stride, pad in zip(spatial_input_shape, kernel_size, strides, paddings):
        output_shape = get_torch_pool_output_shape(
            input_size, kernel, pad, pad, stride)
        torch_spatial_output_shapes.append(output_shape)

    adjusted_padding = paddings
    if count_include_pad:
        input = g.op(
            'Pad',
            input,
            g.op('Constant', value_t=torch.tensor(([0, 0] + paddings) * 2)),
            mode_s='constant',
        )
        adjusted_padding = [0] * len(paddings)
        spatial_input_shape = [(a + 2 * b)
                               for (a, b) in zip(spatial_input_shape, paddings)]

    avg_pool = g.op('AveragePool', input, kernel_shape_i=kernel_size, strides_i=strides,
                    pads_i=(adjusted_padding * 2), ceil_mode_i=ceil_mode)

    slice_ends = []
    for idx, (input_size, kernel, stride, pad) in enumerate(
            zip(spatial_input_shape, kernel_size, strides, adjusted_padding)):
        onnx_output_shape = get_onnx_pool_output_shape(
            input_size, kernel, pad, pad, stride)
        assert len(
            torch_spatial_output_shapes) > idx, 'Meets invalid torch_spatial_output_shapes in convert_avg_pool!'
        if torch_spatial_output_shapes[idx] > onnx_output_shape:
            FATAL(
                '[Parser]: Meets unsupported output shape of avg_pool in convert_avg_pool!')
        slice_ends.append(torch_spatial_output_shapes[idx])
    ends_input = g.op('Constant', value_t=torch.tensor(slice_ends))
    starts_input = g.op(
        'Constant', value_t=torch.tensor([0] * len(slice_ends)))
    axes_input = g.op('Constant', value_t=torch.tensor(
        list(range(2, len(input_shape)))))
    slice_out = g.op('Slice', avg_pool, starts_input, ends_input, axes_input)
    return slice_out


@helper.quantized_args(True, False, False, False, False, False, False)
@helper.parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
def convert_avg_pool1d(g, inp, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
    return convert_avg_pool(g, inp, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, 1)


@helper.quantized_args(True, False, False, False, False, False, False)
@helper.parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
def convert_avg_pool2d(g, inp, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
    return convert_avg_pool(g, inp, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, 2)


@helper.quantized_args(True, False, False, False, False, False, False)
@helper.parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
def convert_avg_pool3d(g, inp, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
    return convert_avg_pool(g, inp, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, 3)


def convert_qat_bn(g, input, weight, bias, running_mean, running_var, eps, s, zp):
    input, _, _, _ = helper.dequantize_helper(g, input)
    weight, bias, running_mean, running_var = helper._batchnorm_helper(
        g, input, weight, bias, running_mean, running_var)
    out = g.op(
        'BatchNormalization',
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon_f=eps,
        outputs=1,
    )
    return out


@helper.parse_args('v', 'v', 'v', 'v', 'v', 'f', 'v', 'v')
def convert_quant_batch_norm_relu_3d(g, x, weight, bias, running_mean, running_var, eps, s, zp, ):
    from torch.onnx.symbolic_opset9 import relu
    out = convert_qat_bn(g, x, weight, bias,
                         running_mean, running_var, eps, s, zp)
    out = relu(g, out)
    return quantize_helper(g, out, s, zp)


@helper.parse_args('v', 'v', 'v', 'v', 'v', 'f', 'v', 'v')
def convert_quant_batch_norm_relu(g, x, weight, bias, running_mean, running_var, eps, s, zp, ):
    from torch.onnx.symbolic_opset9 import relu
    out = convert_qat_bn(g, x, weight, bias,
                         running_mean, running_var, eps, s, zp)
    out = relu(g, out)
    return quantize_helper(g, out, s, zp)


@helper.parse_args('v', 'v', 'v', 'v', 'v', 'f', 'v', 'v')
def convert_quant_batch_norm(g, x, weight, bias, running_mean, running_var, eps, s, zp, ):
    out = convert_qat_bn(g, x, weight, bias,
                         running_mean, running_var, eps, s, zp)
    return quantize_helper(g, out, s, zp)


@helper.parse_args('v', 'v', 'v', 'v', 'v', 'f', 'v', 'v')
def convert_quant_batch_norm3d(g, x, weight, bias, running_mean, running_var, eps, s, zp, ):
    out = convert_qat_bn(g, x, weight, bias,
                         running_mean, running_var, eps, s, zp)
    return quantize_helper(g, out, s, zp)


@helper.parse_args('v')
def hardswish(g, self):
    return g.op('HardSwish', self)


def convert_quantized_hardswish(g, x, op_scale, op_zero_point):
    x, _, _, _ = helper.dequantize_helper(g, x)
    output = hardswish(g, x)
    return quantize_helper(g, output, op_scale, op_zero_point)


@quantized_args(True, False, False, False, False, False, False)
def convert_max_pool(g, input, kernel_size, strides, paddings, dilations, ceil_mode, dim, return_indices=False):
    assert isinstance(dim, int) and dim in [
        1, 2, 3], 'Meets invalid dim in convert_max_pool!'
    if not ceil_mode:
        if return_indices:
            try:
                from torch.onnx.symbolic_opset10 import max_pool1d_with_indices, max_pool2d_with_indices, \
                    max_pool3d_with_indices
                max_pool_func = max_pool1d_with_indices if dim == 1 else (
                    max_pool2d_with_indices if dim == 2 else max_pool3d_with_indices)
            except ImportError:  # torch 2.1
                from torch.onnx.symbolic_opset10 import _max_pool
                if dim == 1:
                    max_pool_func = _max_pool('max_pool1d_with_indices',
                                              torch.nn.modules.utils._single, 1, return_indices=True)
                elif dim == 2:
                    max_pool_func = _max_pool('max_pool2d_with_indices',
                                              torch.nn.modules.utils._pair, 2, return_indices=True)
                else:  # dim == 3
                    max_pool_func = _max_pool('max_pool3d_with_indices',
                                              torch.nn.modules.utils._triple, 3, return_indices=True)
            max_pool, indices = max_pool_func(
                g, input, kernel_size, strides, paddings, dilations, ceil_mode)
            return (max_pool, indices)
        else:
            try:
                from torch.onnx.symbolic_opset10 import max_pool1d, max_pool2d, max_pool3d
                max_pool_func = max_pool1d if dim == 1 else (
                    max_pool2d if dim == 2 else max_pool3d)
            except ImportError:  # torch 2.1
                from torch.onnx.symbolic_opset10 import _max_pool
                if dim == 1:
                    max_pool_func = _max_pool('max_pool1d', torch.nn.modules.utils._single, 1, return_indices=False)
                elif dim == 2:
                    max_pool_func = _max_pool('max_pool2d', torch.nn.modules.utils._pair, 2, return_indices=False)
                else:  # dim == 3
                    max_pool_func = _max_pool('max_pool3d', torch.nn.modules.utils._triple, 3, return_indices=False)
            max_pool = max_pool_func(
                g, input, kernel_size, strides, paddings, dilations, ceil_mode)
            return max_pool

    if not strides:
        strides = kernel_size
    kwargs = {'kernel_shape_i': kernel_size, 'pads_i': paddings * 2,
              'strides_i': strides, 'ceil_mode_i': ceil_mode, 'dilations_i': dilations}
    if return_indices:
        max_pool, indices = g.op('MaxPool', input, outputs=2, **kwargs)
    else:
        max_pool = g.op('MaxPool', input, outputs=1, **kwargs)
        indices = None

    input_shape = helper._get_tensor_sizes(input)
    slice_ends = []
    for input_size, kernel, stride, pad, dilation in zip(input_shape[2:], kernel_size, strides, paddings, dilations):
        torch_output_shape = get_torch_pool_output_shape(
            input_size, kernel, pad, pad, stride, dilation)
        onnx_output_shape = get_onnx_pool_output_shape(
            input_size, kernel, pad, pad, stride, dilation)
        if torch_output_shape > onnx_output_shape:
            FATAL(
                '[Parser]: Meets unsupported output shape of max_pool in convert_max_pool!')
        slice_ends.append(torch_output_shape)
    ends_input = g.op('Constant', value_t=torch.tensor(slice_ends))
    starts_input = g.op(
        'Constant', value_t=torch.tensor([0] * len(slice_ends)))
    axes_input = g.op('Constant', value_t=torch.tensor(
        list(range(2, len(input_shape)))))
    slice_out = g.op('Slice', max_pool, starts_input, ends_input, axes_input)

    if indices is not None:
        ndims = len(input_shape) - 2
        slice_indices = g.op('Slice', indices, starts_input,
                             ends_input, axes_input)
        _, flattened_indices = g.op(
            'MaxPool', input, outputs=2, kernel_shape_i=[1] * ndims)
        sub_indices = helper._slice_helper(g, flattened_indices,
                                           axes=list(
                                               range(2, len(input_shape))),
                                           starts=[0] * ndims, ends=[1] * ndims)
        indices = g.op('Sub', slice_indices, sub_indices)
    return (slice_out, indices) if return_indices else slice_out


@helper.parse_args('v', 'is', 'v', 'v', 'f', 'v')
def convert_layer_norm(g, x, normalized_shape, weight, bias, eps, cudnn_enable):
    axis = -len(normalized_shape)
    if helper._is_none(weight):
        weight = g.op('Constant', value_t=torch.ones(normalized_shape, dtype=torch.float32))
    if helper._is_none(bias):
        bias = g.op('Constant', value_t=torch.zeros(normalized_shape, dtype=torch.float32))
    return g.op('LayerNormalization', x, weight, bias, epsilon_f=eps, axis_i=axis)


@helper.parse_args('v', 'f', 'is', 'i', 'v')
def convert_linalg_vector_norm(
        g,
        self,
        ord,
        dim,
        keepdim,
        dtype,
):
    if dim is None and not keepdim:
        self = helper._reshape_helper(g, self, [-1])

    if ord == math.inf:
        result = g.op('ReduceMax', g.op('Abs', self),
                      axes_i=dim, keepdims_i=keepdim)
    elif ord == -math.inf:
        result = g.op('ReduceMin', g.op('Abs', self),
                      axes_i=dim, keepdims_i=keepdim)
    elif ord == 0:
        return helper._onnx_opset_unsupported_detailed(
            'linalg_vector_norm', 9, 11, 'ord=0 not supported')
    else:
        ord_op = g.op('Constant', value_t=torch.tensor(
            ord, dtype=torch.float32))
        result = helper._reducesum_helper(
            g, g.op('Pow', g.op('Abs', self), ord_op), axes_i=dim, keepdims_i=keepdim
        )
        result = g.op(
            'Pow',
            result,
            g.op(
                'Div',
                g.op('Constant', value_t=torch.tensor(1, dtype=torch.float32)),
                ord_op,
            ),
        )
    return result


@helper.parse_args('v', 'v', 'is', 'i', 'v')
def convert_linalg_norm(
        g,
        self,
        ord,
        dim,
        keepdim,
        dtype,
):
    ord_value = None
    if dim is None:
        if helper._is_none(ord):
            self = helper._reshape_helper(g, self, [-1])
            ord = g.op('Constant', value_t=torch.LongTensor([2]))
        self_dim = helper._get_tensor_rank(self)
        if self_dim is None:
            return helper._unimplemented(
                'linalg_norm', 'Input rank must be known at export time.', self
            )
        if self_dim == 1:
            ord_value = helper._parse_arg(ord, 'f')
        else:
            dim = [0, 1]
    else:
        if len(dim) == 1:
            if helper._is_none(ord):
                ord = g.op('Constant', value_t=torch.LongTensor([2]))
            ord_value = helper._parse_arg(ord, 'f')
    if ord_value:
        return convert_linalg_vector_norm(g, self, ord_value, dim, keepdim, dtype)
    return opset9.linalg_matrix_norm(g, self, ord, dim, keepdim, dtype)


@helper.parse_args('v', 'is', 'is', 'is', 'is', 'i')
def convert_max_pool1d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    return convert_max_pool(g, input, kernel_size, stride, padding, dilation, ceil_mode, 1)


@helper.parse_args('v', 'is', 'is', 'is', 'is', 'i')
def convert_max_pool2d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    return convert_max_pool(g, input, kernel_size, stride, padding, dilation, ceil_mode, 2)


@helper.parse_args('v', 'is', 'is', 'is', 'is', 'i')
def convert_max_pool3d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    return convert_max_pool(g, input, kernel_size, stride, padding, dilation, ceil_mode, 3)


@helper.parse_args('v', 'is', 'is', 'is', 'is', 'i')
def convert_max_pool2d_with_indices(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    return convert_max_pool(g, input, kernel_size, stride, padding, dilation, ceil_mode, 2, return_indices=True)


@helper.parse_args('v', 'v', 'is')
def convert_max_unpool2d(g, input, indices, output_size):
    # Convert indices from HW format to NCHW format
    input_shape = helper._get_tensor_sizes(input)
    n, c, in_h, in_w = input_shape
    if len(output_size) != 2:
        FATAL(
            '[Parser]: Meets invalid output_size of max_unpool2d in convert_max_unpool2d!')
    out_h, out_w = output_size
    offset = np.reshape(np.arange(0, n), (n, 1, 1, 1)) * c * out_h * out_w + \
        np.reshape(np.arange(0, c), (c, 1, 1)) * out_h * out_w
    offset = np.tile(offset, [1, 1, in_h, in_w])
    offset_node = g.op('Constant', value_t=torch.tensor(
        offset, dtype=torch.int64))
    indices = g.op('Add', indices, offset_node)
    output_shape = g.op('Constant', value_t=torch.tensor(
        [n, c] + output_size, dtype=torch.int64))
    # Need set output type for MaxUnpool here because there is no type inference for this op in torch.
    return g.op('MaxUnpool', input, indices, output_shape, kernel_shape_i=[1, 1]).setType(input.type())


@helper.parse_args('v', 's')
def convert_meshgrid(g, tensor_list, indexing='ij'):
    unpacked_inputs = helper._unpack_list(tensor_list)
    if len(unpacked_inputs) == 1:
        return unpacked_inputs[0]
    assert indexing in ('ij', 'xy'), 'Meets unsupported indexing %s in convert_meshgrid!' % indexing
    inputs = []
    output_shape = []
    size_nodes = []
    for idx, t in enumerate(unpacked_inputs):
        input_1d = helper._reshape_helper(g, t, [-1]) if helper._get_tensor_rank(t) != 1 else t
        inputs.append(input_1d)
        size_nodes.append(helper._reshape_helper(g, g.op('Size', t), [1]))
        t_size = helper._get_tensor_sizes(t)
        shape = None if (t_size is None or None in t_size) else int(np.prod(t_size))
        output_shape.append(shape)
    outs = g.op('custom::Meshgrid', *inputs, indexing_s=indexing, outputs=len(inputs))
    if any(shape is None for shape in output_shape):
        concat_node = g.op('Concat', *size_nodes, axis_i=0)
        new_outs = []
        for idx in range(len(inputs)):
            new_outs.append(helper._reshape_helper(g, outs[idx], concat_node))
        outs = new_outs
    else:
        output_type = unpacked_inputs[0].type().with_sizes(output_shape)
        for idx in range(len(inputs)):
            outs[idx].setType(output_type)
    return g.op('prim::ListConstruct', *outs)


@helper.parse_args('v', 'i', 'v', 'v')
def convert_quantized_cat(
        g,
        q_inputs,
        dim,
        op_scale,
        op_zero_point,
):
    unpacked_inputs = helper._unpack_list(q_inputs)
    dequantized = [
        helper.dequantize_helper(g, input)[0] for input in unpacked_inputs
    ]
    concatenated = g.op('Concat', *dequantized, axis_i=dim)
    return helper.quantize_helper(g, concatenated, op_scale, op_zero_point)


def convert_reduce_op(g, x, onnx_op, dim=None, keepdim=None, dtype=None, allow_multi_dim_support=True):
    if dim is None:
        # all-reduce path
        x_rank = helper._get_tensor_rank(x)
        dim_list = list(range(x_rank))
        return g.op(onnx_op, x, axes_i=dim_list, keepdims_i=0)
    else:
        # dim-reduce path
        desc = 'is' if allow_multi_dim_support else 'i'
        dim, keepdim = helper._get_const(
            dim, desc, 'dim'
        ), helper._get_const(keepdim, 'i', 'keepdim')
        dim_list = dim if allow_multi_dim_support else [dim]
        if dtype is not None:
            if dtype.node().kind() == 'onnx::Constant':
                dtype = helper._get_const(dtype, 'i', 'dtype')
                x = g.op('Cast', x, to_i=helper.scalar_type_to_onnx[dtype])
            else:
                WARN('[Parser]: Meets non-constant dtype in convert_reduce_op; dtype will be ignored!')
        return g.op(onnx_op, x, axes_i=dim_list, keepdims_i=keepdim)


@quantized_args(True)
def convert_reduce_mean(g, x, dim_or_dtype=None, keepdim=None, dtype=None):
    # By checking whether keepdim is set, decide which one is used(reduce_nodim or reduce_dim).
    onnx_op = 'ReduceMean'
    if keepdim is None:
        return convert_reduce_op(g, x, onnx_op)
    return convert_reduce_op(g, x, onnx_op, dim_or_dtype, keepdim, dtype, True)


@helper.parse_args('v', 'v', 'f', 'i', 'i', 'i', 'i')
def convert_roi_align(g, x, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    from torch.onnx.symbolic_opset11 import select
    # prepare batch_indices
    indices = select(g, rois, 1, g.op('Constant', value_t=torch.tensor([0], dtype=torch.int64)))
    batch_indices = g.op('Cast', helper._squeeze_helper(g, indices, [1]), to_i=torch.onnx.TensorProtoDataType.INT64)
    # prepare rois, sampling_ratio and coordinate_transformation_mode
    rois = select(g, rois, 1, g.op('Constant', value_t=torch.tensor([1, 2, 3, 4], dtype=torch.int64)))
    sampling_ratio = 0 if sampling_ratio < 0 else sampling_ratio
    coordinate_transformation_mode = 'half_pixel' if aligned else 'output_half_pixel'
    return g.op('RoiAlign', x, rois, batch_indices,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                output_height_i=pooled_height, output_width_i=pooled_width,
                sampling_ratio_i=sampling_ratio, spatial_scale_f=spatial_scale)


@quantized_args(True)
@helper.parse_args('v', 'i')
def convert_round(g, input, decimals=0):
    if decimals == 0:
        return g.op('Round', input)
    pre_mul = g.op('Mul', input, g.op(
        'Constant', value_t=torch.tensor(pow(10, decimals))))
    round_node = g.op('Round', pre_mul)
    return g.op('Mul', round_node, g.op('Constant', value_t=torch.tensor(pow(10, -1 * decimals))))


@helper.parse_args('v', 'i', 'v', 'v', 'v')
def convert_scatter(g, self, dim, index, src, reduce=None):
    if reduce is None:
        from torch.onnx.symbolic_opset11 import scatter
        return scatter(g, self, dim, index, src)
    reduce = helper._parse_arg(reduce, 's')
    assert reduce in (
        'add', 'multiply'), 'Meets invalid reduce (%s) of aten::scatter in convert_scatter' % reduce
    reduction = 'mul' if reduce == 'multiply' else 'add'
    if helper._is_value(src):
        return g.op('ScatterElements', self, index, src, axis_i=dim, reduction_s=reduction)
    return g.op('ScatterElements', self, index, opset9.expand_as(g, src, index), axis_i=dim, reduction_s=reduction)


@helper.parse_args('v')
@quantized_args(True)
def convert_sinh(g, x):
    return g.op('Sinh', x)


def convert_size(g, input, dim=None):
    from torch.onnx.symbolic_opset11 import size
    if input.node().kind() in ('prim::TupleConstruct'):
        # if input is TupleConstruct, then it could be a tuple of quantize info, which will
        # make the output of this op incorrect.
        input = list(input.node().inputs())[0]
    return size(g, input, dim)


def convert_scatter_by_slice(g, input, src, dim=0, start=None, end=None, step=1):
    input_shape = helper._get_tensor_sizes(input)
    shape_at_dim = input_shape[dim]
    dim = (dim + len(input_shape)) if dim < 0 else dim
    start = 0 if start is None else ((start + shape_at_dim) if start < 0 else start)
    end = shape_at_dim if end is None else min(end, shape_at_dim)
    step = 1 if step is None else step
    reshape_dim = [-1 if idx == dim else 1 for idx in range(len(input_shape))]
    indices_val = np.reshape(list(range(start, end, step)), reshape_dim)
    tile_reps = [1 if idx == dim else shape for idx, shape in enumerate(input_shape)]
    indices_tensor = torch.tensor(np.tile(indices_val, tile_reps), dtype=torch.int32)
    indices = g.op('Constant', value_t=indices_tensor)

    # Cast src to the same dtype of input
    input_dtype_str = input.type().scalarType()
    src_dtype_str = src.type().scalarType()
    if input_dtype_str != src_dtype_str:
        src = g.op('Cast', src, to_i=helper.cast_pytorch_to_onnx[input_dtype_str])
    return g.op('ScatterElements', input, indices, src, axis_i=dim)


@helper.parse_args('v', 'v', 'i', 'i')
@quantized_args(True, True)
def convert_select_scatter(g, input, src, dim, index):
    indices = g.op('Unsqueeze', src, g.op('Constant', value_t=torch.tensor([dim], dtype=torch.int64)))
    return convert_scatter_by_slice(g, input, indices, dim, start=index, end=index + 1)


@helper.parse_args('v', 'v', 'i', 'i', 'i', 'i')
@quantized_args(True, True)
def convert_slice_scatter(g, input, src, dim=0, start=None, end=None, step=1):
    return convert_scatter_by_slice(g, input, src, dim, start, end, step)


@helper.parse_args('v', 'v', 'i', 'i')
@quantized_args(True, False, False, False)
def convert_split(g, input, split_size_or_sizes, dim, _outputs=None):
    from torch.onnx.symbolic_opset13 import split
    return split(g, input, split_size_or_sizes, dim, _outputs)


@helper.parse_args('v')
@quantized_args(True)
def convert_t(g, self):
    rank = helper._get_tensor_rank(self)
    if rank is None or rank < 2:
        return g.op('Identity', self)
    return g.op('Transpose', self, perm_i=[1, 0])


@helper.parse_args('v', 'v', 'i')
@quantized_args(True)
def convert_tensor_split(g, x, indices_or_sections, dim=0):
    input_shape = helper._get_tensor_sizes(x)
    shape_at_dim = input_shape[dim]
    indices_rank = helper._get_tensor_rank(indices_or_sections)
    assert indices_rank is not None, 'Meets unsupported non-constant indices_or_sections in convert_tensor_split!'
    if indices_rank != 0:
        indices = helper._parse_arg(indices_or_sections, 'is')
        ends = indices + [shape_at_dim]
        starts = [0] + indices
        splits = [(end - start) for end, start in zip(ends, starts) if end != start]
    else:
        section = helper._parse_arg(indices_or_sections, 'i')
        quotient = shape_at_dim // section
        remainder = shape_at_dim % section
        if remainder == 0:
            splits = [quotient] * section
        else:
            front_split = shape_at_dim // section + 1
            rest_split = shape_at_dim // section
            splits = [front_split] * remainder + [rest_split] * (section - remainder)
    split_sizes = g.op('Constant', value_t=torch.tensor(splits, dtype=torch.int64))
    split_outs = g.op('Split', x, split_sizes, axis_i=dim, outputs=len(splits))
    return g.op('prim::ListConstruct', *split_outs)


def convert_threshold(g, x, threshold, value):
    greater = g.op('Greater', x, threshold)
    where = g.op('Where', greater, x, value)
    return where


def convert_quantized_sigmoid(g, x, op_scale, op_zero_point):
    x, _, _, _ = helper.dequantize_helper(g, x)
    output = opset9.sigmoid(g, x)
    return quantize_helper(g, output, op_scale, op_zero_point)


@helper.parse_args('v', 'is')
@quantized_args(True, False)
def convert_tile(g, input, dims):
    rank = helper._get_tensor_rank(input)
    dims_len = len(dims)
    if dims_len < rank:
        dims = [1] * (rank - dims_len) + dims
    elif dims_len > rank:
        input_shape = helper._get_tensor_sizes(input)
        new_input_shape = [1] * (dims_len - rank) + input_shape
        input = helper._reshape_helper(g, input, new_input_shape)
    return g.op('Tile', input, g.op('Constant', value_t=torch.tensor(dims, dtype=torch.int64)))


@helper.parse_args('v', 'i', 'i')
@quantized_args(True, False, False)
def convert_transpose(g, self, dim0, dim1):
    return opset9.transpose(g, self, dim0, dim1)


@helper.parse_args('v')
def convert_trunc(g, x):
    return g.op('custom::Trunc', x).setType(x.type())


def convert_quantized_leaky_relu(g, x, negative_slope, inplace, op_scale, op_zero_point):
    x, _, _, _ = helper.dequantize_helper(g, x)
    output = opset9.leaky_relu(g, x, negative_slope, inplace)
    return quantize_helper(g, output, op_scale, op_zero_point)


def convert_quantized_elu(g, x, op_scale, op_zero_point, alpha, scale, input_scale):
    x, _, x_zero_point, _ = helper.dequantize_helper(g, x)
    output = opset9.elu(g, x, alpha, scale, input_scale)
    return quantize_helper(g, output, op_scale, op_zero_point, zero_point_scalar_type=x_zero_point.type().scalarType())


@helper.parse_args('v', 'v')
@quantized_args(True, False)
def convert_view(g, self, shape):
    from torch.onnx.symbolic_opset14 import reshape
    return reshape(g, self, shape)


def convert_quantize_per_tensor(g, input, scale, zero_point, dtype):
    dtype = helper._get_const(dtype, 'i', 'dtype')
    zero_point = g.op(
        'Cast', zero_point, to_i=helper.scalar_type_to_onnx[dtype]
    )
    scale = g.op('Cast', scale, to_i=torch._C._onnx.TensorProtoDataType.FLOAT)
    return quantize_helper(g, input, scale, zero_point)


def convert_index_put(g, x, indices_list_value, values, accumulate=False):
    if helper._is_packed_list(indices_list_value):
        indices_list = helper._unpack_list(indices_list_value)
    else:
        indices_list = [indices_list_value]
    if helper.is_caffe2_aten_fallback():
        args = [x] + indices_list + [values, accumulate]
        return g.at('index_put', *args)

    accumulate = helper._parse_arg(accumulate, 'b')

    if len(indices_list) == 0:
        return values

    if len(indices_list) > 1:
        for idx_ in range(len(indices_list)):
            if indices_list[idx_].type().scalarType() == 'Bool':
                indices_list[idx_] = g.op('NonZero', indices_list[idx_])
        index = indices_list[0]

        for ind in indices_list[1:]:
            index = opset9.add(g, index, ind)
        broadcast_index_shape = g.op('Shape', index)
        indices_list = [
            helper._unsqueeze_helper(
                g, opset9.expand(g, ind, broadcast_index_shape, None), [-1]
            )
            for ind in indices_list
        ]
        index = g.op('Concat', *indices_list, axis_i=-1)
    else:
        index = indices_list[0]
        bool_inp = index
        if bool_inp.type() is not None and bool_inp.type().scalarType() == 'Bool':
            rank = helper._get_tensor_rank(values)
            if rank is not None and rank == 0:
                return opset9.masked_fill(g, x, bool_inp, values)
            return masked_scatter(g, x, bool_inp, values)
        broadcast_index_shape = g.op('Shape', index)
        index = helper._unsqueeze_helper(g, index, [-1])
    import sys
    sub_data_shape = helper._slice_helper(
        g, g.op('Shape', x), axes=[0], starts=[len(indices_list)], ends=[sys.maxsize]
    )
    values_shape = g.op('Concat', broadcast_index_shape,
                        sub_data_shape, axis_i=0)
    # Check if values is a singular value and expand accordingly
    rank = helper._get_tensor_rank(values)
    inp_rank = helper._get_tensor_rank(x)
    inp_shape = helper._get_tensor_sizes(x)
    value_shape = helper._get_tensor_sizes(values)
    if rank is not None \
            and rank < inp_rank \
            and all((shape is not None for shape in inp_shape)) \
            and all((shape is not None for shape in value_shape)):
        try:
            values = opset9.expand(g, values, values_shape, None)
        except:
            pass
    values = helper._reshape_helper(g, values, values_shape)

    dtype = x.type().scalarType()
    if dtype is not None and dtype != values.type().scalarType():
        values = g.op('Cast', values, to_i=helper.cast_pytorch_to_onnx[dtype])

    if accumulate:
        zeros = g.op(
            'ConstantOfShape',
            g.op('Shape', x),
            value_t=torch.tensor(
                [0], dtype=helper.pytorch_name_to_type[dtype]),
        )
        result = g.op('ScatterND', zeros, index, values)
        from torch.onnx import symbolic_opset11 as opset11
        result = opset11.add(g, x, result)
    else:
        result = g.op('ScatterND', x, index, values)

    return result


def convert_index_add(g, x, dim, index, other, alpha=None):
    def _has_duplicates(seq):
        return len(seq) != len(set(seq))

    index_duplicates = True
    try:
        index_value = helper._maybe_get_const(index, 'is')
        index_duplicates = _has_duplicates(index_value)
    except:
        pass

    import sys
    dim = helper._maybe_get_const(dim, 'i')
    if dim is None:
        ERROR('ONNX export does NOT support exporting index_add_() function with unknown dim value.')

    x_dim_rank = helper._get_tensor_rank(x)
    other_dim_rank = helper._get_tensor_rank(other)

    if x_dim_rank is None or other_dim_rank is None:
        ERROR(
            'ONNX export does NOT support exporting index_add_() function while, the rank of x tensor or tensor to be added is unknown.')

    if dim < 0:
        dim = dim + x_dim_rank

    if other_dim_rank != x_dim_rank:
        delta = x_dim_rank - other_dim_rank
        for i in range(delta):
            other = helper._unsqueeze_helper(
                g, other, [helper._get_tensor_rank(other)]
            )

    other_dim_size = helper._get_tensor_dim_size(other, dim)
    x_dim_size = helper._get_tensor_dim_size(x, dim)

    if (other_dim_size is not None) and (x_dim_size is not None):
        if other_dim_size > x_dim_size:
            ERROR(
                'ONNX export does not support exporting index_add_() function with duplicated values in index parameter yet.')

    new_shape_axes = list(range(x_dim_rank))
    new_shape_starts = [0 for i in range(x_dim_rank)]
    new_shape_ends = [sys.maxsize if (
        i != dim) else 1 for i in range(x_dim_rank)]

    new_shape = helper._slice_helper(
        g, x, axes=new_shape_axes, starts=new_shape_starts, ends=new_shape_ends
    )

    for i in range(dim):
        index = helper._unsqueeze_helper(g, index, [0])

    for i in range(x_dim_rank - dim - 1):
        index = helper._unsqueeze_helper(
            g, index, [helper._get_tensor_rank(index)]
        )

    scatter_add_indices = opset9.expand_as(g, index, other)

    if alpha and helper._scalar(helper._maybe_get_scalar(alpha)) != 1:
        other = g.op('Neg', other)

    if index_duplicates is False:
        return opset9.scatter_add(g, x, dim, scatter_add_indices, other)
    else:
        return scatter_helper(g, x, dim, scatter_add_indices, other, 'add')


@helper.parse_args('v', 'i', 'v', 'v')
def convert_index_copy(g, self, dim, index, source):
    from torch.onnx import symbolic_opset11 as opset11
    input_rank = helper._get_tensor_rank(self)
    if dim < 0:
        dim = dim + input_rank
    dim_node = g.op('Constant', value_t=torch.tensor(dim, dtype=torch.int64))
    return opset11.index_copy(g, self, dim_node, index, source)


@helper.parse_args('v', 'i', 'v', 'v', 's')
def scatter_helper(g, self, dim, index, src, reduction):
    if helper.is_caffe2_aten_fallback():
        return g.at('scatter', self, dim, index, src, overload_name='src')
    src_type = src.type().scalarType()
    src = helper._maybe_get_scalar(src)
    if helper._is_value(src):
        return g.op('ScatterElements', self, index, src, axis_i=dim, reduction_s=reduction)
    else:
        # TODO: support more reduction type if necessary.

        # Check if scalar 'src' has same type as self (PyTorch allows different
        # type for scalar src (but not when src is tensor)). If not, insert Cast node.
        input_dtype_str = self.type().scalarType()
        if input_dtype_str != src_type:
            src = g.op(
                'Cast',
                src,
                to_i=helper.cast_pytorch_to_onnx[input_dtype_str]
            )
        return g.op(
            'ScatterElements', self, index, opset9.expand_as(g, src, index), axis_i=dim
        )


@helper.parse_args('v', 'i', 'v', 'v', 's', 'b')
def convert_index_reduce(g, x, dim, index, other, reduction, include_self):
    def _has_duplicates(seq):
        return len(seq) != len(set(seq))

    def _get_dtype_min_max_value(torch_dtype, min_or_max):
        ret = None
        assert isinstance(torch_dtype, torch.dtype), 'Expect torch_dtype is torch dtype, but got %s' % str(torch_dtype)
        assert isinstance(min_or_max, str) and min_or_max in ('min', 'max'), 'Expect min_or_max is string min or max'
        if 'int' in str(torch_dtype):
            ret = torch.iinfo(torch_dtype).min if min_or_max == 'min' else torch.iinfo(torch_dtype).max
        elif 'float' in str(torch_dtype):
            ret = torch.finfo(torch_dtype).min if min_or_max == 'min' else torch.finfo(torch_dtype).max
        else:
            ERROR('Meets unsupported torch_dtype(%s) in _get_dtype_min_max_value!' % str(torch_dtype))
        return ret

    reduction_map = {'amin': 'min', 'amax': 'max', 'prod': 'mul', 'mean': 'add'}
    assert reduction in reduction_map, 'Meets invalid reduction(%s) in convert_index_reduce!' % reduction
    onnx_reduction = reduction_map[reduction]

    x_dim_rank = helper._get_tensor_rank(x)
    dim = (dim + x_dim_rank) if dim < 0 else dim
    dim_node = g.op('Constant', value_t=torch.tensor(dim, dtype=torch.int64))
    _, scatter_indices = helper._index_fill_reshape_helper(g, x, dim_node, index)

    other_shape = helper._get_tensor_sizes(other)
    other_dtype = helper.pytorch_name_to_type[other.type().scalarType()]

    if include_self:
        ret = g.op('ScatterElements', x, scatter_indices, other, axis_i=dim, reduction_s=onnx_reduction)
        if reduction == 'mean':
            # get the counts of duplicate indices
            x_shape = helper._get_tensor_sizes(x)
            one_x = g.op('Constant', value_t=torch.full(x_shape, 1, dtype=other_dtype))
            one_value = g.op('Constant', value_t=torch.full(other_shape, 1, dtype=other_dtype))
            index_counts = g.op('ScatterElements', one_x, scatter_indices, one_value, axis_i=dim, reduction_s='add')
            ret = g.op('Div', ret, index_counts)
        return ret

    if reduction == 'mean':
        # set target in x to 0
        zero_value = g.op('Constant', value_t=torch.full(other_shape, 0, dtype=other_dtype))
        x = g.op('ScatterElements', x, scatter_indices, zero_value, axis_i=dim, reduction_s='none')
        # get the counts of duplicate indices(the minumum count is 1)
        x_shape = helper._get_tensor_sizes(x)
        zero_x = g.op('Constant', value_t=torch.full(x_shape, 0, dtype=other_dtype))
        one_value = g.op('Constant', value_t=torch.full(other_shape, 1, dtype=other_dtype))
        index_counts = g.op('ScatterElements', zero_x, scatter_indices, one_value, axis_i=dim, reduction_s='add')
        one_x = g.op('Constant', value_t=torch.full(x_shape, 1, dtype=other_dtype))
        index_counts = g.op('Max', one_x, index_counts)
        # get the sum of targets
        x = g.op('ScatterElements', x, scatter_indices, other, axis_i=dim, reduction_s=onnx_reduction)
        return g.op('Div', x, index_counts)

    if reduction == 'amin':
        max_value = _get_dtype_min_max_value(other_dtype, 'max')
        src_max = g.op('Constant', value_t=torch.full(other_shape, max_value, dtype=other_dtype))
        x = g.op('ScatterElements', x, scatter_indices, src_max, axis_i=dim, reduction_s='none')
    elif reduction == 'amax':
        min_value = _get_dtype_min_max_value(other_dtype, 'min')
        src_min = g.op('Constant', value_t=torch.full(other_shape, min_value, dtype=other_dtype))
        x = g.op('ScatterElements', x, scatter_indices, src_min, axis_i=dim, reduction_s='none')
    elif reduction == 'prod':
        one_value = g.op('Constant', value_t=torch.full(other_shape, 1, dtype=other_dtype))
        x = g.op('ScatterElements', x, scatter_indices, one_value, axis_i=dim, reduction_s='none')
    return g.op('ScatterElements', x, scatter_indices, other, axis_i=dim, reduction_s=onnx_reduction)


@quantized_args(True, False)
@helper.parse_args('v', 'b')
def convert_quantized_relu6(g, x, inplace):
    assert inplace is False
    const_min = g.op('Constant', value_t=torch.tensor(0, dtype=torch.int32))
    const_max = g.op('Constant', value_t=torch.tensor(6, dtype=torch.int32))
    return g.op('Clip', x, const_min, const_max)


def _index_fill_reshape_helper(g, self, dim, index):
    from torch.onnx.symbolic_opset9 import expand

    if self.type().dim() is None:
        return _unimplemented('index_fill input rank not accessible')
    self_dim = self.type().dim()
    dim_value = helper._parse_arg(dim, 'i')
    if dim_value < 0:
        self_dim_rank = helper._get_tensor_rank(self)
        dim_value = dim_value + self_dim_rank
    unsqueezed_index = helper._unsqueeze_helper(
        g, index, [i for i in range(self_dim) if i != dim_value]
    )
    expanded_index_shape = convert_scatter(
        g, g.op('Shape', self), 0, helper._unsqueeze_helper(
            g, dim, [0]), g.op('Shape', index)
    )
    expanded_index = expand(g, unsqueezed_index, expanded_index_shape, None)
    return expanded_index_shape, expanded_index


def convert_index_fill(g, self, dim, index, value):
    dim_value = helper._parse_arg(dim, 'i')
    self_dim_rank = helper._get_tensor_rank(self)

    if self_dim_rank is None:
        ERROR(
            'ONNX export does NOT support exporting index_add_() function while, the rank of self tensor or tensor to be added is unknown.')

    if helper.is_caffe2_aten_fallback():
        return g.at(
            'index_fill',
            self,
            index,
            value,
            overload_name='int_Scalar',
            dim_i=dim_value,
        )

    expanded_index_shape, expanded_index = _index_fill_reshape_helper(
        g, self, dim, index
    )
    value = helper._maybe_get_scalar(value)
    try:
        value = helper._if_scalar_type_as(g, value, self)
    except TypeError:  # torch 2.1
        value = helper._if_scalar_type_as(value, self)
    expanded_value = opset9.expand(g, value, expanded_index_shape, None)

    return convert_scatter(g, self, dim, expanded_index, expanded_value)


def convert_atan2(g, self, other):
    slope = g.op('Div', self, other)
    atan = g.op('Atan', slope)
    const_zero = g.op('Constant', value_t=torch.tensor(0))
    const_pi = g.op('Constant', value_t=torch.tensor(math.pi))

    condition_second_or_third_quadrant = g.op('Greater', self, const_zero)
    second_third_quadrant = g.op(
        'Where',
        condition_second_or_third_quadrant,
        g.op('Add', atan, const_pi),
        g.op('Sub', atan, const_pi),
    )

    condition_14_or_23_quadrant = g.op('Less', other, const_zero)
    result = g.op('Where', condition_14_or_23_quadrant,
                  second_third_quadrant, atan)

    return result


def convert_torch_to_onnx(model_path, params):
    def _export_to_onnx(model,
                        input_tensors,
                        onnx_model_path,
                        input_names,
                        output_names,
                        onnx_opset_version=None):
        custom_opsets = {'opset_11': 11}
        if onnx_opset_version is not None:
            if onnx_opset_version < 18:
                custom_opsets.update({'opset_18': 18, 'opset_19': 19})
            elif onnx_opset_version == 18:
                custom_opsets.update({'opset_19': 19})
        # Note: Use operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        # or torch.onnx.OperatorExportTypes.ONNX_ATEN for debug if export fails.
        # The failure could be caused by unexpected input shapes.
        torch.onnx.export(model,
                          input_tensors,
                          onnx_model_path,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=onnx_opset_version,
                          training=torch._C._onnx.TrainingMode.PRESERVE,
                          custom_opsets=custom_opsets)
        return

    def _flatten_type(torch_type):
        output_types = []
        if isinstance(torch_type, torch._C.TupleType):
            for nested_out in torch_type.elements():
                output_types.extend(_flatten_type(nested_out))
        else:
            output_types.append(torch_type)
        return output_types

    # Check whether inputs and shapes are provided. They must be provided because we cannot get input
    # shapes info from the provided model.
    if not params['input_shapes']:
        FATAL('[Parser]: Input names and shapes must be provided in config file for TorchScript model!')

    # Load torchvision because some models in torchvision need it. If cannot import but model needs it,
    # error will be raised after torch.jit.load.
    try:
        import torchvision
        torchvision_version = torchvision.__version__.split('+', 1)[0]
    except Exception as e:
        DEBUG('[Parser]: Fail to import torchvision because %s!' % str(e))
        torchvision_version = None

    # Load TorchScript/non-TorchScript model
    is_torch_script_model = False
    try:
        try:
            model = torch.jit.load(model_path)
            is_torch_script_model = True
        except RuntimeError:
            model = torch.load(model_path)
            model.eval()
    except Exception as e:
        FATAL('[Parser]: Fail to load model (%s) because %s!' %
              (model_path, str(e)))

    # Get onnx opset version to target
    # From https://onnxruntime.ai/docs/reference/compatibility.html,
    # for onnx version 1.x, onnx opset version=x+5
    onnx_version = str(get_version(onnx)).split('.')
    onnx_opset_version = (
        int(onnx_version[-1]) + 5) if int(onnx_version[0]) == 1 else None
    torch_version = str(torch.onnx.producer_version)
    if onnx_opset_version is not None:
        default_onnx_main_opset = None
        default_onnx_stable_opsets = []
        try:
            if torch_version.startswith('1.11'):
                default_onnx_main_opset = helper._onnx_main_opset
                default_onnx_stable_opsets = helper._onnx_stable_opsets
            elif torch_version >= '1.13.0':
                import torch.onnx._constants as Constant
                default_onnx_main_opset = Constant.ONNX_DEFAULT_OPSET
                default_onnx_stable_opsets = list(range(Constant.ONNX_MIN_OPSET, Constant.ONNX_MAX_OPSET + 1))
            elif torch_version >= '1.12.0':
                import torch.onnx._constants as Constant
                default_onnx_main_opset = Constant.onnx_main_opset
                default_onnx_stable_opsets = Constant.onnx_stable_opsets
        except Exception as e:
            DEBUG(
                '[Parser]: Fail to get default onnx opset version because %s' % str(e))
        if default_onnx_main_opset is None:
            onnx_opset_version = None
        elif onnx_opset_version >= default_onnx_main_opset or onnx_opset_version not in default_onnx_stable_opsets:
            onnx_opset_version = default_onnx_main_opset
    if onnx_opset_version is None:
        onnx_opset_version = 9
    global ONNX_OPSET_VERSION, CUSTOM_OPSET_18, CUSTOM_OPSET_19
    ONNX_OPSET_VERSION = onnx_opset_version
    CUSTOM_OPSET_18 = '' if onnx_opset_version >= 18 else CUSTOM_OPSET_18
    CUSTOM_OPSET_19 = '' if onnx_opset_version >= 19 else CUSTOM_OPSET_19
    if onnx_opset_version < 17:
        WARN('[Parser]: Default onnx opset version (%d) is lower than 17, which may cause some ops failed to convert!' %
             onnx_opset_version)
    else:
        DEBUG('[Parser]: Will convert to onnx opset version (%d)!' %
              onnx_opset_version)

    # Convert torch op to non-custom onnx op
    if torch_version < '2.0.1':
        # The issue of argmax/argmin is fixed in torch 2.0.1.
        # Refer to https://github.com/pytorch/pytorch/pull/79503
        torch.onnx.register_custom_op_symbolic(
            'aten::argmax', convert_argmax, onnx_opset_version)
        torch.onnx.register_custom_op_symbolic(
            'aten::argmin', convert_argmin, onnx_opset_version)
        # The alpha issue of add/sub/rsub is fixed in torch 2.0.1.
        # Refer to https://github.com/pytorch/pytorch/pull/81736
        torch.onnx.register_custom_op_symbolic(
            'aten::add', convert_add, onnx_opset_version)
        torch.onnx.register_custom_op_symbolic(
            'aten::rsub', convert_rsub, onnx_opset_version)
        torch.onnx.register_custom_op_symbolic(
            'aten::sub', convert_sub, onnx_opset_version)
        # The issue of transpose with 0d/1d input is fixed in torch 2.0.1.
        # Refer to https://github.com/pytorch/pytorch/pull/86182
        torch.onnx.register_custom_op_symbolic(
            'aten::t', convert_t, onnx_opset_version)
    if torch_version < '2.1.0':
        # The issue of training is fixed in latest torch.
        # Refer to https://github.com/pytorch/pytorch/pull/86745
        if not hasattr(model, 'training'):
            model.training = False
        # The issue of string padding is fixed in latest torch.
        # Refer to https://github.com/pytorch/pytorch/pull/89107
        for conv_op in ('aten::conv1d', 'aten::conv2d', 'aten::conv3d'):
            torch.onnx.register_custom_op_symbolic(
                conv_op, convert_conv, onnx_opset_version)
        # The issue of logical_not is fixed in latest torch.
        # Refer to https://github.com/pytorch/pytorch/pull/96315
        torch.onnx.register_custom_op_symbolic(
            'aten::logical_not', convert_logical_not, onnx_opset_version)
        torch.onnx.register_custom_op_symbolic(
            'aten::atan2', convert_atan2, onnx_opset_version)

    if onnx_opset_version >= 16:
        torch.onnx.register_custom_op_symbolic(
            'aten::scatter', convert_scatter, onnx_opset_version)
    # if onnx_opset_version < 18:  # not yet supported in torch 2.1
    # The lowest version of onnx BitwiseAnd/Not/Or/Xor is 18
    # (onnx_opset_version is 16 for torch 1.12.0).
    torch.onnx.register_custom_op_symbolic(
        'aten::bitwise_and', convert_bitwise_and, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::bitwise_not', convert_bitwise_not, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::bitwise_or', convert_bitwise_or, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::bitwise_xor', convert_bitwise_xor, onnx_opset_version)
    if onnx_opset_version >= 17:
        # The lowest version of onnx LayerNormalization is 17.
        torch.onnx.register_custom_op_symbolic(
            'aten::layer_norm', convert_layer_norm, onnx_opset_version)
    if onnx_opset_version < 18:
        # The lowest version of onnx Col2Im is 18.
        torch.onnx.register_custom_op_symbolic(
            'aten::col2im', convert_col2im, onnx_opset_version)
    # if onnx_opset_version < 19:  # not yet supported in torch 2.1
    torch.onnx.register_custom_op_symbolic(
        'torchvision::deform_conv2d', convert_deform_conv, onnx_opset_version)

    torch.onnx.register_custom_op_symbolic(
        'aten::acosh', convert_acosh, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::addbmm', convert_addbmm, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::addcdiv', convert_addcdiv, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::addcmul', convert_addcmul, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::addmv', convert_addmv, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::addr', convert_addr, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::asinh', convert_asinh, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::atanh', convert_atanh, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::avg_pool1d', convert_avg_pool1d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::avg_pool2d', convert_avg_pool2d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::avg_pool3d', convert_avg_pool3d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::cosh', convert_cosh, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::index_add', convert_index_add, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::index_copy', convert_index_copy, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::index_fill', convert_index_fill, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::index_put', convert_index_put, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::index_reduce', convert_index_reduce, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::bitwise_left_shift', convert_bitshift_left, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::bitwise_right_shift', convert_bitshift_right, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::chunk', convert_chunk, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::div', convert_div, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::equal', convert_equal, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::erfc', convert_erfc, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::flatten', convert_flatten, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::fliplr', convert_fliplr, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::flipud', convert_flipud, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::gru_cell', convert_gru_cell, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::hardshrink', convert_hardshrink, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::linalg_norm', convert_linalg_norm, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::logaddexp', convert_logaddexp, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::max_pool1d', convert_max_pool1d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::max_pool2d', convert_max_pool2d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::max_pool3d', convert_max_pool3d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::max_pool2d_with_indices', convert_max_pool2d_with_indices, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::max_unpool2d', convert_max_unpool2d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::mean', convert_reduce_mean, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::meshgrid', convert_meshgrid, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::round', convert_round, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::select_scatter', convert_select_scatter, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::sinh', convert_sinh, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::size', convert_size, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::slice_scatter', convert_slice_scatter, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::softshrink', convert_softshrink, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::split', convert_split, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::dsplit', convert_dsplit, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::hsplit', convert_hsplit, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::tensor_split', convert_tensor_split, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::threshold', convert_threshold, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::tile', convert_tile, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::transpose', convert_transpose, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::trunc', convert_trunc, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::view', convert_view, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::__iand_', convert_iand, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'prim::ConstantChunk', convert_constant_chunk, onnx_opset_version)

    # for quantized Ops
    torch.onnx.register_custom_op_symbolic(
        'quantized::add_relu', convert_quantized_add_relu, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::batch_norm2d', convert_quant_batch_norm, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::batch_norm3d', convert_quant_batch_norm3d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::batch_norm2d_relu', convert_quant_batch_norm_relu, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::batch_norm3d_relu', convert_quant_batch_norm_relu_3d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::cat', convert_quantized_cat, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::elu', convert_quantized_elu, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::hardswish', convert_quantized_hardswish, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::leaky_relu', convert_quantized_leaky_relu, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::relu6', convert_quantized_relu6, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::sigmoid', convert_quantized_sigmoid, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::quantize_per_tensor', convert_quantize_per_tensor, onnx_opset_version)

    if is_torch_script_model:
        # Only convert prim::DictConstruct to Identity when it's output node.
        dict_nodes = model.graph.findAllNodes('prim::DictConstruct')
        model_output_names = [out.debugName() for out in model.graph.outputs()]
        if dict_nodes and all(node.output().debugName() in model_output_names for node in dict_nodes):
            torch.onnx.register_custom_op_symbolic(
                'prim::DictConstruct', convert_dict_construct, onnx_opset_version)

    # Convert torch op to custom onnx op
    torch.onnx.register_custom_op_symbolic(
        'aten::adaptive_avg_pool2d', convert_adaptive_avg_pool2d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::adaptive_max_pool2d', convert_adaptive_max_pool2d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::channel_shuffle', convert_channel_shuffle, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::cumprod', convert_cumprod, onnx_opset_version)

    # Convert torchvision op
    if torchvision_version is not None:
        if torchvision_version < '0.15.0':
            # The issue of align=True of RoiAlign has been fixed since torchvision 0.15.0.
            # Refer to https://github.com/pytorch/vision/pull/6685
            torch.onnx.register_custom_op_symbolic(
                'torchvision::roi_align', convert_roi_align, onnx_opset_version)

    # Get input_tensors and input_names
    input_names = []
    tensor_list = []
    input_info_dict = copy.deepcopy(params['input_shapes'])
    input_dtype = params['input_dtype']
    for idx, (input_name, input_shape) in enumerate(input_info_dict.items()):
        # Starting with numbers is not legal in pytorch
        if len(input_name) >= 1 and input_name[0].isdigit():
            new_input_name = 'input_' + input_name
            WARN('[Parser]: Input name %s is invalid; rename it to %s!' %
                 (input_name, new_input_name))
            params['input_shapes'].pop(input_name)
            params['input_shapes'][new_input_name] = input_shape
            input_name = new_input_name
        input_names.append(input_name)
        assert len(
            input_dtype) > idx, 'Meets invalid input_dtype in convert_torch_to_onnx'
        try:
            tensor_dtype = getattr(torch, input_dtype[idx])
            INFO('[Parser]: Input dtype of input %s is set to %s!' %
                 (input_name, input_dtype[idx]))
        except Exception as e:
            tensor_dtype = torch.float32
            WARN('[Parser]: Input dtype %s is changed to float32 because %s' %
                 (input_dtype[idx], str(e)))
        if 'float' in str(tensor_dtype):
            tensor = torch.randn(input_shape, dtype=tensor_dtype)
        else:
            tensor = torch.zeros(input_shape, dtype=tensor_dtype)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        tensor_list.append(tensor)

    jit_model = model if is_torch_script_model else torch.jit.freeze(
        torch.jit.script(model))
    input_tensors = ()
    input_index = 0
    for inp in jit_model.graph.inputs():
        tensors, input_index = get_tuple_from_tensor_type(
            inp.type(), tensor_list, input_index)
        if len(tensors) > 0:
            input_tensors += tensors

    # Get output_names. When the output is a tuple, it's actually multiple outputs constructed in that tuple.
    output_names = []
    for out_idx, out in enumerate(jit_model.graph.outputs()):
        out_name = out.debugName() + '_' + str(out_idx) + '_'
        if isinstance(out.type(), torch._C.DictType):
            inputs_num = len([inp for inp in out.node().inputs()])
            outputs_num = inputs_num // 2
        else:
            outputs_num = len(_flatten_type(out.type()))
        output_names.extend([out_name + str(idx)
                             for idx in range(outputs_num)])
    for idx, output_name in enumerate(output_names):
        if output_name[0].isdigit():
            output_names[idx] = 'output_' + output_name

    # Get the file name of the onnx model to be exported
    onnx_model_path = os.path.join(params.get('output_dir', './'),
                                   os.path.basename(model_path) + '.onnx')
    INFO('[Parser]: Convert torch model (%s) to onnx model...' % model_path)
    if torch.cuda.is_available():
        try:
            parallel_model = nn.DataParallel(model)
            _export_to_onnx(parallel_model.module, input_tensors, onnx_model_path,
                            input_names, output_names, onnx_opset_version)
        except Exception as e:
            FATAL('[Parser]: Fail to convert model (%s) to onnx because %s' %
                  (model_path, str(e)))

    else:

        # Call torch.onnx.export to convert TorchScript model to onnx model
        exit_code = 1
        try:
            # Fix hangs issue by set_num_threads if multiprocessing is used.
            # Refer to https://github.com/pytorch/pytorch/issues/36191
            torch.set_num_threads(1)
            # # Uncomment the following line to debug this code and torch.onnx.export:
            # _export_to_onnx(model, input_tensors, onnx_model_path, input_names, output_names, onnx_opset_version)
            process = Process(target=_export_to_onnx, args=(model,
                                                            input_tensors,
                                                            onnx_model_path,
                                                            input_names,
                                                            output_names,
                                                            onnx_opset_version))
            process.start()
            process.join()
            exit_code = process.exitcode
            try:
                process.close()
            except Exception as e:
                DEBUG('[Parser]: Fail to close process because %s' % str(e))
        except Exception as e:
            FATAL('[Parser]: Fail to convert model (%s) to onnx because %s' %
                  (model_path, str(e)))

        if exit_code != 0:
            FATAL(
                '[Parser]: Fail to convert model (%s) to onnx! Suggest to set env var PYTORCH_JIT_LOG_LEVEL=onnx for debug!' % model_path)

    INFO('[Parser]: Torch model has been converted to onnx model (%s) with opset version (%d)!' %
         (onnx_model_path, 'default' if onnx_opset_version is None else onnx_opset_version))

    # Update params
    updated_params = copy.deepcopy(params)
    updated_params.update({'input_model': onnx_model_path,
                           'original_input_model': model_path,
                           'input_names': input_names,
                           'input_shapes': params['input_shapes'],
                           'output_names': [],
                           'output_tensor_names': output_names,
                           'model_type': 'torch'})

    # Return onnx model path and updated params
    return onnx_model_path, updated_params
