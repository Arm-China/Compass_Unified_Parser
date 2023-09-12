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


def convert_add_sub(g, input, other, alpha, op_type):
    if alpha and not FLOAT_EQUAL(helper._maybe_get_const(alpha, 'f'), 1):
        other = g.op('Mul', other, alpha)
    return g.op(op_type, input, other)


@helper.parse_args('v', 'v', 'v')
def convert_add(g, input, other, alpha=None):
    return convert_add_sub(g, input, other, alpha, 'Add')


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
    return g.op('opset_18::BitwiseAnd', input, other)


@helper.parse_args('v')
def convert_bitwise_not(g, input):
    return g.op('opset_18::BitwiseNot', input)


@helper.parse_args('v', 'v')
def convert_bitwise_or(g, input, other):
    return g.op('opset_18::BitwiseOr', input, other)


@helper.parse_args('v', 'v')
def convert_bitwise_xor(g, input, other):
    return g.op('opset_18::BitwiseXor', input, other)


@helper.parse_args('v', 'i')
@quantized_args(True, False)
def convert_channel_shuffle(g, input, groups):
    return g.op('custom::ChannelShuffle', input, group_i=groups)


@helper.parse_args('v', 'i', 'i')
@quantized_args(True, False, False)
def convert_chunk(g, input, chunks, dim):
    from torch.onnx.symbolic_opset13 import split
    input_shape = helper._get_tensor_sizes(input)
    if input_shape is None or None in input_shape:
        from torch.onnx.symbolic_opset11 import Prim
        return Prim.ConstantChunk(g, input, chunks, dim)
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
    from torch.onnx.symbolic_opset11 import Prim
    return Prim.ConstantChunk(g, input, chunks, dim)


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
    out = g.op('opset_18::Col2Im', input, output_size, kernel_size,
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


@helper.parse_args('v', 'i', 'i')
def convert_cumprod(g, input, dim, dtype):
    if dtype is not None:
        input = g.op('Cast', input, to_i=helper.scalar_type_to_onnx[dtype])
    return g.op('custom::CumProd', input, axis_i=dim)


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
    return g.op('opset_19::DeformConv', input, weight, offset, bias, mask,
                dilations_i=[dilation_h, dilation_w], kernel_shape_i=kernel_shape,
                pads_i=pads, strides_i=[stride_h, stride_w],
                group_i=n_weight_grps, offset_group_i=n_offset_grps)


@helper.parse_args('s', 'v', 's', 'v')
def convert_dict_construct(g, key_0, value_0, key_1=None, value_1=None):
    keys = ', '.join([key_0] + ([] if key_1 is None else [key_1]))
    WARN('[Parser]: prim::DictConstruct is unsupported and is changed to return a tensor or a list of tensors(key(s): %s) instead!' % keys)
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
    if input_rank == 0:
        return helper._reshape_helper(g, input, [1])
    if input_rank == 1:
        return g.op('Identity', input)
    start_dim = (start_dim + input_rank) if start_dim < 0 else start_dim
    end_dim = (end_dim + input_rank) if end_dim < 0 else end_dim
    return helper._flatten_helper(g, input, start_dim, end_dim, input_rank)


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
        from torch.onnx.symbolic_opset11 import avg_pool1d, avg_pool2d, avg_pool3d
        avg_pool_func = avg_pool1d if dim == 1 else (
            avg_pool2d if dim == 2 else avg_pool3d)
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
    for idx, (input_size, kernel, stride, pad) in enumerate(zip(spatial_input_shape, kernel_size, strides, adjusted_padding)):
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
def convert_quant_batch_norm_relu_3d(g, input, weight, bias, running_mean, running_var, eps, s, zp,):
    from torch.onnx.symbolic_opset9 import relu
    out = convert_qat_bn(g, input, weight, bias,
                         running_mean, running_var, eps, s, zp)
    out = relu(g, out)
    return quantize_helper(g, out, s, zp)


@helper.parse_args('v', 'v', 'v', 'v', 'v', 'f', 'v', 'v')
def convert_quant_batch_norm_relu(g, input, weight, bias, running_mean, running_var, eps, s, zp,):
    from torch.onnx.symbolic_opset9 import relu
    out = convert_qat_bn(g, input, weight, bias,
                         running_mean, running_var, eps, s, zp)
    out = relu(g, out)
    return quantize_helper(g, out, s, zp)


@helper.parse_args('v', 'v', 'v', 'v', 'v', 'f', 'v', 'v')
def convert_quant_batch_norm(g, input, weight, bias, running_mean, running_var, eps, s, zp,):
    out = convert_qat_bn(g, input, weight, bias,
                         running_mean, running_var, eps, s, zp)
    return quantize_helper(g, out, s, zp)


@helper.parse_args('v', 'v', 'v', 'v', 'v', 'f', 'v', 'v')
def convert_quant_batch_norm3d(g, input, weight, bias, running_mean, running_var, eps, s, zp,):
    out = convert_qat_bn(g, input, weight, bias,
                         running_mean, running_var, eps, s, zp)
    return quantize_helper(g, out, s, zp)


@quantized_args(True, False, False, False, False, False, False)
def convert_max_pool(g, input, kernel_size, strides, paddings, dilations, ceil_mode, dim, return_indices=False):
    assert isinstance(dim, int) and dim in [
        1, 2, 3], 'Meets invalid dim in convert_max_pool!'
    if not ceil_mode:
        if return_indices:
            from torch.onnx.symbolic_opset10 import max_pool1d_with_indices, max_pool2d_with_indices, max_pool3d_with_indices
            max_pool_func = max_pool1d_with_indices if dim == 1 else (
                max_pool2d_with_indices if dim == 2 else max_pool3d_with_indices)
            max_pool, indices = max_pool_func(
                g, input, kernel_size, strides, paddings, dilations, ceil_mode)
            return (max_pool, indices)
        else:
            from torch.onnx.symbolic_opset10 import max_pool1d, max_pool2d, max_pool3d
            max_pool_func = max_pool1d if dim == 1 else (
                max_pool2d if dim == 2 else max_pool3d)
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


@quantized_args(True)
def convert_reduce_mean(g, input, dim=None, keepdim=None, allow_multi_dim_support=True):
    if dim is None:
        # all-reduce path
        return helper._handle_reduce_dim_none(g, self, 'ReduceMean')
    else:
        # dim-reduce path
        desc = 'is' if allow_multi_dim_support else 'i'
        dim, keepdim = helper._get_const(
            dim, desc, 'dim'
        ), helper._get_const(keepdim, 'i', 'keepdim')
        dim_list = dim if allow_multi_dim_support else [dim]
        return g.op('ReduceMean', input, axes_i=dim_list, keepdims_i=keepdim)


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
    return convert_scatter_by_slice(g, input, indices, dim, start=index, end=index+1)


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
    input_dtype_str = x.type().scalarType()
    cast1 = g.op('Cast', x, to_i=torch._C._onnx.TensorProtoDataType.INT32)
    return g.op('Cast', cast1, to_i=helper.cast_pytorch_to_onnx[input_dtype_str])


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


@quantized_args(True, False)
@helper.parse_args('v', 'b')
def convert_quantized_relu6(g, x, inplace):
    assert inplace is False
    const_min = g.op('Constant', value_t=torch.tensor(0, dtype=torch.int32))
    const_max = g.op('Constant', value_t=torch.tensor(6, dtype=torch.int32))
    return g.op('Clip', x, const_min, const_max)


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
                        opset_version=None):
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
                          custom_opsets={'opset_11': 11, 'opset_18': 18, 'opset_19': 19})
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
    if onnx_opset_version < 16:
        WARN('[Parser]: Default onnx opset version (%d) is lower than 16, which may cause some ops failed to convert!' %
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
    if onnx_opset_version < 18:
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
        # The lowest version of onnx Col2Im is 18.
        torch.onnx.register_custom_op_symbolic(
            'aten::col2im', convert_col2im, onnx_opset_version)
    if onnx_opset_version < 19:
        torch.onnx.register_custom_op_symbolic(
            'torchvision::deform_conv2d', convert_deform_conv, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::avg_pool1d', convert_avg_pool1d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::avg_pool2d', convert_avg_pool2d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::avg_pool3d', convert_avg_pool3d, onnx_opset_version)
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
        'aten::flatten', convert_flatten, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::gru_cell', convert_gru_cell, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::hardshrink', convert_hardshrink, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::linalg_norm', convert_linalg_norm, onnx_opset_version)
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
        'aten::round', convert_round, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::select_scatter', convert_select_scatter, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::size', convert_size, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::slice_scatter', convert_slice_scatter, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::softshrink', convert_softshrink, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::split', convert_split, onnx_opset_version)
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
        'quantized::cat', convert_quantized_cat, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::relu6', convert_quantized_relu6, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::batch_norm2d', convert_quant_batch_norm, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::batch_norm3d', convert_quant_batch_norm3d, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::batch_norm2d_relu', convert_quant_batch_norm_relu, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::batch_norm3d_relu', convert_quant_batch_norm_relu_3d, onnx_opset_version)
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

    jit_model = model if is_torch_script_model else torch.jit.freeze(torch.jit.script(model))
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
            FATAL('[Parser]: Fail to convert model (%s) to onnx! Suggest to set env var PYTORCH_JIT_LOG_LEVEL=onnx for debug!' % model_path)

    INFO('[Parser]: Torch model has been converted to onnx model (%s) with opset version (%d)!' %
         (onnx_model_path, 'default' if onnx_opset_version is None else onnx_opset_version))

    # Update params
    updated_params = copy.deepcopy(params)
    updated_params.update({'input_model': onnx_model_path,
                           'input_names': input_names,
                           'input_shapes': params['input_shapes'],
                           'output_names': [],
                           'output_tensor_names': output_names,
                           'model_type': 'torch'})

    # Return onnx model path and updated params
    return onnx_model_path, updated_params
