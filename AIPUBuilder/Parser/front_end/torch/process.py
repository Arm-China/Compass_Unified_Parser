# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import os
import numpy as np
import onnx
import torch
import torch.onnx.symbolic_helper as helper
from torch.onnx import symbolic_opset9 as opset9
from multiprocessing import Process
from .utils import get_tuple_from_tensor_type, quantized_args, quantize_helper, quantize_helper_multi
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
    if input_dtype.is_signed:
        FATAL('[Parser]: Only BitShift with unsigned input is supported to convert to onnx, but got type %s!' % str(input_dtype))
    return g.op('BitShift', input, other, direction_s=direction)


@helper.parse_args('v', 'v')
def convert_bitshift_left(g, input, other):
    return convert_bitshift(g, input, other, 'LEFT')


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

    kwargs = {'kernel_shape_i': kernel_shape, 'strides_i': stride, 'dilations_i': dilation, 'group_i': groups}

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


@quantized_args(True, False, False)
def convert_constant_chunk(g, self, chunks, dim):
    input_shape = g.op('Shape', self)
    axis = g.op('Constant', value_t=torch.tensor([dim], dtype=torch.long))
    input_shape_dim = g.op('Gather', input_shape, axis, axis_i=0)
    start = g.op('Constant', value_t=torch.tensor([0], dtype=torch.long))
    chunk_size = g.op('Constant', value_t=torch.tensor(
        [chunks], dtype=torch.long))
    chunk_size_minus_1 = g.op(
        'Constant', value_t=torch.tensor([chunks - 1], dtype=torch.long)
    )
    input_shape_dim_shift = g.op(
        'Add', input_shape_dim, chunk_size_minus_1)
    chunk_dim = g.op('Div', input_shape_dim_shift, chunk_size)
    res = []
    for i in range(chunks):
        index = g.op('Constant', value_t=torch.tensor(
            [i + 1], dtype=torch.long))
        end = g.op('Mul', chunk_dim, index)
        res.append(g.op('Slice', self, start, end, axis))
        start = end
    return res


@helper.parse_args('v', 'i', 'i')
def convert_cumprod(g, input, dim, dtype):
    if dtype is not None:
        input = g.op('Cast', input, to_i=helper.scalar_type_to_onnx[dtype])
    return g.op('custom::CumProd', input, axis_i=dim)


@helper.parse_args('s', 'v', 's', 'v')
def convert_dict_construct(g, key_0, value_0, key_1=None, value_1=None):
    keys = ', '.join([key_0] + ([] if key_1 is None else [key_1]))
    WARN('[Parser]: prim::DictConstruct is unsupported and is changed to return a tensor or a list of tensors(key(s): %s) instead!' % keys)
    if value_1 is not None:
        g.registerOutput(value_0)  # value_0 is the first output of graph
        return g.op('Identity', value_1)  # value_1 is the second output of graph if this node is output
    return g.op('Identity', value_0)


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


def convert_to_bool(g, input):
    input_dtype_str = input.type().scalarType()
    if input_dtype_str != 'Bool':
        input = g.op('Cast', input, to_i=torch._C._onnx.TensorProtoDataType.BOOL)
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


@helper.quantized_args(True, False, False, False, False, False, False)
@helper.parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
def convert_avg_pool(
    g,
    inp,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override=None,
):
    if not stride:
        stride = kernel_size
    if count_include_pad:
        padding = tuple(padding)
        inp = g.op(
            'Pad',
            inp,
            g.op('Constant', value_t=torch.tensor(((0,) * 2 + padding) * 2)),
            mode_s='constant',
        )
        padding = (0,) * len(padding)
    output = g.op(
        'AveragePool',
        inp,
        kernel_shape_i=tuple(kernel_size),
        strides_i=tuple(stride),
        pads_i=padding * 2,
        ceil_mode_i=ceil_mode,
    )
    return output


def convert_max_pool(g, input, kernel_size, strides, paddings, dilations, ceil_mode, dim, return_indices=False):
    def _get_torch_output_shape(input_size, kernel_size, pad_head, pad_tail, stride, dilation):
        extra = (stride - 1)
        output_size = int((input_size + pad_head + pad_tail - dilation * (kernel_size - 1) - 1 + extra) / stride) + 1
        if (output_size - 1) * stride >= (input_size + pad_head):
            output_size = output_size - 1
        return output_size

    def _get_onnx_output_shape(input_size, kernel_size, pad_head, pad_tail, stride, dilation):
        output_size = int(np.ceil((input_size + pad_head + pad_tail - dilation * (kernel_size - 1) - 1) / stride)) + 1
        return output_size

    assert isinstance(dim, int) and dim in [1, 2, 3], 'Meets invalid dim in convert_max_pool_in_ceil_mode!'
    if not ceil_mode:
        if return_indices:
            from torch.onnx.symbolic_opset10 import max_pool1d_with_indices, max_pool2d_with_indices, max_pool3d_with_indices
            max_pool_func = max_pool1d_with_indices if dim == 1 else (
                max_pool2d_with_indices if dim == 2 else max_pool3d_with_indices)
            max_pool, indices = max_pool_func(g, input, kernel_size, strides, paddings, dilations, ceil_mode)
            return (max_pool, indices)
        else:
            from torch.onnx.symbolic_opset10 import max_pool1d, max_pool2d, max_pool3d
            max_pool_func = max_pool1d if dim == 1 else (max_pool2d if dim == 2 else max_pool3d)
            max_pool = max_pool_func(g, input, kernel_size, strides, paddings, dilations, ceil_mode)
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
        torch_output_shape = _get_torch_output_shape(input_size, kernel, pad, pad, stride, dilation)
        onnx_output_shape = _get_onnx_output_shape(input_size, kernel, pad, pad, stride, dilation)
        if torch_output_shape > onnx_output_shape:
            FATAL('[Parser]: Meets unsupported output shape of max_pool in convert_max_pool_in_ceil_mode!')
        slice_ends.append(torch_output_shape)
    ends_input = g.op('Constant', value_t=torch.tensor(slice_ends))
    starts_input = g.op('Constant', value_t=torch.tensor([0] * len(slice_ends)))
    axes_input = g.op('Constant', value_t=torch.tensor(list(range(2, len(input_shape)))))
    slice_out = g.op('Slice', max_pool, starts_input, ends_input, axes_input)

    if indices is not None:
        ndims = len(input_shape) - 2
        slice_indices = g.op('Slice', indices, starts_input, ends_input, axes_input)
        _, flattened_indices = g.op('MaxPool', input, outputs=2, kernel_shape_i=[1] * ndims)
        sub_indices = helper._slice_helper(g, flattened_indices,
                                           axes=list(range(2, len(input_shape))),
                                           starts=[0] * ndims, ends=[1] * ndims)
        indices = g.op('Sub', slice_indices, sub_indices)
    return (slice_out, indices) if return_indices else slice_out


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
        FATAL('[Parser]: Meets invalid output_size of max_unpool2d in convert_max_unpool2d!')
    out_h, out_w = output_size
    offset = np.reshape(np.arange(0, n), (n, 1, 1, 1)) * c * out_h * out_w + \
        np.reshape(np.arange(0, c), (c, 1, 1)) * out_h * out_w
    offset = np.tile(offset, [1, 1, in_h, in_w])
    offset_node = g.op('Constant', value_t=torch.tensor(offset, dtype=torch.int64))
    indices = g.op('Add', indices, offset_node)
    output_shape = g.op('Constant', value_t=torch.tensor([n, c] + output_size, dtype=torch.int64))
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
                          custom_opsets={'opset_18': 18})
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
    except:
        DEBUG('[Parser]: Fail to import torchvision!')
        pass

    # Load TorchScript model
    try:
        model = torch.jit.load(model_path)
    except Exception as e:
        FATAL('[Parser]: Fail to load model (%s) because %s! Only TorchScript format is supported.' %
              (model_path, str(e)))

    # Get onnx opset version to target
    # From https://onnxruntime.ai/docs/reference/compatibility.html,
    # for onnx version 1.x, onnx opset version=x+5
    onnx_version = str(get_version(onnx)).split('.')
    onnx_opset_version = (int(onnx_version[-1]) + 5) if int(onnx_version[0]) == 1 else None
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
            DEBUG('[Parser]: Fail to get default onnx opset version because %s' % str(e))
        if default_onnx_main_opset is None:
            onnx_opset_version = None
        elif onnx_opset_version >= default_onnx_main_opset or onnx_opset_version not in default_onnx_stable_opsets:
            onnx_opset_version = default_onnx_main_opset
    if onnx_opset_version is None:
        onnx_opset_version = 9
    DEBUG('[Parser]: Will convert to onnx opset version (%s)!' % str(onnx_opset_version))

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
    if torch_version < '2.1.0':
        # The issue of string padding is fixed in latest torch.
        # Refer to https://github.com/pytorch/pytorch/pull/89107
        for conv_op in ('aten::conv1d', 'aten::conv2d', 'aten::conv3d'):
            torch.onnx.register_custom_op_symbolic(
                conv_op, convert_conv, onnx_opset_version)
        # The issue of logical_not is fixed in latest torch.
        # Refer to https://github.com/pytorch/pytorch/pull/96315
        torch.onnx.register_custom_op_symbolic(
            'aten::logical_not', convert_logical_not, onnx_opset_version)

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
    torch.onnx.register_custom_op_symbolic(
        'aten::avg_pool2d', convert_avg_pool, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::bitwise_left_shift', convert_bitshift_left, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::bitwise_right_shift', convert_bitshift_right, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::equal', convert_equal, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::flatten', convert_flatten, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'aten::gru_cell', convert_gru_cell, onnx_opset_version)
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
        'prim::ConstantChunk', convert_constant_chunk, onnx_opset_version)

    # for quantized Ops
    torch.onnx.register_custom_op_symbolic(
        'quantized::add_relu', convert_quantized_add_relu, onnx_opset_version)
    torch.onnx.register_custom_op_symbolic(
        'quantized::cat', convert_quantized_cat, onnx_opset_version)

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

    # Get input_tensors and input_names
    input_names = []
    tensor_list = []
    input_info_dict = copy.deepcopy(params['input_shapes'])
    input_dtype = params['input_dtype']
    for idx, (input_name, input_shape) in enumerate(input_info_dict.items()):
        if len(input_name) >= 1 and input_name[0].isdigit():  # Starting with numbers is not legal in pytorch
            new_input_name = 'input_' + input_name
            WARN('[Parser]: Input name %s is invalid; rename it to %s!' % (input_name, new_input_name))
            params['input_shapes'].pop(input_name)
            params['input_shapes'][new_input_name] = input_shape
            input_name = new_input_name
        input_names.append(input_name)
        assert len(input_dtype) > idx, 'Meets invalid input_dtype in convert_torch_to_onnx'
        try:
            tensor_dtype = getattr(torch, input_dtype[idx])
            INFO('[Parser]: Input dtype of input %s is set to %s!' % (input_name, input_dtype[idx]))
        except Exception as e:
            tensor_dtype = torch.float32
            WARN('[Parser]: Input dtype %s is changed to float32 because %s' % (input_dtype[idx], str(e)))
        if 'float' in str(tensor_dtype):
            tensor = torch.randn(input_shape, dtype=tensor_dtype)
        else:
            tensor = torch.zeros(input_shape, dtype=tensor_dtype)
        tensor_list.append(tensor)

    input_tensors = ()
    input_index = 0
    for inp in model.graph.inputs():
        tensors, input_index = get_tuple_from_tensor_type(inp.type(), tensor_list, input_index)
        if len(tensors) > 0:
            input_tensors += tensors

    # Get output_names. When the output is a tuple, it's actually multiple outputs constructed in that tuple.
    output_names = []
    for out_idx, out in enumerate(model.graph.outputs()):
        out_name = out.debugName() + '_' + str(out_idx) + '_'
        if isinstance(out.type(), torch._C.DictType):
            inputs_num = len([inp for inp in out.node().inputs()])
            outputs_num = inputs_num // 2
        else:
            outputs_num = len(_flatten_type(out.type()))
        output_names.extend([out_name + str(idx) for idx in range(outputs_num)])
    for idx, output_name in enumerate(output_names):
        if output_name[0].isdigit():
            output_names[idx] = 'output_' + output_name

    # Get the file name of the onnx model to be exported
    onnx_model_path = os.path.join(params.get('output_dir', './'),
                                   os.path.basename(model_path) + '.onnx')
    INFO('[Parser]: Convert TorchScript (%s) to onnx model...' % model_path)

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
        FATAL('[Parser]: Fail to convert model (%s) to onnx because %s' % (model_path, str(e)))

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
