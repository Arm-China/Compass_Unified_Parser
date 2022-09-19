"""
//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
"""

import inspect
import numpy as np
import re
from collections import OrderedDict
from ...common.utils import is_file
from ...common.errors import *
from .tflite.Model import Model
from .tflite.Operator import Operator
from .tflite.BuiltinOptions import BuiltinOptions
from .tflite.BuiltinOperator import BuiltinOperator
from .tflite.Buffer import Buffer

from .tflite.AbsOptions import AbsOptions
from .tflite.ActivationFunctionType import ActivationFunctionType
from .tflite.AddNOptions import AddNOptions
from .tflite.AddOptions import AddOptions
from .tflite.ArgMaxOptions import ArgMaxOptions
from .tflite.ArgMinOptions import ArgMinOptions
from .tflite.BatchToSpaceNDOptions import BatchToSpaceNDOptions
from .tflite.BidirectionalSequenceLSTMOptions import BidirectionalSequenceLSTMOptions
from .tflite.BidirectionalSequenceRNNOptions import BidirectionalSequenceRNNOptions
from .tflite.CallOptions import CallOptions
from .tflite.CastOptions import CastOptions
from .tflite.ConcatEmbeddingsOptions import ConcatEmbeddingsOptions
from .tflite.ConcatenationOptions import ConcatenationOptions
from .tflite.Conv2DOptions import Conv2DOptions
from .tflite.Conv3DOptions import Conv3DOptions
from .tflite.CosOptions import CosOptions
from .tflite.DensifyOptions import DensifyOptions
from .tflite.DepthToSpaceOptions import DepthToSpaceOptions
from .tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from .tflite.DequantizeOptions import DequantizeOptions
from .tflite.DivOptions import DivOptions
from .tflite.EmbeddingLookupSparseOptions import EmbeddingLookupSparseOptions
from .tflite.EqualOptions import EqualOptions
from .tflite.ExpandDimsOptions import ExpandDimsOptions
from .tflite.ExpOptions import ExpOptions
from .tflite.FakeQuantOptions import FakeQuantOptions
from .tflite.FillOptions import FillOptions
from .tflite.FloorDivOptions import FloorDivOptions
from .tflite.FloorModOptions import FloorModOptions
from .tflite.FullyConnectedOptions import FullyConnectedOptions
from .tflite.GatherNdOptions import GatherNdOptions
from .tflite.GatherOptions import GatherOptions
from .tflite.HardSwishOptions import HardSwishOptions
from .tflite.IfOptions import IfOptions
from .tflite.L2NormOptions import L2NormOptions
from .tflite.LeakyReluOptions import LeakyReluOptions
from .tflite.LessEqualOptions import LessEqualOptions
from .tflite.LessOptions import LessOptions
from .tflite.LocalResponseNormalizationOptions import LocalResponseNormalizationOptions
from .tflite.LogicalAndOptions import LogicalAndOptions
from .tflite.LogicalNotOptions import LogicalNotOptions
from .tflite.LogicalOrOptions import LogicalOrOptions
from .tflite.LogSoftmaxOptions import LogSoftmaxOptions
from .tflite.LSHProjectionOptions import LSHProjectionOptions
from .tflite.LSHProjectionType import LSHProjectionType
from .tflite.LSTMKernelType import LSTMKernelType
from .tflite.LSTMOptions import LSTMOptions
from .tflite.MatrixDiagOptions import MatrixDiagOptions
from .tflite.MatrixSetDiagOptions import MatrixSetDiagOptions
from .tflite.MaximumMinimumOptions import MaximumMinimumOptions
from .tflite.MirrorPadMode import MirrorPadMode
from .tflite.MirrorPadOptions import MirrorPadOptions
from .tflite.MulOptions import MulOptions
from .tflite.NegOptions import NegOptions
from .tflite.NonMaxSuppressionV4Options import NonMaxSuppressionV4Options
from .tflite.NonMaxSuppressionV5Options import NonMaxSuppressionV5Options
from .tflite.NotEqualOptions import NotEqualOptions
from .tflite.OneHotOptions import OneHotOptions
from .tflite.PackOptions import PackOptions
from .tflite.Padding import Padding
from .tflite.PadOptions import PadOptions
from .tflite.PadV2Options import PadV2Options
from .tflite.Pool2DOptions import Pool2DOptions
from .tflite.PowOptions import PowOptions
from .tflite.QuantizeOptions import QuantizeOptions
from .tflite.RangeOptions import RangeOptions
from .tflite.RankOptions import RankOptions
from .tflite.ReducerOptions import ReducerOptions
from .tflite.ReshapeOptions import ReshapeOptions
from .tflite.ResizeBilinearOptions import ResizeBilinearOptions
from .tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions
from .tflite.ReverseSequenceOptions import ReverseSequenceOptions
from .tflite.ReverseV2Options import ReverseV2Options
from .tflite.RNNOptions import RNNOptions
from .tflite.ScatterNdOptions import ScatterNdOptions
from .tflite.SegmentSumOptions import SegmentSumOptions
from .tflite.SelectOptions import SelectOptions
from .tflite.SelectV2Options import SelectV2Options
from .tflite.SequenceRNNOptions import SequenceRNNOptions
from .tflite.ShapeOptions import ShapeOptions
from .tflite.SkipGramOptions import SkipGramOptions
from .tflite.SliceOptions import SliceOptions
from .tflite.SoftmaxOptions import SoftmaxOptions
from .tflite.SpaceToDepthOptions import SpaceToDepthOptions
from .tflite.SpaceToBatchNDOptions import SpaceToBatchNDOptions
from .tflite.SparseToDenseOptions import SparseToDenseOptions
from .tflite.SplitOptions import SplitOptions
from .tflite.SplitVOptions import SplitVOptions
from .tflite.SquaredDifferenceOptions import SquaredDifferenceOptions
from .tflite.SquareOptions import SquareOptions
from .tflite.SqueezeOptions import SqueezeOptions
from .tflite.StridedSliceOptions import StridedSliceOptions
from .tflite.SubOptions import SubOptions
from .tflite.SVDFOptions import SVDFOptions
from .tflite.TileOptions import TileOptions
from .tflite.TopKV2Options import TopKV2Options
from .tflite.TransposeConvOptions import TransposeConvOptions
from .tflite.TransposeOptions import TransposeOptions
from .tflite.UnidirectionalSequenceLSTMOptions import UnidirectionalSequenceLSTMOptions
from .tflite.UniqueOptions import UniqueOptions
from .tflite.UnpackOptions import UnpackOptions
from .tflite.WhereOptions import WhereOptions
from .tflite.WhileOptions import WhileOptions
from .tflite.ZerosLikeOptions import ZerosLikeOptions


# FLOAT32 = 0,
# FLOAT16 = 1,
# INT32 = 2,
# UINT8 = 3,
# INT64 = 4,
# STRING = 5,
# BOOL = 6,
# INT16 = 7,
# COMPLEX64 = 8,
# INT8 = 9,
# FLOAT64 = 10,
# COMPLEX128 = 11,
# UINT64 = 12,
# RESOURCE = 13,
# VARIANT = 14,
# UINT32 = 15,

tensor_type_map = {
    0: np.float32,
    1: np.float16,
    2: np.int32,
    3: np.uint8,
    4: np.int64,
    5: np.str,
    6: np.bool,
    7: np.int16,
    8: np.complex64,
    9: np.int8,
    10: np.float64,
    11: np.complex128,
    12: np.uint64,
    13: lambda x: x,
    14: lambda x: x,
    15: np.uint32
}


def get_class_variables_map(class_type):
    return {getattr(class_type, v): v for v in dir(class_type) if not callable(v) and not v.startswith('__')}


def get_valid_option_attribute(option_obj):
    ret = {}
    attr_set = {v for v in dir(option_obj) if not v.startswith(
        '_') and not v.startswith('GetRootAs') and v != 'Init'}
    for attr in attr_set:
        attr_call = getattr(option_obj, attr)
        if callable(attr_call) and len(inspect.getfullargspec(attr_call)[0]) == 1:
            value = attr_call()
            if attr == 'Padding':
                value = get_class_variables_map(Padding)[value]
            elif attr == 'FusedActivationFunction':
                value = get_class_variables_map(ActivationFunctionType)[value]
            elif attr in ['InDataType', 'OutDataType', 'OutType', 'OutputType']:
                value = np.dtype(tensor_type_map[value]).name
            elif attr == 'Mode':
                value = get_class_variables_map(MirrorPadMode)[value]
            ret.update({attr: value})
    return ret


def parse_operator(operator, tflite_model, buffer):
    linear_ops = ('CONV_2D', 'CONV_3D', 'CONV_3D_TRANSPOSE', 'DEPTHWISE_CONV_2D',
                  'FULLY_CONNECTED', 'TRANSPOSE_CONV')
    opcode_index = operator.OpcodeIndex()
    assert 0 <= opcode_index < tflite_model.OperatorCodesLength(
    ), 'The opcode_index of tflite model is invalid in parse_operator.'

    opcode = tflite_model.OperatorCodes(opcode_index)
    if hasattr(opcode, 'DeprecatedBuiltinCode'):
        builtin_op_code = max(opcode.BuiltinCode(),
                              opcode.DeprecatedBuiltinCode())
    else:
        builtin_op_code = opcode.BuiltinCode()
    builtin_op_type = get_class_variables_map(BuiltinOperator)[builtin_op_code]
    builtin_op_version = opcode.Version()

    if builtin_op_type != 'CUSTOM':
        option_type_code = operator.BuiltinOptionsType()
        option_type = get_class_variables_map(BuiltinOptions)[option_type_code]
        if option_type != 'NONE':
            option_offset = operator.BuiltinOptions().Pos
            try:
                option = eval(option_type + '()')
                option.Init(buffer, option_offset)
                op_attr = get_valid_option_attribute(option)
            except Exception as e:
                WARN(str(e) + ' in parse_operator!')
                op_attr = {}
        else:
            op_attr = {}
    else:
        custum_op_code = opcode.CustomCode()
        op_attr = {'method': custum_op_code.decode('utf-8')}

    inputs = operator.InputsAsNumpy().tolist() if operator.InputsLength() != 0 else []
    outputs = operator.OutputsAsNumpy().tolist(
    ) if operator.OutputsLength() != 0 else []

    ret = {'type': builtin_op_type, 'opcode_version': builtin_op_version,
           'attr': op_attr, 'inputs': inputs, 'outputs': outputs}
    ret.update({'linear_weights': (
        {inputs[1]: builtin_op_type} if builtin_op_type in linear_ops else {})})
    return ret


def parse_quantization_info(quant_info):
    ret = OrderedDict()
    if quant_info is not None and quant_info.Details() is None:
        info_names = ['Scale', 'ZeroPoint', 'Min', 'Max']
        for name in info_names:
            meta_info_length = getattr(quant_info, name + 'Length')()
            if meta_info_length != 0:
                value = getattr(quant_info, name + 'AsNumpy')()
                value = value.astype(np.float32) if isinstance(
                    value, np.ndarray) else np.array(value, np.float32)
                ret.update({name: value})
    return ret


def parse_tensor(tensor_info, tflite_model):
    tensor, is_const, linear_type = tensor_info
    buffer_index = tensor.Buffer()
    assert 0 <= buffer_index < tflite_model.BuffersLength(
    ), 'The buffer_index of tflite model is invalid in parse_tensor.'
    data = tflite_model.Buffers(buffer_index).DataAsNumpy()
    data_shape = tensor.ShapeAsNumpy().tolist() if tensor.ShapeLength() > 0 else []
    data_type = tensor_type_map[tensor.Type()]

    try:
        parsed_data = np.reshape(data.view(dtype=data_type).copy(), data_shape) \
            if isinstance(data, np.ndarray) \
            else np.full(data_shape, data, dtype=data_type)
    except:
        parsed_data = np.empty(data_shape, dtype=data_type)

    quant_info_dict = parse_quantization_info(tensor.Quantization())
    if quant_info_dict:
        if is_const:
            if linear_type == 'DEPTHWISE_CONV_2D':
                scale = quant_info_dict['Scale']
                zp = quant_info_dict['ZeroPoint']
            else:
                expand_dims = len(parsed_data.shape) - \
                    len(quant_info_dict['Scale'].shape)
                new_scale_zp_shape = list(
                    quant_info_dict['Scale'].shape) + [1 for _ in range(expand_dims)]
                scale = np.reshape(
                    quant_info_dict['Scale'], newshape=new_scale_zp_shape)
                zp = np.reshape(
                    quant_info_dict['ZeroPoint'], newshape=new_scale_zp_shape)
            parsed_data = (parsed_data - zp) * scale
        else:
            if 'ZeroPoint' in quant_info_dict and 'Scale' in quant_info_dict:
                parsed_data = (
                    parsed_data - quant_info_dict['ZeroPoint']) * quant_info_dict['Scale']
        if parsed_data.dtype == np.float64:
            parsed_data = parsed_data.astype(np.float32)

    ret = {'name': tensor.Name().decode('utf-8'), 'data': parsed_data,
           'is_const': is_const}
    if quant_info_dict:
        ret.update({'quant_info': quant_info_dict})
    return ret


def get_act_info_from_tensor(tensor_dict):
    ret = {}
    if 'quant_info' in tensor_dict and 'Min' in tensor_dict['quant_info'] and 'Max' in tensor_dict['quant_info']:
        act_type_names = ['RELU', 'RELU6', 'HARD_SWISH']
        for act_type in act_type_names:
            if re.search(r'%s$' % act_type, tensor_dict['name'].upper()):
                if act_type in ('RELU', 'RELU6'):
                    ret.update({'act_type': act_type})
                    if act_type == 'RELU' and tensor_dict['quant_info']['Min'] < 0:
                        tensor_dict['quant_info']['Min'] = np.array(
                            [0], np.float32)
                    elif act_type == 'RELU6':
                        if tensor_dict['quant_info']['Min'] < 0:
                            tensor_dict['quant_info']['Min'] = np.array(
                                [0], np.float32)
                        if tensor_dict['quant_info']['Max'] > 6:
                            tensor_dict['quant_info']['Max'] = np.array(
                                [6], np.float32)
    return ret


def read_tflite_model(tflite_path):
    ret = False, None, None
    if is_file(tflite_path):
        try:
            tflite_model_buf = open(tflite_path, 'rb').read()
            tflite_model = Model.GetRootAsModel(tflite_model_buf, 0)
            if tflite_model is not None and tflite_model.SubgraphsLength() > 0:
                if tflite_model.Subgraphs(0).OperatorsLength() > 0:
                    ret = True, tflite_model, tflite_model_buf
                else:
                    WARN('[Parser]: Empty operators in tflite model (%s)!' %
                         tflite_path)
            else:
                WARN(
                    '[Parser]: Meets invalid tflite model(%s) in read_tflite_model!' % tflite_path)
        except IOError:
            ERROR('[Parser]: Meets error when reading tflite file(%s) in read_tflite_model!' %
                  tflite_path)
    else:
        WARN('[Parser]: Invalid tflite file (%s) in read_tflite_model!' %
             tflite_path)
    return ret
