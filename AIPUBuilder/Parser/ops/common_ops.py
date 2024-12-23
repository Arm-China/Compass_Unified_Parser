# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import tensorflow as tf
import numpy as np
import torch

from .op import *
from ..common.utils import get_random_array, list_list_to_string
from ..plugin_loader import PARSER_OP_DICT


class AccidentalHitsOp(OpHasMultipleOutPorts, CommonOp):
    @classmethod
    def attributes(cls):
        return {'num_true': {'type': AttrType.INT, 'required': True},
                }

    def __init__(self, graph, attr_dict=None):
        super(AccidentalHitsOp, self).__init__(graph, attr_dict)
        self.update_attributes(AccidentalHitsOp, attr_dict)
        assert self.check_required(), 'AccidentalHitsOp is missing a required parameter.'

    def infer_shape(self):
        super(AccidentalHitsOp, self).infer_shape()
        input_shape = self.get_input_shapes()[0]
        num_hits = min(np.prod(input_shape), 32768)
        output_indices = get_random_array([num_hits], 'int32')
        output_ids = get_random_array([num_hits], 'int32')
        output_effective_len = get_random_array([1], 'int32')
        self.set_out_tensor([output_indices, output_ids, output_effective_len])


class AdaptivePoolOp(LayoutConcernedOp, OpHasMethod, OpHasOneOutPort, CommonOp):
    @classmethod
    def attributes(cls):
        return {'output_size': {'type': AttrType.INTS, 'required': True},
                'method': {'options': ['AVG', 'MAX'], 'required': True}}

    def __init__(self, graph, attr_dict=None):
        super(AdaptivePoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(AdaptivePoolOp, attr_dict)
        assert self.check_required(), 'AdaptivePoolOp is missing a required parameter.'

    def infer_shape(self):
        super(AdaptivePoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_dim = len(inputs[0].shape)
        assert len(inputs) == 1 and input_dim in (3, 4, 5), 'The input is invalid in AdaptivePoolOp.'
        if self.data_format.startswith('NC'):
            perm = None
            input_tensor = inputs[0]
        else:
            perm = [0, (input_dim - 1)] + list(range(1, input_dim - 1))
            input_tensor = np.transpose(inputs[0], perm)
        if input_dim == 3:
            m = torch.nn.AdaptiveAvgPool1d(tuple(self.output_size)) if self.method == 'AVG' \
                else torch.nn.AdaptiveMaxPool1d(tuple(self.output_size))
        elif input_dim == 4:
            m = torch.nn.AdaptiveAvgPool2d(tuple(self.output_size)) if self.method == 'AVG' \
                else torch.nn.AdaptiveMaxPool2d(tuple(self.output_size))
        else:  # input_dim == 5
            m = torch.nn.AdaptiveAvgPool3d(tuple(self.output_size)) if self.method == 'AVG' \
                else torch.nn.AdaptiveMaxPool3d(tuple(self.output_size))
        input_dtype = input_tensor.dtype
        # torch adaptive_avg_pool/adaptive_max_pool doesn't support int input
        out_tensor = m(torch.from_numpy(input_tensor.astype(np.float32))).numpy()
        if perm is not None:
            out_tensor = np.transpose(out_tensor, Op.cal_inverse_perm(perm))
        self.set_out_tensor(out_tensor.astype(input_dtype))


class BatchGatherOp(OpHasAxis, OpHasOneOutPort, CommonOp):
    @classmethod
    def attributes(cls):
        return {'axis': {'default': 0},
                'batch_dims': {'type': AttrType.INT, 'default': 0}}

    def __init__(self, graph, attr_dict=None):
        super(BatchGatherOp, self).__init__(graph, attr_dict)
        self.update_attributes(BatchGatherOp, attr_dict)
        assert self.check_required(), 'BatchGatherOp is missing a required parameter.'

    def infer_shape(self):
        super(BatchGatherOp, self).infer_shape()
        inputs = self.get_input_tensors()
        indices = inputs[1].tolist()
        ref_shape = inputs[0].shape
        indices = np.array(indices, np.int64)
        negative_axes = indices < 0
        if np.any(negative_axes):
            len_shape = ref_shape[self.axis]
            indices[negative_axes] += len_shape
        indices = indices.tolist()
        out_tensor = tf.gather(inputs[0],
                               indices,
                               axis=self.axis,
                               batch_dims=self.batch_dims).numpy()
        self.set_out_tensor(out_tensor)


class BNLLOp(OpHasOneOutPort, CommonOp):
    def infer_shape(self):
        super(BNLLOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.log(1. + np.exp(*inputs))
        self.set_out_tensor(out_tensor)


class CumProdOp(OpHasOneOutPort, OpHasAxis, CommonOp):
    @classmethod
    def attributes(cls):
        return {'exclusive': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]},
                'reverse': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}
                }

    def __init__(self, graph, attr_dict=None):
        super(CumProdOp, self).__init__(graph, attr_dict)
        self.update_attributes(CumProdOp, attr_dict)
        assert self.check_required(), 'CumprodOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item in ('exclusive', 'reverse'):
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(CumProdOp, self).__getattr__(item)
        return ret

    def __setattr__(self, item, value):
        if item in ('exclusive', 'reverse'):
            self.__dict__['_attr'][item].value = int(value)
        else:
            super(CumProdOp, self).__setattr__(item, value)

    def infer_shape(self):
        super(CumProdOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.math.cumprod(
            inputs[0], axis=self.axis, exclusive=self.exclusive, reverse=self.reverse).numpy()
        self.set_out_tensor(out_tensor)


class CropAndResizeOp(OpHasMethod, LayoutConcernedOp, OpHasOneOutPort, CommonOp):
    @classmethod
    def attributes(cls):
        return {'crop_size': {'type': AttrType.INTS, 'default': []},
                'method': {'default': 'BILINEAR', 'options': ['BILINEAR', 'NEAREST']},
                'extrapolation_value': {'type': AttrType.FLOAT, 'default': 0.},
                }

    def __init__(self, graph, attr_dict=None):
        super(CropAndResizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(CropAndResizeOp, attr_dict)
        assert self.check_required(), 'CropAndResizeOp is missing a required parameter.'

    def infer_shape(self):
        super(CropAndResizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.image.crop_and_resize(image=inputs[0],
                                              boxes=inputs[1],
                                              box_indices=inputs[2],
                                              crop_size=self.crop_size,
                                              method=self.method.lower(),
                                              extrapolation_value=self.extrapolation_value).numpy()
        self.set_out_tensor(out_tensor)


class ChannelShuffleOp(LayoutConcernedOp, OpHasMultipleOutPorts, CommonOp):
    @classmethod
    def attributes(cls):
        return {'group': {'type': AttrType.INT, 'required': True, 'default': 2},
                'splits': {'type': AttrType.INT, 'required': True, 'default': 1}
                }

    def __init__(self, graph, attr_dict=None):
        super(ChannelShuffleOp, self).__init__(graph, attr_dict)
        self.update_attributes(ChannelShuffleOp, attr_dict)
        assert self.check_required(), 'ChannelShuffleOp is missing a required parameter.'

    def infer_shape(self):
        super(ChannelShuffleOp, self).infer_shape()
        inputs = self.get_input_tensors()
        pre_perm = None
        if self.data_format == 'NHWC':
            max_dim = len(inputs[0].shape) - 1
            pre_perm = [0, max_dim] + list(range(1, max_dim))
            inp = np.transpose(inputs[0], pre_perm)
        else:
            inp = inputs[0]
        out_tensor = torch.nn.functional.channel_shuffle(
            torch.from_numpy(inp), self.group).numpy()
        if pre_perm is not None:
            out_tensor = np.transpose(out_tensor, Op.cal_inverse_perm(pre_perm))
            out_tensors = np.split(out_tensor, self.splits, axis=-1)
        else:
            out_tensors = np.split(out_tensor, self.splits, axis=1)
        self.set_out_tensor(out_tensors)


class CTCGreedyDecoderOp(OpHasOneOutPort, CommonOp):
    @classmethod
    def attributes(cls):
        return {'merge_repeated': {'type': AttrType.INT, 'default': 1},
                'sequence_lens': {'type': AttrType.INT},
                'input_size': {'type': AttrType.INT}
                }

    def __init__(self, graph, attr_dict=None):
        super(CTCGreedyDecoderOp, self).__init__(graph, attr_dict)
        self.update_attributes(CTCGreedyDecoderOp, attr_dict)
        assert self.check_required(), 'CTCGreedyDecoderOp is missing a required parameter.'

    def infer_shape(self):
        super(CTCGreedyDecoderOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.data_format == 'NHWC':
            batch_size, self.sequence_lens, self.input_size = inputs[0].shape
        else:
            self.sequence_lens, batch_size, self.input_size = inputs[0].shape
        out_tensor = get_random_array([batch_size, 4096, 1, 1], 'int64')
        self.set_out_tensor(out_tensor)


class DilationOp(OpHasPaddingStrides, OpHasWeights, OpHasOneOutPort, LayoutConcernedOp, CommonOp):
    @classmethod
    def attributes(cls):
        return {'dilations': {'default': [1, 1, 1, 1]}
                }

    def __init__(self, graph, attr_dict=None):
        super(DilationOp, self).__init__(graph, attr_dict)
        self.update_attributes(DilationOp, attr_dict)
        assert self.check_required(), 'DilationOp is missing a required parameter.'

    def infer_shape(self):
        super(DilationOp, self).infer_shape()
        inputs = self.get_input_tensors()

        if self.auto_pad == 'VALID':
            padding = 'VALID'
        else:  # 'SAME_UPPER', 'SAME_LOWER', 'NOTSET'
            padding = 'SAME'
        inp = np.transpose(inputs[0], [0, 2, 3, 1]
                           ) if self.data_format == 'NCHW' else inputs[0]
        out_tensor = tf.nn.dilation2d(inp,
                                      np.transpose(self.weights, [1, 2, 0]),
                                      strides=[1] + self.strides + [1],
                                      padding=padding,
                                      dilations=[1] + self.dilations + [1],
                                      data_format='NHWC').numpy()

        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                inputs[0].shape[1:3],
                out_tensor.shape[1:3],
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
                zero_minimum=True,

            )
            self.auto_pad = 'NOTSET'

        if self.data_format == 'NCHW':
            out_tensor = np.transpose(out_tensor, [0, 3, 1, 2])
        self.set_out_tensor(out_tensor)


class DivModOp(OpNeedBroadcast, LayoutUnawareOp, OpHasDivisor, OpHasMultipleOutPorts, CommonOp):
    @classmethod
    def attributes(cls):
        return {'mode': {'type': AttrType.STRING, 'default': 'floor', 'options': ['trunc', 'floor']}}

    def __init__(self, graph, attr_dict=None):
        super(DivModOp, self).__init__(graph, attr_dict)
        self.update_attributes(DivModOp, attr_dict)
        assert self.check_required(), 'DivModOp is missing a required parameter.'

    def infer_shape(self):
        super(DivModOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 2, 'Expects 2 inputs for ArmDivMod op (%s), but got %d' % (self.name, len(inputs))
        if np.any(inputs[1] == 0):
            in_consts = self.sorted_in_consts()
            if len(in_consts) == 2 \
                    or (len(in_consts) == 1 and in_consts[0][1] == 1):
                ERROR('[Parser]: Meets invalid second input of DivMod Op (%s) in infer_shape!' % self.name)
                out0, out1 = None, None
            else:
                WARN('[Parser]: The second input of DivMod Op (%s) is replaced by ones in infer_shape!' % self.name)
                second_input = np.ones_like(inputs[1])
                out0, out1 = np.divmod(inputs[0], second_input)
        else:
            out0 = torch.div(torch.tensor(inputs[0].astype(np.int64)), torch.tensor(inputs[1].astype(np.int64)),
                             rounding_mode=self.mode).numpy().astype(inputs[0].dtype)
            out1 = (inputs[0] - out0 * inputs[1])
        self.set_out_tensor([out0, out1])


class DummyOp(OpHasOneOutPort, ConstLikeOp, CommonOp):
    def __init__(self, graph, attr_dict=None):
        super(DummyOp, self).__init__(graph, attr_dict)

    def infer_shape(self, input_tensor=None):
        super(DummyOp, self).infer_shape()
        if input_tensor is not None:
            out_tensor = input_tensor.copy()
            self.set_out_tensor(out_tensor)


class DummyInputOp(OpHasOneOutPort, InputLikeOp, CommonOp):
    @classmethod
    def attributes(cls):
        return {'target_graph': {'type': AttrType.STRING, 'default': '', 'required': False},
                }

    def __init__(self, graph, attr_dict=None):
        super(DummyInputOp, self).__init__(graph, attr_dict)
        self.update_attributes(DummyInputOp, attr_dict)
        assert self.check_required(), 'DummyInputOp is missing a required parameter.'

    def infer_shape(self, input_tensor=None, is_const=False):
        super(DummyInputOp, self).infer_shape()
        assert input_tensor is not None, 'input tensor is empty in DummyInputOp.'
        out_tensor = input_tensor.copy()
        self.set_out_tensor(out_tensor, is_const)

    def set_out_tensor(self, tensor_data, is_const):
        '''set the out tensor of this op.'''
        try:
            for _, _, d in self._graph.sorted_out_edges(self.name, data=True):
                if d.get('tensor', None) is not None:
                    d['tensor'].value = tensor_data
                    if tensor_data is not None:
                        d['tensor'].shape = d['tensor'].value.shape
                        d['tensor'].is_const = is_const
                        if not self.quantize or d['tensor'].dtype is None:
                            d['tensor'].dtype = str(d['tensor'].value.dtype)
                else:
                    d['tensor'] = Tensor(value=tensor_data)

            self.clear_unused_tensor(is_const)

        except KeyError as e:
            ERROR('[Parser]: Node(%s) meets key error in set_out_tensor (%s)!' %
                  (self.name, str(e)))
        except Exception as e:
            ERROR('[Parser]: Node(%s) meets exception in set_out_tensor (%s)!' %
                  (self.name, str(e)))


class ErosionOp(OpHasPaddingStrides, OpHasWeights, OpHasOneOutPort, LayoutConcernedOp, CommonOp):
    @classmethod
    def attributes(cls):
        return {'dilations': {'default': [1, 1, 1, 1]},
                }

    def __init__(self, graph, attr_dict=None):
        super(ErosionOp, self).__init__(graph, attr_dict)
        self.update_attributes(ErosionOp, attr_dict)
        assert self.check_required(), 'ErosionOp is missing a required parameter.'

    def infer_shape(self):
        super(ErosionOp, self).infer_shape()
        inputs = self.get_input_tensors()

        if self.auto_pad == 'VALID':
            padding = 'VALID'
        elif self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            padding = 'SAME'
        inp = np.transpose(inputs[0], [0, 2, 3, 1]
                           ) if self.data_format == 'NCHW' else inputs[0]
        out_tensor = tf.compat.v1.nn.erosion2d(inp,
                                               np.transpose(self.weights, [1, 2, 0]),
                                               strides=[1] + self.strides + [1],
                                               rates=[1] + self.dilations + [1],
                                               padding='VALID' if self.auto_pad == 'NOTSET' else 'SAME').numpy()

        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                inputs[0].shape[1:3],
                out_tensor.shape[1:3],
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
                zero_minimum=True,

            )
            self.auto_pad = 'NOTSET'

        if self.data_format == 'NCHW':
            out_tensor = np.transpose(out_tensor, [0, 3, 1, 2])
        self.set_out_tensor(out_tensor)


class FillOp(OpHasOneOutPort, CommonOp):
    def infer_shape(self):
        super(FillOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.fill(*inputs).numpy()
        self.set_out_tensor(out_tensor)


class FilterOp(OpHasMultipleOutPorts, CommonOp):
    def infer_shape(self):
        super(FilterOp, self).infer_shape()
        inputs = self.get_input_tensors()
        mask = inputs[-1].astype(bool)
        out_tensors = [np.zeros_like(inp) for inp in inputs[:-1]]
        for i, ot in enumerate(inputs[:-1]):
            true_indices = np.where(mask)[0]
            out_tensors[i][mask] = np.take(ot, true_indices, axis=0)
        valid_num = np.array([np.sum(mask)], np.int32)
        out_tensors.append(valid_num)
        self.set_out_tensor(out_tensors)


class FractionalPoolOp(OpHasMethod, OpHasMultipleOutPorts, CommonOp):
    @classmethod
    def attributes(cls):
        return {'method': {'options': ['AVG', 'MAX'], 'required': True},
                'pooling_ratio': {'type': AttrType.FLOATS, 'required': True},
                'pseudo': {'type': AttrType.BOOL, 'default': False},
                'overlap': {'type': AttrType.BOOL, 'default': False},
                'seed': {'type': AttrType.INT, 'default': 0}
                }

    def __init__(self, graph, attr_dict=None):
        super(FractionalPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(FractionalPoolOp, attr_dict)
        assert self.check_required(), 'FractionalPoolOp is missing a required parameter.'

    def infer_shape(self):
        super(FractionalPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tuple = tf.nn.fractional_avg_pool(
            inputs[0],
            self.pooling_ratio,
            pseudo_random=self.pseudo,
            overlapping=self.overlap,
            seed=self.seed
        )
        out_tensors = [t.numpy() if i == 0 else t.numpy().astype(np.int32) for i, t in enumerate(out_tuple)]
        self.set_out_tensor(out_tensors)


class FullyConnectedOp(BaseLinearOp, CommonOp):
    @classmethod
    def attributes(cls):
        return {}

    @classmethod
    def perm_onnx_to_tf(cls):
        return [1, 0]

    def __init__(self, graph, attr_dict=None):
        super(FullyConnectedOp, self).__init__(graph, attr_dict)
        self.update_attributes(FullyConnectedOp, attr_dict)
        assert self.check_required(), 'FullyConnectedOp is missing a required parameter.'

    def infer_shape(self):
        super(FullyConnectedOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if inputs[0].dtype in ('int8', 'uint8'):
            inp = inputs[0].astype(np.float32)
        else:
            inp = inputs[0]
        out_tensor = (tf.matmul(inp, np.transpose(self.weights, axes=type(self).perm_onnx_to_tf()))
                      + self.biases
                      ).numpy().astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class GenerateProposalsOp(LayoutConcernedOp, OpHasWeights, OpHasMultipleOutPorts, CommonOp):
    @classmethod
    def attributes(cls):
        return {'pre_nms_topn': {'type': AttrType.INT, 'default': 6000},
                'post_nms_topn': {'type': AttrType.INT, 'default': 300},
                'min_size': {'type': AttrType.INT, 'default': 16},
                'nms_threshold': {'type': AttrType.FLOAT, 'default': 0.7},
                'feature_stride': {'type': AttrType.INT, 'default': 16},
                'image_width': {'type': AttrType.INT, 'default': 600},
                'image_height': {'type': AttrType.INT, 'default': 600},
                }

    def __init__(self, graph, attr_dict=None):
        super(GenerateProposalsOp, self).__init__(graph, attr_dict)
        self.update_attributes(GenerateProposalsOp, attr_dict)
        assert self.check_required(), 'GenerateProposalsOp is missing a required parameter.'

    def infer_shape(self):
        super(GenerateProposalsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        batch_size = inputs[0].shape[0]
        scores = np.random.ranf(
            (batch_size, self.post_nms_topn)).astype(np.float32)
        boxes = np.random.ranf(
            (batch_size, self.post_nms_topn, 4)).astype(np.float32)
        indices = np.random.ranf(
            (batch_size, self.post_nms_topn, 1)).astype(np.float32)
        box_num = np.random.ranf((batch_size, 1)).astype(np.float32)
        self.set_out_tensor([scores, boxes, indices, box_num])


class HardSwishOp(LayoutUnawareOp, OpHasOneOutPort, CommonOp):
    @classmethod
    def attributes(cls):
        return {}

    def __init__(self, graph, attr_dict=None):
        super(HardSwishOp, self).__init__(graph, attr_dict)
        self.update_attributes(HardSwishOp, attr_dict)
        assert self.check_required(), 'HardSwishOp is missing a required parameter.'

    def infer_shape(self):
        super(HardSwishOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = (inputs[0] * tf.nn.relu6(inputs[0] + 3) / 6).numpy().astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class InputOp(OpHasOneOutPort, InputLikeOp, CommonOp):
    def infer_shape(self, input_tensor=None):
        super(InputOp, self).infer_shape()
        assert input_tensor is not None, 'input shape is empty in InputOp.'
        out_tensor = input_tensor.copy()
        self.set_out_tensor(out_tensor)


class InTopKOp(LayoutUnawareOp, OpHasOneOutPort, CommonOp):
    @classmethod
    def attributes(cls):
        return {'k': {'type': AttrType.INT, 'required': True}}

    def __init__(self, graph, attr_dict=None):
        super(InTopKOp, self).__init__(graph, attr_dict)
        self.update_attributes(InTopKOp, attr_dict)
        assert self.check_required(), 'InTopKOp is missing a required parameter.'

    def infer_shape(self):
        super(InTopKOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 2, 'InTopKOp expects two inputs, but got %d.' % len(inputs)
        out_tensor = tf.raw_ops.InTopK(
            predictions=inputs[0], targets=inputs[1], k=self.k).numpy()
        self.set_out_tensor(out_tensor)


class MeshgridOp(OpHasMultipleOutPorts, CommonOp):
    @classmethod
    def attributes(cls):
        return {'indexing': {'type': AttrType.STRING, 'options': ['ij', 'xy'], 'default': 'xy'}}

    def __init__(self, graph, attr_dict=None):
        super(MeshgridOp, self).__init__(graph, attr_dict)
        self.update_attributes(MeshgridOp, attr_dict)
        assert self.check_required(), 'MeshgridOp is missing a required parameter.'

    def infer_shape(self):
        super(MeshgridOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = tf.meshgrid(*inputs, indexing=self.indexing)
        out_tensors = [t.numpy() for t in out_tensors]
        self.set_out_tensor(out_tensors)


class MMCVModulatedDeformConv2dOp(BaseConvOp, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'stride': {'type': AttrType.INTS, 'required': True},
                    'dilation': {'type': AttrType.INTS, 'required': True},
                    'padding': {'type': AttrType.INTS, 'required': True},
                    'groups': {'type': AttrType.INT, 'default': 1},
                    'deform_groups': {'type': AttrType.INT, 'default': 1}},
                }

    def __init__(self, graph, attr_dict=None):
        super(MMCVModulatedDeformConv2dOp, self).__init__(graph, attr_dict)
        self.update_attributes(MMCVModulatedDeformConv2dOp, attr_dict)
        assert self.check_required(), 'MMCVModulatedDeformConv2dOp is missing a required parameter.'
        self.group = self.groups
        assert len(self.padding) in (2, 4), 'Expects the length of padding is 2 or 4, but got %d' % len(self.padding)
        self.pads = self.padding * 2 if len(self.padding) == 2 else self.padding
        self.dilations = self.dilation
        self.strides = self.stride

    def infer_shape(self):
        super(MMCVModulatedDeformConv2dOp, self).infer_shape()
        inputs = self.get_input_tensors()  # input, offset, mask, weight, bias(optional)
        assert len(inputs) >= 4, 'Expects at least 4 inputs for MMCVModulatedDeformConv2dOp but got %d' % len(inputs)
        batch = inputs[0].shape[0]
        num_output = inputs[3].shape[0]
        if self.data_format == 'NHWC':
            out_shape = [batch] + list(inputs[1].shape[1:-1]) + [num_output]
        else:
            out_shape = [batch, num_output] + list(inputs[1].shape[2:])
        out_tensor = np.random.ranf(size=out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class MomentsOp(OpHasMultipleOutPorts, OpHasAxis, CommonOp):
    def infer_shape(self):
        super(MomentsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = tf.nn.moments(
            inputs[0], self.axes, keepdims=self.keepdims)
        out_tensors = [out_tensor.numpy() for out_tensor in out_tensors]
        self.set_out_tensor(out_tensors)


class NormalizedMomentsOp(OpHasMultipleOutPorts, CommonOp):
    @classmethod
    def attributes(cls):
        return {'counts': {'type': AttrType.FLOAT, 'required': True},
                }

    def __init__(self, graph, attr_dict=None):
        super(NormalizedMomentsOp, self).__init__(graph, attr_dict)
        self.update_attributes(NormalizedMomentsOp, attr_dict)
        assert self.check_required(), 'NormalizedMomentsOp is missing a required parameter.'

    def infer_shape(self):
        super(NormalizedMomentsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        count = np.array(self.counts).astype(np.float32)
        out_tensors = tf.nn.normalize_moments(counts=count,
                                              mean_ss=inputs[0],
                                              variance_ss=inputs[1],
                                              shift=inputs[2])
        out_tensors = [out_tensor.numpy() for out_tensor in out_tensors]
        self.set_out_tensor(out_tensors)


class OutOp(OpHasOneOutPort, CommonOp):
    def __init__(self, graph, attr_dict=None):
        super(OutOp, self).__init__(graph, attr_dict)

    def infer_shape(self):
        super(OutOp, self).infer_shape()


class OverlapAddOp(OpHasOneOutPort, CommonOp):
    @classmethod
    def attributes(cls):
        return {'frame_step': {'type': AttrType.INT}
                }

    def __init__(self, graph, attr_dict=None):
        super(OverlapAddOp, self).__init__(graph, attr_dict)
        self.update_attributes(OverlapAddOp, attr_dict)
        assert self.check_required(), 'OverlapAddOp is missing a required parameter.'

    def infer_shape(self):
        super(OverlapAddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs[0].shape) >= 2, 'The rank of OverlapAddOp inp0 must be at least 2., but got %d' % len(
            inputs[0].shape)
        out_tensor = tf.signal.overlap_and_add(
            signal=inputs[0], frame_step=self.frame_step).numpy()
        self.set_out_tensor(out_tensor)


class RepeatOp(OpHasAxis, OpHasOneOutPort, CommonOp):
    @classmethod
    def attributes(cls):
        return {'max_dim': {'type': AttrType.INT, 'default': None}}

    def __init__(self, graph, attr_dict=None):
        super(RepeatOp, self).__init__(graph, attr_dict)
        self.update_attributes(RepeatOp, attr_dict)
        assert self.check_required(), 'RepeatOp is missing a required parameter.'

    def infer_shape(self):
        super(RepeatOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 2, 'RepeatOp expects 2 inputs, but got %d' % len(inputs)
        out_tensor = np.repeat(inputs[0], inputs[1].tolist(), self.axis)
        if self.axis is not None:
            out_shape = list(out_tensor.shape)
            if self.max_dim is None:
                self.max_dim = out_tensor.shape[self.axis]
            elif out_shape[self.axis] > self.max_dim:
                obj = tuple(slice(0, e if i != self.axis else self.max_dim)
                            for (i, e) in enumerate(out_shape))
                out_tensor = out_tensor[obj]
            elif out_shape[self.axis] < self.max_dim:
                zeros_shape = copy.deepcopy(out_shape)
                zeros_shape[self.axis] = self.max_dim - out_shape[self.axis]
                zeros = np.zeros(zeros_shape, inputs[0].dtype)
                out_tensor = np.concatenate([out_tensor, zeros], axis=self.axis)
        self.set_out_tensor(out_tensor)


class PluginOp(OpHasVariableOutPorts, CommonOp):
    @classmethod
    def num_in_ports(cls):
        return 1

    def __init__(self, graph, attr_dict=None):
        super(PluginOp, self).__init__(graph, attr_dict)
        self._type = 'Plugin' + attr_dict.get('type', 'Plugin')
        self.constants = {}
        self.constants_offset_dict = {}
        self.out_tensors = attr_dict.get('out_tensors', [])  # record output tensors
        fw = graph._attr['framework'].name
        try:
            plugin_type = PARSER_OP_DICT[re.sub(
                r'^Plugin', '', self._type, count=1)]
            nest_input = None
            if '_nest_inputs' in attr_dict:
                nest_input = attr_dict['_nest_inputs']
                attr_dict.pop('_nest_inputs')
            self._plugin = plugin_type(fw, attr_dict)
            if nest_input is not None:
                self._plugin._nest_inputs = nest_input
            if self._plugin is not None:
                self.constants = getattr(self._plugin, 'constants', {})
        except Exception as e:
            self._plugin = None
            ERROR('[Parser]: Creating plugin type (%s) meets error %s! ' %
                  (self.type, str(e)))
            import traceback
            print(traceback.format_exc())

    def remove_inputs(self, nest_input_index, inputs):
        # remove input edges if _inputs_to_remove is set in ParserOp
        remove_input_indexes = getattr(self._plugin, '_inputs_to_remove', [])
        remove_edge_indexes = []
        for node_index, tensor_index in remove_input_indexes:
            if nest_input_index is None:
                assert node_index == 0, 'Expect node_index == 0 in non-subgraph plugin, but got %d' % node_index
                assert tensor_index < len(inputs), 'Expect tensor_index < inputs length (%d), but got %d' % (
                    len(inputs), tensor_index)
                remove_edge_index = tensor_index
            else:
                assert node_index < len(
                    nest_input_index), 'Expect node_index < inputs length (%d) in subgraph plugin, but got %d' % (
                    len(nest_input_index), node_index)
                assert tensor_index < len(
                    nest_input_index[node_index]), 'Expect tensor_index < inputs length (%d), but got %d' % (
                    len(nest_input_index[node_index]), tensor_index)
                remove_edge_index = np.sum([len(nest_input_index[idx]) for idx in range(node_index)]) + tensor_index
            remove_edge_indexes.append(remove_edge_index)
        in_edges = self._graph.sorted_in_edges(self.name)
        self._graph.remove_edges_from([edge for idx, edge in enumerate(in_edges) if idx in remove_edge_indexes])

    def infer_shape(self, final=False):
        # Do infer_shape only once for plugin
        if len(self.out_tensors) > 0:
            if final:
                for i, t in enumerate(self.out_tensors):
                    if isinstance(t, np.ndarray) and hasattr(t, 'dtype'):
                        if t.dtype == np.int64:
                            self.out_tensors[i] = t.astype(np.int32)
                        elif t.dtype == np.float64:
                            self.out_tensors[i] = t.astype(np.float32)
            self.set_out_tensor(self.out_tensors)
            return
        super(PluginOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = []
        if self._plugin is not None:
            # check if is subgraph plugin, use nest list
            nest_input_index = None
            if hasattr(self._plugin, '_subgraph_type'):
                nest_input_index = getattr(self._plugin, '_nest_inputs', [])
                inputs = [[inputs[i] for i in inps]
                          for inps in nest_input_index]
            try:
                DEBUG('[Parser]: Call plugin infer_shape!')
                out_tensors = self._plugin.infer_shape(inputs)
                if final:
                    for i, t in enumerate(out_tensors):
                        if isinstance(t, np.ndarray) and hasattr(t, 'dtype'):
                            if t.dtype == np.int64:
                                out_tensors[i] = t.astype(np.int32)
                            elif t.dtype == np.float64:
                                out_tensors[i] = t.astype(np.float32)
            except Exception as e:
                ERROR('[Parser]: plugin type (%s) infer shape meets error %s! ' %
                      (self.type, str(e)))
                import traceback
                print(traceback.format_exc())
            try:
                self.remove_inputs(nest_input_index, inputs)
            except Exception as e:
                ERROR('[Parser]: plugin type (%s) meets error in removing input tensors: %s!' %
                      (self.type, str(e)))
        else:
            ERROR('[Parser]: Invalid Plugin op (%s) for infer_shape!' % self.name)
        self.set_out_tensor(out_tensors)
        self.out_tensors = out_tensors
        # self.constants could be used in self._plugin.infer_shape, so check constants here
        self.constants = getattr(self._plugin, 'constants', {})
        if not all((isinstance(key, str) and isinstance(val, np.ndarray))
                   for key, val in self.constants.items()):
            ERROR(
                '[Parser]: Invalid constants in Plugin op (%s). Expect key is string and value is numpy array!' % self.name)
            ERROR('constants: %s' % str(self.constants))

    def write_attrs(self, txt_file):
        ret = super(PluginOp, self).write_attrs(txt_file)
        if ret:
            if self._plugin is not None:
                for k, v in getattr(self._plugin, 'params', {}).items():
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    if isinstance(v, (list,)):
                        try:
                            v = str(v).replace(' ', '')
                        except Exception as e:
                            ERROR('[Parser]: Node(%s) meets error for write_attrs: %s' %
                                  (self.name, str(e)))
                            continue
                    elif isinstance(v, (int, float)):
                        v = str(v)
                    elif isinstance(v, str):
                        pass
                    else:
                        try:
                            v = str(v)
                        except Exception as e:
                            ERROR('[Parser]: Node(%s) meets error for write_attrs: %s' %
                                  (self.name, str(e)))
                            continue
                    txt_file.write('%s=%s\n' % (k, v))
                for key, np_array in self.constants.items():
                    if key not in self.constants_offset_dict:
                        ERROR('[Parser]: Fail to save constants (%s) for unknown offset in write_attrs!' % key)
                        continue
                    txt_file.write('%s_type=%s\n' % (key, str(np_array.dtype)))
                    txt_file.write('%s_offset=%d\n' % (key, self.constants_offset_dict[key]))
                    txt_file.write('%s_size=%d\n' % (key,
                                                     (np_array.size * np_array.dtype.itemsize)))
                    txt_file.write('%s_shape=[%s]\n' % (key, num_list_to_string(
                        list(np_array.shape))))
            else:
                ERROR('[Parser]: Invalid Plugin op(%s) for write_attrs!' %
                      self.name)
                ret = False
        return ret

    def write_constants(self, bin_file):
        '''Write value of constants in IR binfile.'''
        ret = True
        if not bin_file.closed and bin_file.mode == 'ab':
            if not self.constants:
                pass
            elif not self.constants_offset_dict \
                    or any(key not in self.constants_offset_dict for key in self.constants):
                ERROR('[Parser]: Node(%s) has invalid offset for constants in write_constants!' % self.name)
                ret = False
            else:
                start = bin_file.tell()
                for key, value in self.constants.items():
                    offset = self.constants_offset_dict[key]
                    assert start == offset, 'constants offset not match! layer name: %s, %d' % (self.name, offset)
                    value.tofile(bin_file)
                    end = bin_file.tell()
                    assert (value.dtype.itemsize * int(np.prod(value.shape))) == (end - start), \
                        'Node(%s) has error in writing constants (%s) to bin in write_constants!' % (self.name, key)
                    start = end
        else:
            FATAL(
                '[Parser]: Invalid file to write constants for Node(%s) in write_constants!' % self.name)
        return ret


class QueryRebatchOp(OpHasOneOutPort, CommonOp):
    def infer_shape(self):
        super(QueryRebatchOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cam_num = len(inputs) - 1
        out_shape = list(inputs[0].shape[:])
        out_shape.insert(1, cam_num)
        if self.is_all_inputs_const():
            batch = inputs[0].shape[0]
            query = torch.from_numpy(inputs[0])
            query_rebatch = torch.zeros(out_shape)
            for j in range(batch):
                for i in range(cam_num):
                    index_query_per_img = torch.from_numpy(inputs[i + 1])
                    query_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
            out_tensor = query_rebatch.numpy().astype(inputs[0].dtype)
        else:
            out_tensor = np.random.ranf(out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class ReduceAllOp(OpHasAxis, OpHasOneOutPort, CommonOp):
    @classmethod
    def attributes(cls):
        return {'keepdims': {'required': True}}

    def __init__(self, graph, attr_dict=None):
        super(ReduceAllOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceAllOp, attr_dict)
        assert self.check_required(), 'ReduceAllOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                inputs = self.get_input_tensors()
                if len(inputs) > 1:
                    ret = inputs[1].tolist() if inputs[1].size > 0 else None
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(ReduceAllOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ReduceAllOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = tf.reduce_all(inputs[0], axis=tuple(
            self.axes), keepdims=self.keepdims).numpy()
        self.set_out_tensor(out_tensor)


class ReduceAnyOp(OpHasAxis, OpHasOneOutPort, CommonOp):
    @classmethod
    def attributes(cls):
        return {'keepdims': {'default': 1}}

    def __init__(self, graph, attr_dict=None):
        super(ReduceAnyOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceAnyOp, attr_dict)
        assert self.check_required(), 'ReduceAnyOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                inputs = self.get_input_tensors()
                if len(inputs) > 1:
                    ret = inputs[1].tolist() if inputs[1].size > 0 else None
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(ReduceAnyOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ReduceAnyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = tf.reduce_any(inputs[0], axis=tuple(
            self.axes), keepdims=self.keepdims).numpy()
        self.set_out_tensor(out_tensor)


class ReduceVarianceOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'keepdims': {'default': 1},
                    'unbiased': {'type': AttrType.INT, 'default': 0, 'options': [0, 1]}}
                }

    def __init__(self, graph, attr_dict=None):
        super(ReduceVarianceOp, self).__init__(graph, attr_dict)
        self.update_attributes(ReduceVarianceOp, attr_dict)
        assert self.check_required(), 'ReduceVarianceOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item in ('keepdims', 'unbiased'):
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(ReduceVarianceOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(ReduceVarianceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.axes is None:
            self.axes = list(range(len(inputs[0].shape)))
        out_tensor = np.var(
            inputs[0], axis=tuple(self.axes), keepdims=self.keepdims, ddof=self.unbiased)
        self.set_out_tensor(out_tensor)


class RollOp(OpHasAxis, OpHasOneOutPort, CommonOp):

    @staticmethod
    def cal_roll_parm(axis_value, roll_shift, roll_shape):
        if axis_value < 0:
            axis_value = len(roll_shape) + axis_value

        if abs(roll_shift) > roll_shape[axis_value]:
            roll_shift = roll_shift % roll_shape[axis_value]

        roll_shift = roll_shape[axis_value] + roll_shift
        slice_num = roll_shape[axis_value] - roll_shift

        start1 = np.zeros((axis_value + 1)).astype(np.int32).tolist()
        start1[-1] = slice_num
        end1 = np.array([roll_shape[i]
                         for i in range(0, axis_value + 1)]).tolist()
        steps1 = np.ones((axis_value + 1)).astype(np.int32).tolist()
        axes1 = np.arange(axis_value + 1).astype(np.int32).tolist()

        start2 = np.zeros((axis_value + 1)).astype(np.int32).tolist()
        end2 = np.array([roll_shape[i]
                         for i in range(0, axis_value + 1)]).tolist()
        end2[-1] = slice_num
        steps2 = np.ones((axis_value + 1)).astype(np.int32).tolist()
        axes2 = np.arange(axis_value + 1).astype(np.int32).tolist()

        return roll_shift, start1, end1, steps1, axes1, start2, end2, steps2, axes2

    @classmethod
    def attributes(cls):
        return {'axes': {'type': AttrType.INTS, 'required': True},
                'shift': {'type': AttrType.INTS, 'required': True}
                }

    def __init__(self, graph, attr_dict=None):
        super(RollOp, self).__init__(graph, attr_dict)
        self.update_attributes(RollOp, attr_dict)
        assert self.check_required(), 'RollOp is missing a required parameter.'

    def infer_shape(self):
        super(RollOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.roll(inputs[0], self.shift, self.axes).numpy()
        self.set_out_tensor(out_tensor)


class SegmentReduceOp(OpHasMethod, OpHasOneOutPort, CommonOp):
    FUNC_MAP = {'SUM': tf.math.segment_sum,
                'PROD': tf.math.segment_prod,
                'MIN': tf.math.segment_min,
                'MAX': tf.math.segment_max,
                'MEAN': tf.math.segment_mean,
                }

    @classmethod
    def attributes(cls):
        return {'method': {'options': ['SUM', 'PROD', 'MIN', 'MAX', 'MEAN'], 'default': 'SUM'}}

    def __init__(self, graph, attr_dict=None):
        super(SegmentReduceOp, self).__init__(graph, attr_dict)
        self.update_attributes(SegmentReduceOp, attr_dict)
        assert self.check_required(), 'SegmentReduceOp is missing a required parameter.'

    def infer_shape(self):
        super(SegmentReduceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = SegmentReduceOp.FUNC_MAP[self.method](*inputs).numpy()
        self.set_out_tensor(out_tensor)


class SiluOp(LayoutUnawareOp, ActivationOnlyOp, CommonOp):
    def infer_shape(self):
        super(SiluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = (inputs[0]) * tf.sigmoid(inputs[0].astype(np.float32)).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))


class SlotUpdateOp(OpHasOneOutPort, CommonOp):
    def infer_shape(self):
        super(SlotUpdateOp, self).infer_shape()
        inputs = self.get_input_tensors()
        cam_num = len(inputs) - 1
        assert cam_num == inputs[0].shape[1]
        out_shape = list(inputs[0].shape[:])
        out_shape.pop(1)
        if self.is_all_inputs_const():
            batch = inputs[0].shape[0]
            queries = torch.from_numpy(inputs[0])
            slots = torch.zeros_like(queries)
            for j in range(batch):
                for i, index_query_per_img in enumerate(inputs[1:]):
                    slots[j, torch.from_numpy(index_query_per_img)] += queries[j, i, :len(index_query_per_img)]
            out_tensor = slots.numpy().astype(inputs[0].dtype)
        else:
            out_tensor = np.random.ranf(out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class SufficientStatisticsOp(OpHasAxis, OpHasMultipleOutPorts, CommonOp):
    def __init__(self, graph, attr_dict=None):
        super(SufficientStatisticsOp, self).__init__(graph, attr_dict)
        self.update_attributes(SufficientStatisticsOp, attr_dict)
        assert self.check_required(), 'SufficientStatisticsOp is missing a required parameter.'

    def infer_shape(self):
        super(SufficientStatisticsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor_list = tf.nn.sufficient_statistics(
            inputs[0], self.axes, shift=inputs[1], keepdims=True)
        out_tensor_list = [ot.numpy() for ot in out_tensor_list[1:3]]
        self.set_out_tensor(out_tensor_list)


class SwishOp(OpHasOneOutPort, CommonOp):
    @classmethod
    def attributes(cls):
        return {'alpha': {'type': AttrType.FLOAT, 'default': 1}
                }

    def __init__(self, graph, attr_dict=None):
        super(SwishOp, self).__init__(graph, attr_dict)
        self.update_attributes(SwishOp, attr_dict)
        assert self.check_required(), 'SwishOp is missing a required parameter.'

    def infer_shape(self):
        super(SwishOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_dtype = str(inputs[0].dtype)
        inp = inputs[0].astype(np.float32) if input_dtype != 'float32' else inputs[0]
        out_tensor = inp * tf.sigmoid(self.alpha * inp).numpy()
        if input_dtype != 'float32':
            out_tensor = out_tensor.astype(input_dtype)
        self.set_out_tensor(out_tensor)


class TempOp(LayoutUnawareOp, CommonOp):
    '''
    This OP is just used for pettern match and won't appear in the final graph.
    '''

    def infer_shape(self, input_tensor=None):
        super(TempOp, self).infer_shape()


class TruncOp(LayoutUnawareOp, OpHasOneOutPort, CommonOp):
    def __init__(self, graph, attr_dict=None):
        super(TruncOp, self).__init__(graph, attr_dict)
        self.update_attributes(TruncOp, attr_dict)
        assert self.check_required(), 'TruncOp is missing a required parameter.'

    def infer_shape(self):
        super(TruncOp, self).infer_shape()
        inputs = self.get_input_tensors()
        torch_input = torch.from_numpy(inputs[0])
        out_tensor = torch.trunc(torch_input).numpy()
        self.set_out_tensor(out_tensor)


class UndefinedOp(OpHasVariableOutPorts, CommonOp):
    def __init__(self, graph, attr_dict=None):
        super(UndefinedOp, self).__init__(graph, attr_dict)
        if 'type' in attr_dict:
            self.type = attr_dict['type']
        for k, v in attr_dict.items():
            if k not in ('name', 'type', 'data_format') and k not in self._attr.keys():
                attr_param = {'type': AttrType.UNDEFINED,
                              'dafault': None, 'required': False, 'value': v}
                self._attr[k] = Attribute(k, attr_param)

    def infer_shape(self, input_tensor=None):
        super(UndefinedOp, self).infer_shape()

    def write_attrs(self, txt_file):
        ret = super(UndefinedOp, self).write_attrs(txt_file)
        if ret:
            for k, v in self._attr.items():
                if v.type == AttrType.UNDEFINED:
                    txt_file.write('%s=%s\n' % (k, str(v.value)))
        return ret


class ZeroFractionOp(OpHasOneOutPort, CommonOp):
    def infer_shape(self):
        super(ZeroFractionOp, self).infer_shape()
        input_tensor = self.get_input_tensors()[0]
        out_tensor = np.array(tf.math.zero_fraction(input_tensor).numpy())
        self.set_out_tensor(out_tensor)
