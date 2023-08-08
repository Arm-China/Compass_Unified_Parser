# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfAvgPoolOp(TfHasPaddingStrides, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {}, 2: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfAvgPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfAvgPoolOp, attr_dict)
        assert self.check_required(), 'TfAvgPoolOp is missing a required parameter.'

    def infer_shape(self):
        super(TfAvgPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = np.transpose(inputs[0], [0, 2, 3, 1]
                           ) if self.data_format == 'NCHW' else inputs[0]
        out_tensor = tf.nn.avg_pool(inp,
                                    ksize=[1] + self.kernel_shape + [1],
                                    strides=[1] + self.strides + [1],
                                    padding='VALID' if self.auto_pad in (
                                        'VALID', 'NOTSET') else 'SAME',
                                    data_format='NHWC').numpy()
        if self.data_format == 'NCHW':
            out_tensor = np.transpose(out_tensor, [0, 3, 1, 2])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'AveragePool', 'version': 10}


class TfAvgPool3DOp(TfHasPaddingStrides, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'data_format': {'default': 'NDHWC', 'options': ['NDHWC', 'NCDHW']}
                    }}

    def __init__(self, graph, attr_dict=None):
        super(TfAvgPool3DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfAvgPool3DOp, attr_dict)
        assert self.check_required(), 'TfAvgPool3DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfAvgPool3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = inputs[0] if self.data_format == 'NDHWC' else np.transpose(inputs[0], [
                                                                         0, 2, 3, 4, 1])
        out_tensor = tf.nn.avg_pool3d(
            inp,
            ksize=self.kernel_shape,
            strides=[1] + self.strides + [1],
            padding='SAME' if self.auto_pad in (
                'SAME_UPPER', 'SAME_LOWER') else 'VALID',
            data_format='NDHWC').numpy()
        if self.data_format == 'NCDHW':
            out_tensor = np.transpose(out_tensor, [0, 4, 1, 2, 3])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'AveragePool', 'version': 10}


class TfBiasAddOp(LayoutConcernedOp, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfBiasAddOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfBiasAddOp, attr_dict)
        assert self.check_required(), 'TfBiasAddOp is missing a required parameter.'

    def infer_shape(self):
        super(TfBiasAddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.bias_add(
            *inputs, data_format=self.data_format).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Add', 'version': 7}


class TfComputeAccidentalHitsOp(OpHasMultipleOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'num_true': {'type': AttrType.INT, 'required': True},
                    'seed': {'type': AttrType.INT, 'default': 0},
                    'seed2': {'type': AttrType.INT, 'default': 0}},
                }

    def __init__(self, graph, attr_dict=None):
        super(TfComputeAccidentalHitsOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfComputeAccidentalHitsOp, attr_dict)
        assert self.check_required(), 'TfComputeAccidentalHitsOp is missing a required parameter.'

    def infer_shape(self):
        super(TfComputeAccidentalHitsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensors = tf.raw_ops.ComputeAccidentalHits(true_classes=inputs[0],
                                                       sampled_candidates=inputs[1],
                                                       num_true=self.num_true,
                                                       seed=self.seed,
                                                       seed2=self.seed2)
        self.set_out_tensor([out_tensor.numpy() for out_tensor in out_tensors])

    @property
    def correspond_onnx_op(self):
        return {'type': 'AccidentalHits', 'version': 1}


class TfConv2DOp(TfHasPaddingStrides, OpHasWeights, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'dilations': {'default': [1, 1, 1, 1]}}
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [3, 2, 0, 1]

    def __init__(self, graph, attr_dict=None):
        super(TfConv2DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfConv2DOp, attr_dict)
        assert self.check_required(), 'TfConv2DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfConv2DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None:
            assert isinstance(
                self.weights, np.ndarray), 'TfConv2DOp only supports constant filter'
            self.kernel_shape = self.weights.shape[0:2]

        if self.auto_pad == 'NOTSET':
            padding = np.reshape(np.array(self.explicit_paddings), (4, 2)).tolist()
        else:
            padding = [0] * (len(inputs[0].shape) - 2)
        if self.data_format == 'NHWC':
            inp = inputs[0]
        else:
            inp = np.transpose(inputs[0], [0, 2, 3, 1])
            padding = padding[0:1] + padding[2:4] + padding[1:2]
        out_shape = BaseConvOp.cal_out_shape(inp.shape[1:-1],
                                             padding,
                                             self.strides,
                                             self.kernel_shape,
                                             self.auto_pad,
                                             dilations=self.dilations,
                                             data_format='NHWC')
        full_out_shape = [inputs[0].shape[0]] + out_shape + [self.weights.shape[-1]]
        out_tensor = np.random.ranf(size=full_out_shape).astype(np.float32)
        if self.data_format == 'NCHW':
            out_tensor = np.transpose(out_tensor, [0, 3, 1, 2])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class TfConv2DBackpropInputOp(TfHasPaddingStrides, OpHasWeights, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'dilations': {'default': [1, 1, 1, 1]},
                    'use_cudnn_on_gpu': {'type': AttrType.INT, 'default': 1},
                    'output_padding': {'type': AttrType.INTS, 'required': False, 'default': None}
                    }
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [3, 2, 0, 1]

    def __init__(self, graph, attr_dict=None):
        super(TfConv2DBackpropInputOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfConv2DBackpropInputOp, attr_dict)
        assert self.check_required(), 'TfConv2DBackpropInputOp is missing a required parameter.'

    def infer_shape(self):
        super(TfConv2DBackpropInputOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None:
            self.kernel_shape = self.weights.shape[0:2]

        if self.auto_pad == 'VALID':
            padding = 'VALID'
        elif self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            padding = 'SAME'
        else:
            padding = np.reshape(
                np.array(self.explicit_paddings), (4, 2)).tolist()

        out_tensor = tf.compat.v1.nn.conv2d_backprop_input(
            input_sizes=inputs[0],
            filter=self.weights,
            out_backprop=inputs[1],
            strides=([1] + self.strides + [1]
                     ) if self.data_format == 'NHWC' else ([1, 1] + self.strides),
            padding=padding,
            use_cudnn_on_gpu=bool(self.use_cudnn_on_gpu),
            data_format=self.data_format,
            dilations=([1] + self.dilations + [1]
                       ) if self.data_format == 'NHWC' else ([1, 1] + self.dilations)
        ).numpy()
        self.set_out_tensor(out_tensor)


class TfConv3DOp(TfHasPaddingStrides, OpHasWeights, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'dilations': {'default': [1, 1, 1, 1, 1]}
                    }
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [4, 3, 0, 1, 2]

    def __init__(self, graph, attr_dict=None):
        super(TfConv3DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfConv3DOp, attr_dict)
        assert self.check_required(), 'TfConv3DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfConv3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None:
            self.kernel_shape = self.weights.shape[0:3]
        inp = inputs[0] if self.data_format == 'NDHWC' else np.transpose(inputs[0], [
                                                                         0, 2, 3, 4, 1])
        out_tensor = tf.nn.conv3d(
            inp,
            self.weights,
            [1] + self.strides + [1],
            'SAME' if self.auto_pad in (
                'SAME_UPPER', 'SAME_LOWER') else 'VALID',
            data_format='NDHWC',
            dilations=[1] + self.dilations + [1]).numpy()
        if self.data_format == 'NCDHW':
            out_tensor = np.transpose(out_tensor, [0, 4, 1, 2, 3])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class TfConv3DBackpropInputV2Op(TfHasPaddingStrides, OpHasWeights, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'dilations': {'default': [1, 1, 1, 1, 1]},
                    'data_format': {'default': 'NDHWC', 'options': ['NDHWC', 'NCDHW']},
                    'output_padding': {'type': AttrType.INTS, 'required': False, 'default': None}
                    }
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [4, 3, 0, 1, 2]

    def __init__(self, graph, attr_dict=None):
        super(TfConv3DBackpropInputV2Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfConv3DBackpropInputV2Op, attr_dict)
        assert self.check_required(), 'TfConv3DBackpropInputV2Op is missing a required parameter.'

    def infer_shape(self):
        super(TfConv3DBackpropInputV2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None:
            self.kernel_shape = self.weights.shape[0:3]
        input_sizes = inputs[0] if self.data_format == 'NDHWC' else np.take(inputs[0], [
                                                                            0, 2, 3, 4, 1])
        inp = inputs[1] if self.data_format == 'NDHWC' else np.transpose(inputs[1], [
                                                                         0, 2, 3, 4, 1])
        out_tensor = tf.raw_ops.Conv3DBackpropInputV2(
            input_sizes=input_sizes,
            filter=self.weights,
            out_backprop=inp,
            strides=([1] + self.strides + [1]),
            padding='SAME' if self.auto_pad in (
                'SAME_UPPER', 'SAME_LOWER') else 'VALID',
            data_format='NDHWC',
            dilations=([1] + self.dilations + [1])
        ).numpy()
        if self.data_format == 'NCDHW':
            out_tensor = np.transpose(out_tensor, [0, 4, 1, 2, 3])
        self.set_out_tensor(out_tensor)


class TfCTCGreedyDecoderOp(OpHasVariableOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'merge_repeated': {'type': AttrType.INT, 'default': 0},
                    }}

    def __init__(self, graph, attr_dict=None):
        super(TfCTCGreedyDecoderOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfCTCGreedyDecoderOp, attr_dict)
        assert self.check_required(), 'TfCTCGreedyDecoderOp is missing a required parameter.'

    def infer_shape(self):
        super(TfCTCGreedyDecoderOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_list = tf.raw_ops.CTCGreedyDecoder(
            inputs=inputs[0],
            sequence_length=inputs[1],
            merge_repeated=bool(self.merge_repeated))
        out_tensor_list = [o.numpy() for o in out_list]
        self.set_out_tensor(out_tensor_list)

    @property
    def correspond_onnx_op(self):
        return {'type': 'CTCGreedyDecoder', 'version': 1}


class TfDepthToSpaceOp(LayoutConcernedOp, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'block_size': {'type': AttrType.INT, 'required': True},
                    'data_format': {'options': ['NHWC', 'NCHW', 'NCHW_VECT_C']}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfDepthToSpaceOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfDepthToSpaceOp, attr_dict)
        assert self.check_required(), 'TfDepthToSpaceOp is missing a required parameter.'

    @staticmethod
    def cal_out_tensor(input_data, block_size, data_format):
        ret = None
        inp = TfOp.convert_to_nhwc(input_data, data_format)
        out_tensor = tf.nn.depth_to_space(inp, block_size, 'NHWC').numpy()
        ret = TfOp.convert_from_nhwc(out_tensor, data_format)
        return ret

    def infer_shape(self):
        super(TfDepthToSpaceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 1, 'TfDepthToSpaceOp expects 1 inputs, but got %d.' % len(inputs)
        out_tensor = TfDepthToSpaceOp.cal_out_tensor(inputs[0], self.block_size, self.data_format)
        self.set_out_tensor(out_tensor)


class TfDilation2DOp(TfHasPaddingStrides, OpHasWeights, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'dilations': {'default': [1, 1, 1, 1]},
                    }
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [2, 0, 1]

    def __init__(self, graph, attr_dict=None):
        super(TfDilation2DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfDilation2DOp, attr_dict)
        assert self.check_required(), 'TfDilation2DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfDilation2DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None:
            assert isinstance(
                self.weights, np.ndarray), 'TfDilation2DOp only supports constant filter'
            self.kernel_shape = self.weights.shape[0:2]

        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            padding = 'SAME'
        else:
            padding = 'VALID'

        inp = np.transpose(inputs[0], [0, 2, 3, 1]
                           ) if self.data_format == 'NCHW' else inputs[0]

        out_tensor = tf.nn.dilation2d(inp,
                                      self.weights,
                                      strides=[1] + self.strides + [1],
                                      padding=padding,
                                      dilations=[1] + self.dilations + [1],
                                      data_format='NHWC').numpy()

        if self.data_format == 'NCHW':
            out_tensor = np.transpose(out_tensor, [0, 3, 1, 2])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Dilation', 'version': 1}


class TfDepthwiseConv2dNativeOp(TfHasPaddingStrides, OpHasWeights, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'dilations': {'default': [1, 1, 1, 1]},
                    'group': {'type': AttrType.INT, 'default': 1},
                    }
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [2, 3, 0, 1]

    def __init__(self, graph, attr_dict=None):
        super(TfDepthwiseConv2dNativeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfDepthwiseConv2dNativeOp, attr_dict)
        assert self.check_required(), 'TfDepthwiseConv2dNativeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfDepthwiseConv2dNativeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.kernel_shape is None:
            self.kernel_shape = self.weights.shape[0:2]
        if self.auto_pad == 'VALID':
            padding = 'VALID'
        elif self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            padding = 'SAME'
        else:
            padding = np.reshape(
                np.array(self.explicit_paddings), (4, 2)).tolist()
            if self.data_format == 'NCHW':
                padding = padding[0:1] + padding[2:4] + padding[1:2]
        inp = np.transpose(inputs[0], [0, 2, 3, 1]
                           ) if self.data_format == 'NCHW' else inputs[0]
        self.group = inp.shape[-1]
        out_tensor = tf.compat.v1.nn.depthwise_conv2d_native(inp,
                                                             self.weights,
                                                             strides=[1] + self.strides + [1],
                                                             padding=padding,
                                                             dilations=[1] + self.dilations + [1],
                                                             data_format='NHWC').numpy()
        if self.data_format == 'NCHW':
            out_tensor = np.transpose(out_tensor, [0, 3, 1, 2])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class TfEluOp(LayoutUnawareOp, ActivationOnlyOp, TfOp):
    def infer_shape(self):
        super(TfEluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.elu(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Elu', 'version': 6}


class TfFractionalAvgPoolOp(OpHasMultipleOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'pooling_ratio': {'type': AttrType.FLOATS, 'required': True},
                    'pseudo_random': {'type': AttrType.BOOL, 'default': False},
                    'overlapping': {'type': AttrType.BOOL, 'default': False},
                    'deterministic': {'type': AttrType.BOOL, 'default': False},
                    'seed': {'type': AttrType.INT, 'default': 0},
                    'seed2': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfFractionalAvgPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfFractionalAvgPoolOp, attr_dict)
        assert self.check_required(), 'TfFractionalAvgPoolOp is missing a required parameter.'

    def infer_shape(self):
        super(TfFractionalAvgPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tuple = tf.compat.v1.nn.fractional_avg_pool(
            inputs[0],
            self.pooling_ratio,
            pseudo_random=self.pseudo_random,
            overlapping=self.overlapping,
            deterministic=self.deterministic,
            seed=self.seed,
            seed2=self.seed2
        )
        out_tensors = [t.numpy() for t in out_tuple]
        self.set_out_tensor(out_tensors)

    @property
    def correspond_onnx_op(self):
        return {'type': 'FractionalPool', 'version': 1}


class TfFractionalMaxPoolOp(OpHasMultipleOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'pooling_ratio': {'type': AttrType.FLOATS, 'required': True},
                    'pseudo_random': {'type': AttrType.BOOL, 'default': False},
                    'overlapping': {'type': AttrType.BOOL, 'default': False},
                    'deterministic': {'type': AttrType.BOOL, 'default': False},
                    'seed': {'type': AttrType.INT, 'default': 0},
                    'seed2': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfFractionalMaxPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfFractionalMaxPoolOp, attr_dict)
        assert self.check_required(), 'TfFractionalMaxPoolOp is missing a required parameter.'

    def infer_shape(self):
        super(TfFractionalMaxPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tuple = tf.compat.v1.nn.fractional_max_pool(
            inputs[0],
            self.pooling_ratio,
            pseudo_random=self.pseudo_random,
            overlapping=self.overlapping,
            deterministic=self.deterministic,
            seed=self.seed,
            seed2=self.seed2
        )
        out_tensors = [t.numpy() for t in out_tuple]
        self.set_out_tensor(out_tensors)

    @property
    def correspond_onnx_op(self):
        return {'type': 'FractionalPool', 'version': 1}


class TfFusedBatchNormOp(LayoutConcernedOp, OpHasVariableOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-4},
                    'is_training': {'type': AttrType.INT, 'default': 1},
                    'exponential_avg_factor': {'type': AttrType.FLOAT, 'default': 1}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfFusedBatchNormOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfFusedBatchNormOp, attr_dict)
        assert self.check_required(), 'TfFusedBatchNormOp is missing a required parameter.'

    def infer_shape(self):
        super(TfFusedBatchNormOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 5, 'TfFusedBatchNormOp expects 5 inputs, but got %d.' % len(inputs)
        out_list = tf.raw_ops.FusedBatchNorm(x=inputs[0], scale=inputs[1], offset=inputs[2],
                                             mean=inputs[3], variance=inputs[4],
                                             epsilon=self.epsilon,
                                             data_format=self.data_format,
                                             is_training=self.is_training)
        out_tensor_list = [o.numpy() for o in out_list]
        self.set_out_tensor(out_tensor_list)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BatchNormalization', 'version': 15}


class TfFusedBatchNormV3Op(LayoutConcernedOp, OpHasVariableOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'epsilon': {'type': AttrType.FLOAT, 'default': 1e-4},
                    'is_training': {'type': AttrType.INT, 'default': 1},
                    'exponential_avg_factor': {'type': AttrType.FLOAT, 'default': 1}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfFusedBatchNormV3Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfFusedBatchNormV3Op, attr_dict)
        assert self.check_required(), 'TfFusedBatchNormV3Op is missing a required parameter.'

    def infer_shape(self):
        super(TfFusedBatchNormV3Op, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 5, 'TfFusedBatchNormV3Op expects 5 inputs, but got %d.' % len(inputs)
        out_list = tf.raw_ops.FusedBatchNormV3(x=inputs[0], scale=inputs[1], offset=inputs[2],
                                               mean=inputs[3], variance=inputs[4],
                                               epsilon=self.epsilon,
                                               data_format=self.data_format,
                                               is_training=self.is_training)
        out_tensor_list = [o.numpy() for o in out_list]
        self.set_out_tensor(out_tensor_list)

    @property
    def correspond_onnx_op(self):
        return {'type': 'BatchNormalization', 'version': 15}


class TfLeakyReluOp(BaseReluOp, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 0.2}},
                }

    def __init__(self, graph, attr_dict=None):
        super(TfLeakyReluOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfLeakyReluOp, attr_dict)
        assert self.check_required(), 'TfLeakyReluOp is missing a required parameter.'
        self.activations = 'LEAKYRELU'
        self.negative_slope = self.alpha

    def infer_shape(self):
        super(TfLeakyReluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = self.cal_activation(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LeakyRelu', 'version': 6}


class TfLogSoftmaxOp(OpHasOneOutPort, TfOp):
    def infer_shape(self):
        super(TfLogSoftmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.log_softmax(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LogSoftmax', 'version': 13}


class TfLRNOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                    'beta': {'type': AttrType.FLOAT, 'default': 0.5},
                    'bias': {'type': AttrType.FLOAT, 'default': 1.0},
                    'depth_radius': {'type': AttrType.INT, 'default': 5}},
                }

    def __init__(self, graph, attr_dict=None):
        super(TfLRNOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfLRNOp, attr_dict)
        assert self.check_required(), 'TfLRNOp is missing a required parameter.'

    def infer_shape(self):
        super(TfLRNOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.LRN(input=inputs[0],
                                    depth_radius=self.depth_radius,
                                    bias=self.bias,
                                    alpha=self.alpha,
                                    beta=self.beta).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LRN', 'version': 1}


class TfMaxPoolOp(TfHasPaddingStrides, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'data_format': {'type': AttrType.STRING, 'options': ['NHWC', 'NCHW', 'NCHW_VECT_C']},
                    'dilations': {'default': [1, 1, 1, 1]},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfMaxPoolOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfMaxPoolOp, attr_dict)
        assert self.check_required(), 'TfMaxPoolOp is missing a required parameter.'

    def infer_shape(self):
        super(TfMaxPoolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.auto_pad == 'VALID':
            padding = 'VALID'
        elif self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            padding = 'SAME'
        else:
            padding = np.reshape(
                np.array(self.explicit_paddings), (4, 2)).tolist()
            if self.data_format == 'NCHW':
                padding = padding[0:1] + padding[2:4] + padding[1:2]
        inp = np.transpose(inputs[0], [0, 2, 3, 1]
                           ) if self.data_format == 'NCHW' else inputs[0]
        out_tensor = tf.nn.max_pool(inp,
                                    ksize=self.kernel_shape,
                                    strides=[1] + self.strides + [1],
                                    padding=padding,
                                    data_format='NHWC').numpy()
        if self.data_format == 'NCHW':
            out_tensor = np.transpose(out_tensor, [0, 3, 1, 2])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'MaxPool', 'version': 10}


class TfMaxPool3DOp(TfHasPaddingStrides, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'data_format': {'default': 'NDHWC', 'options': ['NDHWC', 'NCDHW']}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfMaxPool3DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfMaxPool3DOp, attr_dict)
        assert self.check_required(), 'TfMaxPool3DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfMaxPool3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = inputs[0] if self.data_format == 'NDHWC' else np.transpose(inputs[0], [
                                                                         0, 2, 3, 4, 1])
        out_tensor = tf.nn.max_pool3d(
            inp,
            ksize=self.kernel_shape,
            strides=[1] + self.strides + [1],
            padding='SAME' if self.auto_pad in (
                'SAME_UPPER', 'SAME_LOWER') else 'VALID',
            data_format='NDHWC').numpy()
        if self.data_format == 'NCDHW':
            out_tensor = np.transpose(out_tensor, [0, 4, 1, 2, 3])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'MaxPool', 'version': 12}


class TfMaxPoolWithArgmaxOp(TfHasPaddingStrides, OpHasMultipleOutPorts):
    @classmethod
    def attributes(cls):
        return {1: {'include_batch_in_index': {'type': AttrType.INT, 'default': False},
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfMaxPoolWithArgmaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfMaxPoolWithArgmaxOp, attr_dict)
        assert self.check_required(), 'TfMaxPoolWithArgmaxOp is missing a required parameter.'

    def infer_shape(self):
        super(TfMaxPoolWithArgmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        # FIXME: tf 1.13 does not have include_batch_in_index and assumes True. Ignore tf 1.13 for now.
        out_tensors = tf.nn.max_pool_with_argmax(inputs[0],
                                                 ksize=[1] +
                                                 self.kernel_shape + [1],
                                                 strides=[1] +
                                                 self.strides + [1],
                                                 padding='VALID' if self.auto_pad in (
                                                     'VALID', 'NOTSET') else 'SAME',
                                                 data_format='NHWC',
                                                 include_batch_in_index=self.include_batch_in_index)
        out_tensors = [t.numpy() for t in out_tensors]
        self.set_out_tensor(out_tensors)


class TfReluOp(BaseReluOp, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfReluOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfReluOp, attr_dict)
        assert self.check_required(), 'TfReluOp is missing a required parameter.'
        self.activations = 'RELU'

    def infer_shape(self):
        super(TfReluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = self.cal_activation(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Relu', 'version': 6}


class TfRelu6Op(BaseReluOp, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfRelu6Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfRelu6Op, attr_dict)
        assert self.check_required(), 'TfRelu6Op is missing a required parameter.'
        self.activations = 'CLIP'

    def infer_shape(self):
        super(TfRelu6Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.relu6(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Clip', 'version': 6}


class TfSeluOp(LayoutUnawareOp, ActivationOnlyOp, TfOp):
    def infer_shape(self):
        super(TfSeluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.selu(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Selu', 'version': 6}


class TfSoftmaxOp(OpHasAxis, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': -1}}}

    def __init__(self, graph, attr_dict=None):
        super(TfSoftmaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSoftmaxOp, attr_dict)
        assert self.check_required(), 'TfSoftmaxOp is missing a required parameter.'

    def infer_shape(self):
        super(TfSoftmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.softmax(inputs[0], axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Softmax', 'version': 13}


class TfSpaceToDepthOp(LayoutConcernedOp, OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'block_size': {'type': AttrType.INT, 'required': True},
                    'data_format': {'options': ['NHWC', 'NCHW', 'NCHW_VECT_C']}},
                }

    def __init__(self, graph, attr_dict=None):
        super(TfSpaceToDepthOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfSpaceToDepthOp, attr_dict)
        assert self.check_required()

    def infer_shape(self):
        super(TfSpaceToDepthOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = TfOp.convert_to_nhwc(inputs[0], self.data_format)
        out_tensor = tf.nn.space_to_depth(
            inp, self.block_size, data_format='NHWC').numpy()
        out_tensor = TfOp.convert_from_nhwc(out_tensor, self.data_format)
        self.set_out_tensor(out_tensor)
