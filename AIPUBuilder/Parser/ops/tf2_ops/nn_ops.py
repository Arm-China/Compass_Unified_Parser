# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ..tf_ops.math_ops import TfSoftsignOp
from ..tf_ops.nn_ops import TfSeluOp, TfRelu6Op
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL
from ...common.defs import FLOAT_EQUAL


class Tfconv2dOp(Tf2HasPaddingStrides, OpHasWeights, OpHasOneOutPort):
    @classmethod
    def attributes(cls):
        return {1: {'padding': {'default': None},
                    'dilations': {'default': [1, 1, 1, 1]},
                    'use_cudnn_on_gpu': {'type': AttrType.INT, 'default': 1, 'options': [0, 1]},
                    },
                2: {'padding': {'default': None},
                    'dilations': {'default': [1, 1, 1, 1]},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(Tfconv2dOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tfconv2dOp, attr_dict)
        assert self.check_required(), 'Tfconv2dOp is missing a required parameter.'

    @classmethod
    def perm_tf_to_onnx(cls):
        return [3, 2, 0, 1]

    def __getattr__(self, item):
        ret = None
        try:
            if self.cur_version == 1:
                input_args = ['input', 'weights', 'strides', 'padding', 'use_cudnn_on_gpu', 'data_format', 'dilations']
            else:
                input_args = ['input', 'weights', 'strides', 'padding', 'data_format', 'dilations']
            if item in input_args[3:]:
                inputs = self.get_input_tensors()
                item_idx = input_args.index(item)
                if len(inputs) > item_idx:
                    ret = inputs[item_idx].item() if inputs[item_idx].size == 1 else list(inputs[item_idx])
                    if item == 'use_cudnn_on_gpu':
                        ret = int(ret)
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(Tfconv2dOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Tfconv2dOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 4, 'Tfconv2dOp expects 4 inputs, but got %d' % len(inputs)
        self.weights = inputs[1]
        self.strides = [inputs[2].item()] if inputs[2].size == 1 else list(inputs[2])
        self.kernel_shape = self.weights.shape[0:2]
        padding = self.padding
        if isinstance(padding, str):
            self.auto_pad = 'SAME_UPPER' if padding == 'SAME' else 'VALID'
        else:
            padding = padding.tolist()
            if self.data_format == 'NCHW':
                padding = padding[0:1] + padding[2:4] + padding[1:2]
        inp = np.transpose(inputs[0], [0, 2, 3, 1]
                           ) if self.data_format == 'NCHW' else inputs[0]
        out_tensor = tf.nn.conv2d(inp,
                                  self.weights,
                                  strides=self.strides,
                                  padding=padding,
                                  dilations=self.dilations,
                                  data_format='NHWC').numpy()
        if self.data_format == 'NCHW':
            out_tensor = np.transpose(out_tensor, [0, 3, 1, 2])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class TfcreluOp(OpHasAxis, OpHasOneOutPort, Tf2Op):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': -1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfcreluOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfcreluOp, attr_dict)
        assert self.check_required(), 'TfcreluOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axis':
                inputs = self.get_input_tensors()
                if len(inputs) >= 3:
                    ret = int(inputs[2])
                    self.__dict__['_attr'][item].value = ret
                else:
                    ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        if ret is None:
            ret = super(TfcreluOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfcreluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.crelu(inputs[0], axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)


class Tfdepth_to_spaceOp(LayoutConcernedOp, OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(Tfdepth_to_spaceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 3, 'Tfdepth_to_spaceOp expects 3 inputs, but got %d.' % len(inputs)
        self.block_size = inputs[1].item(0)
        self.data_format = inputs[2].item(0)

        from ..tf_ops.nn_ops import TfDepthToSpaceOp
        out_tensor = TfDepthToSpaceOp.cal_out_tensor(inputs[0], self.block_size, self.data_format)
        self.set_out_tensor(out_tensor)


class Tffractional_avg_poolOp(OpHasMultipleOutPorts, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'pseudo_random': {'type': AttrType.BOOL, 'default': False},
                    'overlapping': {'type': AttrType.BOOL, 'default': False},
                    'seed': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(Tffractional_avg_poolOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tffractional_avg_poolOp, attr_dict)
        assert self.check_required(), 'Tffractional_avg_poolOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'pseudo_random':
                inputs = self.get_input_tensors()
                if len(inputs) >= 3:
                    ret = bool(inputs[2])
                    self.__dict__['_attr'][item].value = ret
                else:
                    ret = self.__dict__['_attr'][item].value
            elif item == 'overlapping':
                inputs = self.get_input_tensors()
                if len(inputs) >= 4:
                    ret = bool(inputs[3])
                    self.__dict__['_attr'][item].value = ret
                else:
                    ret = self.__dict__['_attr'][item].value
            elif item == 'seed':
                inputs = self.get_input_tensors()
                if len(inputs) >= 5:
                    ret = int(inputs[4])
                    self.__dict__['_attr'][item].value = ret
                else:
                    ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        if ret is None:
            ret = super(Tffractional_avg_poolOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Tffractional_avg_poolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tuple = tf.nn.fractional_avg_pool(
            inputs[0],
            inputs[1].tolist(),
            pseudo_random=self.pseudo_random,
            overlapping=self.overlapping,
            seed=self.seed
        )
        out_tensors = [t.numpy() for t in out_tuple]
        self.set_out_tensor(out_tensors)

    @property
    def correspond_onnx_op(self):
        return {'type': 'FractionalPool', 'version': 1}


class Tffractional_max_poolOp(OpHasMultipleOutPorts, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'pseudo_random': {'type': AttrType.BOOL, 'default': False},
                    'overlapping': {'type': AttrType.BOOL, 'default': False},
                    'seed': {'type': AttrType.INT, 'default': 0}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(Tffractional_max_poolOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tffractional_max_poolOp, attr_dict)
        assert self.check_required(), 'Tffractional_max_poolOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'pseudo_random':
                inputs = self.get_input_tensors()
                if len(inputs) >= 3:
                    ret = bool(inputs[2])
                    self.__dict__['_attr'][item].value = ret
                else:
                    ret = self.__dict__['_attr'][item].value
            elif item == 'overlapping':
                inputs = self.get_input_tensors()
                if len(inputs) >= 4:
                    ret = bool(inputs[3])
                    self.__dict__['_attr'][item].value = ret
                else:
                    ret = self.__dict__['_attr'][item].value
            elif item == 'seed':
                inputs = self.get_input_tensors()
                if len(inputs) >= 5:
                    ret = int(inputs[4])
                    self.__dict__['_attr'][item].value = ret
                else:
                    ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        if ret is None:
            ret = super(Tffractional_max_poolOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Tffractional_max_poolOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tuple = tf.nn.fractional_max_pool(
            inputs[0],
            inputs[1].tolist(),
            pseudo_random=self.pseudo_random,
            overlapping=self.overlapping,
            seed=self.seed
        )
        out_tensors = [t.numpy() for t in out_tuple]
        self.set_out_tensor(out_tensors)

    @property
    def correspond_onnx_op(self):
        return {'type': 'FractionalPool', 'version': 1}


class TfgeluOp(ActivationOnlyOp, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'approximate': {'type': AttrType.BOOL, 'default': False},
                    }}

    def __init__(self, graph, attr_dict=None):
        super(TfgeluOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfgeluOp, attr_dict)
        assert self.check_required(), 'TfgeluOp is missing a required parameter.'
        self.activations = 'GELU'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'approximate':
                inputs = self.get_input_tensors()
                if len(inputs) >= 2:
                    ret = bool(inputs[1])
                    self.__dict__['_attr'][item].value = ret
                else:
                    ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        if ret is None:
            ret = super(TfgeluOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfgeluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.gelu(inputs[0], approximate=self.approximate).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Gelu', 'version': 1}


class Tflocal_response_normalizationOp(OpHasOneOutPort, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                    'beta': {'type': AttrType.FLOAT, 'default': 0.5},
                    'bias': {'type': AttrType.FLOAT, 'default': 1.0},
                    'depth_radius': {'type': AttrType.INT, 'default': 5}},
                }

    def __init__(self, graph, attr_dict=None):
        super(Tflocal_response_normalizationOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tflocal_response_normalizationOp, attr_dict)
        assert self.check_required(), 'Tflocal_response_normalizationOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            input_args = ['input', 'depth_radius', 'bias', 'alpha', 'beta']
            if item in input_args[1:]:
                inputs = self.get_input_tensors()
                item_idx = input_args.index(item)
                if len(inputs) > item_idx:
                    ret = inputs[item_idx].item(0)
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(Tflocal_response_normalizationOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Tflocal_response_normalizationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.local_response_normalization(input=inputs[0],
                                                        depth_radius=self.depth_radius,
                                                        bias=self.bias,
                                                        alpha=self.alpha,
                                                        beta=self.beta).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LRN', 'version': 1}


class Tflog_softmaxOp(OpHasAxis, OpHasOneOutPort, Tf2Op):
    @classmethod
    def attributes(cls):
        return {1: {'axis': {'default': -1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(Tflog_softmaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tflog_softmaxOp, attr_dict)
        assert self.check_required(), 'Tflog_softmaxOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axis':
                inputs = self.get_input_tensors()
                if len(inputs) >= 2:
                    ret = inputs[1].item(0)
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(Tflog_softmaxOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Tflog_softmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.log_softmax(inputs[0], axis=self.axis).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LogSoftmax', 'version': 13}


class Tfmax_pool_with_argmaxOp(Tf2HasPaddingStrides, OpHasMultipleOutPorts):
    @classmethod
    def attributes(cls):
        return {2: {'padding': {'type': AttrType.STRING},
                    'data_format': {'default': 'NHWC', 'options': ['NHWC']},
                    'output_dtype': {'type': AttrType.STRING, 'default': 'int64', 'options': ['int64', 'int32']},
                    'include_batch_in_index': {'type': AttrType.BOOL, 'default': False},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(Tfmax_pool_with_argmaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tfmax_pool_with_argmaxOp, attr_dict)
        assert self.check_required(), 'Tfmax_pool_with_argmaxOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            input_args = ['input', 'ksize', 'strides', 'padding',
                          'data_format', 'output_dtype', 'include_batch_in_index']
            if item in input_args[3:]:
                inputs = self.get_input_tensors()
                item_idx = input_args.index(item)
                if len(inputs) > item_idx and inputs[item_idx].size == 1:
                    ret = inputs[item_idx].item()
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(Tfmax_pool_with_argmaxOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Tfmax_pool_with_argmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 4, 'Tfmax_pool_with_argmaxOp expects at least 4 inputs, but got %d' % len(inputs)
        self.kernel_shape = [inputs[1].item()] if inputs[1].size == 1 else list(inputs[1])
        self.strides = [inputs[2].item()] if inputs[2].size == 1 else list(inputs[2])
        padding = self.padding
        self.auto_pad = 'SAME_UPPER' if padding == 'SAME' else 'VALID'
        output_dtype = self.output_dtype
        if output_dtype == 'int64':
            WARN('[Parser]: Output dtype int64 in Tfmax_pool_with_argmaxOp (%s) is not supported and will be set to int32!' % self.name)
        out_tensors = tf.nn.max_pool_with_argmax(inputs[0],
                                                 ksize=self.kernel_shape,
                                                 strides=self.strides,
                                                 padding=padding,
                                                 data_format=self.data_format,
                                                 output_dtype=output_dtype,
                                                 include_batch_in_index=self.include_batch_in_index)
        out_tensors = [t.numpy() for t in out_tensors]
        self.set_out_tensor(out_tensors)


class Tfrelu6Op(TfRelu6Op, Tf2Op):
    pass


class TfseluOp(TfSeluOp, Tf2Op):
    pass


class TfsiluOp(LayoutUnawareOp, ActivationOnlyOp, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'beta': {'type': AttrType.FLOAT, 'default': 1.0},
                    }}

    def __init__(self, graph, attr_dict=None):
        super(TfsiluOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfsiluOp, attr_dict)
        assert self.check_required(), 'TfsiluOp is missing a required parameter.'

    def infer_shape(self):
        super(TfsiluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = (inputs[0]) * tf.sigmoid(self.beta * inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        if FLOAT_EQUAL(self.beta, 1):
            return {'type': 'Silu', 'version': 1}
        else:
            return {'type': 'Swish', 'version': 1}


class TfsoftsignOp(TfSoftsignOp, Tf2Op):
    pass
