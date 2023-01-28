# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class Tfconv2dOp(TfHasPaddingStrides, Tf2Op, OpHasWeights, OpHasOneOutPort):
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
            if item in input_args[1:]:
                inputs = self.get_input_tensors()
                item_idx = input_args.index(item)
                if len(inputs) > item_idx:
                    if item_idx == 1:
                        ret = inputs[item_idx]
                    else:
                        ret = inputs[item_idx].item() if inputs[item_idx].size == 1 else list(
                            inputs[item_idx])
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
        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                inp.shape[1:3],
                out_tensor.shape[1:3],
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
                zero_minimum=True,
            )
            self.auto_pad = 'NOTSET'
        elif self.auto_pad == 'NOTSET':
            pad_slice = slice(
                1, 3) if self.data_format == 'NHWC' else slice(2, 4)
            self.pads = np.transpose(
                np.reshape(np.array(self.explicit_paddings), (4, 2))[
                    pad_slice, :]
            ).flatten().tolist()
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
