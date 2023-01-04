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
