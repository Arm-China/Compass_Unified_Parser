# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfKerasAddOp(OpHasOneOutPort, KerasNeedBroadcast):
    @classmethod
    def attributes(cls):
        return {2: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfKerasAddOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasAddOp, attr_dict)
        assert self.check_required(), 'TfKerasAddOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasAddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.keras.layers.Add()([*inputs]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sum', 'version': 6}


class TfKerasAverageOp(OpHasOneOutPort, KerasNeedBroadcast):
    def infer_shape(self):
        super(TfKerasAverageOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.keras.layers.Average()([*inputs]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Mean', 'version': 13}


class TfKerasAveragePooling1DOp(KerasPoolingOp):
    @classmethod
    def ufunc(cls):
        return tf.keras.layers.AveragePooling1D

    def infer_shape(self):
        super(TfKerasAveragePooling1DOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'AveragePool', 'version': 10}


class TfKerasAveragePooling2DOp(KerasPoolingOp):
    @classmethod
    def ufunc(cls):
        return tf.keras.layers.AveragePooling2D

    def infer_shape(self):
        super(TfKerasAveragePooling2DOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'AveragePool', 'version': 10}


class TfKerasAveragePooling3DOp(KerasPoolingOp):
    @classmethod
    def ufunc(cls):
        return tf.keras.layers.AveragePooling3D

    def infer_shape(self):
        super(TfKerasAveragePooling3DOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'AveragePool', 'version': 10}


class TfKerasBatchNormalizationOp(KerasNormalizationOp):
    @classmethod
    def attributes(cls):
        return {2: {'training_mode': {'type': AttrType.INT, 'default': 0}
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasBatchNormalizationOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasBatchNormalizationOp, attr_dict)
        assert self.check_required(), 'TfKerasBatchNormalizationOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'training_mode':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(TfKerasBatchNormalizationOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfKerasBatchNormalizationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 1, 'TfKerasBatchNormalizationOp expects at least 1 input, but got %d.' % len(inputs)
        if len(inputs) >= 2 and inputs[1].size == 1:
            self.training_mode = int(inputs[1].item())
        if self.axes is not None and len(self.axes) == 1:
            self.axis = self.axes[0]
        batchnorm = tf.keras.layers.BatchNormalization(self.axis,
                                                       epsilon=self.epsilon,
                                                       center=self.center,
                                                       scale=self.scale)
        out_tensor = batchnorm(inputs[0], training=self.training_mode).numpy()
        self.set_out_tensor(out_tensor)


class TfKerasCenterCropOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'height': {'type': AttrType.INT, 'required': True},
                    'width': {'type': AttrType.INT, 'required': True}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasCenterCropOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasCenterCropOp, attr_dict)
        assert self.check_required(), 'TfKerasCenterCropOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasCenterCropOp, self).infer_shape()
        inputs = self.get_input_tensors()
        crop = tf.keras.layers.CenterCrop(self.height, self.width)
        out_tensor = crop(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class TfKerasConcatenateOp(OpHasAxis, OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'axis': {'type': AttrType.INT, 'required': False, 'default': -1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasConcatenateOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasConcatenateOp, attr_dict)
        assert self.check_required(), 'TfKerasConcatenateOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasConcatenateOp, self).infer_shape()
        inputs = self.get_input_tensors()
        concat = tf.keras.layers.Concatenate(axis=self.axis)
        out_tensor = concat([*inputs]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Concat', 'version': 4}


class TfKerasConv1DOp(KerasBaseConvOp):
    @classmethod
    def attributes(cls):
        return {2: {'filters': {'type': AttrType.INT, 'required': True}}
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [2, 1, 0]

    def __init__(self, graph, attr_dict=None):
        super(TfKerasConv1DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasConv1DOp, attr_dict)
        assert self.check_required(), 'TfKerasConv1DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasConv1DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = np.transpose(inputs[0], [0, 2, 1]
                           ) if self.data_format.startswith('NC') else inputs[0]
        self.num_output = self.filters
        activation = None if self.activations == 'NONE' else self.activations.lower()
        biases = 'zeros' if self.biases is None else tf.keras.initializers.Constant(self.biases)
        conv1d = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_shape,
            strides=self.strides,
            padding='valid' if self.auto_pad == 'VALID' else 'same',
            data_format='channels_last',
            dilation_rate=self.dilations,
            groups=self.group,
            activation=activation,
            use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.Constant(self.weights),
            bias_initializer=biases)
        out_tensor = conv1d(inp).numpy()
        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                inp.shape[1:-1],
                out_tensor.shape[1:-1],
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
                zero_minimum=True,
            )
            self.auto_pad = 'NOTSET'
        if self.data_format.startswith('NC'):
            out_tensor = np.transpose(out_tensor, [0, 2, 1])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class TfKerasConv1DTransposeOp(KerasBaseDeconvOp):
    @classmethod
    def attributes(cls):
        return {2: {'filters': {'type': AttrType.INT, 'required': True},
                    },
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [2, 1, 0]

    def __init__(self, graph, attr_dict=None):
        super(TfKerasConv1DTransposeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasConv1DTransposeOp, attr_dict)
        assert self.check_required(), 'TfKerasConv1DTransposeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasConv1DTransposeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = np.transpose(inputs[0], [0, 2, 1]
                           ) if self.data_format.startswith('NC') else inputs[0]
        self.num_output = self.filters
        activation = None if self.activations == 'NONE' else self.activations.lower()
        biases = 'zeros' if self.biases is None else tf.keras.initializers.Constant(self.biases)
        conv1d_trans = tf.keras.layers.Conv1DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_shape,
            strides=self.strides,
            padding='valid' if self.auto_pad == 'VALID' else 'same',
            output_padding=self.output_padding,
            data_format='channels_last',
            dilation_rate=self.dilations,
            activation=activation,
            use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.Constant(self.weights),
            bias_initializer=biases)
        out_tensor = conv1d_trans(inp).numpy()
        self.update_pads(list(inp.shape[1:-1]), list(out_tensor.shape[1:-1]))
        if self.data_format.startswith('NC'):
            out_tensor = np.transpose(out_tensor, [0, 2, 1])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ConvTranspose', 'version': 11}


class TfKerasConv2DOp(KerasBaseConvOp):
    @classmethod
    def attributes(cls):
        return {2: {'filters': {'type': AttrType.INT, 'required': True}}
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [3, 2, 0, 1]

    def __init__(self, graph, attr_dict=None):
        super(TfKerasConv2DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasConv2DOp, attr_dict)
        assert self.check_required(), 'TfKerasConv2DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasConv2DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = np.transpose(inputs[0], [0, 2, 3, 1]
                           ) if self.data_format == 'NCHW' else inputs[0]
        self.num_output = self.filters
        conv2d = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_shape,
            strides=self.strides,
            padding='valid' if self.auto_pad == 'VALID' else 'same',
            data_format='channels_last',
            dilation_rate=self.dilations,
            groups=self.group)
        out_tensor = conv2d(inp).numpy()
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
        if self.data_format == 'NCHW':
            out_tensor = np.transpose(out_tensor, [0, 3, 1, 2])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class TfKerasConv2DTransposeOp(KerasBaseDeconvOp):
    @classmethod
    def attributes(cls):
        return {2: {'filters': {'type': AttrType.INT, 'required': True},
                    },
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [3, 2, 0, 1]

    def __init__(self, graph, attr_dict=None):
        super(TfKerasConv2DTransposeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasConv2DTransposeOp, attr_dict)
        assert self.check_required(), 'TfKerasConv2DTransposeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasConv2DTransposeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = np.transpose(inputs[0], [0, 2, 3, 1]
                           ) if self.data_format == 'NCHW' else inputs[0]
        self.num_output = self.filters
        conv2d = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_shape,
            strides=self.strides,
            padding='valid' if self.auto_pad == 'VALID' else 'same',
            output_padding=self.output_padding,
            data_format='channels_last',
            dilation_rate=self.dilations)
        out_tensor = conv2d(inp).numpy()
        self.update_pads(list(inp.shape[1:3]), list(out_tensor.shape[1:3]))
        if self.data_format == 'NCHW':
            out_tensor = np.transpose(out_tensor, [0, 3, 1, 2])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ConvTranspose', 'version': 11}


class TfKerasConv3DOp(KerasBaseConvOp):
    @classmethod
    def attributes(cls):
        return {2: {'filters': {'type': AttrType.INT, 'required': True}}
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [4, 3, 0, 1, 2]

    def __init__(self, graph, attr_dict=None):
        super(TfKerasConv3DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasConv3DOp, attr_dict)
        assert self.check_required(), 'TfKerasConv3DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasConv3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = np.transpose(inputs[0], [0, 2, 3, 4, 1]
                           ) if self.data_format == 'NCDHW' else inputs[0]
        self.num_output = self.filters
        conv3d = tf.keras.layers.Conv3D(
            filters=self.filters,
            kernel_size=self.kernel_shape,
            strides=self.strides,
            padding='valid' if self.auto_pad == 'VALID' else 'same',
            data_format='channels_last',
            dilation_rate=self.dilations,
            groups=self.group)
        out_tensor = conv3d(inp).numpy()
        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                inp.shape[1:4],
                out_tensor.shape[1:4],
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
                zero_minimum=True,
            )
            self.auto_pad = 'NOTSET'
        if self.data_format == 'NCDHW':
            out_tensor = np.transpose(out_tensor, [0, 4, 1, 2, 3])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class TfKerasConv3DTransposeOp(KerasBaseDeconvOp):
    @classmethod
    def attributes(cls):
        return {2: {'filters': {'type': AttrType.INT, 'required': True},
                    },
                }

    @classmethod
    def perm_tf_to_onnx(cls):
        return [4, 3, 0, 1, 2]

    def __init__(self, graph, attr_dict=None):
        super(TfKerasConv3DTransposeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasConv3DTransposeOp, attr_dict)
        assert self.check_required(), 'TfKerasConv3DTransposeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasConv3DTransposeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = np.transpose(inputs[0], [0, 2, 3, 4, 1]
                           ) if self.data_format == 'NCDHW' else inputs[0]
        self.num_output = self.filters
        conv3d = tf.keras.layers.Conv3DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_shape,
            strides=self.strides,
            padding='valid' if self.auto_pad == 'VALID' else 'same',
            output_padding=self.output_padding,
            data_format='channels_last',
            dilation_rate=self.dilations)
        out_tensor = conv3d(inp).numpy()
        self.update_pads(list(inp.shape[1:4]), list(out_tensor.shape[1:4]))
        if self.data_format == 'NCDHW':
            out_tensor = np.transpose(out_tensor, [0, 4, 1, 2, 3])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ConvTranspose', 'version': 11}


class TfKerasCropping1DOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'cropping': {'required': False, 'default': [1, 1]}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasCropping1DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasCropping1DOp, attr_dict)
        assert self.check_required(), 'TfKerasCropping1DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasCropping1DOp, self).infer_shape()
        if isinstance(self.cropping, int):
            self.cropping = [self.cropping, self.cropping]
        inputs = self.get_input_tensors()
        crop = tf.keras.layers.Cropping1D(self.cropping)
        out_tensor = crop(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Slice', 'version': 1}


class TfKerasCropping2DOp(OpHasOneOutPort, LayoutConcernedOp, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'cropping': {'required': False, 'default': [[0, 0], [0, 0]]}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasCropping2DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasCropping2DOp, attr_dict)
        assert self.check_required(), 'TfKerasCropping2DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasCropping2DOp, self).infer_shape()
        if isinstance(self.cropping, int):
            self.cropping = [[self.cropping] * 2] * 2
        elif isinstance(self.cropping, (list, tuple)):
            assert len(self.cropping) == 2, 'Meet invalid cropping in TfKerasCropping2DOp!'
            if all(isinstance(crop_val, int) for crop_val in self.cropping):
                self.cropping = [[self.cropping[0]] * 2, [self.cropping[1]] * 2]
        inputs = self.get_input_tensors()
        crop = tf.keras.layers.Cropping2D(
            self.cropping, 'channels_first' if self.data_format.startswith('NC') else 'channels_last')
        out_tensor = crop(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Slice', 'version': 1}


class TfKerasCropping3DOp(OpHasOneOutPort, LayoutConcernedOp, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'cropping': {'required': False, 'default': [[1, 1], [1, 1], [1, 1]]}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasCropping3DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasCropping3DOp, attr_dict)
        assert self.check_required(), 'TfKerasCropping3DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasCropping3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if isinstance(self.cropping, int):
            self.cropping = [[self.cropping] * 2] * 3
        elif isinstance(self.cropping, (list, tuple)):
            assert len(self.cropping) == 3, 'Meet invalid padding in TfKerasCropping3DOp!'
            if all(isinstance(crop_val, int) for crop_val in self.cropping):
                self.cropping = [[self.cropping[0]] * 2, [self.cropping[1]] * 2, [self.cropping[2]] * 2]
        crop = tf.keras.layers.Cropping3D(
            self.cropping, 'channels_first' if self.data_format.startswith('NC') else 'channels_last')
        out_tensor = crop(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Slice', 'version': 1}


class TfKerasELUOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 1.}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasELUOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasELUOp, attr_dict)
        assert self.check_required(), 'TfKerasELUOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasELUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0])
        mask = out_tensor <= 0
        out_tensor[mask] = self.alpha * (np.exp(out_tensor[mask]) - 1)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Elu', 'version': 6}


class TfKerasFlattenOp(OpHasOneOutPort, LayoutConcernedOp, KerasOp):
    def infer_shape(self):
        super(TfKerasFlattenOp, self).infer_shape()
        inputs = self.get_input_tensors()
        data_format = 'channels_first' if self.data_format.startswith('NC') else 'channels_last'
        flatten = tf.keras.layers.Flatten(data_format)
        out_tensor = flatten(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reshape', 'version': 1}


class TfKerasGlobalAveragePooling1DOp(KerasGlobalPoolingOp):
    def infer_shape(self):
        super(TfKerasGlobalAveragePooling1DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        global_avg_pool = tf.keras.layers.GlobalAveragePooling1D(self.keras_data_format, keepdims=self.keepdims)
        out_tensor = global_avg_pool(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'GlobalAveragePool', 'version': 1}


class TfKerasGlobalAveragePooling2DOp(KerasGlobalPoolingOp):
    def infer_shape(self):
        super(TfKerasGlobalAveragePooling2DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(self.keras_data_format, keepdims=self.keepdims)
        out_tensor = global_avg_pool(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'GlobalAveragePool', 'version': 1}


class TfKerasGlobalAveragePooling3DOp(KerasGlobalPoolingOp):
    def infer_shape(self):
        super(TfKerasGlobalAveragePooling3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        global_avg_pool = tf.keras.layers.GlobalAveragePooling3D(self.keras_data_format, keepdims=self.keepdims)
        out_tensor = global_avg_pool(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'GlobalAveragePool', 'version': 1}


class TfKerasGlobalMaxPooling1DOp(KerasGlobalPoolingOp):
    def infer_shape(self):
        super(TfKerasGlobalMaxPooling1DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        global_max_pool = tf.keras.layers.GlobalMaxPool1D(self.keras_data_format, keepdims=self.keepdims)
        out_tensor = global_max_pool(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'GlobalMaxPool', 'version': 1}


class TfKerasGlobalMaxPooling2DOp(KerasGlobalPoolingOp):
    def infer_shape(self):
        super(TfKerasGlobalMaxPooling2DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        global_max_pool = tf.keras.layers.GlobalMaxPool2D(self.keras_data_format, keepdims=self.keepdims)
        out_tensor = global_max_pool(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'GlobalMaxPool', 'version': 1}


class TfKerasGlobalMaxPooling3DOp(KerasGlobalPoolingOp):
    def infer_shape(self):
        super(TfKerasGlobalMaxPooling3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        global_max_pool = tf.keras.layers.GlobalMaxPool3D(self.keras_data_format, keepdims=self.keepdims)
        out_tensor = global_max_pool(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'GlobalMaxPool', 'version': 1}


class TfKerasGRUOp(KerasRecurrentOp):
    @classmethod
    def attributes(cls):
        return {2: {'reset_after': {'type': AttrType.INT, 'required': False, 'default': 1, 'options': [0, 1]},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasGRUOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasGRUOp, attr_dict)
        assert self.check_required(), 'TfKerasGRUOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'reset_after':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(TfKerasGRUOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfKerasGRUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 1, 'The length of inputs is invalid in TfKerasGRUOp infer shape.'
        assert inputs[0] is not None and len(
            inputs[0].shape) == 3, 'The first input is invalid in TfKerasGRUOp infer shape.'
        hidden_size = self.units
        if not self.time_major:
            batch, timesteps, feature = inputs[0].shape
            whole_out_shape = [batch, timesteps, hidden_size]
        else:
            timesteps, batch, feature = inputs[0].shape
            whole_out_shape = [timesteps, batch, hidden_size]
        state_out_shape = [batch, hidden_size]
        input_dtype = inputs[0].dtype
        if self.return_sequences:
            whole_or_state_out = np.random.ranf(whole_out_shape).astype(input_dtype)
        else:
            whole_or_state_out = np.random.ranf(state_out_shape).astype(input_dtype)
        if self.return_state:
            state_out = np.random.ranf(state_out_shape).astype(input_dtype)
            self.set_out_tensor([whole_or_state_out, state_out])
        else:
            self.set_out_tensor([whole_or_state_out])


class TfKerasInputLayerOp(OpHasOneOutPort, InputLikeOp, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'input_shape': {'type': AttrType.INTS, 'required': False, 'default': None},
                    'batch_size': {'type': AttrType.INT, 'required': False, 'default': None},
                    'dtype': {'type': AttrType.STRING, 'required': False, 'default': 'float32'},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasInputLayerOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasInputLayerOp, attr_dict)
        assert self.check_required(), 'TfKerasInputLayerOp is missing a required parameter.'

    def infer_shape(self, input_tensor=None):
        super(TfKerasInputLayerOp, self).infer_shape()
        out_tensor = input_tensor
        if out_tensor is None:
            try:
                if self.batch_size is None or self.input_shape is None:
                    out_tensor = None
                else:
                    shape = [self.batch_size] + self.input_shape[1:]
                    out_tensor = ((np.random.ranf(shape) - 0.5)
                                  * 100).astype(np.dtype(self.dtype))
            except:
                out_tensor = None
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Input', 'version': None}


class TfKerasLayerNormalizationOp(KerasNormalizationOp):
    def infer_shape(self):
        super(TfKerasLayerNormalizationOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 1, 'TfKerasLayerNormalizationOp expects at least 1 input, but got %d.' % len(inputs)
        layernorm = tf.keras.layers.LayerNormalization(self.axes,
                                                       epsilon=self.epsilon,
                                                       center=self.center,
                                                       scale=self.scale)
        out_tensor = layernorm(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LayerNorm', 'version': None}


class TfKerasLeakyReLUOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'alpha': {'type': AttrType.FLOAT, 'required': False, 'default': 0.3},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasLeakyReLUOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasLeakyReLUOp, attr_dict)
        assert self.check_required(), 'TfKerasLeakyReLUOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasLeakyReLUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        leaky_relu = tf.keras.layers.LeakyReLU(self.alpha)
        out_tensor = leaky_relu(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'LeakyRelu', 'version': 6}


class TfKerasLSTMOp(KerasRecurrentOp):
    @classmethod
    def attributes(cls):
        return {2: {},
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasLSTMOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasLSTMOp, attr_dict)
        assert self.check_required(), 'TfKerasLSTMOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasLSTMOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 1, 'The length of inputs is invalid in TfKerasLSTMOp infer shape.'
        assert inputs[0] is not None and len(
            inputs[0].shape) == 3, 'The first input is invalid in TfKerasLSTMOp infer shape.'
        hidden_size = self.units
        if not self.time_major:
            batch, timesteps, feature = inputs[0].shape
            whole_out_shape = [batch, timesteps, hidden_size]
        else:
            timesteps, batch, feature = inputs[0].shape
            whole_out_shape = [timesteps, batch, hidden_size]
        state_out_shape = [batch, hidden_size]
        input_dtype = inputs[0].dtype
        if self.return_sequences:
            whole_or_state_out = np.random.ranf(whole_out_shape).astype(input_dtype)
        else:
            whole_or_state_out = np.random.ranf(state_out_shape).astype(input_dtype)
        if self.return_state:
            hidden_state_out = np.random.ranf(state_out_shape).astype(input_dtype)
            cell_state_out = np.random.ranf(state_out_shape).astype(input_dtype)
            self.set_out_tensor([whole_or_state_out, hidden_state_out, cell_state_out])
        else:
            self.set_out_tensor([whole_or_state_out])


class TfKerasMaxPooling1DOp(KerasPoolingOp):
    @classmethod
    def ufunc(cls):
        return tf.keras.layers.MaxPool1D

    def infer_shape(self):
        super(TfKerasMaxPooling1DOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'MaxPool', 'version': 10}


class TfKerasMaxPooling2DOp(KerasPoolingOp):
    @classmethod
    def ufunc(cls):
        return tf.keras.layers.MaxPool2D

    def infer_shape(self):
        super(TfKerasMaxPooling2DOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'MaxPool', 'version': 10}


class TfKerasMaxPooling3DOp(KerasPoolingOp):
    @classmethod
    def ufunc(cls):
        return tf.keras.layers.MaxPool3D

    def infer_shape(self):
        super(TfKerasMaxPooling3DOp, self).infer_shape()

    @property
    def correspond_onnx_op(self):
        return {'type': 'MaxPool', 'version': 10}


class TfKerasMaximumOp(OpHasOneOutPort, KerasNeedBroadcast):
    @classmethod
    def attributes(cls):
        return {2: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfKerasMaximumOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasMaximumOp, attr_dict)
        assert self.check_required(), 'TfKerasMaximumOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasMaximumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.keras.layers.Maximum()([*inputs]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Max', 'version': 6}


class TfKerasMinimumOp(OpHasOneOutPort, KerasNeedBroadcast):
    @classmethod
    def attributes(cls):
        return {2: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfKerasMinimumOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasMinimumOp, attr_dict)
        assert self.check_required(), 'TfKerasMinimumOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasMinimumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.keras.layers.Minimum()([*inputs]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Min', 'version': 6}


class TfKerasMultiplyOp(OpHasOneOutPort, KerasNeedBroadcast):
    @classmethod
    def attributes(cls):
        return {2: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfKerasMultiplyOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasMultiplyOp, attr_dict)
        assert self.check_required(), 'TfKerasMultiplyOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasMultiplyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.keras.layers.Multiply()([*inputs]).numpy()
        self.set_out_tensor(out_tensor)


class TfKerasPermuteOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'dims': {'type': AttrType.INTS, 'required': True},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasPermuteOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasPermuteOp, attr_dict)
        assert self.check_required(), 'TfKerasPermuteOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasPermuteOp, self).infer_shape()
        inputs = self.get_input_tensors()
        permute = tf.keras.layers.Permute(self.dims)
        out_tensor = permute(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Transpose', 'version': 1}


class TfKerasPReLUOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'alpha': {'type': AttrType.TENSOR, 'required': False, 'default': None},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasPReLUOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasPReLUOp, attr_dict)
        assert self.check_required(), 'TfKerasPReLUOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasPReLUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.alpha is None:
            self.alpha = np.zeros_like(inputs[0])
        pos = tf.keras.layers.ReLU()(inputs[0]).numpy()
        neg = -self.alpha * (tf.keras.layers.ReLU()(-inputs[0]).numpy())
        out_tensor = pos + neg
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'PRelu', 'version': 6}


class TfKerasReshapeOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'target_shape': {'type': AttrType.INTS, 'required': True},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasReshapeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasReshapeOp, attr_dict)
        assert self.check_required(), 'TfKerasReshapeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasReshapeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        reshape = tf.keras.layers.Reshape(self.target_shape)
        out_tensor = reshape(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reshape', 'version': 1}


class TfKerasResizingOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'height': {'type': AttrType.INT, 'required': True},
                    'width': {'type': AttrType.INT, 'required': True},
                    'interpolation': {'type': AttrType.STRING, 'required': False, 'default': 'bilinear'},
                    'crop_to_aspect_ratio': {'type': AttrType.INT, 'required': False, 'default': 0, 'options': [0, 1]}
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasResizingOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasResizingOp, attr_dict)
        assert self.check_required(), 'TfKerasResizingOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'crop_to_aspect_ratio':
                ret = bool(self.__dict__['_attr'][item].value)
        except:
            ret = None
        if ret is None:
            ret = super(TfKerasResizingOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfKerasResizingOp, self).infer_shape()
        inputs = self.get_input_tensors()
        resize = tf.keras.layers.Resizing(self.height, self.width, self.interpolation, self.crop_to_aspect_ratio)
        out_tensor = resize(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class TfKerasSubtractOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {}}

    def __init__(self, graph, attr_dict=None):
        super(TfKerasSubtractOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasSubtractOp, attr_dict)
        assert self.check_required(), 'TfKerasSubtractOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasSubtractOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.keras.layers.Subtract()([*inputs]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sub', 'version': 6}


class TfKerasThresholdedReLUOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'theta': {'type': AttrType.FLOAT, 'required': False, 'default': 1.0},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasThresholdedReLUOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasThresholdedReLUOp, attr_dict)
        assert self.check_required(), 'TfKerasThresholdedReLUOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasThresholdedReLUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        thresholded_relu = tf.keras.layers.ThresholdedReLU(self.theta)
        out_tensor = thresholded_relu(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ThresholdedRelu', 'version': 10}


class TfKerasUpSampling1DOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'size': {'type': AttrType.INT, 'required': False, 'default': 2},
                    'interpolation': {'type': AttrType.STRING, 'required': False, 'default': 'nearest'},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasUpSampling1DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasUpSampling1DOp, attr_dict)
        assert self.check_required(), 'TfKerasUpSampling1DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasUpSampling1DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        upsample = tf.keras.layers.UpSampling1D(self.size)
        out_tensor = upsample(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class TfKerasUpSampling2DOp(OpHasOneOutPort, LayoutConcernedOp, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'size': {'required': False, 'default': [2, 2]},
                    'interpolation': {'type': AttrType.STRING, 'required': False, 'default': 'nearest'},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasUpSampling2DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasUpSampling2DOp, attr_dict)
        assert self.check_required(), 'TfKerasUpSampling2DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasUpSampling2DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if isinstance(self.size, int):
            self.size = [self.size] * 2
        resize = tf.keras.layers.UpSampling2D(self.size,
                                              'channels_first' if self.data_format.startswith(
                                                  'NC') else 'channels_last',
                                              self.interpolation)
        out_tensor = resize(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class TfKerasUpSampling3DOp(OpHasOneOutPort, LayoutConcernedOp, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'size': {'required': False, 'default': [2, 2, 2]},
                    'interpolation': {'type': AttrType.STRING, 'required': False, 'default': 'nearest'},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasUpSampling3DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasUpSampling3DOp, attr_dict)
        assert self.check_required(), 'TfKerasUpSampling3DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasUpSampling3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if isinstance(self.size, int):
            self.size = [self.size] * 3
        upsample = tf.keras.layers.UpSampling3D(self.size,
                                                'channels_first' if self.data_format.startswith('NC') else 'channels_last')
        out_tensor = upsample(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)


class TfKerasZeroPadding1DOp(OpHasOneOutPort, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'padding': {'required': False, 'default': 1}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasZeroPadding1DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasZeroPadding1DOp, attr_dict)
        assert self.check_required(), 'TfKerasZeroPadding1DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasZeroPadding1DOp, self).infer_shape()
        if isinstance(self.padding, int):
            self.padding = [self.padding, self.padding]
        inputs = self.get_input_tensors()
        zero_pad = tf.keras.layers.ZeroPadding1D(self.padding)
        out_tensor = zero_pad(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pad', 'version': 1}


class TfKerasZeroPadding2DOp(OpHasOneOutPort, LayoutConcernedOp, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'padding': {'required': False, 'default': [1, 1]}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasZeroPadding2DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasZeroPadding2DOp, attr_dict)
        assert self.check_required(), 'TfKerasZeroPadding2DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasZeroPadding2DOp, self).infer_shape()
        if isinstance(self.padding, int):
            self.padding = [[self.padding] * 2] * 2
        elif isinstance(self.padding, (list, tuple)):
            assert len(self.padding) == 2, 'Meet invalid padding in TfKerasZeroPadding2DOp!'
            if all(isinstance(pad, int) for pad in self.padding):
                self.padding = [[self.padding[0]] * 2, [self.padding[1]] * 2]
        inputs = self.get_input_tensors()
        zero_pad = tf.keras.layers.ZeroPadding2D(
            self.padding, 'channels_first' if self.data_format.startswith('NC') else 'channels_last')
        out_tensor = zero_pad(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pad', 'version': 1}


class TfKerasZeroPadding3DOp(OpHasOneOutPort, LayoutConcernedOp, KerasOp):
    @classmethod
    def attributes(cls):
        return {2: {'padding': {'required': False, 'default': [1, 1, 1]}}
                }

    def __init__(self, graph, attr_dict=None):
        super(TfKerasZeroPadding3DOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfKerasZeroPadding3DOp, attr_dict)
        assert self.check_required(), 'TfKerasZeroPadding3DOp is missing a required parameter.'

    def infer_shape(self):
        super(TfKerasZeroPadding3DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if isinstance(self.padding, int):
            self.padding = [[self.padding] * 2] * 3
        elif isinstance(self.padding, (list, tuple)):
            assert len(self.padding) == 3, 'Meet invalid padding in TfKerasZeroPadding3DOp!'
            if all(isinstance(pad, int) for pad in self.padding):
                self.padding = [[self.padding[0]] * 2, [self.padding[1]] * 2, [self.padding[2]] * 2]
        zero_pad = tf.keras.layers.ZeroPadding3D(
            self.padding, 'channels_first' if self.data_format.startswith('NC') else 'channels_last')
        out_tensor = zero_pad(inputs[0]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pad', 'version': 1}
