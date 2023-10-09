# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import tensorflow as tf
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfCropAndResizeOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'method': {'type': AttrType.STRING, 'default': 'bilinear', 'options': ['bilinear', 'nearest']},
                    'extrapolation_value': {'type': AttrType.FLOAT, 'default': 0.},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfCropAndResizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfCropAndResizeOp, attr_dict)
        assert self.check_required(), 'TfCropAndResizeOp is missing a required parameter.'

    def infer_shape(self):
        super(TfCropAndResizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.image.crop_and_resize(*inputs, method=self.method,
                                              extrapolation_value=self.extrapolation_value).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'CropAndResize', 'version': 1}


class TfNonMaxSuppressionV2Op(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'max_output_size': {'type': AttrType.INT, 'required': False},
                    'iou_threshold': {'type': AttrType.FLOAT, 'required': False}
                    }
                }

    def __init__(self, graph, attr_dict=None):
        super(TfNonMaxSuppressionV2Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfNonMaxSuppressionV2Op, attr_dict)
        assert self.check_required(), 'TfNonMaxSuppressionV2Op is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        if item in ('max_output_size', 'iou_threshold'):
            inputs = self.get_input_tensors()
            try:
                if item == 'max_output_size':
                    ret = int(inputs[2].item())
                    self.__dict__['_attr'][item].value = ret
                elif item == 'iou_threshold':
                    ret = float(inputs[3].item())
                    self.__dict__['_attr'][item].value = ret
            except:
                ret = None
        if ret is None:
            ret = super(TfNonMaxSuppressionV2Op, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(TfNonMaxSuppressionV2Op, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.raw_ops.NonMaxSuppressionV2(boxes=inputs[0],
                                                    scores=inputs[1],
                                                    max_output_size=self.max_output_size,
                                                    iou_threshold=self.iou_threshold).numpy()
        self.set_out_tensor(out_tensor)


class TfNonMaxSuppressionV3Op(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'max_output_size': {'type': AttrType.INT, 'required': False},
                    'iou_threshold': {'type': AttrType.FLOAT, 'required': False},
                    'score_threshold': {'type': AttrType.FLOAT, 'required': False},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfNonMaxSuppressionV3Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfNonMaxSuppressionV3Op, attr_dict)
        assert self.check_required(), 'TfNonMaxSuppressionV3Op is missing a required parameter.'

    def infer_shape(self):
        super(TfNonMaxSuppressionV3Op, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) >= 5, 'The length of inputs is invalid in TfNonMaxSuppressionV3Op.'
        self.max_output_size, self.iou_threshold, self.score_threshold = [
            np.asscalar(inp) for inp in inputs[2:5]]
        # Do not use tf.image.non_max_suppression because the actual output shape is different
        # with the inferred output shape.
        pad_to_max_output_size = False if self.is_all_inputs_const() else True
        out_tensors = tf.image.non_max_suppression_padded(*(inputs[:2]),
                                                          self.max_output_size,
                                                          self.iou_threshold,
                                                          self.score_threshold,
                                                          pad_to_max_output_size=pad_to_max_output_size)
        out_tensor = out_tensors[0].numpy()
        self.set_out_tensor(out_tensor)


class TfNonMaxSuppressionV4Op(OpHasMultipleOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'max_output_size': {'type': AttrType.INT, 'required': False},
                    'iou_threshold': {'type': AttrType.FLOAT, 'required': False},
                    'score_threshold': {'type': AttrType.FLOAT, 'required': False},
                    'pad_to_max_output_size': {'type': AttrType.INT, 'required': False, 'default': 0}
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfNonMaxSuppressionV4Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfNonMaxSuppressionV4Op, attr_dict)
        assert self.check_required(), 'TfNonMaxSuppressionV4Op is missing a required parameter.'

    def infer_shape(self):
        super(TfNonMaxSuppressionV4Op, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 5, 'The length of inputs is invalid in TfNonMaxSuppressionV4Op.'
        self.max_output_size, self.iou_threshold, self.score_threshold = [
            np.asscalar(inp) for inp in inputs[2:5]]
        pad_to_max_output_size = self.pad_to_max_output_size
        if not self.is_all_inputs_const():
            if self.pad_to_max_output_size:
                WARN('[Parser]: TfNonMaxSuppressionV5Op does not support pad_to_max_output_size=True! Will treat it as False!')
            else:
                pad_to_max_output_size = True
        out_tensors = tf.raw_ops.NonMaxSuppressionV4(
            boxes=inputs[0],
            scores=inputs[1],
            max_output_size=self.max_output_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            pad_to_max_output_size=pad_to_max_output_size)
        out_tensors = [t.numpy() for t in out_tensors]
        self.set_out_tensor(out_tensors)


class TfNonMaxSuppressionV5Op(OpHasMultipleOutPorts, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'max_output_size': {'type': AttrType.INT, 'required': False},
                    'iou_threshold': {'type': AttrType.FLOAT, 'required': False},
                    'score_threshold': {'type': AttrType.FLOAT, 'required': False},
                    'soft_nms_sigma': {'type': AttrType.FLOAT, 'required': False, 'default': 0.0},
                    'pad_to_max_output_size': {'type': AttrType.INT, 'required': False, 'default': 0}
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfNonMaxSuppressionV5Op, self).__init__(graph, attr_dict)
        self.update_attributes(TfNonMaxSuppressionV5Op, attr_dict)
        assert self.check_required(), 'TfNonMaxSuppressionV5Op is missing a required parameter.'

    def infer_shape(self):
        super(TfNonMaxSuppressionV5Op, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) >= 6, 'The length of inputs is invalid in TfNonMaxSuppressionV5Op.'
        self.max_output_size, self.iou_threshold, self.score_threshold, self.soft_nms_sigma = [
            np.asscalar(inp) for inp in inputs[2:6]]
        pad_to_max_output_size = bool(self.pad_to_max_output_size)
        if not self.is_all_inputs_const():
            if pad_to_max_output_size:
                WARN('[Parser]: TfNonMaxSuppressionV5Op does not support pad_to_max_output_size=True! Will treat it as False!')
            else:
                pad_to_max_output_size = True
        out_tensors = tf.raw_ops.NonMaxSuppressionV5(
            boxes=inputs[0],
            scores=inputs[1],
            max_output_size=self.max_output_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            soft_nms_sigma=self.soft_nms_sigma,
            pad_to_max_output_size=pad_to_max_output_size)
        out_tensors = [t.numpy() for t in out_tensors]
        self.set_out_tensor(out_tensors)


class TfResizeBilinearOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'align_corners': {'type': AttrType.INT, 'required': False, 'default': 0},
                    'half_pixel_centers': {'type': AttrType.INT, 'required': False, 'default': 0},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfResizeBilinearOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfResizeBilinearOp, attr_dict)
        assert self.check_required(), 'TfResizeBilinearOp is missing a required parameter.'

    def infer_shape(self):
        super(TfResizeBilinearOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.compat.v1.image.resize_bilinear(
            *inputs, align_corners=self.align_corners, half_pixel_centers=self.half_pixel_centers).numpy()
        self.set_out_tensor(out_tensor)


class TfResizeNearestNeighborOp(OpHasOneOutPort, TfOp):
    @classmethod
    def attributes(cls):
        return {1: {'align_corners': {'type': AttrType.INT, 'required': False, 'default': 0},
                    'half_pixel_centers': {'type': AttrType.INT, 'required': False, 'default': 0},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(TfResizeNearestNeighborOp, self).__init__(graph, attr_dict)
        self.update_attributes(TfResizeNearestNeighborOp, attr_dict)
        assert self.check_required(), 'TfResizeNearestNeighborOp is missing a required parameter.'

    def infer_shape(self):
        super(TfResizeNearestNeighborOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.compat.v1.image.resize_nearest_neighbor(
            *inputs, align_corners=self.align_corners, half_pixel_centers=self.half_pixel_centers).numpy()
        self.set_out_tensor(out_tensor)
