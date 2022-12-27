# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from functools import cmp_to_key
import torch
from ..op import *


class NonMaxSuppressionOp(OpHasOneOutPort, OnnxOp):

    @staticmethod
    def min_max(lhs, rhs):
        if lhs >= rhs:
            min = rhs
            max = lhs
        else:
            min = lhs
            max = rhs
        return min, max

    @staticmethod
    def suppress_by_iou(boxes_data, box_index1, box_index2, center_point_box, iou_threshold):
        box1 = boxes_data[box_index1]
        box2 = boxes_data[box_index2]

        # // center_point_box_ only support 0 or 1
        if 0 == center_point_box:
            # boxes data format [y1, x1, y2, x2],
            x1_min, x1_max = NonMaxSuppressionOp.min_max(box1[1], box1[3])
            x2_min, x2_max = NonMaxSuppressionOp.min_max(box2[1], box2[3])

            intersection_x_min = max(x1_min, x2_min)
            intersection_x_max = min(x1_max, x2_max)
            if intersection_x_max <= intersection_x_min:
                return False

            y1_min, y1_max = NonMaxSuppressionOp.min_max(box1[0], box1[2])
            y2_min, y2_max = NonMaxSuppressionOp.min_max(box2[0], box2[2])
            intersection_y_min = max(y1_min, y2_min)
            intersection_y_max = min(y1_max, y2_max)
            if intersection_y_max <= intersection_y_min:
                return False
        else:
            # // 1 == center_point_box_ => boxes data format [x_center, y_center, width, height]
            box1_width_half = box1[2] / 2
            box1_height_half = box1[3] / 2
            box2_width_half = box2[2] / 2
            box2_height_half = box2[3] / 2

            x1_min = box1[0] - box1_width_half
            x1_max = box1[0] + box1_width_half
            x2_min = box2[0] - box2_width_half
            x2_max = box2[0] + box2_width_half

            intersection_x_min = max(x1_min, x2_min)
            intersection_x_max = min(x1_max, x2_max)
            if intersection_x_max <= intersection_x_min:
                return False

            y1_min = box1[1] - box1_height_half
            y1_max = box1[1] + box1_height_half
            y2_min = box2[1] - box2_height_half
            y2_max = box2[1] + box2_height_half

            intersection_y_min = max(y1_min, y2_min)
            intersection_y_max = min(y1_max, y2_max)
            if intersection_y_max <= intersection_y_min:
                return False

        intersection_area = (intersection_x_max - intersection_x_min) * \
            (intersection_y_max - intersection_y_min)
        if intersection_area <= .0:
            return False

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - intersection_area
        if area1 <= .0 or area2 <= .0 or union_area <= .0:
            return False

        intersection_over_union = intersection_area / union_area
        return intersection_over_union > iou_threshold

    @staticmethod
    def compute(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box):
        # boxes: [num_batches, num_boxes, 4]
        # scores: [num_batches, num_classes, num_boxes]
        def _box_comparator(a, b):
            if a[0] < b[0] or (a[0] == b[0] and a[1] > b[1]):
                return -1
            elif a[0] == b[0] and a[1] == b[1]:
                return 0
            else:
                return 1

        last_dim = 3
        if max_output_boxes_per_class == 0:
            return np.array([0, last_dim], np.int64)

        num_batches = boxes.shape[0]
        num_classes = scores.shape[1]
        num_boxes = boxes.shape[1]
        selected_indices = []
        for b in range(num_batches):
            for c in range(num_classes):
                class_scores = scores[b, c, ...]
                batch_boxes = boxes[b, ...]
                candidate_boxes = []
                for box_index in range(num_boxes):
                    cur_class_score = class_scores[box_index]
                    if cur_class_score > score_threshold:
                        candidate_boxes.append((cur_class_score, box_index))
                sorted_boxes = sorted(
                    candidate_boxes, key=cmp_to_key(_box_comparator))
                selected_boxes_inside_class = []
                while sorted_boxes and len(selected_boxes_inside_class) < max_output_boxes_per_class:
                    next_top_score = sorted_boxes.pop()
                    selected = True
                    for i, selected_index in enumerate(selected_boxes_inside_class):
                        if NonMaxSuppressionOp.suppress_by_iou(batch_boxes, next_top_score[1], selected_index[1], center_point_box, iou_threshold):
                            selected = False
                            break
                    if selected:
                        selected_boxes_inside_class.append(next_top_score)
                        selected_indices.append((b, c, next_top_score[1]))
        num_selected = len(selected_indices)
        outputs = np.zeros(
            (max(max_output_boxes_per_class, num_selected), last_dim), np.int64)
        for i in range(num_selected):
            if i < max_output_boxes_per_class:
                outputs[i] = np.array(selected_indices[i], np.int64)
        return outputs

    @classmethod
    def attributes(cls):
        return {10: {'center_point_box': {'type': AttrType.INT, 'default': 0},
                     'max_output_boxes_per_class': {'type': AttrType.INT, 'default': 0},
                     'iou_threshold': {'type': AttrType.FLOAT, 'default': 0.},
                     'score_threshold': {'type': AttrType.FLOAT, 'default': 0.}
                     },
                11: {'center_point_box': {'type': AttrType.INT, 'default': 0},
                     'max_output_boxes_per_class': {'type': AttrType.INT, 'default': 0},
                     'iou_threshold': {'type': AttrType.FLOAT, 'default': 0.},
                     'score_threshold': {'type': AttrType.FLOAT, 'default': 0.}
                     }
                }

    def __init__(self, graph, attr_dict=None):
        super(NonMaxSuppressionOp, self).__init__(graph, attr_dict)
        self.update_attributes(NonMaxSuppressionOp, attr_dict)
        assert self.check_required(), 'NonMaxSuppressionOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        if item == 'center_point_box':
            ret = bool(self.__dict__['_attr'][item].value)
        elif item in ('max_output_boxes_per_class', 'iou_threshold', 'score_threshold'):
            inputs = self.get_input_tensors()
            try:
                if item == 'max_output_boxes_per_class':
                    ret = int(np.asscalar(inputs[2]))
                elif item == 'iou_threshold':
                    ret = float(np.asscalar(inputs[3]))
                elif item == 'score_threshold':
                    ret = float(np.asscalar(inputs[4]))
                self.__dict__['_attr'][item].value = ret
            except:
                ret = None
        if ret is None:
            ret = super(NonMaxSuppressionOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(NonMaxSuppressionOp, self).infer_shape()
        inputs = self.get_input_tensors()
        boxes, scores = inputs[0:2]
        box_shape = inputs[0].shape
        score_shape = inputs[1].shape
        max_output_boxes_per_class = min(
            int(np.asscalar(inputs[2])), box_shape[0] * box_shape[1] * score_shape[1])
        out_tensor = NonMaxSuppressionOp.compute(boxes,
                                                 scores,
                                                 max_output_boxes_per_class,
                                                 self.iou_threshold,
                                                 self.score_threshold,
                                                 self.center_point_box)
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            self.cur_version = max_ver

        from ...front_end.onnx.passes.common_passes import insert_constant
        in_edges = self._graph.sorted_in_edges(self.name, data=True)
        iou_threshold = self.iou_threshold
        score_threshold = self.score_threshold
        max_output_boxes_per_class = self.max_output_boxes_per_class
        if len(in_edges) <= 4:
            insert_constant(self._graph, self.name + '_score_threshold',
                            np.array(score_threshold), self.name, in_port=4)
        if len(in_edges) <= 3:
            insert_constant(self._graph, self.name + '_iou_threshold',
                            np.array(iou_threshold), self.name, in_port=3)
        if len(in_edges) <= 2:
            insert_constant(self._graph, self.name + '_max_output_boxes_per_class',
                            np.array(max_output_boxes_per_class), self.name, in_port=2)


class RoiAlignOp(LayoutConcernedOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {10: {'mode': {'type': AttrType.STRING, 'default': 'avg'},
                     'output_height': {'type': AttrType.INT, 'default': 1},
                     'output_width': {'type': AttrType.INT, 'default': 1},
                     'sampling_ratio': {'type': AttrType.INT, 'default': 0},
                     'spatial_scale': {'type': AttrType.FLOAT, 'default': 1.0},
                     },
                16: {'coordinate_transformation_mode': {'type': AttrType.STRING, 'default': 'half_pixel', 'options': ['half_pixel', 'output_half_pixel']},
                     'mode': {'type': AttrType.STRING, 'default': 'avg'},
                     'output_height': {'type': AttrType.INT, 'default': 1},
                     'output_width': {'type': AttrType.INT, 'default': 1},
                     'sampling_ratio': {'type': AttrType.INT, 'default': 0},
                     'spatial_scale': {'type': AttrType.FLOAT, 'default': 1.0},
                     },
                }

    def __init__(self, graph, attr_dict=None):
        super(RoiAlignOp, self).__init__(graph, attr_dict)
        self.update_attributes(RoiAlignOp, attr_dict)
        assert self.check_required(), 'RoiAlignOp is missing a required parameter.'

    def infer_shape(self):
        super(RoiAlignOp, self).infer_shape()
        inputs = self.get_input_tensors()
        rois = inputs[1].shape[0]
        if self.data_format == 'NHWC':
            channels = inputs[0].shape[-1]
            out_tensor = np.random.ranf(
                (rois, self.output_height, self.output_width, channels)).astype(np.float32)
        else:
            channels = inputs[0].shape[1]
            out_tensor = np.random.ranf(
                (rois, channels, self.output_height, self.output_width)).astype(np.float32)
        self.set_out_tensor(out_tensor)

    def convert_version(self):
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            if cur_ver == 10:
                self._attr['coordinate_transformation_mode'] = Attribute(
                    'coordinate_transformation_mode', {'type': AttrType.STRING, 'value': 'output_half_pixel'})
            self.cur_version = max_ver
