# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ..tf_ops.image_ops import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class Tfnon_max_suppressionOp(TfNonMaxSuppressionV3Op, Tf2Op):
    pass


class Tfnon_max_suppression_with_scoresOp(OpHasMultipleOutPorts, Tf2Op):
    @classmethod
    def attributes(cls):
        return {2: {'max_output_size': {'type': AttrType.INT, 'required': False},
                    'iou_threshold': {'type': AttrType.FLOAT, 'required': False, 'default': 0.5},
                    'score_threshold': {'type': AttrType.FLOAT, 'required': False, 'default': float('-inf')},
                    'soft_nms_sigma': {'type': AttrType.FLOAT, 'required': False, 'default': 0.0},
                    },
                }

    def __init__(self, graph, attr_dict=None):
        super(Tfnon_max_suppression_with_scoresOp, self).__init__(graph, attr_dict)
        self.update_attributes(Tfnon_max_suppression_with_scoresOp, attr_dict)
        assert self.check_required(), 'Tfnon_max_suppression_with_scoresOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            input_args = ['boxes', 'scores', 'max_output_size', 'iou_threshold', 'score_threshold', 'soft_nms_sigma']
            if item in input_args[2:]:
                item_idx = input_args.index(item)
                inputs = self.get_input_tensors()
                if len(inputs) > item_idx and inputs[item_idx].size == 1:
                    ret = inputs[item_idx].item()
                    if ret is not None:
                        self.__dict__['_attr'][item].value = ret
        except:
            ret = None
        if ret is None:
            ret = super(Tfnon_max_suppression_with_scoresOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Tfnon_max_suppression_with_scoresOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) >= 3, 'Tfnon_max_suppression_with_scoresOp expects at least 3 inputs, but got %d' % len(inputs)
        out_tensors = tf.raw_ops.NonMaxSuppressionV5(
            boxes=inputs[0],
            scores=inputs[1],
            max_output_size=self.max_output_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            soft_nms_sigma=self.soft_nms_sigma,
            pad_to_max_output_size=False if self.is_all_inputs_const() else True)
        # non_max_suppression_with_scores has only 2 outputs: selected_indices, selected_scores;
        # while NonMaxSuppressionV5 has 3 outputs: selected_indices, selected_scores, valid_outputs
        out_tensors = [t.numpy() for t in out_tensors[:2]]
        self.set_out_tensor(out_tensors)
