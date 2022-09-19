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

from .onnx_ops.array_ops import *
from .onnx_ops.control_ops import *
from .onnx_ops.detection_ops import *
from .onnx_ops.logic_ops import *
from .onnx_ops.math_ops import *
from .onnx_ops.nn_ops import *
from .onnx_ops.quantize_ops import *
from .onnx_ops.random_ops import *
from .onnx_ops.reduce_ops import *
from .onnx_ops.sequence_ops import *
from .tf_ops.array_ops import *
from .tf_ops.bitwise_ops import *
from .tf_ops.control_flow_ops import *
from .tf_ops.data_flow_ops import *
from .tf_ops.image_ops import *
from .tf_ops.linalg_ops import *
from .tf_ops.math_ops import *
from .tf_ops.nn_ops import *
from .tf_ops.random_ops import *
from .tf_ops.sparse_ops import *
from .tflite_ops import *
from .caffe_ops import *
from .common_ops import *
from .release_ops import *
from ..common.errors import *


def op_factory(graph, op_type, node_attr=None):
    ret = None
    try:
        tmp_type = copy.deepcopy(op_type)
        if graph._attr.get('framework', Framework.NONE) == Framework.TFLITE:
            tmp_type = re.sub(r'^Lite', '', tmp_type, count=1)
        elif graph._attr.get('framework', Framework.NONE) == Framework.CAFFE:
            tmp_type = re.sub(r'^Caffe', '', tmp_type, count=1)
        elif graph._attr.get('framework', Framework.NONE) == Framework.TENSORFLOW:
            tmp_type = re.sub(r'^Tf', '', tmp_type, count=1)
        if tmp_type in PARSER_OP_DICT:
            node_attr.update({'type': tmp_type})
            ret = PluginOp(graph, node_attr)
        else:
            ret = eval(op_type + 'Op(graph, node_attr)')
    except Exception as e:
        WARN('[Parser]: Node(%s) creating operator failed (%s)!' %
             (node_attr.get('name', ''), str(e)))
        node_attr.update({'type': op_type})
        ret = eval('Undefined' + 'Op(graph, node_attr)')
    return ret
