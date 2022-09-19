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

from .load import convert_onnx_to_graph
from ...graph.graph_algo import infer
from .passes.front_passes import fuse_weights_const, convert_special_prelu
from .passes.common_passes import remove_useless_op, apply_subgraph_plugin, record_output_tensors
from ...common.errors import *


def process_onnx(model_path, params):
    '''Do some preprocessing on the graph under the onnx framework.'''
    graph = convert_onnx_to_graph(model_path, params)
    record_output_tensors(graph)
    if graph is not None and len(graph) > 0:
        apply_subgraph_plugin(graph)

        for i in range(2):
            remove_useless_op(
                graph, ['Dummy', 'Transpose', 'Reshape', 'Upsample', 'Identity'])
        infer(graph, partial=True)
        fuse_weights_const(graph)
        convert_special_prelu(graph)
        infer(graph)
    else:
        WARN('[Parser]: Got empty graph in process_onnx!')
    return graph
