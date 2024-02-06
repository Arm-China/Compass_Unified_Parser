# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


from .load import convert_onnx_to_graph
from ...graph.graph_algo import infer
from .passes.front_passes import fuse_weights_const, convert_special_prelu, merge_qconv, merge_qmatmul, \
    merge_q_multiple, merge_q_unary, convert_special_sequence_construct, merge_sequence_construct_and_at, \
    decompose_loop, merge_rcnn, convert_mmcv_deform_conv
from .passes.common_passes import remove_useless_op, apply_subgraph_plugin, record_output_tensors
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


def process_onnx(graph, model_path, params):
    '''Do some preprocessing on the graph under the onnx framework.'''
    graph = convert_onnx_to_graph(graph, model_path, params)
    record_output_tensors(graph, params)
    if graph is not None and len(graph) > 0:
        apply_subgraph_plugin(graph)

        for i in range(2):
            remove_useless_op(
                graph, ['Dummy', 'Transpose', 'Reshape', 'Upsample', 'Identity', 'Cast', 'Concat'])

        infer(graph, partial=True)
        merge_rcnn(graph, params)
        merge_qconv(graph)
        merge_qmatmul(graph)
        merge_q_multiple(graph, ['Add', 'Concat', 'Gemm', 'Mul'])
        merge_q_unary(graph, ['AdaptivePool', 'AveragePool', 'Celu', 'Clip', 'Elu', 'Flatten',
                              'GlobalAveragePool', 'GlobalMaxPool',
                              'HardSwish', 'HardSigmoid', 'LeakyRelu', 'LRN', 'MaxPool',
                              'ReduceMean', 'Relu', 'Reshape', 'Slice', 'Sigmoid', 'Transpose',
                              ])
        convert_mmcv_deform_conv(graph)

        fuse_weights_const(graph)
        convert_special_prelu(graph)
        merge_sequence_construct_and_at(graph)
        convert_special_sequence_construct(graph)
        decompose_loop(graph, params)
        infer(graph)

    else:
        WARN('[Parser]: Got empty graph in process_onnx!')
    return graph
