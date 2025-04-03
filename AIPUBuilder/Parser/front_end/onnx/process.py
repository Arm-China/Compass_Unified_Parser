# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from .load import convert_onnx_to_graph
from ...graph.graph_algo import infer
from .passes.front_passes import fuse_weights_const, convert_special_prelu, merge_qconv, merge_qmatmul, \
    merge_q_multiple, merge_q_unary, convert_special_sequence_construct, merge_sequence_construct_and_at, \
    merge_sequence_construct_and_concat, merge_rcnn, convert_mmcv_deform_conv, \
    merge_qgemm, uplift_quant, uplift_quant_through_concat, merge_qconv_no_bias
from .passes.common_passes import remove_useless_op, apply_subgraph_plugin, record_output_tensors, \
    merge_same_op_at_out_port
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


def front_process_onnx(graph, params):
    record_output_tensors(graph, params)
    if graph is not None and len(graph) > 0:
        apply_subgraph_plugin(graph)

        for i in range(2):
            remove_useless_op(
                graph, ['Blank', 'Transpose', 'Reshape', 'Upsample', 'Identity', 'Cast', 'Concat'])

        infer(graph, partial=True)
        merge_rcnn(graph, params)
        uplift_quant_through_concat(graph)
        merge_same_op_at_out_port(graph, op_types=['QuantizeLinear', 'DequantizeLinear'])
        uplift_quant(graph)
        merge_qconv(graph)
        merge_qconv_no_bias(graph)
        merge_qmatmul(graph)
        merge_q_multiple(graph, ['Add', 'Concat', 'Gemm', 'Mul', 'Split', 'Gather'])
        merge_q_unary(graph, ['AdaptivePool', 'AveragePool', 'Celu', 'Clip', 'Elu', 'Expand', 'Flatten',
                              'GlobalAveragePool', 'GlobalMaxPool',
                              'HardSwish', 'HardSigmoid', 'LeakyRelu', 'LRN', 'MaxPool',
                              'ReduceMean', 'Relu', 'Reshape',
                              'Slice', 'Sigmoid', 'Softmax', 'Squeeze', 'Transpose',
                              ])
        from ..lite.passes.front_passes import split_quatized_mean
        split_quatized_mean(graph, 'ReduceMean')

        convert_mmcv_deform_conv(graph)

        fuse_weights_const(graph)
        convert_special_prelu(graph)
        merge_sequence_construct_and_at(graph)
        convert_special_sequence_construct(graph)
        merge_sequence_construct_and_concat(graph)
        infer(graph)

        merge_qgemm(graph)

    else:
        WARN('[Parser]: Got empty graph in front_process_onnx!')


def process_onnx(graph, model_path, params):
    '''Do some preprocessing on the graph under the onnx framework.'''
    graph = convert_onnx_to_graph(graph, model_path, params)
    front_process_onnx(graph, params)

    return graph
