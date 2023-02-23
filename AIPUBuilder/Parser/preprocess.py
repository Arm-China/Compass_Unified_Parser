# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# cython: language_level=3
import numpy as np
import re
import copy
from .common.defs import Tensor
from .common.utils import list_string_to_list
from .logger import ERROR, WARN, DEBUG, INFO
from .graph.node_wrap import NodeWrap
from .graph.graph_algo import get_valid_node_name
from .graph.pattern_match import single_node_matcher
from .front_end.onnx.passes.common_passes import insert_constant


coefficient_shift_map = {
    'int8': 8,
    'int10': 10,
    'int16': 16,
    'float32': 0
}

coefficient_default_map = {
    'BT709': {
        'RgbToYuv': [0, 0, 0, 0, 128, 128, 0.212600, 0.715200, 0.072200, -0.114572, -0.385428, 0.500000, 0.500000, -0.454153, -0.045847],
        'YuvToRgb': [0, 128, 128, 0, 0, 0, 1.000000, 0.000000, 1.574800, 1.000000, -0.187324, -0.468124, 1.000000, 1.855600, 0.000000]
    },
}


def gen_gamut_params(params):
    ret = dict()
    preprocess = params['gamut_preprocess']
    if preprocess.upper() in ['RGBTOYUV', 'YUVTORGB']:
        # get preprocess type and rgb_shape
        if preprocess.upper() == 'RGBTOYUV':
            preprocess_type = 'RgbToYuv'
            shape = params.get('rgb_shape', None)
            if shape is None:
                ERROR('RgbToYuv must provide origin RGB shape. Ingore gamut_preprocess!')
                return ret
            ret['shape'] = shape
        else:
            preprocess_type = 'YuvToRgb'
        # get format
        gamut_format = params.get('gamut_format', None)
        if gamut_format is None:
            INFO('gamut_format is not set. Set to default value I420!')
            gamut_format = 'I420'
        # get bits
        bits = params.get('gamut_bits', None)
        if bits is None:
            INFO('gamut_bits is not set. Set to default value 8!')
            bits = 8
        # get conversion, coefficient_dtype and coefficient_shift
        conversion = params.get('gamut_conversion', None)
        if conversion is None:
            INFO('gamut_conversion is not set. Set to default value BT709!')
            conversion = 'BT709'
        else:
            conversion = conversion.upper()
        if conversion == 'SELF':
            coefficient = params.get('coefficient', None)
            coefficient_dtype = params.get('coefficient_dtype', None)
            coefficient_shift = params.get('coefficient_shift', None)
            if coefficient is None or coefficient_dtype is None or coefficient_shift is None:
                WARN('gamut conversion is SELF defined, but coefficient/coefficient_dtype/'
                     'coefficient_shift is not set. Set to default value BT709!')
                conversion = 'BT709'
            else:
                try:
                    coefficient = list(map(float,
                                           coefficient.lstrip('[').rstrip(']').split(',')))
                except Exception as e:
                    WARN('Unsupported coefficient format: %s. Set conversion to default value BT709!' % str(e))
                    conversion = 'BT709'
        elif conversion not in coefficient_default_map:
            WARN(
                'Currently conversion %s is not supported. Use BT709 instead!' % conversion)
            conversion = 'BT709'
        if conversion in coefficient_default_map:
            coefficient = coefficient_default_map[conversion][preprocess_type]
            coefficient_shift = 0
            coefficient_dtype = 'float32'
        ret['type'] = preprocess_type
        ret['format'] = gamut_format
        ret['bits'] = bits
        ret['conversion'] = conversion
        ret['coefficient'] = coefficient
        ret['coefficient_dtype'] = coefficient_dtype
        ret['coefficient_shift'] = coefficient_shift
    else:
        ERROR('Meet unsupported preprocess type (%s). Ingore it!' % preprocess)
    return ret


def gamut_preprocess(graph, params):
    '''Optimization for RGB2YUV, YUV2RGB op.'''
    inputs = []
    for n in graph.nodes:
        node_obj = NodeWrap(graph, n)['object']
        if node_obj.type == 'ArmInput':
            inputs.append(n)

    preprocess = params.get('gamut_preprocess', None)
    if preprocess is not None:
        gamut = gen_gamut_params(params)
        if gamut:
            if len(inputs) != 1:
                ERROR(
                    'Currently for RgbToYuv, graph with more than 1 inputs is not supported!')
                return
            ori_inp = inputs[0]
            input_out_edges = graph.sorted_out_edges(inputs[0], data=True)
            inp_tensor = input_out_edges[0][2]['tensor'].value
            inp_shape = inp_tensor.shape
            gamut['dtype'] = inp_tensor.dtype
            raw_input = get_valid_node_name(graph, inputs[0] + '_raw')

            def replace_input():
                graph.add_node(raw_input)
                for edge in input_out_edges:
                    s, d, attr = edge
                    graph.remove_edge(s, d)
                    graph.add_edge(raw_input, d, **attr)

            if gamut['type'] == 'RgbToYuv':
                shape = gamut['shape']
                shape = re.findall('\[[\s*\d+,]*\d+\]|\[\s*\]', shape)[0]
                shape = [int(i) for i in re.findall('\d+', shape)]
                n, h, w, c = shape
                if c != 3:
                    ERROR('For RgbToYuv the rgb shape must be [n,h,w,3] but got %s, '
                          'the channel is wrong.' % (str(shape)))
                    return
                if len(inp_shape) != 2:
                    ERROR('For RgbToYuv, the original model input must has a 2 '
                          'dimensions input, but got a shape: %s ' % (str(inp_shape)))
                    return
                if int(h * w * 1.5) != inp_shape[1]:
                    ERROR('For RgbToYuv the input size mismatches with rgb_shape. '
                          'Expect h*w*3/2 == input\'s_size, but got h*w*3/2=%d, and '
                          'input size=%d' % (int(h * w * 1.5), inp_shape[1]))
                    return
                replace_input()
                t = Tensor(value=np.random.randn(
                    *tuple(shape)).astype(np.uint8))
                graph.add_edge(ori_inp, raw_input, **{'tensor': t})
                graph._attr['input_tensors'][ori_inp] = t
                gamut.pop('shape')
                gamut['name'] = raw_input
                NodeWrap(graph, raw_input).replace_obj('ArmRgbToYuv', gamut)
            else:
                if len(inp_shape) != 4:
                    ERROR('To insert YUV2RGB layer, the input must be 4 dimensions, '
                          'but got a shape %s!' % (str(inp_shape)))
                    return
                need_insert_transpose = False
                n, h, w, c = inp_shape
                if params['model_type'].upper() in ['CAFFE', 'ONNX']:
                    need_insert_transpose = True
                    _, c, h, w = inp_shape
                if c != 3:
                    ERROR('To insert YUV2RGB layer, the input channel must be 3, '
                          'but got %d!' % (c))
                    return
                replace_input()
                shape = [n, int(h * w * 1.5)]
                gamut['shape'] = [n, h, w, 3]
                gamut['name'] = raw_input
                t = Tensor(value=np.random.randn(
                    *tuple(shape)).astype(np.uint8))
                graph.add_edge(ori_inp, raw_input, **{'tensor': t})
                graph._attr['input_tensors'][ori_inp] = t
                NodeWrap(graph, raw_input).replace_obj('ArmYuvToRgb', gamut)
                if need_insert_transpose:
                    trans = get_valid_node_name(graph, raw_input + '_trans')
                    transpose_attr = {'name': trans, 'opset_version': 1}
                    transpose_attr.update({'perm': [0, 3, 1, 2]})
                    graph.add_node(trans)
                    NodeWrap(graph, trans).replace_obj(
                        'ArmTranspose', transpose_attr)
                    for edge in graph.sorted_out_edges(raw_input, data=True):
                        s, d, attr = edge
                        graph.remove_edge(s, d)
                        graph.add_edge(trans, d, **attr)
                    graph.add_edge(raw_input, trans)


def standardization_preprocess(graph, params, hooking_node):
    ret = hooking_node
    if not graph.has_node(hooking_node):
        ERROR('[Parser]: Invalid hooking Node(%s) that dose not exist in graph in standardization_preprocess!', hooking_node)
        return ret
    hooking_obj = NodeWrap(graph, hooking_node)['object']
    if hooking_obj is None:
        ERROR(
            '[Parser]: Meets invalid hooking Node(%s) in standardization_preprocess!' % hooking_node)
        return ret
    if len(hooking_obj.get_out_ports()) != 1:
        ERROR(
            '[Parser]: Only support hooking Node(%s) with 1 out port in standardization_preprocess!' % hooking_node)
        return ret
    out_tensors = hooking_obj.get_output_tensors()
    if len(out_tensors) == 0:
        ERROR(
            '[Parser]: Meets invalid hooking output tensor(%s) in standardization_preprocess!' % hooking_node)
        return ret
    input_tensor_value = out_tensors[0]
    if input_tensor_value is None or input_tensor_value.dtype != 'float32':
        ERROR(
            '[Parser]: Only support float32 input tensor for standardization_preprocess!')
        return ret
    input_tensor_shape = list(input_tensor_value.shape)
    if any([s is None for s in input_tensor_shape]):
        ERROR(
            '[Parser]: Invalid tensor shape in standardization_preprocess!')
        return ret

    axes = []
    if params:
        axes_pattern = re.compile(r'\[[\s*\d+\s*,]*\s*\d*\s*\]')
        if axes_pattern is not None:
            axes_matched = re.search(axes_pattern, params)
            axes = list_string_to_list(axes_matched[0])[0]

    if len(input_tensor_shape) < len(axes):
        ERROR('[Parser]: Length of axes exceeds the length of tensor shape in resize_preprocess! Please check config file.')

    input_dim = len(input_tensor_shape)
    np_axes = np.array(axes, np.int32)
    negative_mask = np_axes < 0
    np_axes[negative_mask] = (np_axes + input_dim)[negative_mask]
    if np.any(np_axes >= input_dim):
        ERROR(
            '[Parser]: Invalid axes in standardization_preprocess! Please check config file.')
        return ret
    axes = sorted(np_axes.tolist())
    out_port = hooking_obj.get_out_ports()[0]
    mvn = get_valid_node_name(graph, hooking_node + '_mvn')
    for _, dst, out_attr in graph.sorted_out_edges(hooking_node, data=True):
        graph.remove_edge(hooking_node, dst)
        graph.add_edge(mvn, dst, **out_attr)
    graph.add_edge(hooking_node, mvn, **{'src_out_port': out_port, 'dst_in_port': 0})
    mvn_attr = {'name': mvn,
                'opset_version': 13,
                'axes': axes
                }
    NodeWrap(graph, mvn).replace_obj('MeanVarianceNormalization', mvn_attr)
    return mvn


def resize_preprocess(graph, params, hooking_node):
    ret = hooking_node
    if graph.has_node(hooking_node):
        if params is not None:
            resize_pattern = re.compile(
                r'^\s*[a-zA-Z_]*\s*,\s*\[[\s*\d+\s*,]*\s*\d+\s*\]\s*$')
            resize_matched = re.search(resize_pattern, params)
            if resize_matched is not None:
                method_pattern = re.compile(r'[a-zA-Z_]*')
                shape_pattern = re.compile(r'\[[\s*\d+\s*,]*\s*\d+\s*\]')
                method_matched = re.search(method_pattern, resize_matched[0])
                shape_matched = re.search(shape_pattern, resize_matched[0])
                if method_matched is not None and shape_matched is not None:
                    method = method_matched[0]
                    shape = list_string_to_list(shape_matched[0])[0]

                    if method.upper() not in ('BILINEAR', 'NEAREST'):
                        ERROR(
                            '[Parser]: Meets invalid Resize method (%s) in resize_preprocess! Please check config file.' % method)
                        return ret

                    if not shape or len(shape) < 3:
                        ERROR(
                            '[Parser]: Meets invalid Resize input shape (%s) in resize_preprocess! Please check config file.' % str(shape))
                        return ret

                    hooking_obj = NodeWrap(graph, hooking_node)['object']
                    if hooking_obj is None:
                        ERROR(
                            '[Parser]: Meets invalid hooking Node(%s) in resize_preprocess!' % hooking_node)
                        return ret

                    if len(hooking_obj.get_out_ports()) != 1:
                        ERROR(
                            '[Parser]: Only support hooking Node(%s) with 1 out port in resize_preprocess!' % hooking_node)
                        return ret

                    out_tensors = hooking_obj.get_output_tensors()
                    if len(out_tensors) == 0:
                        ERROR(
                            '[Parser]: Meets invalid hooking output tensor(%s) in resize_preprocess!' % hooking_node)
                        return ret

                    input_tensor_value = out_tensors[0]
                    if input_tensor_value is None or input_tensor_value.dtype != 'float32':
                        ERROR(
                            '[Parser]: Only support float32 input tensor for resize_preprocess!')
                        return ret

                    input_tensor_shape = list(input_tensor_value.shape)
                    if len(input_tensor_shape) != len(shape) \
                            or not (input_tensor_shape[0:2] == shape[0:2]
                                    or (input_tensor_shape[0] == shape[0]
                                        and input_tensor_shape[-1] == shape[-1])):
                        ERROR(
                            '[Parser]: Resize input shape (%s) does not match the model input in resize_preprocess! Please check config file.' % str(shape))
                        return ret

                    new_input_value = np.resize(input_tensor_value, shape)
                    input_node_name = hooking_node
                    resize = get_valid_node_name(
                        graph, input_node_name + '_resize')
                    for _, dst, out_attr in graph.sorted_out_edges(input_node_name, data=True):
                        graph.remove_edge(input_node_name, dst)
                        graph.add_edge(resize, dst, **out_attr)
                    new_input_out_attr = {'src_out_port': 0,
                                          'dst_in_port': 0,
                                          'tensor': Tensor(value=new_input_value)}
                    graph.add_edge(input_node_name, resize,
                                   **new_input_out_attr)

                    if hooking_node in graph._attr['input_tensors']:
                        graph._attr['input_tensors'][hooking_node].value = new_input_value
                        graph._attr['input_tensors'][hooking_node].shape = new_input_value.shape

                    if input_tensor_shape[0:2] == shape[0:2] \
                            and input_tensor_shape[2:] != shape[2:]:
                        data_format = 'NCHW'
                    else:
                        data_format = 'NHWC'
                    scales = np.array(input_tensor_shape, np.float32) / \
                        np.array(shape, np.float32)
                    resize_attr = {
                        'name': resize,
                        'data_format': data_format,
                        'opset_version': 13,
                        'mode': 'nearest' if method.upper() == 'NEAREST' else 'linear',
                    }
                    NodeWrap(graph, resize).replace_obj('Resize', resize_attr)
                    insert_constant(graph,
                                    resize + '_roi',
                                    np.ones([len(shape) * 2], np.float32),
                                    resize,
                                    in_port=1,
                                    data_format=data_format)
                    insert_constant(graph,
                                    resize + '_scales',
                                    scales,
                                    resize,
                                    in_port=2,
                                    data_format=data_format)

                    ret = resize
                else:
                    ERROR('[Parser]: Resize Preprocees was set in config file, but got invalid parameters (%s)!'
                          ' Please check config file!' % params['preprocess_params'])
            else:
                ERROR('[Parser]: Resize Preprocees was set in config file, but got invalid parameters (%s)!'
                      ' Please check config file!' % params['preprocess_params'])
        else:
            ERROR(
                '[Parser]: Meets invalid preprocess_params in resize_preprocess, please check config file.')
    else:
        ERROR('[Parser]: Invalid hooking Node(%s) that dosenot exist in graph in resize_preprocess!', hooking_node)
    return ret


def rgb2bgr_preprocess(graph, method, hooking_node):
    ret = hooking_node
    if graph.has_node(hooking_node):
        if method.upper() not in ['RGB2BGR', 'BGR2RGB']:
            ERROR('[Parser]: Meets invalid method (%s) in rgb2bgr_preprocess!' %
                  method.upper())
            return ret

        hooking_obj = NodeWrap(graph, hooking_node)['object']
        if hooking_obj is None:
            ERROR(
                '[Parser]: Meets invalid hooking Node(%s) in rgb2bgr_preprocess!' % hooking_node)
            return ret

        if len(hooking_obj.get_out_ports()) != 1:
            ERROR(
                '[Parser]: Only support hooking Node(%s) with 1 out port in rgb2bgr_preprocess!' % hooking_node)
            return ret

        out_tensors = hooking_obj.get_output_tensors()
        if len(out_tensors) == 0 \
                or out_tensors[0] is None \
                or len(out_tensors[0].shape) < 3 \
                or (out_tensors[0].shape[-1] != 3 and out_tensors[0].shape[1] != 3):
            ERROR(
                '[Parser]: Meets invalid hooking output tensor(%s) in rgb2bgr_preprocess!' % hooking_node)
            return ret
        inp_shape = list(out_tensors[0].shape)
        data_format = 'NHWC' if inp_shape[-1] == 3 else 'NCHW'
        spatial_dim = inp_shape[1:-
                                1] if data_format == 'NHWC' else inp_shape[2:]
        batch_axis, time_axis = 0, 1
        need_transpose = data_format == 'NHWC'
        need_reshape = len(out_tensors[0].shape) != 3

        reverse = get_valid_node_name(graph, '%s_reverse' % method.lower())

        if need_transpose:
            pre_transpose = get_valid_node_name(
                graph, reverse + '_pre_transpose')
            post_transpose = get_valid_node_name(
                graph, reverse + '_post_transpose')
            pre_perm = [0, len(inp_shape) - 1] + list(range(1, len(inp_shape) - 1))
            post_perm = [0] + list(range(2, len(inp_shape))) + [1]
        else:
            pre_transpose, post_transpose = None, None
            pre_perm, post_perm = None, None

        if need_reshape:
            pre_reshape = get_valid_node_name(graph, reverse + '_pre_reshape')
            post_reshape = get_valid_node_name(
                graph, reverse + '_post_reshape')
            pre_dim = np.array([inp_shape[0], 3, -1], np.int32)
            post_dim = np.array([inp_shape[0], 3] + spatial_dim, np.int32)
        else:
            pre_reshape, post_reshape = None, None
            pre_dim, post_dim = None, None

        hooking_out_edges = graph.sorted_out_edges(hooking_node, data=True)
        _, _, edge_attr = hooking_out_edges[0]
        graph.remove_edges_from(hooking_out_edges)

        edge_attr_used, current_node = False, hooking_node
        if need_transpose:
            in_attr = copy.deepcopy(edge_attr)
            in_attr['dst_in_port'] = 0
            graph.add_edge(hooking_node, pre_transpose, **in_attr)
            edge_attr_used, current_node = True, pre_transpose
            NodeWrap(graph, pre_transpose).replace_obj('Transpose',
                                                       {'name': pre_transpose,
                                                        'opset_version': 1,
                                                        'perm': pre_perm})
        if need_reshape:
            if edge_attr_used:
                in_attr = {'src_out_port': 0, 'dst_in_port': 0}
            else:
                in_attr = copy.deepcopy(edge_attr)
                in_attr['dst_in_port'] = 0
                edge_attr_used = True
            graph.add_edge(current_node, pre_reshape, **in_attr)
            current_node = pre_reshape
            NodeWrap(graph, pre_reshape).replace_obj(
                'Reshape', {'name': pre_reshape, 'opset_version': 5})
            insert_constant(graph, pre_reshape + '_shape',
                            pre_dim, pre_reshape, in_port=1)

        if edge_attr_used:
            in_attr = {'src_out_port': 0, 'dst_in_port': 0}
        else:
            in_attr = copy.deepcopy(edge_attr)
            in_attr['dst_in_port'] = 0
        graph.add_edge(current_node, reverse, **in_attr)

        current_node = reverse
        sequence_lens = np.array([3] * inp_shape[0], np.int32)
        insert_constant(graph, reverse + '_seq_len',
                        sequence_lens, reverse, in_port=1)
        NodeWrap(graph, reverse).replace_obj('ReverseSequence',
                                             {'name': reverse,
                                              'opset_version': 10,
                                              'batch_axis': batch_axis,
                                              'time_axis': time_axis
                                              })
        if need_reshape:
            graph.add_edge(current_node, post_reshape)
            current_node = post_reshape
            NodeWrap(graph, post_reshape).replace_obj(
                'Reshape', {'name': post_reshape, 'opset_version': 5})
            insert_constant(graph, post_reshape + '_shape',
                            post_dim, post_reshape, in_port=1)
        if need_transpose:
            graph.add_edge(current_node, post_transpose)
            current_node = post_transpose
            NodeWrap(graph, post_transpose).replace_obj('Transpose',
                                                        {'name': post_transpose,
                                                         'opset_version': 1,
                                                         'perm': post_perm})

        for _, dst, out_attr in hooking_out_edges:
            graph.add_edge(current_node, dst, **out_attr)

        ret = current_node
    else:
        ERROR('[Parser]: Invalid hooking Node(%s) that does not exist in graph in rgb2bgr_preprocess!', hooking_node)
    return ret


def rgb2gray_preprocess(graph, method, hooking_node):
    ret = hooking_node

    if not graph.has_node(hooking_node):
        ERROR('[Parser]: Invalid hooking Node(%s) that does not exist in graph in rgb2gray_preprocess!', hooking_node)
        return ret

    hooking_obj = NodeWrap(graph, hooking_node)['object']
    if hooking_obj is None:
        ERROR(
            '[Parser]: Meets invalid hooking Node(%s) in rgb2gray_preprocess!' % hooking_node)
        return ret

    if len(hooking_obj.get_out_ports()) != 1:
        ERROR(
            '[Parser]: Only support hooking Node(%s) with 1 out port in rgb2gray_preprocess!' % hooking_node)
        return ret

    out_tensors = hooking_obj.get_output_tensors()
    if len(out_tensors) == 0:
        ERROR(
            '[Parser]: Meets invalid hooking output tensor(%s) in rgb2gray_preprocess!' % hooking_node)
        return ret

    input_tensor_value = out_tensors[0]
    if input_tensor_value is None or input_tensor_value.dtype != 'float32':
        ERROR(
            '[Parser]: Only support float32 input tensor for rgb2gray_preprocess!')
        return ret

    input_tensor_shape = list(input_tensor_value.shape)
    if len(input_tensor_shape) < 3:
        ERROR(
            '[Parser]: Meets invalid input shape in rgb2gray_preprocess! Please check config file.')
        return ret

    if hooking_obj.type == 'Input' \
            and input_tensor_shape[-1] not in (1, 3) \
            and input_tensor_shape[1] not in (1, 3):
        ERROR(
            '[Parser]: Meets invalid input shape in rgb2gray_preprocess! Please check config file.')
        return ret

    if hooking_obj.type != 'Input' \
            and input_tensor_shape[-1] != 3 \
            and input_tensor_shape[1] != 3:
        ERROR(
            '[Parser]: Meets invalid input shape in rgb2gray_preprocess! Please check config file.')
        return ret

    if hooking_obj.type == 'Input':
        if (input_tensor_shape[1] == 1 and input_tensor_shape[-1] != 1) \
                or (input_tensor_shape[1] == 3 and input_tensor_shape[-1] != 3):
            data_format = 'NCHW'
        else:
            data_format = 'NHWC'
    else:
        if input_tensor_shape[1] == 3 and input_tensor_shape[-1] != 3:
            data_format = 'NCHW'
        else:
            data_format = 'NHWC'

    coefficients = [0.3, 0.59, 0.11] if method.upper() == 'RGB2GRAY' else [
        0.11, 0.59, 0.3]
    input_dim = len(input_tensor_shape)
    need_tile = input_tensor_shape[1] == 3 if data_format == 'NCHW' else input_tensor_shape[-1] == 3
    reps = [1, 3] + [1] * \
        (input_dim -
         2) if data_format == 'NCHW' else [1] * (input_dim - 1) + [3]

    hooking_out_edges = graph.sorted_out_edges(hooking_node, data=True)
    _, _, edge_attr = hooking_out_edges[0]
    graph.remove_edges_from(hooking_out_edges)

    bn = get_valid_node_name(graph, '%s_bn' % method.lower())
    reduce = get_valid_node_name(graph, '%s_reduce' % method.lower())

    new_edge_attr = copy.deepcopy(edge_attr)
    new_edge_attr['dst_in_port'] = 0
    graph.add_edge(hooking_node, bn, **new_edge_attr)
    graph.add_edge(bn, reduce)

    if need_tile:
        tile = get_valid_node_name(graph, reduce + '_post_tile')
        graph.add_edge(reduce, tile)
        NodeWrap(graph, tile).replace_obj(
            'Tile', {'name': tile, 'opset_version': 13})
        insert_constant(graph,
                        tile + '_repeats',
                        np.array(reps, np.int32),
                        tile,
                        in_port=1,
                        data_format=data_format)
        ret = tile
    else:
        if hooking_node in graph._attr['input_tensors']:
            tiled_input_tensor = np.tile(input_tensor_value, reps)
            graph._attr['input_tensors'][hooking_node].value = tiled_input_tensor
            graph._attr['input_tensors'][hooking_node].shape = tiled_input_tensor.shape
        ret = reduce

    for _, dst, out_attr in hooking_out_edges:
        new_out_attr = copy.deepcopy(out_attr)
        new_out_attr['src_out_port'] = 0
        graph.add_edge(ret, dst, **new_out_attr)

    NodeWrap(graph, bn).replace_obj('BatchNormalization',
                                    {'name': bn,
                                     'opset_version': 9,
                                     'data_format': data_format,
                                     'epsilon': 0.,
                                     })
    scale = np.array(coefficients, np.float32)
    B = np.zeros((3,), np.float32)
    mean = np.zeros((3,), np.float32)
    var = np.ones((3,), np.float32)
    insert_constant(graph, bn + '_scale', scale, bn,
                    in_port=1, data_format=data_format)
    insert_constant(graph, bn + '_B', B, bn, in_port=2, data_format=data_format)
    insert_constant(graph, bn + '_mean', mean, bn,
                    in_port=3, data_format=data_format)
    insert_constant(graph, bn + '_var', var, bn,
                    in_port=4, data_format=data_format)

    NodeWrap(graph, reduce).replace_obj('ReduceSum',
                                        {'name': reduce,
                                         'opset_version': 11,
                                         'data_format': data_format,
                                         'axes': [1] if data_format == 'NCHW' else [-1],
                                         'keepdims': True
                                         })

    return ret


def preprocess(graph, params):
    if params.get('preprocess_type', None) is not None:
        input_nodes = [m['target']
                       for m in single_node_matcher(graph, 'Input')]
        if len(input_nodes) != 1:
            ERROR('[Parser]: Only support one input for preprocess!')
            return
        if params.get('preprocess_params', None) is not None:
            types_pattern = re.compile(r'^[\s*\w+\s*,]*\s*\w+\s*$')
            params_pattern = re.compile(
                r'^[\s*\{\s*.*\s*\}\s*,]*\s*\{\s*.*\s*\}\s*$')
            types_matched = re.search(types_pattern, params['preprocess_type'])
            params_matched = re.search(
                params_pattern, params['preprocess_params'])
            if types_matched is not None \
                    and params_matched is not None:
                meta_type_pattern = re.compile(r'\w+')
                meta_param_pattern = re.compile(r'\{\s*[^\{^\}]*\s*\}')
                all_types = re.findall(meta_type_pattern, types_matched[0])
                all_params = re.findall(meta_param_pattern, params_matched[0])
                if len(all_types) > 0 and len(all_types) == len(all_params):
                    all_params = [p.lstrip('{').lstrip(' ').rstrip(
                        '}').rstrip(' ') for p in all_params]
                    hooking = input_nodes[0]
                    for t, p in zip(all_types, all_params):
                        if t.upper() == 'RESIZE':
                            hooking = resize_preprocess(graph, p, hooking)
                        elif t.upper() in ['RGB2BGR', 'BGR2RGB']:
                            hooking = rgb2bgr_preprocess(graph, t, hooking)
                        elif t.upper() in ['RGB2GRAY', 'BGR2GRAY']:
                            hooking = rgb2gray_preprocess(graph, t, hooking)
                        elif t.upper() == 'STANDARDIZATION':
                            hooking = standardization_preprocess(graph, p, hooking)
                        else:
                            ERROR(
                                '[Parser]: Meets invalid preprocess type(%s) set in config file!' % t)
                else:
                    ERROR(
                        '[Parser]: Meets invalid preprocess_type/params, please check config file.')
            else:
                ERROR(
                    '[Parser]: Meets invalid preprocess_type/params, please check config file.')
        else:
            ERROR('[Parser]: Meets invalid preprocess_params when preprocess_type was set, please check config file.')
