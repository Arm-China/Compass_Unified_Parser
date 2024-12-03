# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import os
import re
import numpy as np
import onnx
import torch
from collections import OrderedDict
from .common.utils import is_file, is_dir, multi_string_to_list, list_string_to_list, get_dict_params
from .graph.graph import Graph
from .logger import *


def univ_parser(params):
    ret = True

    if params:
        '''Set the necessary parameters.'''
        model_path = params.get('input_model', '')
        output_dir = params.get('output_dir', './')
        model_type = params.get('model_type', '')
        if 'input_names' not in params and 'input' in params:
            params['input_names'] = params['input']
            params.pop('input')
        if 'output_names' not in params and 'output' in params:
            params['output_names'] = params['output']
            params.pop('output')
        if 'input_shapes' not in params and 'input_shape' in params:
            params['input_shapes'] = params['input_shape']
            params.pop('input_shape')

        params['input_names'] = multi_string_to_list(
            params['input_names']) if 'input_names' in params else []
        if len(params['input_names']) > 0 and any(not name for name in params['input_names']):
            WARN(
                '[Parser]: Meets empty name in input(%s) and it will be ignored!' % str(params['input_names']))
            params['input_names'] = [name for name in params['input_names'] if name]
        # use input_tensor_map({str:str}) to record input node/tensor name(s) from cfg and input tensor name(s) from IR
        params['input_tensor_map'] = {input_from_cfg: None for input_from_cfg in params['input_names']}

        params['output_names'] = multi_string_to_list(
            params['output_names']) if 'output_names' in params else []
        out_names_dict = OrderedDict(
            {k: i for (i, k) in enumerate(params['output_names'])})
        params['output_names'] = list(out_names_dict.keys())
        # use output_tensor_map({str:list}) to record output node/tensor name(s) from cfg and output tensors from IR
        params['output_tensor_map'] = {out_name: [] for out_name in params['output_names']}
        params['input_shapes'] = list_string_to_list(
            params['input_shapes']) if 'input_shapes' in params else []

        if 'ds_compat' in params:
            ds_compat = str(params['ds_compat']).lower() == 'true'
            if ds_compat:
                WARN('[Parser]: dynamic_shape compatibility mode is enabled, some passes will be disabled!')
            params['ds_compat'] = ds_compat
        else:
            params['ds_compat'] = False

        def _parse_npy(key_name):
            input_npy = {}
            if params.get(key_name, ''):
                npy_pattern = re.compile(r'^[\s*.*\s*,]*\s*.*s*$')
                npy_matched = re.search(npy_pattern, params[key_name])
                if npy_matched is not None:
                    npy_paths = npy_matched[0].split(',')
                    for p in npy_paths:
                        try:
                            p = p.rstrip(' ').lstrip(' ')
                            data = np.load(p, allow_pickle=True)
                            if isinstance(data, np.ndarray):
                                if data.size == 1 \
                                        and isinstance(data.item(), dict):
                                    input_npy.update(data.item())
                            elif isinstance(data, np.lib.npyio.NpzFile):
                                for key in list(data.keys()):
                                    input_npy.update({key: data[key]})
                        except Exception as e:
                            WARN('[Parser]: Load input npy(%s) failed because %s' % (os.path.basename(p), str(e)))
            return input_npy

        params['input_npy'] = _parse_npy('input_npy')
        params['similarity_input_npy'] = _parse_npy('similarity_input_npy')

        if model_type in ('torch', 'pytorch'):
            # For torch, input_names and output_names are useless because it's not allowed to change
            # input nodes or output nodes for TorchScript. They are just names assigned to the input
            # and output nodes of the graph in order. So, only providing input_shapes is allowed.
            # If input_names is not set, set it to ['x0', 'x1', ...].
            if not params['input_shapes']:
                FATAL('[Parser]: input_shapes must be provided in config file for torch model!')
            if params['input_names'] or params['output_names']:
                INFO('[Parser]: input_names and output_names in config file won\'t change input '
                     'and output nodes for torch model!')
            input_num = len(params['input_shapes'])
            if not params['input_names']:
                params['input_names'] = [('x' + str(idx)) for idx in range(input_num)]
            if 'input_dtype' in params:
                params['input_dtype'] = multi_string_to_list(params['input_dtype'])
                if len(params['input_dtype']) != input_num:
                    FATAL('[Parser]: Length of input_dtype should be equal to length of input_shapes! '
                          'Please check config file!')
            else:
                params['input_dtype'] = ['float32'] * input_num
                INFO('[Parser]: Input dtype is not set; default to float32 for torch model!')
            if params.get('force_cpu', None) is not None:
                params['force_cpu'] = True if str(params['force_cpu']).lower() == 'true' else False
            else:
                params['force_cpu'] = False

        input_shapes_cnt = len(params['input_shapes'])
        if len(params['input_names']) == input_shapes_cnt:
            params['input_shapes'] = {
                params['input_names'][i]: v for i, v in enumerate(params['input_shapes'])}
        else:
            if input_shapes_cnt == 0:
                params['input_shapes'] = {name: None for name in params['input_names']}
            else:
                FATAL(
                    '[Parser]: Length of input_names should be equal to length of input_shapes! '
                    'Please check config file!')

        if len(params['input_names']) == 0 and len(params['input_shapes']) == 0 and 'input_dimensions' in params:
            params['input_dimensions'] = get_dict_params(params['input_dimensions'])
        else:
            params['input_dimensions'] = {}

        if 'batch_size' in params:
            WARN(
                '[Parser]: batch_size in config file will be deprecated and has no effect!')
        if 'input_data_format' in params:
            WARN('[Parser]: input_data_format in config file will be deprecated!')
        params['input_data_format'] = 'NCHW' if model_type in ('onnx', 'caffe', 'torch') else 'NHWC'
        params['output_tensor_names'] = params['output_names'][:]
        params['proposal_normalized'] = (params.get('proposal_normalized', 'true').lower() in ('1', 'true'))

        if not is_file(model_path) and not is_dir(model_path):
            ERROR('[Parser]: Meets invalid model file/directory(%s)!' % model_path)
            ret = False
        elif not is_dir(output_dir):
            ERROR('[Parser]: Meets invalid output directory(%s)!' % output_dir)
            ret = False
        else:
            if not params.get('model_name', ''):
                params['model_name'] = model_type + '_model'
            graph = Graph(name=params['model_name'])

            tmp_tensors_dir = '.%s_tmp_tensors' % params.get('model_name', '')
            tmp_tensors_path = os.path.join(output_dir, tmp_tensors_dir)
            os.makedirs(tmp_tensors_path, exist_ok=True)  # folder could possibly be created by other processes
            params['tmp_tensors_path'] = tmp_tensors_path

            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            if int(tf.__version__.split('.')[0]) < 2:
                WARN('Require tensorflow version==2.6 but now is in version %s!' % str(tf.__version__))

            try:
                # Convert torch model to onnx before processing
                if model_type in ('torch', 'pytorch'):
                    from .front_end.torch.process import convert_torch_to_onnx
                    model_path, params = convert_torch_to_onnx(model_path, params)
                    model_type = 'onnx'

                '''The models under different frameworks are parsed and finally converted into representations under the onnx framework.'''
                if model_type == 'onnx':
                    from .front_end.onnx.process import process_onnx
                    graph = process_onnx(graph, model_path, params)
                elif model_type == 'tflite':
                    from .front_end.lite.process import process_tflite
                    graph = process_tflite(graph, model_path, params)
                elif model_type == 'caffe':
                    from .front_end.caffe.process import process_caffe
                    graph = process_caffe(graph, model_path, params)
                elif model_type in ('tf', 'tensorflow'):
                    is_keras_model = model_path.endswith('.h5') or model_path.endswith('.hdf5') or model_path.endswith(
                        '.keras') or is_dir(model_path)
                    if is_keras_model:
                        from .front_end.tf2.process import process_tf2
                        graph = process_tf2(graph, model_path, params)
                    else:
                        from .front_end.tf.process import process_tf
                        graph = process_tf(graph, model_path, params)
                else:
                    ERROR('[Parser]: Framework %s is not supported!' %
                          params.get('model_type', ''))
            except Exception as e:
                ERROR('[Parser]: Meets error when processing models, %s!' % str(e))
                ret = False

            if graph:
                from .front_end.onnx.passes.back_passes import trim_weights, assign_top_range_scale_zp
                from .front_end.onnx.passes.common_passes import remove_useless_op, convert_dummyinput_to_input
                from .graph.graph_algo import infer, has_path
                from .graph.pattern_match import single_node_matcher
                from .writer import serialize, show_in_out_map
                from .misc import check_similarity

                '''Check if it is a connected graph.'''
                input_names = []
                input_names_list = single_node_matcher(graph, 'Input')
                for input_name in input_names_list:
                    input_names.append(input_name['target'])
                output_names = list(set(graph._attr.get('output_names', [])).difference(
                    list(graph._attr.get('subgraph_output_names', []))))
                for output_name in output_names:
                    has_path_flag = False
                    for input_name in input_names:
                        if has_path(graph, input_name, output_name):
                            has_path_flag = True
                            break
                        else:
                            inp_obj = graph.nodes[input_name]['object']
                            if len(inp_obj.subgraphs) > 0 and \
                                    'subgraphs' in graph._attr and \
                                    len(graph._attr['subgraphs']) > 0:
                                for sub in inp_obj.subgraphs:
                                    for k, v in graph._attr['subgraphs'].items():
                                        if sub in v:
                                            has_path_flag = True
                                            break
                    if has_path_flag is False and len(input_names) > 0:
                        out_edges = graph.sorted_out_edges(output_name, data=True)
                        if len(out_edges) > 0 and all((out_attr['tensor'] is not None and out_attr['tensor'].is_const) for _, _, out_attr in out_edges):
                            WARN('[Parser]: Meets const node %s in outputs! It could be removed from graph!' % output_name)
                        else:
                            ERROR('[Parser]: Graph is not a connected one!')
                            break

                process_graph(graph, params)

                if 'subgraphs' in graph._attr and graph._attr['subgraphs']:
                    for n_name, v in graph._attr['subgraphs'].items():
                        for subgraph_name, subgraph in v.items():
                            front_process_graph(model_type, model_path, subgraph, params)
                            process_graph(subgraph, params)
                            convert_dummyinput_to_input(subgraph)

                txt_path, bin_path = '', ''
                try:
                    assign_top_range_scale_zp(graph)
                    root_offset = trim_weights(graph)
                    init_offset = root_offset
                    if 'subgraphs' in graph._attr and graph._attr['subgraphs']:
                        for n_name, v in graph._attr['subgraphs'].items():
                            for subgraph_name, subgraph in v.items():
                                assign_top_range_scale_zp(subgraph)
                                sub_offset = trim_weights(subgraph, init_offset)
                                init_offset = sub_offset
                    ret, txt_path, bin_path = serialize(graph, params)
                    ret = show_in_out_map(graph, params) and ret
                except Exception as e:
                    ERROR('[Parser]: Meets exception in serialize (%s)!' % str(e))

                if ret:
                    try:
                        # Do not check exit code or raise error for similarity here because
                        # it could be caused by non-parser issues.
                        check_similarity(graph, params, txt_path, bin_path)
                    except Exception as e:
                        ERROR('[Parser]: Meets exception in check_similarity (%s)!' % str(e))
                        ret = False
            else:
                WARN('[Parser]: Got invalid or empty graph from model!')
                ret = True

            if os.path.exists(tmp_tensors_path):
                import shutil
                shutil.rmtree(tmp_tensors_path)
    else:
        ERROR('[Parser]: Meets invalid parameters for universal parser!')
        ret = False
    return ret


def front_process_graph(model_type, model_path, graph, params):
    if model_type == 'onnx':
        from .front_end.onnx.process import front_process_onnx
        graph = front_process_onnx(graph, params)
    elif model_type == 'tflite':
        from .front_end.lite.process import front_process_tflite
        graph = front_process_tflite(graph, params)
    elif model_type in ('tf', 'tensorflow'):
        is_keras_model = model_path.endswith('.h5') or model_path.endswith('.hdf5') or model_path.endswith(
            '.keras') or is_dir(model_path)
        if is_keras_model:
            from .front_end.tf2.process import front_process_tf2
            graph = front_process_tf2(graph, params)
        else:
            from .front_end.tf.process import front_process_tf
            graph = front_process_tf(graph, params)
    elif model_type == 'caffe':
        from .front_end.caffe.process import front_process_caffe
        graph = front_process_caffe(graph, params)
    else:
        ERROR('[Parser]: Framework %s is not supported!' %
              params.get('model_type', ''))
    return graph


def process_graph(graph, params):
    from .front_end.onnx.passes.middle_passes import middle_passes, convert_onnx_version
    from .front_end.onnx.passes.back_passes import back_passes, trim_weights, assign_top_range_scale_zp
    from .front_end.onnx.passes.transform import transform_to_nhwc
    from .front_end.onnx.passes.common_passes import remove_useless_op, convert_64bit_const
    from .graph.graph_algo import infer
    from .preprocess import gamut_preprocess, preprocess
    from .misc import special_character_conversion
    '''Gives a 'may be time consuming' hint for huge models.'''
    if len(graph) >= 2000:
        WARN(
            '[Parser]: Begin to process large model (number of nodes = %d) and maybe cost quite a lot of time!' % len(
                graph))

    try:
        preprocess(graph, params)
    except Exception as e:
        ERROR(
            '[Parser]: Meets exception in insert preprocess (%s)!' % str(e))

    try:
        convert_onnx_version(graph)
    except Exception as e:
        ERROR(
            '[Parser]: Meets exception in convert_onnx_version (%s)!' % str(e))

    try:
        middle_passes(graph, params)
    except Exception as e:
        ERROR('[Parser]: Meets exception in middle_passes (%s)!' % str(e))

    infer(graph)

    try:
        transform_to_nhwc(graph, params)
    except Exception as e:
        ERROR(
            '[Parser]: Meets exception in transform_to_nhwc (%s)!' % str(e))

    try:
        back_passes(graph, params)
    except Exception as e:
        ERROR('[Parser]: Meets exception in back_passes (%s)!' % str(e))

    try:
        gamut_preprocess(graph, params)
    except Exception as e:
        ERROR(
            '[Parser]: Meets exception in insert gamut preprocess (%s)!' % str(e))

    try:
        special_character_conversion(graph, params)
    except Exception as e:
        ERROR(
            '[Parser]: Meets exception in insert special character conversion (%s)!' % str(e))

    try:
        convert_64bit_const(graph)
        infer(graph, final=True)
        remove_useless_op(graph, ['ArmCast'])
    except Exception as e:
        ERROR('[Parser]: Meets exception in last infer (%s)!' % str(e))
