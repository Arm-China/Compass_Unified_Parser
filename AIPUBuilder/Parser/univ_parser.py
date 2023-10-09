# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import os
import re
import numpy as np
import onnx
import torch
from collections import OrderedDict
from .common.utils import is_file, is_dir, multi_string_to_list, list_string_to_list
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
        params['output_names'] = multi_string_to_list(
            params['output_names']) if 'output_names' in params else []
        out_names_dict = OrderedDict(
            {k: i for (i, k) in enumerate(params['output_names'])})
        params['output_names'] = list(out_names_dict.keys())
        params['input_shapes'] = list_string_to_list(
            params['input_shapes']) if 'input_shapes' in params else []

        input_npy = {}
        if params.get('input_npy', ''):
            npy_pattern = re.compile(r'^[\s*.*\s*,]*\s*.*s*$')
            npy_matched = re.search(npy_pattern, params['input_npy'])
            if npy_matched is not None:
                npy_paths = npy_matched[0].split(',')
                for p in npy_paths:
                    try:
                        p = p.rstrip(' ').lstrip(' ')
                        data = np.load(p, allow_pickle=True)
                        if isinstance(data, np.ndarray) \
                                and data.size == 1 \
                                and isinstance(data.item(), dict):
                            input_npy.update(data.item())
                    except:
                        WARN('[Parser]: Load input npy(%s) failed!' % os.path.basename(p))
        params['input_npy'] = input_npy

        if model_type == 'torch':
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

        if len(params['input_names']) == len(params['input_shapes']):
            params['input_shapes'] = {
                params['input_names'][i]: v for i, v in enumerate(params['input_shapes'])}
        else:
            FATAL(
                '[Parser]: Length of input_names should be equal to length of input_shapes! '
                'Please check config file!')

        if 'batch_size' in params:
            WARN(
                '[Parser]: batch_size in config file will be deprecated and has no effect!')
        if 'input_data_format' in params:
            WARN('[Parser]: input_data_format in config file will be deprecated!')
        params['input_data_format'] = 'NCHW' if model_type in ('onnx', 'caffe', 'torch') else 'NHWC'
        params['output_tensor_names'] = params['output_names'][:]

        if not is_file(model_path) and not is_dir(model_path):
            ERROR('[Parser]: Meets invalid model file/directory(%s)!' % model_path)
            ret = False
        elif not is_dir(output_dir):
            ERROR('[Parser]: Meets invalid output directory(%s)!' % output_dir)
            ret = False
        else:
            graph = None

            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            if int(tf.__version__.split('.')[0]) < 2:
                WARN('Require tensorflow version==2.6 but now is in version %s!' % str(tf.__version__))

            try:
                # Convert torch model to onnx before processing
                if model_type == 'torch':
                    from .front_end.torch.process import convert_torch_to_onnx
                    model_path, params = convert_torch_to_onnx(model_path, params)
                    model_type = 'onnx'

                '''The models under different frameworks are parsed and finally converted into representations under the onnx framework.'''
                if model_type == 'onnx':
                    from .front_end.onnx.process import process_onnx
                    graph = process_onnx(model_path, params)
                elif model_type == 'tflite':
                    from .front_end.lite.process import process_tflite
                    graph = process_tflite(model_path, params)
                elif model_type == 'caffe':
                    from .front_end.caffe.process import process_caffe
                    graph = process_caffe(model_path, params)
                elif model_type in ('tf', 'tensorflow'):
                    is_keras_model = model_path.endswith('.h5') or model_path.endswith('.hdf5') or model_path.endswith(
                        '.keras') or is_dir(model_path)
                    if is_keras_model:
                        from .front_end.tf2.process import process_tf2
                        graph = process_tf2(model_path, params)
                    else:
                        from .front_end.tf.process import process_tf
                        graph = process_tf(model_path, params)
                else:
                    ERROR('[Parser]: Framework %s is not supported!' %
                          params.get('model_type', ''))
            except Exception as e:
                ERROR('[Parser]: Meets error when processing models, %s!' % str(e))
                ret = False

            if graph:
                from .front_end.onnx.passes.middle_passes import middle_passes, convert_onnx_version
                from .front_end.onnx.passes.back_passes import back_passes, trim_weights
                from .front_end.onnx.passes.transform import transform_to_nhwc
                from .front_end.onnx.passes.common_passes import remove_useless_op
                from .graph.graph_algo import infer, has_path
                from .graph.pattern_match import single_node_matcher
                from .writer import serialize
                from .preprocess import gamut_preprocess, preprocess
                from .misc import special_character_conversion

                '''Check if it is a connected graph.'''
                input_names = []
                input_names_list = single_node_matcher(graph, 'Input')
                for input_name in input_names_list:
                    input_names.append(input_name['target'])
                output_names = graph._attr.get('output_names')
                for output_name in output_names:
                    has_path_flag = False
                    for input_name in input_names:
                        if has_path(graph, input_name, output_name):
                            has_path_flag = True
                            break
                    if has_path_flag is False and len(input_names) > 0:
                        out_edges = graph.sorted_out_edges(output_name, data=True)
                        if len(out_edges) > 0 and all((out_attr['tensor'] is not None and out_attr['tensor'].is_const) for _, _, out_attr in out_edges):
                            WARN('[Parser]: Meets const node %s in outputs! It could be removed from graph!' % output_name)
                        else:
                            ERROR('[Parser]: Graph is not a connected one!')
                            break

                '''Gives a 'may be time consuming' hint for huge models.'''
                if len(graph) >= 2000:
                    WARN(
                        '[Parser]: Begin to process large model (number of nodes = %d) and maybe cost quite a lot of time!' % len(graph))

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
                    infer(graph)
                    remove_useless_op(graph, ['ArmCast'])
                except Exception as e:
                    ERROR('[Parser]: Meets exception in last infer (%s)!' % str(e))

                try:
                    trim_weights(graph)
                    serialize(graph, params)
                except Exception as e:
                    ERROR('[Parser]: Meets exception in serialize (%s)!' % str(e))

            else:
                WARN('[Parser]: Got invalid or empty graph from model!')
                ret = True
    else:
        ERROR('[Parser]: Meets invalid parameters for universal parser!')
        ret = False
    return ret
