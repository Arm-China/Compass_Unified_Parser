# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


# cython: language_level=3
import re
import os
from .logger import ERROR, WARN, DEBUG, INFO
from .graph.node_wrap import NodeWrap
from .graph.graph_algo import get_valid_node_name
from .common.utils import is_file, is_dir


def special_character_conversion(graph, params):
    '''Convert characters that are not allowed in IR.'''
    newname_dict = {}
    for n in graph.nodes:
        node_obj = NodeWrap(graph, n)['object']
        newname = node_obj.name
        newname = re.sub(r'[^0-9a-zA-Z\.\:\/\_\;\'\x22]', '_', newname)
        for value in newname_dict.values():
            while value == newname:
                newname = newname + '_'
        for key in newname_dict.keys():
            while key == newname:
                newname = newname + '_'
        if node_obj.name != newname:
            DEBUG('[Parser]: Duplicate layer name found! Convert layer name:(%s) to layer name: (%s)!' % (
                str(node_obj.name), str(newname)))
            newname = get_valid_node_name(graph, newname)
        newname_dict[n] = newname
    graph._attr['duplicate_name'] = newname_dict


def check_similarity(graph, params, txt_path, bin_path):
    if not params.get('similarity_input_npy', ''):
        return True

    if not txt_path or not bin_path or not is_file(txt_path) or not is_file(bin_path):
        ERROR('[Parser]: Meets invalid txt_path (%s) or bin_path (%s) in check_similarity!' % (txt_path, bin_path))
        return False

    model_path = params.get('original_input_model', params.get('input_model', ''))
    if not model_path:
        ERROR('[Parser]: Meets empty input_model in check_similarity!')
        return False

    feed_dict = params['similarity_input_npy']
    DEBUG('[Parser]: Input names from feed_dict: %s' % str(list(feed_dict.keys())))
    forward_type = 'float'

    if graph._attr.get('quantize', False):
        from .utils.quantize import generate_symm_quant_cfg, generate_symm_quant_ir
        model_name = params.get('model_name', 'quant')
        symm_quant_cfg_file = generate_symm_quant_cfg(model_name, txt_path, bin_path)
        if not symm_quant_cfg_file:
            ERROR('[Parser]: Fail to generate symm quant cfg file!')
            return False
        if not generate_symm_quant_ir(symm_quant_cfg_file):
            ERROR('[Parser]: Fail to generate symm quant IR!')
            return False
        new_base_path = os.path.join(os.path.dirname(txt_path), model_name + '_opt')
        txt_path = new_base_path + '.txt'
        if not is_file(txt_path):
            ERROR('[Parser]: Meets invalid symm quant txt(%s) file!' % txt_path)
            return False
        INFO('[Parser]: symm quant txt file: %s' % txt_path)
        bin_path = new_base_path + '.bin'
        if not is_file(bin_path):
            ERROR('[Parser]: Meets invalid symm quant bin(%s) file!' % bin_path)
            return False
        INFO('[Parser]: symm quant bin file: %s' % bin_path)
        forward_type = 'quantized'

    ret = True
    rt_output_dict, opt_output_dict = {}, {}
    try:
        from .utils.forward import rt_forward
        # Get model output using runtime of original framework(tf, onnx, caffe and etc)
        output_names = params.get('output_names', None)
        rt_output_dict = rt_forward(model_path, feed_dict, output_names=output_names if output_names else None,
                                    proto_path=params.get('caffe_prototxt', ''))
    except Exception as e:
        ERROR('[Parser]: Meets Exception (%s) in framework runtime forward!' % str(e))
        ret = False

    transfer_to_float = True
    if forward_type == 'quantized':
        for value in rt_output_dict.values():
            if 'int' in value.dtype.name:
                transfer_to_float = False
                break

    try:
        from .utils.forward import opt_forward
        # Get model output using opt forward
        opt_output_dict = opt_forward(txt_path, bin_path, feed_dict, forward_type=forward_type,
                                      transfer_to_float=transfer_to_float)
    except Exception as e:
        ERROR('[Parser]: Meets Exception (%s) in opt forward!' % str(e))
        ret = False

    if not rt_output_dict or not opt_output_dict:
        ERROR('[Parser]: Fail to check similarity due to missing runtime/opt forward outputs!')
        ret = False
    else:
        from .utils.compare import compare_data_dict
        # Compare outputs
        INFO('[Parser]: Comparing outputs(first runtime, second opt)')
        ret = compare_data_dict(rt_output_dict, opt_output_dict)

        # Report result
        if ret:
            INFO('[Parser]: Similarity checking is passed!')
        else:
            WARN('[Parser]: Similarity checking is failed!')
            ret = False
    return ret
