# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os

import numpy as np

from UnifiedParser.logger import ERROR, INFO, WARN


def check_float_ir(ir_txt_path, expected_keywords, unexpected_keywords=[]):
    ''' Check whether expected_keywords exist and unexpected_keywords not exist
    in ir file. Return True if ir file matches the expectation, otherwise return
    False.
    '''
    if not os.path.exists(ir_txt_path):
        ERROR('File %s does not exist!' % ir_txt_path)
    if not isinstance(expected_keywords, list):
        expected_keywords = [expected_keywords]
    if not isinstance(unexpected_keywords, list):
        unexpected_keywords = [unexpected_keywords]
    expected_res = [False] * len(expected_keywords)
    unexpected_res = [False] * len(unexpected_keywords)
    with open(ir_txt_path) as f:
        for line in f:
            ex_rets = [True if word in line else False for word in expected_keywords]
            if any(ex_rets):
                for idx, ret in enumerate(ex_rets):
                    expected_res[idx] |= ret
            unex_rets = [True if word not in line else False for word in unexpected_keywords]
            if any(unex_rets):
                for idx, ret in enumerate(unex_rets):
                    unexpected_res[idx] |= ret
    if all(expected_res) and all(unexpected_res):
        INFO('No issue found in checking IR file!')
        return True
    if not all(expected_res):
        words_not_found = [expected_keywords[idx] for idx, res in enumerate(expected_res) if not res]
        WARN('Cannot find expected words(%s) in IR!' % str(words_not_found))
    if not all(unexpected_res):
        words_found = [unexpected_keywords[idx] for idx, res in enumerate(unexpected_res) if not res]
        WARN('Find unexpected words(%s) in IR!' % str(words_not_found))
    return False


def get_feed_dict(data_path):
    ''' Return a dict from the provided numpy file.
    '''
    feed_dict = dict()

    if not os.path.exists(data_path):
        ERROR('File %s does not exist!' % data_path)

    try:
        data = np.load(data_path, allow_pickle=True)
        for k, v in data.item().items():
            feed_dict[k] = v
    except:
        data = np.load(data_path)
        for k in data.files:
            feed_dict[k] = data[k]

    return feed_dict


def get_model_type(model_path):
    ''' Return type of the model according to the postfix of model's path.
    '''
    model_type = None
    if model_path.endswith('.onnx'):
        model_type = 'onnx'
    elif model_path.endswith('.pb') or model_path.endswith('.h5') or os.path.isdir(model_path):
        model_type = 'tensorflow'
    elif model_path.endswith('.caffemodel'):
        model_type = 'caffe'
    elif model_path.endswith('.tflite'):
        model_type = 'tflite'
    else:
        # TODO: Support other models
        ERROR('Unsupported model type!')
    return model_type


def match_node_name(node_name, ir_name, data_input_name=None, strict_mode=False):
    ''' Check whether node_name exists in ir_name after adding/removing postfix '_0'
    and ':0' in ir_name.
    '''
    for _name in [data_input_name, ir_name]:
        if _name:
            tmp_name_0 = _name
            available_names = [tmp_name_0]
            if _name[-2:] == '_0':
                tmp_name_1 = _name[:-2]
                if not strict_mode:
                    available_names.append(_name.replace('_0', ':0'))
            else:
                tmp_name_1 = _name + '_0'
            if _name[-2:] == ':0':
                tmp_name_2 = _name[:-2]
            else:
                tmp_name_2 = _name + ':0'
            if not strict_mode:
                available_names.append(tmp_name_1)
                available_names.append(tmp_name_2)
            if node_name in available_names:
                return True
    return False


def save_data_to_file(file_path, data_dict):
    ''' Save data dict to file. Remove the existing file if it already exists.
    '''
    if os.path.exists(file_path):
        WARN('Original file %s is overwriten!' % file_path)
        os.remove(file_path)
    np.save(file_path, data_dict)
    INFO('Save output data to file %s' % file_path)
