# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import os
import re
import numpy as np
from functools import reduce
from ..logger import ERROR


def is_file(file_path):
    return True if (file_path and os.path.isfile(file_path)) else False


def is_dir(dir_path):
    return True if (dir_path and os.path.isdir(dir_path)) else False


def get_absolute_path(file_path):
    file_path = os.path.expanduser(file_path)
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    return file_path


def get_file_name(file_path):
    return os.path.basename(file_path).split('.')[0]


def readable_file(path):
    if not os.path.isfile(path):
        raise Exception('The "{}" is not existing file'.format(path))
    elif not os.access(path, os.R_OK):
        raise Exception('The "{}" is not readable'.format(path))
    else:
        return path


def readable_dir(path):
    if not os.path.isdir(path):
        raise Exception('The "{}" is not existing directory'.format(path))
    elif not os.access(path, os.R_OK):
        raise Exception('The "{}" is not readable'.format(path))
    else:
        return path


def writable_dir(path):
    if path is None:
        raise Exception('The directory parameter is None')
    if os.path.exists(path):
        if os.path.isdir(path):
            if os.access(path, os.W_OK):
                return path
            else:
                raise Exception(
                    'The directory "{}" is not writable'.format(path))
        else:
            raise Exception('The "{}" is not a directory'.format(path))
    else:
        cur_path = path
        while os.path.dirname(cur_path) != cur_path:
            if os.path.exists(cur_path):
                break
            cur_path = os.path.dirname(cur_path)
        if cur_path == '':
            cur_path = os.path.curdir
        if os.access(cur_path, os.W_OK):
            return path
        else:
            raise Exception(
                'The directory "{}" is not writable'.format(cur_path))


# '[1,2,3],[3,5]' to [[1,2,3],[3,5]]
def list_string_to_list(list_string):
    ret = []
    if list_string:
        items = re.findall('(\[.*?\])', list_string)
        for meta_item in items:
            inner_str = meta_item.lstrip('[').rstrip(']')
            if inner_str:
                meta_list = [int(m) for m in inner_str.split(',')]
            else:
                meta_list = list()
            ret.append(meta_list)
    return ret


# ['aa', 'bb'] to 'aa,bb'
def string_list_to_string(string_list):
    return reduce(lambda x, y: (x + ',' + y), string_list) if string_list else ''


# 'AA,BB' to ['AA','BB']
def multi_string_to_list(multi_string):
    if multi_string:
        multi_string = re.sub(r' ', '', multi_string)
        return [item for item in multi_string.split(',')]
    else:
        return []


# '1.0,2.0,3' to [1.0, 2.0, 3]
def float_string_to_list(num_str):
    ret = re.sub(r' ', '', num_str).split(',')
    return [float(r) for r in ret]


# [1, 2, 3] to '1,2,3'
def list_list_to_string(list_list):
    if not isinstance(list_list, list):
        list_list = [list_list]
    return str(list_list)[1:-1].replace(' ', '') if list_list is not None else ''


num_list_to_string = list_list_to_string


def get_random_array(shape, type_str):
    return np.random.ranf(size=shape).astype(dtype=np.dtype(type_str))


def extend_lists(lists):
    return list(reduce(lambda x, y: x + y, lists))


def read_config_txt(txt_path):
    ret = {}
    if is_file(txt_path):
        valid_line_pattern = re.compile(
            r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*(.*)\s*$')
        item_pattern = re.compile(r'(?<=\s)*[a-zA-Z_][a-zA-Z0-9_]*(?=\s)*')
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                line_matched = re.search(valid_line_pattern, l)
                item_matched = re.search(item_pattern, l)
                if line_matched is not None and item_matched is not None:
                    ret.update({item_matched[0]: line_matched[1]})
    return ret


def get_version(module):
    if hasattr(module, '__version__'):
        vers = [('%02d' % int(v))
                for v in getattr(module, '__version__').split('.')[0:2]]
        return float('.'.join(vers))
    else:
        ERROR('[Parser]: Invalid module (%s) that does not have __version__  in get_version!' %
              module.__name__)


def get_converted_dtype(original_dtype, return_type_string=False):
    if 'float' in original_dtype:
        to_dtype = np.float32
    elif 'uint' in original_dtype:
        to_dtype = np.uint32
    elif 'int' in original_dtype:
        to_dtype = np.int32
    elif 'bool' in original_dtype:
        to_dtype = np.uint8
    else:
        to_dtype = None
    if return_type_string:
        to_dtype = to_dtype.__name__ if to_dtype is not None else ''
    return to_dtype
