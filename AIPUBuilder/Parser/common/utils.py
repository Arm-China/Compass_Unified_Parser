# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import re
import numpy as np
from functools import reduce
from ..logger import ERROR, WARN


def is_file(file_path):
    return True if (file_path and os.path.isfile(file_path)) else False


def is_dir(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except PermissionError:
            ERROR('[Parser]: Cannot Access this dir path: %s' % dir_path)
    return True if (dir_path and os.path.isdir(dir_path)) else False


def get_absolute_path(file_path):
    file_path = os.path.expanduser(file_path)
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    return file_path


def get_file_name(file_path):
    return os.path.basename(file_path).split('.')[0]


def get_target_graph(target_g_name, root_graph, parent_graph=None):
    for _, v in root_graph._attr['subgraphs'].items():
        if target_g_name in v:
            return v[target_g_name]
    if parent_graph is not None and target_g_name == parent_graph.name:
        return parent_graph
    else:
        return root_graph


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
    return reduce(lambda x, y: (str(x) + ',' + str(y)), string_list) if string_list else ''


# 'AA,BB' to ['AA','BB']
def multi_string_to_list(multi_string):
    if multi_string:
        multi_string = re.sub(r' ', '', multi_string)
        multi_string = multi_string.rstrip(',')
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


def version_to_tuple(version_str):
    version_tuple = tuple()
    try:
        version_tuple = tuple(map(int, version_str.split('.')))
    except Exception as e:
        ERROR('[Parser]: Cannot get version tuple for %s in version_to_tuple because %s!' % (version_str, str(e)))
    return version_tuple


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


def get_closest_dtype(origin_dtype, available_dtypes):
    def _loop_ava_dtypes(_is_int, _is_unsign, _start_exp, _end_exp, _step):
        can_list = []
        if _is_int:
            for e in range(_start_exp + _step, _end_exp + 1, _step):
                can_list.append(str(dtype) + str(int(np.power(2, e))))
                if _is_unsign:
                    can_list.append(
                        'int' + str(int(np.power(2, min(e + 1, _end_exp) if _step > 0 else max(e + 1, _end_exp)))))
        else:
            for e in range(_start_exp + _step, _end_exp + 1, _step):
                can_list.append(str(dtype) + str(int(np.power(2, e))))
        return can_list

    closest_dtype = None
    dtype = re.findall(r'[a-zA-Z]+', origin_dtype)[0]
    bits = re.findall(r'\d+', origin_dtype)[0]
    is_int = 'int' in dtype
    exp = int(np.log2(int(bits)))
    is_unsign = dtype[0] == 'u'
    final_step = 1
    for max_exp, step in ((5, 1), (1, -1)):  # increase bits and decrease bits
        can_cast_list = _loop_ava_dtypes(is_int, is_unsign, exp, max_exp, step)
        for d in can_cast_list:
            if d in available_dtypes:
                closest_dtype = d
                break
        if closest_dtype:
            final_step = step
            break

    matched_dtype = available_dtypes[0] if closest_dtype is None else closest_dtype
    if closest_dtype is None or final_step < 0:
        WARN(f'[Parser]: Cast from {origin_dtype} to {matched_dtype} here may cause similarity down!')
    return matched_dtype


def get_dict_params(params):
    ret = {}
    if isinstance(params, str) and len(params) > 0:
        param_pattern = re.compile(r'\{\s*[^\{^\}]*\s*\}')
        all_params = re.findall(param_pattern, params)
        if len(all_params) > 0:
            all_params = all_params[0].lstrip('{').lstrip(' ').rstrip('}').rstrip(' ')
            meta_pattern = re.compile(r'[\s*[a-zA-Z_]*\s*\:\s*[0-9]*\s*,*]*')
            meta_found = re.findall(meta_pattern, all_params)
            for m in meta_found:
                key, value = m.split(':')[:]
                key = key.lstrip(' ').rstrip(' ')
                value = value.lstrip(' ').rstrip(' ').rstrip(',')
                ret.update({key: int(value)})
    return ret
