# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

from .common import get_feed_dict
from ..logger import DEBUG, ERROR, INFO, WARN


def compare_data(data1, data2):
    ''' Compare data1 and data2. Return True if they have the same shape and the differences
    of data are not bigger than 1e-4. Otherwise, return False.
    '''
    import torch

    INFO('-------------- COMPARE DATA ----------------')
    if np.isscalar(data1):
        data1 = np.array(data1)
    if np.isscalar(data2):
        data2 = np.array(data2)
    if data1.dtype != data2.dtype:
        DEBUG('Different dtype! The output dtype of first data is %s, output dtype of second data is %s.' %
              (str(data1.dtype), str(data2.dtype)))
    if data1.dtype != np.float32:
        data1 = data1.astype(np.float32)
    if data2.dtype != np.float32:
        data2 = data2.astype(np.float32)
    if data1.shape == data2.shape:
        # Convert nan to num -1
        data1 = np.nan_to_num(data1, nan=-1)
        data2 = np.nan_to_num(data2, nan=-1)
        cos = torch.nn.CosineSimilarity(dim=1)
        sim = cos(torch.from_numpy(data2.reshape([1, -1])),
                  torch.from_numpy(data1.reshape([1, -1]))).numpy().item()
        mean_value = np.mean(abs(data1 - data2))
        non_zero_num = np.count_nonzero(data2)
        l1_error = np.sum(abs(data1 - data2)) / non_zero_num if non_zero_num != 0 else np.nan
        INFO('Similarity is %.6f, mean is %.6f, l1 error is %f' % (sim, mean_value, l1_error))
        ret = True
        # Check similarity: pass if sim > 0.9, or data and sim are both 0
        if sim > 0.9 or (np.allclose(sim, 0) and np.allclose(data1, 0)):
            pass
        else:
            ret = False
        # Check mean: pass if mean <= 1e-2
        if mean_value > 1e-2:
            ret = False
        if not ret:
            INFO('Former data output: ')
            print(data1.flatten()[:50])
            INFO('Latter data output: ')
            print(data2.flatten()[:50])
            return False
        return True
    else:
        WARN('Different shape! The output shape of first data is %s, output shape of second data is %s.' %
             (str(data1.shape), str(data2.shape)))
        return False


def compare_data_dict(data_dict1, data_dict2):
    ''' Compare two data dicts and return True if they are almost same(the differences
    of data are not bigger than 1e-2). The keys of dicts are ignored.
    '''
    data1_cnt = len(data_dict1)
    data2_cnt = len(data_dict2)
    if data1_cnt != data2_cnt:
        ERROR('Two data dicts have different length! list1: %s, list2: %s' %
              (str(data1_cnt), str(data2_cnt)))
    # Ignore dict keys. Only compare dicts' value.
    is_passed = True
    for n1, d1, n2, d2 in zip(data_dict1.keys(), data_dict1.values(), data_dict2.keys(), data_dict2.values()):
        INFO('Comparing Former %s and latter %s' % (n1, n2))
        if not compare_data(d1, d2):
            is_passed = False
    return is_passed


def compare_data_from_file(first_file, second_file):
    ''' Compare two data files and return True if they are almost same(the differences
    of data are not bigger than 1e-4). The files store a dict, whose keys are ignored.
    '''
    data1 = get_feed_dict(first_file)
    data2 = get_feed_dict(second_file)
    return compare_data_dict(data1, data2)
