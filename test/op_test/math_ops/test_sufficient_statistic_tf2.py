# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from itertools import combinations
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.run import run_parser
import itertools


def create_suff_model(model_path, x1_size, x2_size, keepdims, axes, one_input, output_list_key):
    ''' Create tensorflow model for sufficient_statistics op.
    '''
    x1 = keras.Input(shape=x1_size[1:], batch_size=x1_size[0], name='X1')
    x2 = keras.Input(shape=x2_size[1:], batch_size=x2_size[0], name='X2')

    if one_input == 1:
        suff = tf.nn.sufficient_statistics(
            x1, axes=axes, shift=None, keepdims=bool(keepdims), name='suff')
        list_map = {'0': 'relu0', '1': 'relu1', '2': 'relu2'}
        output_lists = []
        for key in output_list_key:
            relu = tf.nn.relu6(suff[int(key)], name=list_map[key])
            output_lists.append(relu)
        model = keras.models.Model([x1], output_lists)
    else:

        suff = tf.nn.sufficient_statistics(
            x1, axes=axes, shift=x2, keepdims=bool(keepdims), name='suff')
        list_map = {'0': 'relu0', '1': 'relu1', '2': 'relu2', '3': 'relu3'}
        output_lists = []
        for key in output_list_key:
            relu = tf.nn.relu6(suff[int(key)], name=list_map[key])
            output_lists.append(relu)
        model = keras.models.Model([x1, x2], output_lists)

    model.save(model_path)


TEST_NAME = 'sufficient_statistics'
input_shape1 = [1, 2, 3]
input_shape2 = [1, 2, 3]
keep_dims = [True, False]
axes_list = [[0, 2], [], [0], [0, 1, 2]]
one_input_list = [1, 0]

items_2 = ['0', '1', '2', '3']
items_1 = ['0', '1', '2']
output_lists_2 = []
output_lists_1 = []
for num in range(1, 5):
    for i in combinations(items_2, num):
        # pick vaild model
        if '1' in i or '2' in i:
            output_lists_2.append(i)

for num in range(1, 4):
    for i in combinations(items_1, num):
        if '1' in i or '2' in i:
            output_lists_1.append(i)

for keep_dim in keep_dims:
    for axes in axes_list:
        for one_input in one_input_list:
            # Generate input data
            feed_dict = {}
            feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
            feed_dict['X2:0'] = np.random.ranf(input_shape2).astype(np.float32)

            # Create model
            if one_input == 0:
                for output_list in output_lists_2:
                    model_path = TEST_NAME + \
                        str(axes) + str(keep_dim) + \
                        str(one_input)+str(output_list) + '.h5'
                    # model_path = TEST_NAME + str(output_list) + '.h5'
                    create_suff_model(
                        model_path, input_shape1, input_shape2, keep_dim, axes, one_input, output_list)
                    # Run tests with parser and compare result with runtime
                    exit_status = run_parser(
                        model_path, feed_dict, model_type='tf', verify=False)
                    assert exit_status
            else:
                for output_list in output_lists_1:
                    model_path = TEST_NAME + \
                        str(axes) + str(keep_dim) + \
                        str(one_input)+str(output_list) + '.h5'

                    # model_path = TEST_NAME + str(output_list) + '.h5'
                    create_suff_model(
                        model_path, input_shape1, input_shape2, keep_dim, axes, one_input, output_list)
                    # Run tests with parser and compare result with runtime
                    exit_status = run_parser(
                        model_path, feed_dict, model_type='tf', verify=False)
                    assert exit_status
