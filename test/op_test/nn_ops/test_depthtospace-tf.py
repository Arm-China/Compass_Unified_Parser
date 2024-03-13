# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_depth_to_space_model(pb_file_path, input_size, data_format):
    ''' Create tensorflow model for depth_to_space op.
    '''
    try:
        with tf.Session(graph=tf.Graph()) as sess:
            x = tf.placeholder(tf.float32, shape=input_size, name='X')
            op1 = tf.nn.depth_to_space(x, 2, data_format=data_format, name='d2s')
            y = tf.math.multiply(op1, 1.5, name='Y')

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ['Y'])

            # save to pb file
            with tf.gfile.GFile(pb_file_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
        return True
    except Exception as e:
        print("Fail to create model because %s" % str(e))
        return False


TEST_NAME = 'depth_to_space'
input_shape = [2, 224, 224, 16]
feed_dict = dict()

for data_format in ['NCHW', 'NHWC', 'NCHW_VECT_C']:
    if data_format == 'NCHW_VECT_C':
        input_shape = input_shape + [4]
    # Generate input data
    feed_dict.clear()
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

    model_name = TEST_NAME + '-' + data_format
    model_path = model_name + '.pb'
    # Create model
    model_created = create_depth_to_space_model(model_path, input_shape, data_format)
    if data_format.startswith('NC') and not model_created:
        # Model cannot be created for NCHW and NCHW_VECT_C if running on a host without GPU
        continue

    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=True, verify=True)
    assert exit_status
