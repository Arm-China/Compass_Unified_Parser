# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_fused_batch_norm_model(pb_file_path, input_size, data_format, is_training):
    ''' Create tensorflow model for fused_batch_norm op.
    '''
    try:
        with tf.Session(graph=tf.Graph()) as sess:
            x = tf.placeholder(tf.float32, shape=input_size, name='X')
            channel_size = input_size[-1] if data_format == 'NHWC' else input_size[1]
            scale = (np.random.ranf([channel_size]) * 10).astype(np.float32)
            offset = (np.random.ranf([channel_size]) * 20).astype(np.float32)
            if is_training:
                op1 = tf.compat.v1.nn.fused_batch_norm(
                    x, scale, offset, data_format=data_format, is_training=True, name='fused_bn')
            else:
                op1 = tf.compat.v1.nn.fused_batch_norm(
                    x, scale, offset, mean=scale, variance=offset, data_format=data_format, is_training=False, name='fused_bn')
            y = tf.math.multiply(op1[0], 1.1, name='Y')

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


TEST_NAME = 'fused_batch_norm'
input_shape = [2, 224, 224, 16]
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for data_format in ['NCHW', 'NHWC', ]:
    for is_training in [False, True, ]:
        model_name = TEST_NAME + '-' + data_format + '-' + str(is_training)
        model_path = model_name + '.pb'
        # Create model
        model_created = create_fused_batch_norm_model(model_path, input_shape, data_format, is_training)
        assert model_created, 'Fail to create model!'

        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', output_names=['Y:0'], save_output=True, verify=True)
        assert exit_status
