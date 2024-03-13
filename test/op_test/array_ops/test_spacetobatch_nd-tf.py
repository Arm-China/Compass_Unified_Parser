# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_spacetobatchnd_model(pb_file_path, input_size, pad_is_zero=True):
    ''' Create tensorflow model for spacetobatch_nd op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        if pad_is_zero:
            paddings = np.zeros([1, 2], dtype=np.int32).tolist()
        else:
            paddings = np.ones([1, 2], dtype=np.int32).tolist()
        op1 = tf.raw_ops.SpaceToBatchND(input=x, block_shape=[2], paddings=paddings, name='spacetobatch_nd')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'spacetobatch_nd'
input_shape = [2, 50, 4]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for pad_is_zero in (False, True, ):
    model_path = '-'.join([TEST_NAME, str(pad_is_zero)]) + '.pb'
    # Create model
    create_spacetobatchnd_model(model_path, input_shape, pad_is_zero)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', verify=True)
    assert exit_status
