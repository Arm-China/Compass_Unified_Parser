# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_topk_model(model_path, input_shape, largest, input_dtype):
    ''' Create tensorflow model for topk op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(input_dtype, shape=input_shape, name='X')
        values, indices = tf.math.top_k(x, k=largest, sorted=True)

        sess.run(tf.global_variables_initializer())

        # save to tflite file
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x], output_tensors=[values, indices])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        open(model_path, 'wb').write(tflite_model)


TEST_NAME = 'topk'
input_shape = [3, 5, 6]

# Generate input data
feed_dict = {}
data = np.random.randint(0, 3, input_shape) * 100
for input_dtype in ('float32', ):
    feed_dict['X'] = data.astype(input_dtype)

    model_path = '-'.join([TEST_NAME, input_dtype]) + '.tflite'
    # Create model
    create_topk_model(model_path, input_shape, 4, input_dtype)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, output_names=['TopKV2', 'TopKV2:1'], verify=True,
                             expected_logs=['Output TopKV2 from cfg is shown as tensor TopKV2_0 in IR',
                                            'Output TopKV2:1 from cfg is shown as tensor TopKV2_1 in IR'])
    assert exit_status
