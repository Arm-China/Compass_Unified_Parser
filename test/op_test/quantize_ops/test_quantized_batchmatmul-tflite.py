# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_batchmatmul_model(tflite_file_path, input_size1, input_size2, adj_x, adj_y, input_type=tf.int8):
    ''' Create tflite model for batchmatmul op.
    '''
    def representative_dataset():
        fake_x1 = np.random.random(input_size1).astype(np.float32)
        fake_x2 = np.random.random(input_size2).astype(np.float32)
        yield [fake_x1, fake_x2]

    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.float32, shape=input_size1, name='X1')
        x2 = tf.placeholder(tf.float32, shape=input_size2, name='X2')
        op1 = tf.raw_ops.BatchMatMul(x=x1, y=x2, adj_x=adj_x, adj_y=adj_y)
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())

        # save to tflite file
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x1, x2], output_tensors=[y])
        converter._experimental_disable_batchmatmul_unfold = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = input_type
        converter.inference_output_type = input_type
        tflite_model = converter.convert()
        open(tflite_file_path, 'wb').write(tflite_model)


TEST_NAME = 'batch_matmul'
input_shapes1 = [[1, 2, 1, 192], [3, 4, 5, 6], [2, 3, 4, 5], [3, 2, 5, 4]]
input_shapes2 = [[1, 2, 1, 192], [3, 4, 5, 7], [2, 3, 5, 6], [3, 2, 7, 5]]
adj_x_vals = [False, True, False, True]
adj_y_vals = [True, False, False, True]

# Generate input data
feed_dict = dict()

for input_shape1, input_shape2, adj_x, adj_y in zip(input_shapes1, input_shapes2, adj_x_vals, adj_y_vals):
    feed_dict['X1'] = (np.random.ranf(input_shape1) * 10).astype(np.float32)
    feed_dict['X2'] = (np.random.ranf(input_shape2) * 10).astype(np.float32)

    # for idx, cond_shape in enumerate(cond_shapes):
    model_name = '-'.join([TEST_NAME, str(adj_x), str(adj_y)])
    model_path = model_name + '.tflite'

    for input_dtype in (tf.int8, tf.uint8):
        # Create model
        create_batchmatmul_model(model_path, input_shape1, input_shape2, adj_x, adj_y, input_dtype)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, verify=False,
            expected_keywords=['compat_quantized_model=true', 'layer_type=MatMul',
                               'layer_top_type=[' + input_dtype.name + ']'])
        assert exit_status
