# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_nms_model(tflite_file_path, input_size, scores, max_output_size, iou_threshold, score_threshold, version):
    ''' Create tensorflow model for nms op.
    '''
    assert version in (4, 5)
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        if version == 4:
            out1, out2 = tf.raw_ops.NonMaxSuppressionV4(boxes=x, scores=scores, max_output_size=max_output_size,
                                                        iou_threshold=iou_threshold, score_threshold=score_threshold,
                                                        pad_to_max_output_size=True, name='nms')
            out3 = tf.add(out1, 1, name='Y')
        else:
            out1, out2, out3 = tf.raw_ops.NonMaxSuppressionV5(boxes=x, scores=scores, max_output_size=max_output_size,
                                                              iou_threshold=iou_threshold, score_threshold=score_threshold,
                                                              soft_nms_sigma=0.0, pad_to_max_output_size=True, name='nms')

        sess.run(tf.global_variables_initializer())

        # save to tflite file
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x], output_tensors=[out1, out2, out3])
        tflite_model = converter.convert()
        open(tflite_file_path, "wb").write(tflite_model)


TEST_NAME = 'nms'

input_shape = [3, 4]
feed_dict = dict()
# The coordinates of box corners are not in a fixed format:
# For example, the first box is in format [y_upperright, x_upperright, y_lowerleft, x_lowerleft],
# while the second box is [y_upperleft, x_upperleft, y_lowerright, x_lowerright].
feed_dict['X'] = np.array([[26.219383, 28.419462, 14.341243, -5.540409],
                           [47.862896, 6.328754, -2.2302818, 26.386961],
                           [46.344173, 42.946133, 10.519033, 0.6209858]]).astype(np.float32)
# The coordinates of box corners are in a fixed format:
# feed_dict['X'] = np.array([[26.219383, 28.419462, 14.341243, -5.540409],
#                            [47.862896, 6.328754, -2.2302818, 5.386961],
#                            [46.344173, 42.946133, 10.519033, 0.6209858]]).astype(np.float32)
scores = np.array([43.468216, 45.562622, 41.179985]).astype(np.float32)
max_output_size = 3
iou_threshold = 0.3

for version in (4, 5):
    for score_threshold in (2.5, ):
        model_name = '-'.join([TEST_NAME, str(version), str(score_threshold)])
        model_path = model_name + '.tflite'
        # Create model
        create_nms_model(
            model_path, input_shape, scores, max_output_size, iou_threshold, score_threshold, version)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, verify=True)
        assert exit_status
