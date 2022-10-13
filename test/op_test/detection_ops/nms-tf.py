import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_nms_model(pb_file_path, input_size, scores, max_output_size, iou_threshold, score_threshold, version):
    ''' Create tensorflow model for nms op.
    '''
    assert version in (3, 4, 5)
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        if version == 3:
            y = tf.raw_ops.NonMaxSuppressionV3(boxes=x, scores=scores, max_output_size=max_output_size,
                                               iou_threshold=iou_threshold, score_threshold=score_threshold,
                                               name='nms')
            out = tf.add(y, 10, name='Y')
        elif version == 4:
            y = tf.raw_ops.NonMaxSuppressionV4(boxes=x, scores=scores, max_output_size=max_output_size,
                                               iou_threshold=iou_threshold, score_threshold=score_threshold,
                                               pad_to_max_output_size=True, name='nms')
            out1 = tf.add(y[0], 10, name='Y1')
            out = tf.add(out1, y[1], name='Y')
        else:
            y = tf.raw_ops.NonMaxSuppressionV5(boxes=x, scores=scores, max_output_size=max_output_size,
                                               iou_threshold=iou_threshold, score_threshold=score_threshold,
                                               soft_nms_sigma=3.0, pad_to_max_output_size=True, name='nms')
            out1 = tf.add(y[0], 10, name='Y1')
            y1 = tf.cast(y[1], tf.int32)
            out2 = tf.add(out1, y1, name='Y2')
            out = tf.add(out2, y[2], name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'nms'
#input_shape = [20, 4]

# Generate input data
#feed_dict = dict()
#feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.int32) * 300
#scores = np.random.ranf([input_shape[0]]).astype(np.float32)

# For the following example, tf nms output shape is [2] while opt output shape is [3] if
# pad_to_max_output_size is False. For tf nms v4 and v5, we can set pad_to_max_output_size
# to True, but for tf nms v3(NonMaxSuppressionV3) it doesn't have pad_to_max_output_size
# so we are not able to keep output shapes from tf and opt same.
input_shape = [3, 4]
feed_dict = dict()
feed_dict['X:0'] = np.array([[26.219383, 28.419462, 14.341243, -5.540409],
                             [47.862896, 6.328754, -2.2302818, 26.386961],
                             [46.344173, 42.946133, 10.519033, 0.6209858]]).astype(np.float32)
scores = np.array([1.0, 2.0, 3.0]).astype(np.float32)
max_output_size = 3
iou_threshold = 0.5

for version in (5, 4, 3):
    for score_threshold in (0.5, 1.5):
        model_name = '-'.join([TEST_NAME, str(version), str(score_threshold)])
        model_path = model_name + '.pb'
        # Create model
        create_nms_model(
            model_path, input_shape, scores, max_output_size, iou_threshold, score_threshold, version)

        # Run tests with parser and compare result with runtime
        if version == 3:
            verify = False
        else:
            verify = True
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', verify=verify)
        assert exit_status
