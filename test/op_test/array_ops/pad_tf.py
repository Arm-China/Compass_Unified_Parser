import numpy as np

import tensorflow.compat.v1 as tf

from AIPUBuilder.Parser.tool_utils.run import run_parser


def create_pad_model(pb_file_path, input_size, paddings, constant_values):
    ''' Create tensorflow model for pad op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        op1 = tf.raw_ops.PadV2(input=x, paddings=paddings,
                               constant_values=constant_values, name='padv2')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


def create_mirror_pad_model(pb_file_path, input_size, paddings, constant_values):
    ''' Create tensorflow model for pad op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        op1 = tf.raw_ops.MirrorPad(
            input=x, paddings=paddings, mode='REFLECT', name='padv2')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'pad'

input_shape = [[10, 20, 30, 40, 50], 10, [1, 8, 26, 512]]
paddings = [[[0, 0], [3, 4], [5, 6], [0, 0], [9, 10]],
            [[1, 2]], [[0, 0], [1, 1], [1, 1], [0, 0]]]
constant_values = [0, 0, 0]

for i in range(0, len(input_shape)):

    # Generate input data
    feed_dict = dict()
    feed_dict['X:0'] = np.random.ranf(input_shape[i]).astype(np.float32)

    model_path = TEST_NAME + '.pb'
    # Create model
    #create_pad_model(model_path, input_shape,paddings,constant_values)
    create_mirror_pad_model(
        model_path, input_shape[i], paddings[i], constant_values[i])

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=False, verify=True)
    assert exit_status

    # Generate input data
    feed_dict = dict()
    feed_dict['X:0'] = np.random.ranf(input_shape[i]).astype(np.float32)

    model_path = TEST_NAME + '.pb'
    # Create model
    create_pad_model(
        model_path, input_shape[i], paddings[i], constant_values[i])

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=False, verify=True)
    assert exit_status
