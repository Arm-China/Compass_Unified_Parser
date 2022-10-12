import numpy as np

import tensorflow.compat.v1 as tf

from AIPUBuilder.Parser.tool_utils.run import run_parser


def create_splitv_model(pb_file_path, input_size, size_splits, axis):
    ''' Create tensorflow model for splitv op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        num_split = len(size_splits)
        op1 = tf.raw_ops.SplitV(value=x, size_splits=size_splits, axis=axis, num_split=num_split, name='splitv')
        y = tf.concat([op1[index] for index in reversed(range(num_split))], axis=axis, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'splitv'
input_shape = [1, 3, 10, 16, 32]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for size_splits in ([2, 3, 5], [2, 3, 4, 1], [2, -1]):
    model_name = '-'.join([TEST_NAME, str(len(size_splits))])
    model_path = model_name + '.pb'
    # Create model
    create_splitv_model(
        model_path, input_shape, size_splits, axis=2)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=False, verify=True)
    assert exit_status
