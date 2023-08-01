import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_zero_fraction_model(pb_file_path, input_size):
    ''' Create tensorflow model for zero_fraction op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        op1 = tf.math.zero_fraction(x, name='zero_fraction')
        y = tf.math.add(op1, x, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'zero_fraction'
input_shapes = [[], [2, 3, 4], [10], ]
feed_dict = dict()

for input_shape in input_shapes:
    # Generate input data
    feed_dict.clear()
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

    model_name = TEST_NAME + '-' + str(len(input_shape))
    model_path = model_name + '.pb'
    # Create model
    create_zero_fraction_model(model_path, input_shape)

    # FIXME: opt crashes for ZeroFraction. Enable verify after it's fixed.
    verify = False
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=False, verify=verify)
    assert exit_status
