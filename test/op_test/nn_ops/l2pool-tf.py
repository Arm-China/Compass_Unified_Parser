import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_l2pool_model(pb_file_path, input_size, padding):
    ''' Create tensorflow model for l2pool op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        ksize = [1, 3, 5, 1]
        op1 = tf.nn.avg_pool(tf.square(x),
                             ksize=ksize,
                             strides=[1, 7, 7, 1],
                             padding=padding,
                             name='avgpool')
        op2 = tf.math.multiply(op1, ksize[1] * ksize[2], name='mul')
        y = tf.sqrt(op2, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'l2pool'
input_shape = [2, 224, 224, 3]
feed_dict = dict()

for padding in ['SAME', 'VALID']:
    # Generate input data
    feed_dict.clear()
    feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

    model_name = TEST_NAME + '-' + padding
    model_path = model_name + '.pb'
    # Create model
    create_l2pool_model(model_path, input_shape, padding)

    # Run tests with parser and compare result with runtime
    # Set verify to False as this op's output shape is not fixed.
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=True, verify=True)
    assert exit_status
