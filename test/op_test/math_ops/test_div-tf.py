import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_div_model(pb_file_path, x1_size, x2_size):
    ''' Create tensorflow model for div op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.float32, shape=x1_size, name='X1')
        x2 = tf.placeholder(tf.float32, shape=x2_size, name='X2')
        op1 = tf.raw_ops.Div(x=x1, y=x2, name='div')
        y = tf.div(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'div'
input_shape1 = [2, 1, 1, 1, 2]
input_shape2 = [2, 1, 2]

# Generate input data
feed_dict = dict()
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)
feed_dict['X2:0'] = np.random.ranf(input_shape2).astype(np.float32)

model_path = TEST_NAME + '.pb'
# Create model
create_div_model(
    model_path, input_shape1, input_shape2)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
