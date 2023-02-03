import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_leaky_relu_model(pb_file_path, x1_size):
    ''' Create tensorflow model for LeakyRelu op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.float32, shape=x1_size, name='X1')
        #op1 = tf.raw_ops.LeakyRelu(features=x1, alpha=0.2,name='relu')
        op1 = tf.raw_ops.LeakyRelu(features=x1, name='relu')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'leakyrelu'
input_shape1 = [2, 3]

# Generate input data
feed_dict = dict()
feed_dict['X1:0'] = np.random.ranf(input_shape1).astype(np.float32)

model_path = TEST_NAME + '.pb'
# Create model
create_leaky_relu_model(
    model_path, input_shape1)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', verify=True)
assert exit_status
