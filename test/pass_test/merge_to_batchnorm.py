import numpy as np

import tensorflow.compat.v1 as tf

from AIPUBuilder.Parser.tool_utils.run import run_parser


def create_mul_add_model(pb_file_path, input_size):
    ''' Create tensorflow model for mul_add op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        mul = tf.math.multiply(x, 1.23, name='mul')
        y = tf.add(mul, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'mul_add'
input_shape = [1, 3, 10, 16]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

model_path = TEST_NAME + '.pb'
# Create model
create_mul_add_model(model_path, input_shape)

# Run tests with parser and compare result with runtime
exit_status = run_parser(
    model_path, feed_dict, model_type='tensorflow', save_output=False, verify=True)
assert exit_status
