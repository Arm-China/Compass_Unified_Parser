import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_floordiv_model(pb_file_path, input_size, div_num):
    ''' Create tensorflow model for floordiv op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        op1 = tf.math.floordiv(x=x, y=div_num, name='floordiv')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'floordiv'
input_shape = [1, 3, 10, 16, 32]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for div_num in (10, 2.3, ):
    model_name = '-'.join([TEST_NAME, str(div_num)])
    model_path = model_name + '.pb'
    # Create model
    create_floordiv_model(
        model_path, input_shape, div_num)
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tf', save_output=False, verify=True)
    assert exit_status
