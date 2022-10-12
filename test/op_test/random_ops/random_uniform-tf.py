import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_RandomUniform_model(pb_file_path, input_size, seed=0):
    ''' Create tensorflow model for RandomUniform op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        shape = tf.shape(x)
        # Use shape as the input of RandomUniform op
        out = tf.raw_ops.RandomUniform(shape=shape, dtype=tf.float32, seed=seed, seed2=0, name='RandomUniform')
        y = tf.math.add(out, x, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'RandomUniform'
input_shapes = [[10, 20], [3, 5, 10]]
feed_dict = dict()

for input_shape in input_shapes:
    for seed in (5, 0):
        # Generate input data
        feed_dict.clear()
        feed_dict['X:0'] = (np.random.ranf(input_shape) * 100).astype(np.int64)

        model_name = '-'.join([TEST_NAME, str(len(input_shape)), str(seed)])
        model_path = model_name + '.pb'
        # Create model
        create_RandomUniform_model(model_path, input_shape, seed)

        # Run tests with parser and compare result with runtime
        # When seed and seed2 are both 0, it will get different outputs for the same input each time running the model
        verify = True if seed != 0 else False
        exit_status = run_parser(
            model_path, feed_dict, model_type='tf', save_output=True, verify=verify)
        assert exit_status
