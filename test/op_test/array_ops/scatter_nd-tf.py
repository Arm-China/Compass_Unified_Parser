import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_scatter_nd_model(pb_file_path, indices_shape, updates_shape, shape):
    ''' Create tensorflow model for scatter_nd op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.int32, shape=indices_shape, name='X1')
        x2 = tf.placeholder(tf.float32, shape=updates_shape, name='X2')
        x3 = tf.constant(shape)
        op1 = tf.scatter_nd(x1, x2, x3, name='scatter_nd')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'scatter_nd'
indices_shape = [1728, 4]
updates_shape = [1728]
shape = [4, 18, 24, 4]
# Generate input data
feed_dict = dict()
feed_dict['X1:0'] = np.ones(indices_shape).astype(np.int32)
feed_dict['X2:0'] = np.random.ranf(updates_shape).astype(np.float32)

model_path = TEST_NAME + '.pb'
# Create model
create_scatter_nd_model(model_path, indices_shape, updates_shape, shape)

# non-constant indices is not supported
exit_status = run_parser(
    model_path, feed_dict, model_type='tf', expected_keywords=['TfScatterNd'], verify=False)
assert exit_status
