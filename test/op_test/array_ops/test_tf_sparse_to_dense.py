import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_SparseToDense_model_tf2_api(pb_file_path, sparse_shape, indices_shape, output_shape, default_value=None, validate_indices=False):
    ''' Create tensorflow model for SparseToDense op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        if indices_shape == [3, 2]:
            x0 = np.array(
                [[0, 2], [2, 3], [3, 4]]).astype(np.int32)
        elif indices_shape == []:
            x0 = np.array(2).astype(np.int32)
        else:
            x0 = np.array([2, 3, 4, 5]).astype(np.int32)

        x1 = tf.placeholder(tf.float32, shape=sparse_shape, name='X1')
        sp_input = tf.sparse.SparseTensor(
            dense_shape=output_shape,
            values=x1,
            indices=x0)
        if default_value == -1:
            out = tf.sparse.to_dense(sp_input,
                                     validate_indices=validate_indices,
                                     name='SparseToDense')
        else:
            out = tf.sparse.to_dense(sp_input,
                                     default_value=default_value,
                                     validate_indices=validate_indices,
                                     name='SparseToDense')

        y = tf.math.add(out, 11.2, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'SparseToDense'

default_values = [None, 1, -1]
validate_indices_s = [False, True]

sparse_shapes = [[3], ]  # [4], []]
indices_shapes = [[3, 2], ]  # [4], []]
output_shapes = [[10, 20], ]  # [15], [15]]

feed_dict = dict()

for sparse_shape, indices_shape, output_shape in zip(sparse_shapes, indices_shapes, output_shapes):
    for default_value in default_values:
        for validate_indices in validate_indices_s:
            # Generate input data
            feed_dict.clear()
            feed_dict['X1:0'] = (np.random.ranf(
                sparse_shape) * 100).astype(np.int64)

            model_path = '-'.join([TEST_NAME, str(len(sparse_shape)),
                                   str(len(indices_shape))]) + str(default_value) + '.pb'
            # Create model
            create_SparseToDense_model_tf2_api(
                model_path, sparse_shape, indices_shape, output_shape, default_value, validate_indices)

            exit_status = run_parser(model_path, feed_dict)
            assert exit_status
