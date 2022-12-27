import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa

from utils.run import run_parser


def create_instancenorm_model(pb_file_path, input_size, axis, scale, center):
    ''' Create tensorflow model for instancenorm op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        norm_func = tfa.layers.InstanceNormalization(axis=axis, scale=scale, center=center,
                                                     gamma_initializer='glorot_uniform', beta_initializer='glorot_uniform')
        norm = norm_func(x)
        y = tf.add(norm, 10., name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'instancenorm'
input_shape = [2, 180, 224, 3]
feed_dict = dict()
# Generate input data
feed_dict['X:0'] = np.random.randint(-100, 200, input_shape).astype(np.float32)

for axis in (-1, 2):
    for scale in (False, True):
        for center in (True, ):
            model_name = '-'.join([TEST_NAME, str(axis), str(scale), str(center)])
            model_path = model_name + '.pb'
            # Create model
            create_instancenorm_model(model_path, input_shape, axis, scale, center)

            exit_status = run_parser(
                model_path, feed_dict, verify=True, expected_keywords=['InstanceNorm'])
            assert exit_status
