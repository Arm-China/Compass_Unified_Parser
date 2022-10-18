import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_batchnorm_model(pb_file_path, input_size, data_format):
    ''' Create tensorflow model for batchnorm op.
    '''
    assert data_format in ('NDHWC', 'NHWC', 'NCHW', 'NCDHW')
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        if data_format in ('NDHWC', 'NHWC'):
            scale = np.tile([1], x.shape[-1]).astype(np.float32)
            offset = np.tile([0], x.shape[-1]).astype(np.float32)
        else:
            scale = np.tile([1], x.shape[1]).astype(np.float32)
            offset = np.tile([0], x.shape[1]).astype(np.float32)
        mean = offset
        variance = scale
        op1 = tf.raw_ops.FusedBatchNormV3(x=x, scale=scale, offset=offset, mean=mean,
                                          variance=variance, data_format=data_format, is_training=False, name='batchnorm')
        y = tf.add(op1.y, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'batchnorm-5d'
input_shape = [1, 3, 10, 16, 32]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

for data_format in ('NDHWC', 'NCDHW', ):
    model_name = '-'.join([TEST_NAME, data_format])
    model_path = model_name + '.pb'
    # Create model
    create_batchnorm_model(
        model_path, input_shape, data_format)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tensorflow',
        output_names=['Y:0'], save_output=False, verify=True)
    assert exit_status
