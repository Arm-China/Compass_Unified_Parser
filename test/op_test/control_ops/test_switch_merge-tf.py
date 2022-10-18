import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_switch_merge_model(pb_file_path, input_size):
    ''' Create tensorflow model for switch/merge op.
    '''
    from tensorflow.python.ops import control_flow_ops
    with tf.Session(graph=tf.Graph()) as sess:
        x1 = tf.placeholder(tf.float32, shape=input_size, name='X1')
        x2 = tf.placeholder(tf.float32, shape=input_size, name='X2')
        y0 = tf.add(x1, x2)
        output_false, output_true = control_flow_ops.switch(y0, True)
        y1 = tf.add(output_false, 10.0)
        y2 = tf.add(output_true, 20.0)
        y3 = tf.raw_ops.Merge(inputs=[y1, y2])
        out = tf.add(y3[0], 30, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'switch_merge'
input_shapes = [[2, 3, 4], [10]]
feed_dict = dict()

for input_shape in input_shapes:
    # Generate input data
    feed_dict.clear()
    feed_dict['X1:0'] = np.random.ranf(input_shape).astype(np.float32)
    feed_dict['X2:0'] = (np.random.ranf(input_shape) * 100).astype(np.float32)

    model_name = TEST_NAME + '-' + str(len(input_shape))
    model_path = model_name + '.pb'
    # Create model
    create_switch_merge_model(model_path, input_shape)

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, output_names=['Y:0'], model_type='tf', verify=True)
    assert exit_status
