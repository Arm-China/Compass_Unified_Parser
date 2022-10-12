import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_where_model(tflite_file_path, cond_shape, input_size):
    ''' Create tensorflow model for where op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        cond = tf.placeholder(tf.bool, shape=cond_shape, name='X1')
        x_true = tf.placeholder(tf.float32, shape=input_size, name='X2')
        x_false = tf.placeholder(tf.float32, shape=input_size, name='X3')
        op1 = tf.where(cond, x_true, x_false, name='where')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())

        # save to tflite file
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[cond, x_true, x_false], output_tensors=[y])
        tflite_model = converter.convert()
        open(tflite_file_path, "wb").write(tflite_model)


TEST_NAME = 'where'
cond_shapes = [[20, 20, 30], [20], ]
input_shape = [20, 20, 30]

# Generate input data
feed_dict = dict()
feed_dict['X2'] = (np.random.ranf(input_shape) * 100).astype(np.float32)
feed_dict['X3'] = (np.random.ranf(input_shape) * 10).astype(np.float32)

for idx, cond_shape in enumerate(cond_shapes):
    model_name = '-'.join([TEST_NAME, str(idx)])
    model_path = model_name + '.tflite'

    # Create model
    create_where_model(model_path, cond_shape, input_shape)

    feed_dict['X1'] = np.random.randint(0, 2, cond_shape).astype(bool)
    # Run tests with parser and compare result with runtime
    # FIXME: Has issue in similarity. Enable verify after fixing.
    exit_status = run_parser(
        model_path, feed_dict, save_output=True, verify=False)
    assert exit_status
