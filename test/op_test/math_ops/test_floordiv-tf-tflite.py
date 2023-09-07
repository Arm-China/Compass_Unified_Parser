import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_floordiv_model(model_path, input_size, div_num, input_dtype=tf.float32, is_tflite_model=False):
    ''' Create tensorflow/tflite model for floordiv op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(input_dtype, shape=input_size, name='X')
        op1 = tf.math.floordiv(x=x, y=div_num, name='floor')
        op2 = tf.math.floor(tf.cast(op1, 'float32'), name='floor_1')
        op3 = tf.cast(op2, 'int32')
        y = tf.add(op3, 10, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        if is_tflite_model:
            # save to tflite file
            converter = tf.lite.TFLiteConverter.from_session(sess,
                                                             input_tensors=[x], output_tensors=[y])
            tflite_model = converter.convert()
            open(model_path, 'wb').write(tflite_model)
        else:
            # save to pb file
            with tf.gfile.GFile(model_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())


TEST_NAME = 'floordiv'
input_shape = [1, 3, 10, 16, 32]

# Generate input data
feed_dict = dict()

for div_num in (10, 11.2):
    if isinstance(div_num, int):
        input_dtype = tf.int32
        feed_dict['X'] = np.random.randint(-20, 20, input_shape).astype(np.int32)
    else:
        input_dtype = tf.float32
        feed_dict['X'] = np.random.ranf(input_shape).astype(np.float32) * 20
    for model_type in ('tf', 'tflite'):
        model_name = '-'.join([TEST_NAME, str(div_num), input_dtype.name, model_type])
        model_path = model_name + ('.pb' if model_type == 'tf' else '.tflite')
        # Create model
        create_floordiv_model(
            model_path, input_shape, div_num, input_dtype, model_type == 'tflite')
        # Run tests with parser and compare result with runtime
        exit_status = run_parser(
            model_path, feed_dict, verify=True)
        assert exit_status
