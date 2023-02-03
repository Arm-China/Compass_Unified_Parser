import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_zeroslike_model(tflite_file_path, input_size1):
    ''' Create tflite model for zeroslike op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, shape=input_size1, name='X')

        op1 = tf.raw_ops.ZerosLike(x=x, name='zeroslike')
        op2 = tf.raw_ops.Shape(input=op1)
        y1 = tf.add(op1, x)
        y2 = tf.add(op2, x)
        y = tf.add(y1, y2, name='Y')

        sess.run(tf.global_variables_initializer())

        # save to tflite file
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x], output_tensors=[y])
        tflite_model = converter.convert()
        open(tflite_file_path, "wb").write(tflite_model)


TEST_NAME = 'zeroslike'
input_shape1 = [4, 2, 3, 1]


# Generate input data
feed_dict = dict()
feed_dict['X'] = (np.random.ranf(input_shape1) * 10).astype(np.int32)


# for idx, cond_shape in enumerate(cond_shapes):
model_name = '-'.join([TEST_NAME])
model_path = model_name + '.tflite'

# Create model
create_zeroslike_model(model_path, input_shape1)

# Run tests with parser and compare result with runtime
exit_status = run_parser(model_path, feed_dict, save_output=True, verify=True)
assert exit_status
