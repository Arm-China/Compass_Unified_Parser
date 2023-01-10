import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_depthwise_conv_model(pb_file_path, input_size):
    ''' Create tensorflow model for depthwise_conv op.
    '''
    try:
        with tf.Session(graph=tf.Graph()) as sess:

            x = tf.compat.v1.placeholder(
                tf.float32, shape=input_size, name='X')
            filters = np.array(np.random.ranf(
                [10, 6, 1, 10])).astype(np.float32)

            A2 = tf.raw_ops.DepthwiseConv2dNative(input=x,
                                                  filter=filters,
                                                  strides=[1, 1, 1, 1],
                                                  padding='EXPLICIT',
                                                  explicit_paddings=[
                                                      0, 0, 1, 2, 3, 4, 0, 0],
                                                  dilations=[1, 1, 1, 1],
                                                  data_format='NHWC',
                                                  name='depthwiseconv')

            y = tf.add(A2, 10.0, name='Y')

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ['Y'])

            # save to pb file
            with tf.gfile.GFile(pb_file_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())

        return True
    except Exception as e:
        print("Fail to create model because %s" % str(e))
        return False


TEST_NAME = 'depthwise_conv'
input_shape = [1, 200, 100, 1]
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32)

model_name = TEST_NAME
model_path = model_name + '.pb'
# Create model
model_created = create_depthwise_conv_model(model_path, input_shape)
assert model_created, 'Fail to create model!'

exit_status = run_parser(
    model_path, feed_dict, model_type='tf', output_names=['Y:0'], save_output=True, verify=True)
assert exit_status
