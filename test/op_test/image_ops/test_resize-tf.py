import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import run_parser


def create_resize_model(pb_file_path, input_size, mode, align_corners=False, half_pixel_centers=False):
    ''' Create tensorflow model for resize op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        if mode == 'bilinear':
            op1 = tf.raw_ops.ResizeBilinear(images=x, size=[65, 65],
                                            align_corners=align_corners,
                                            half_pixel_centers=half_pixel_centers,
                                            name='resize_bilinear')
        else:
            op1 = tf.raw_ops.ResizeNearestNeighbor(images=x, size=[65, 65],
                                                   align_corners=align_corners,
                                                   half_pixel_centers=half_pixel_centers,
                                                   name='resize_nearest')
        y = tf.add(op1, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['Y'])

        # save to pb file
        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


TEST_NAME = 'resize'
input_shape = [1, 12, 20, 256]

# Generate input data
feed_dict = dict()
feed_dict['X:0'] = np.random.ranf(input_shape).astype(np.float32) * 100

for mode in ('bilinear', 'nearest'):
    for align_corners in (True, False):
        for half_pixel_centers in (True, False):
            # tf: If half_pixel_centers is True, align_corners must be False
            if half_pixel_centers and align_corners:
                continue
            model_path = '-'.join([TEST_NAME, mode, str(align_corners), str(half_pixel_centers)]) + '.pb'
            # Create model
            create_resize_model(model_path, input_shape, mode, align_corners, half_pixel_centers)

            # Run tests with parser and compare result with runtime
            exit_status = run_parser(
                model_path, feed_dict, model_type='tf', verify=True)
            assert exit_status
