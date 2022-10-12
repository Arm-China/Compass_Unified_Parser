import numpy as np

import tensorflow.compat.v1 as tf

from AIPUBuilder.Parser.tool_utils.run import run_parser


def create_ss_model(pb_file_path, input_size, ss_input,
                    begin_mask,
                    ellipsis_mask,
                    end_mask,
                    Index,
                    new_axis_mask,
                    shrink_axis_mask,
                    T,
                    begin,
                    end,
                    stride):
    ''' Create tensorflow model for ss op.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=input_size, name='X')
        x1 = np.array([1, 1, 1, 1])
        #op1 = tf.zeros_like(zeroslike_input, dtype=tf.float32, name='zeroslike')

        y = tf.strided_slice(x,
                             begin=begin,
                             end=end,
                             strides=stride,
                             begin_mask=begin_mask,
                             end_mask=end_mask,
                             ellipsis_mask=ellipsis_mask,
                             new_axis_mask=new_axis_mask,
                             shrink_axis_mask=shrink_axis_mask,
                             name='Y')

        sess.run(tf.global_variables_initializer())

        # # save to pb file
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x], output_tensors=[y])
        tflite_model = converter.convert()
        open(pb_file_path, "wb").write(tflite_model)


TEST_NAME = 'stride_slice'
# #######################
input_shape = [[1, 1, 1, 1], [2, 28, 1], [10, 12, 20, 30], [10, 12, 20, 30], [3], [30, 40], [30, 40], [30, 40], [10, 100, 100, 10], [
    10, 100, 100, 10], [10, 100, 100, 10], [10, 100, 100, 10], [30, 40], [10, 12, 20, 30], [10, 12, 20, 30]]
begin_mask = [15, 6, 0, 0, 0, 0, 0, 0, 0, 15, 0, 15, 0, 0, 0]
ellipsis_mask = [0, 0, 0, 0, 0, 1, 4, 0, 0, 4, 0, 4, 8, 0, 0]
end_mask = [15, 6, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0]
new_axis_mask = [0, 0, 2, 0, 0, 3, 2, 3, 0, 0, 0, 0, 2, 0, 2]
shrink_axis_mask = [0, 1, 2, 3, 1, 3, 1, 5, 0, 0, 0, 0, 0, 2, 2]
begin = [[0, 0, 0, 0], [-1, 0, 0], [-1, 5, 3, 4], [-1, 5, 3, 4], [1], [2, 5, 3], [2, 5, 3], [2, 5, 3], [5, 4],
         [0, 1, 2, 3], [-3], [9, 99, 98, 9], [2, 5, 3], [-1, 5, 3, 4], [-1, 5, 3, 4]]
end = [[0, 0, 0, 0], [0, 0, 0], [10, 6, 30, 30], [10, 6, 30, 30], [2], [10, 16, 20], [10, 16, 20], [10, 16, 20], [9, 99],
       [9, 90, 91, 8], [-10], [1, 2, 3, 0], [10, 16, 20], [10, 6, 30, 30], [10, 6, 30, 30]]
stride = [[1, 1, 1, 1], [1, 1, 1], [1, 1, 2, 3], [1, 1, 2, 3], [1], [3, 2, 4], [3, 2, 4], [3, 2, 4], [2, 8], [
    4, 3, 2, 1], [-1], [-3, -3, -2, -1], [3, 2, 4], [1, 1, 2, 3], [1, 1, 2, 3]]
Index = 'int32'
T = 'float32'

for i in range(0, len(begin_mask)):
    print(i, " :............")

    feed_dict = dict()
    # Generate input data
    feed_dict.clear()
    feed_dict['X'] = np.random.ranf(input_shape[i]).astype(np.float32)

    model_name = TEST_NAME + '-' + str(len(input_shape[i]))
    model_path = model_name + '.tflite'
    # Create model
    create_ss_model(model_path, input_shape[i], 10.0,
                    begin_mask[i],
                    ellipsis_mask[i],
                    end_mask[i],
                    Index,
                    new_axis_mask[i],
                    shrink_axis_mask[i],
                    T,
                    begin[i],
                    end[i],
                    stride[i])

    # Run tests with parser and compare result with runtime
    exit_status = run_parser(
        model_path, feed_dict, model_type='tflite', save_output=False, verify=True)
    assert exit_status
