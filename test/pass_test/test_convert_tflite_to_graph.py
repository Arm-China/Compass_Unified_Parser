import os
import numpy as np

import tensorflow.compat.v1 as tf

from utils.run import generate_ir
from utils.forward import opt_forward, tflite_forward
from utils.compare import compare_data_dict


def create_batch_is_none_model(tflite_file_path, input_size):
    ''' Create tflite model in which batch is None.
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.float32, shape=[None] + input_size[1:], name='X')
        y = tf.add(x, 10.0, name='Y')

        sess.run(tf.global_variables_initializer())

        # save to tflite file
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         input_tensors=[x], output_tensors=[y])
        tflite_model = converter.convert()
        open(tflite_file_path, 'wb').write(tflite_model)


TEST_NAME = 'change_batch_size'
input_shape = [1, 2, 1, 192]

# Generate input data
feed_dict = dict()
feed_dict['X'] = (np.random.ranf(input_shape) * 10).astype(np.float32)

# for idx, cond_shape in enumerate(cond_shapes):
model_name = '-'.join([TEST_NAME])
model_path = model_name + '.tflite'

# Create model
create_batch_is_none_model(model_path, input_shape)

# Generate cfg
output_dir = './output_dir'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cfg_path = TEST_NAME + '.cfg'
cfg_content = '''[Common]
model_type = tflite
model_name = {0}
detection_postprocess = 
input_model = {1}
input = X
input_shape = [4, 2, 1, 192]
output_dir = {2}
'''.format(TEST_NAME, model_path, output_dir)
with open(cfg_path, 'w') as txt_file:
    txt_file.write(cfg_content)

# Run tests with parser and compare result with runtime
exit_status = generate_ir(cfg_path, verbose=True)
assert exit_status

# opt forward
opt_outputs_dict = opt_forward(os.path.join(output_dir, TEST_NAME + '.txt'),
                               os.path.join(output_dir, TEST_NAME + '.bin'),
                               feed_dict)

# tflite forward
tflite_outputs_dict = tflite_forward(model_path, feed_dict)

# compare results
same_outputs = compare_data_dict(tflite_outputs_dict, opt_outputs_dict)
assert same_outputs, 'Expect tflite forward and opt forward getting same outputs!'
