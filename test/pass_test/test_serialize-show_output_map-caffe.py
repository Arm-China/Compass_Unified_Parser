# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.

import sys
import subprocess
import numpy as np
import caffe
from utils.run import run_parser


def create_pooling_model(model_path, prototxt_path):
    ''' Create caffe model.
    '''
    prototxt_content = '''
name: "test"
input: "data"
input_shape { dim: 1  dim: 56  dim: 56  dim: 24}

layer {
    name: "pooling_node"
    type: "Pooling"
    bottom: "data"
    top: "pooling_out"
    pooling_param {
        pool: AVE
        kernel_size: 3
        stride: 2
    }
}
layer {
    name: 'ip_node'
    type: 'InnerProduct'
    bottom: 'pooling_out'
    top: 'ip_out'
    inner_product_param {
        num_output: 64
        weight_filler {
        type: 'gaussian'
        std: 0.1
        }
        bias_filler {
        type: 'constant'
        }
    }
}'''
    with open(prototxt_path, 'w') as txt_file:
        txt_file.write(prototxt_content)
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_path, caffe.TEST)
    net.save(model_path)
    return model_path


TEST_NAME = 'pooling'

model_path = TEST_NAME + '.caffemodel'
prototxt_path = TEST_NAME + '.prototxt'
create_pooling_model(model_path, prototxt_path)
feed_dict = {'data': np.random.ranf([1, 56, 56, 24]).astype(np.float32)}

for output_names in (['pooling_node', 'ip_node'], ['pooling_out', 'ip_out']):
    exit_status = run_parser(model_path, feed_dict, proto_path=prototxt_path,
                             output_names=output_names, verify=False, expected_logs=[
                                 'Output ' + output_names[0] +
                                 ' from cfg is shown as tensor pooling_node_post_transpose_0 in IR',
                                 'Output ' + output_names[1] + ' from cfg is shown as tensor ip_node_0 in IR'])
    assert exit_status
