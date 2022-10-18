import sys
import subprocess
import numpy as np
from utils.run import run_parser


def create_pooling_model(model_path, prototxt_path):
    """ Create caffe model.
    """
    prototxt_content = """
name: "test"
input: "data"
input_shape { dim: 1  dim: 56  dim: 56  dim: 24}

layer {
    name: "pooling1"
    type: "Pooling"
    bottom: "data"
    top: "pooling1"
    pooling_param {
        pool: AVE
        kernel_size: 3
        stride: 2
    }
}"""
    with open(prototxt_path, "w") as txt_file:
        txt_file.write(prototxt_content)
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_path, caffe.TEST)
    net.save(model_path)
    return model_path


TEST_NAME = 'pooling'
try:
    import caffe
    model_path = TEST_NAME + '.caffemodel'
    prototxt_path = TEST_NAME + '.prototxt'
    create_pooling_model(model_path, prototxt_path)
    feed_dict = {'data': np.random.ranf([1, 56, 56, 24]).astype(np.float32)}
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, proto_path=prototxt_path, save_output=True, verify=True)
    assert exit_status
except Exception as e:
    print('Fail to run tests because %s' % str(e))
