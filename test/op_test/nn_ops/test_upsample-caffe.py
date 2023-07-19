import sys
import subprocess
import numpy as np
from utils.run import run_parser


def create_upsamplebyindex_model(model_path, prototxt_path):
    """ Create caffe model.
    """
    prototxt_content = """
name: "test"
input: "data"
input_shape { dim: 1  dim: 16  dim: 36  dim: 24}
input: "indices"
input_shape { dim: 1  dim: 16  dim: 36  dim: 24}

layer {
    name: "upsamplebyindex1"
    type: "Upsample"
    bottom: "data"
    bottom: "indices"
    top: "upsamplebyindex1"
    upsample_param {
        scale: 3
    }
}"""
    with open(prototxt_path, "w") as txt_file:
        txt_file.write(prototxt_content)
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_path, caffe.TEST)
    net.save(model_path)
    return model_path


TEST_NAME = 'upsamplebyindex'
try:
    import caffe
    model_path = TEST_NAME + '.caffemodel'
    prototxt_path = TEST_NAME + '.prototxt'
    create_upsamplebyindex_model(model_path, prototxt_path)
    feed_dict = {'data': np.random.ranf([1, 16, 36, 24]).astype(np.float32),
                 'indices': np.tile(np.arange(24), [1, 16, 36, 1]).astype(np.int32)}
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, proto_path=prototxt_path, save_output=True, verify=True)
    assert exit_status
except Exception as e:
    print('Fail to run tests because %s' % str(e))
