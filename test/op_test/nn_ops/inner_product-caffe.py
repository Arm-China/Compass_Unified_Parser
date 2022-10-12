import sys
import subprocess
import numpy as np
from AIPUBuilder.Parser.tool_utils.run import run_parser
from AIPUBuilder.Parser.tool_utils.common import get_feed_dict


def create_inner_product_model(model_path, prototxt_path):
    """ Create caffe model.
    """
    prototxt_content = """
name: "test"
input: "data"
input_shape { dim: 10  dim: 5  dim: 3  dim: 4}

layer {
    name: "ip1"
    type: "InnerProduct"
    bottom: "data"
    top: "ip1"
    inner_product_param {
        num_output: 64
        weight_filler {
        type: "gaussian"
        std: 0.1
        }
        bias_filler {
        type: "constant"
        }
    }
}"""
    with open(prototxt_path, "w") as txt_file:
        txt_file.write(prototxt_content)
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_path, caffe.TEST)
    net.save(model_path)
    return model_path

# input_args_info = '1) cfg path; 2) caffe model path; 3) prototxt path; 4) input numpy file'
# assert len(sys.argv) > 3, 'Expect at least 4 inputs:\n %s' % input_args_info

# cfg_path = sys.argv[1]
# model_path = sys.argv[2]
# prototxt_path = sys.argv[3]
# input_data_file = sys.argv[4]


# feed_dict = get_feed_dict(input_data_file)
TEST_NAME = 'inner_product'
exit_status = True
try:
    # conda activate caffe
    import caffe
    model_path = TEST_NAME + '.caffemodel'
    prototxt_path = TEST_NAME + '.prototxt'
    create_inner_product_model(model_path, prototxt_path)
    feed_dict = {'data': np.random.ranf([10, 5, 3, 4]).astype(np.float32)}
    # conda deactivate
    # Run tests with parser and compare result with runtime
    exit_status = run_parser(model_path, feed_dict, proto_path=prototxt_path, save_output=True, verify=True)
except Exception as e:
    print('Fail to run tests because %s' % str(e))
assert exit_status
