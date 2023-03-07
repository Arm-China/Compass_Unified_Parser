import sys
import subprocess
import numpy as np
from os.path import exists
from utils.run import run_parser
from utils.common import get_feed_dict


def create_lstm_model(model_path, prototxt_path, expose_hidden):
    ''' Create caffe model.
    '''
    if expose_hidden:
        prototxt_content = '''
name: 'test'
input: 'data'
input_shape { dim: 10  dim: 2  dim: 50}
input: 'initial_h'
input_shape { dim: 1  dim: 2  dim: 64}
input: 'initial_c'
input_shape { dim: 1  dim: 2  dim: 64}

layer {
    name: 'cond'
    type: 'DummyData'
    top: 'cond'
    dummy_data_param {
        shape: { dim: 10  dim: 2 }
        data_filler { type: 'constant' value: 1 }
    }
}

layer {
    name: 'lstm'
    type: 'LSTM'
    bottom: 'data'
    bottom: 'cond'
    bottom: 'initial_h'
    bottom: 'initial_c'
    top: 'lstm'
    top: 'hout'
    top: 'cout'
    recurrent_param {
        num_output: 64
        weight_filler {
            type: 'constant'
            value: 1.3
        }
        bias_filler {
            type: 'constant'
            value: 4.1
        }
        expose_hidden: true
    }
}'''
    else:
        prototxt_content = '''
name: 'test'
input: 'data'
input_shape { dim: 10  dim: 2  dim: 50}

layer {
    name: 'cond'
    type: 'DummyData'
    top: 'cond'
    dummy_data_param {
        shape: { dim: 10  dim: 2 }
        data_filler { type: 'constant' value: 1 }
    }
}

layer {
    name: 'lstm'
    type: 'LSTM'
    bottom: 'data'
    bottom: 'cond'
    top: 'lstm'
    recurrent_param {
        num_output: 64
        weight_filler {
            type: 'constant'
            value: 1.3
        }
        bias_filler {
            type: 'constant'
            value: 4.1
        }
    }
}'''
    with open(prototxt_path, 'w') as txt_file:
        txt_file.write(prototxt_content)
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_path, caffe.TEST)
    net.save(model_path)
    return model_path


TEST_NAME = 'lstm'
exit_status = True
input_data = np.random.ranf([10, 2, 50]).astype(np.float32) * 100
initial_h = np.random.ranf([1, 2, 64]).astype(np.float32) * 10
initial_c = np.random.ranf([1, 2, 64]).astype(np.float32) * 10
try:
    # conda activate caffe
    import caffe
    for expose_hidden in (True, False):
        model_name = '-'.join([TEST_NAME, str(expose_hidden)])
        model_path = model_name + '.caffemodel'
        prototxt_path = model_name + '.prototxt'
        create_lstm_model(model_path, prototxt_path, expose_hidden)
        feed_dict = {'data': input_data}
        if expose_hidden:
            feed_dict.update({'initial_h': initial_h, 'initial_c': initial_c})
        # conda deactivate
        # Run tests with parser and compare result with runtime
        exit_status = run_parser(model_path, feed_dict, proto_path=prototxt_path, verify=True)
except Exception as e:
    print('Fail to run tests because %s' % str(e))
assert exit_status
