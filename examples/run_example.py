# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import configparser
import sys
import os
import re
import argparse
import numpy as np

sys.path.append("..")  # noqa: E402
from utils.compare import compare_data_dict
from utils.run_ir_forward import run_ir_forward
from utils.forward import rt_forward
from utils.run import generate_ir


def list_string_to_list(list_string):
    ret = []
    if list_string:
        items = re.findall('(\\[.*?\\])', list_string)
        for meta_item in items:
            inner_str = meta_item.lstrip('[').rstrip(']')
            if inner_str:
                meta_list = [int(m) for m in inner_str.split(',')]
            else:
                meta_list = list()
            ret.append(meta_list)
    return ret


def run_example(framework, input_path):
    current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), framework)

    if not os.path.exists(current_dir):
        raise FileNotFoundError(f"Cannot find {framework} example")

    example_cfg = os.path.join(current_dir, "example.cfg")

    if not os.path.exists(example_cfg):
        raise FileNotFoundError("Cannot find cfg file")

    # checking cfg and convert relative path to abs path
    config = configparser.ConfigParser()
    try:
        config.read(example_cfg)
    except configparser.MissingSectionHeaderError as e:
        print('Config file error: %s' % (str(e)))
        raise RuntimeError("Read Config Error")

    if 'Common' in config:
        common = config['Common']
    else:
        raise RuntimeError(f"Config Format Error")

    model_type = common['model_type']
    if model_type.upper() not in ('ONNX', 'TFLITE', 'CAFFE', 'TENSORFLOW', 'TF'):
        raise RuntimeError(f"Unsupported model type: {model_type}")

    input_tensor_names = common['input'].split(',')
    input_tensor_shapes = list_string_to_list(common['input_shape'])
    output_tensor_names = common['output'].split(',')

    model_path = common['input_model']
    model_name = common['model_name']
    output_ir_dir = common['output_dir']

    if not os.path.isabs(model_path) or not os.path.isabs(output_ir_dir):
        config['Common']['input_model'] = model_path = os.path.realpath(os.path.join(current_dir, model_path))
        config['Common']['output_dir'] = output_ir_dir = os.path.realpath(os.path.join(current_dir, output_ir_dir))
        cfg_path = os.path.join(current_dir, "example_new.cfg")
        with open(cfg_path, "w") as f:
            config.write(f)
    else:
        cfg_path = example_cfg

    ret = generate_ir(cfg_path)

    if not ret:
        raise RuntimeError("Error happened during Parser")

    ir_txt = os.path.join(output_ir_dir, model_name + '.txt')
    ir_bin = os.path.join(output_ir_dir, model_name + '.bin')

    if input_path == '':
        inputs_num = len(input_tensor_names)
        feed_dict = {}
        for i in range(inputs_num):
            feed_dict[input_tensor_names[i]] = np.random.uniform(low=-10., high=50.,
                                                                 size=tuple(input_tensor_shapes[i])).astype(np.float32)
    else:
        # you should provide a npy and it's a dict. key: input_tensor_name  value: ndarray data
        feed_dict = np.load(input_path, allow_pickle=True).item()

    gt_outputs = rt_forward(model_path, feed_dict.copy(), output_tensor_names, save_output=False)

    ir_outputs = run_ir_forward(ir_txt, ir_bin, feed_dict.copy())

    ir_output_dict = {}
    for name, value in zip(output_tensor_names, ir_outputs):
        ir_output_dict[name] = value

    ret &= compare_data_dict(gt_outputs, ir_output_dict)

    if ret:
        print("Float IR generated and the acc is ok...")
    else:
        print("Float IR generated but have similarity issue...")


if __name__ == '__main__':
    cfg = argparse.ArgumentParser(description='example script args')
    cfg.add_argument('--framework', type=str, default="tensorflow")
    cfg.add_argument('--input_data', type=str, default="")
    args = cfg.parse_args()

    framework = args.framework
    input_data_path = args.input_data

    run_example(framework, input_data_path)
