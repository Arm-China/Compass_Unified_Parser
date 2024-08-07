# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from job_mamanger import JobManager
import os
import sys
import time
import configparser

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "/project/ai/zhouyi_compass/Model"
fast_test = os.path.abspath(os.path.join(WORKING_DIR, "fast_parser_test.py"))
jm = JobManager()
print("Parser sanity test working in %s" % WORKING_DIR)
TF_BENCHMARK = {
    '1_13': [
    ],
    '1_15': [
        'deeplab_v3',
        'gru_l',
        'inception_v3',
        'inception_v4',
        'maskrcnn',
        'mobilenet_v2',
        'mobilenet_v2_ssd',
        'resnet_v1_50',
        'resnet_v2_50',
        'shufflenet_v2',
        'transformer_mini',
        # 'transformer_official',
        'wavenet',
        'yolo_v2_416',
    ],
    '2_6': [
        'efficientnet_b5',  # SavedModel
        'inception_v3',     # keras
        'mobilenet_v2',     # hdf5
        'resnet_v2_50',     # h5
    ]
}

TFLITE_BENCHMARK = {
    '1_13': [
    ],
    '1_15': [
        'mobilenet_v1_ssd'
    ],
}

ONNX_BENCHMARK = {
    '1_6': [
        # 'swin_transformer',  will cost ~40+ mins
        'unet_3d',
        'vision_transformer',
        'yolo_v3_tiny',
        'yolo_v4',
        'yolo_v5',
    ]
}

CAFFE_BENCHMARK = {
    '1_0': [
        'faster_rcnn',
        'inception_v3',
        'inception_v4',
        'mobilenet_v2',
        'mobilenet_v2_ssd',
        'mtcnn_o',
        'mtcnn_p',
        'mtcnn_r',
        'peleenet',
        'resnet_v1_50',
        'shufflenet_v2',
        'yolo_v2_416',
        'yolo_v3',
    ]
}

TORCH_BENCHMARK = {
    '1_12': [
        'alexnet',
        'deeplab_v3',
        'efficientnet_b5',
        'fcn',
        'resnet_v1_50',
        'shufflenet_v2',
        'swin_transformer_tiny_224',
        'vgg_16',
        'ViT_B_16',
    ]
}

AIB_MODELS = {
    '1_15': [
        'mobilebert_quant',
        'lstm_quant',
        'crnn_quant',
        'deeplab_v3_plus_quant',
        'dped_instance_quant',
        'dped_quant',
        'efficientnet_b4_quant',
        'esrgan_quant',
        'imdn_quant',
        'inception_v3_quant',
        'mobilenet_v2_b8_quant',
        'mobilenet_v2_quant',
        'mobilenet_v3_b4_quant',
        'mobilenet_v3_quant',
        'mv3_depth_quant',
        'punet_quant',
        'pynet_quant',
        'srgan_quant',
        'unet_quant',
        'vgg_quant',
        'vsr_quant',
        'xlsr_quant',
        'yolo_v4_tiny_quant'
    ]
}

SANITY_MODELS = []

for k, models in TF_BENCHMARK.items():
    for m in models:
        SANITY_MODELS.append(f"tf-{k}-{m}")

for k, models in TFLITE_BENCHMARK.items():
    for m in models:
        SANITY_MODELS.append(f"tflite-{k}-{m}")

for k, models in ONNX_BENCHMARK.items():
    for m in models:
        SANITY_MODELS.append(f"onnx-{k}-{m}")

for k, models in CAFFE_BENCHMARK.items():
    for m in models:
        SANITY_MODELS.append(f"caffe-{k}-{m}")

for k, models in TORCH_BENCHMARK.items():
    for m in models:
        SANITY_MODELS.append(f"torch-{k}-{m}")

for k, models in AIB_MODELS.items():
    for m in models:
        SANITY_MODELS.append(f"tflite-{k}-{m}")


def parser_auto_fast_test(name):
    framework = name.split('-')[0]
    ver = name.split('-')[1]
    model_name = name.split('-')[-1]
    dir_path = os.path.join(WORKING_DIR, "models", model_name, framework, ver)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    # use shared cfg in Model
    if model_name in AIB_MODELS['1_15']:  # AIB model
        cfg = os.path.join(MODEL_DIR, "AI_Benchmark", model_name, framework, ver,
                           "config", f"{model_name}_parser.cfg")
    else:
        cfg = os.path.join(MODEL_DIR, model_name, framework, ver,
                           "config", f"{model_name}_parser.cfg")
    if not os.path.exists(cfg):
        print(f"Cannot find {cfg}")
        cfg = os.path.join(dir_path, model_name + ".cfg")
    config = configparser.ConfigParser()
    config.read(cfg)
    config["Common"]["model_name"] = model_name
    config["Common"]["output_dir"] = dir_path
    IR_dir = os.path.abspath(os.path.join(
        config["Common"]["input_model"], "../../IR/release"))
    ref_ir = os.path.join(IR_dir, model_name + "_parser.txt")
    bk_ref_ir = os.path.join(dir_path, model_name + "_ref.def2")
    ref_w = os.path.join(IR_dir, model_name + "_parser.bin")
    log = os.path.join(dir_path, f"{model_name}_parser.log")
    cfg = os.path.join(dir_path, model_name + ".cfg")
    with open(cfg, "w") as f:
        config.write(f)
    fmt = "python3 %s %s %s %s %s %s" % (
        fast_test, ref_ir, ref_w, cfg, log, bk_ref_ir)
    jm.add_job(name, fmt)


def main():
    t0 = time.time()
    for name in SANITY_MODELS:
        parser_auto_fast_test(name)
    try:
        jm.mt.start()
        jm.mt.join()
    except (KeyboardInterrupt, Exception) as e:
        jm.__stop__ = True
        raise e
    print("-" * 30 + "testes result" + "-" * 30)
    print("idx\t%18s\t%s\t%s(s)" % ("name", "status", "time"))
    print("-" * 73)
    for i, job in enumerate(jm.finished):
        tag = "failed"
        if job not in jm.failed:
            tag = "passed"
        print("%3d\t%18s\t%s\t%04.5fs" %
              (i, job["name"], tag, job["cost_time"]))
    if jm.failed:
        print("-" * 30 + "failed summary" + "-" * 30)
        for i, job in enumerate(jm.failed):
            tag = "failed"
            print("%3d\t%18s\t%s\t%04.5fs" %
                  (i, job["name"], tag, job["cost_time"]))
    print("-" * 73)
    passed = jm.pass_num
    failed = jm.fail_num
    total = passed + failed
    pass_rate = passed / total * 100.
    print("All testes cost %f s. Total testes:%d, passed:%d, failed: %d, pass rate:%.2f%%" % (time.time() - t0,
                                                                                              total, passed, failed,
                                                                                              pass_rate))
    if failed != 0:
        print("----------testes Failed!!!!-------------")
        sys.exit(-1)
    else:
        print("----------testes passed-------------")
        sys.exit(0)


if __name__ == "__main__":
    main()
