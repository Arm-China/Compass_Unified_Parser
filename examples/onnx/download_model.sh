#!/bin/sh

cd `dirname $0`; pwd
wget -O mobilenet_v2.onnx https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-10.onnx?raw=true
