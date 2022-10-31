#!/bin/sh

cd `dirname $0`; pwd
wget -O alexnet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel

wget -O alexnet.prototxt https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt
