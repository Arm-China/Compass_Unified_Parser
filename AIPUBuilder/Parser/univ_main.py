# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import os
import sys
import argparse
import configparser
from .logger import *


def main():
    class _argparse_formatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    args = argparse.ArgumentParser(formatter_class=_argparse_formatter,
                                   epilog='''
required arguments in Common section of <net.cfg>:
    model_name          The name for the input model.
    input_model         File path of the input model. The following models are supported:
                            tensorflow frozen pb/h5/saved model
                            tflite model
                            caffe model
                            onnx model
                            TorchScript model
    input               The input(s) node(s)' name of the model. Use comma to separate for several inputs.
    input_shape         The input shape(s) of model. For multiple inputs, use comma the separate them.
                        Example:
                            One input: input_shape=[1,224,224,3]
                            Multiple inputs: input_shape=[1,224,224,3],[1,112,112,3]
    output              The output(s) node(s)' name of the model. Use comma to separate for several outputs.

optional arguments in Common section of <net.cfg>:
    model_type          The model format of the input model. (default: tensorflow)
                        The supported types are(case insensitive): tensorflow, tflite, onnx, caffe, torch.
    model_domain        The domain of the model. (default: image_classification)
                        Example:
                            image_classification
                            object_detection
                            keyword_spotting
                            speech_recognition
                            image_segmentation
    detection_postprocess
                        The type of post process if model_domain is object_detection and official detection
                        models are used. (default: None)
                        The following types of post process are supported(case insensitive):
                            SSD
                            SSD_RESNET
                            YOLO2
                            YOLO3_TINY
                            YOLO3_FULL
                            CAFFE_FASTERRCNN
                            FASTERRCNN (Only for torch models or onnx models converted from torch)
                            MASKRCNN (Only for torch models or onnx models converted from torch)
    caffe_prototxt      Prototxt file path of the Caffe model. Required for caffe model. (default: None)
    input_dtype         The dtype of input(s) for torch models. Use comma to separate for several dtype string.
                        The sequence of several dtype should be aligned with inputs. (default: float32)
                        Example:
                        1) One input and the dtype of the only one input is int8:
                           input_dtype=int8
                        2) Three inputs, in which the dtype of the first input is int32, the second is uint8,
                           and the third is float32:
                           input_dtype=int32,uint8,float32
    iou_threshold       Overlap threshold value between two boxes. Required if detection_postprocess is used.
    obj_threshold       Confidence threshold value of the current box. Required if detection_postprocess in
                        (case insensitive):
                            YOLO2
                            YOLO3_TINY
                            YOLO3_FULL
    max_box_num         Maximum box number of some detection layers (such as Region/DecodeBox). Required if
                        detection_postprocess is used.
    image_width         Input image width. Required if detection_postprocess is used.
    image_height        Input image height. Required if detection_postprocess is used.
    box_format          The format of box used as the input of SSD. Format 'xy' and 'yx' are supported.
                        The default format is 'yx'.
    ssd_activation      The activation applied to class input of SSD. It's ignored if activation has been
                        applied. 'softmax'(default) and 'sigmoid' is supported.
    proposal_normalized It's used to indicate whether box/anchor is normalized for SSD.
                        '1'(or 'true', default) and '0'(or 'false') are supported.
    preprocess_type     The pre-processing to be applied on the model input(s). The following types are
                        supported(case insensitive):
                        RESIZE, RGB2BGR, BGR2RGB, RGB2GRAY, and BGR2GRAY.
                        When several pre-processing types are set, they will run on the NPU sequentially.
    preprocess_params   The parameters needed in the pre-processing. The parameters of each pre-processing
                        should be set with a brace. When the pre-processing does not need parameters, an
                        empty brace {} should be used.
                        For RGB2BGR, BGR2RGB, RGB2GRAY, and BGR2GRAY, there is no need of parameters.
                        For RESIZE, it needs two parameters:
                        1) the resize method, which can be bilinear or nearest
                        2) the input image shape. For example, the setting of a bilinear RESIZE, with an
                           input image shape [1,448,448,3], should be {bilinear,[1,448,448,3]}.
    force_float_ir      Convert the quantized op in the quantized model to float op and the output IR will
                        be a float IR if set to True. (default: False)
                        For non-quantized models, this field should be ignored. For quantized model, if
                        this field is set to False or not set, the quantized op will be kept and the output
                        IR will be a quantized IR.
    similarity_input_npy
                        Show similarity and mean errors if the file path of the feeded inputs is provided.
                        The file should be a binary file in NumPy .npy format. A dictionary in format
                        {input tensor name 1: input tensor 1, ..., input tensor name N: input tensor N}
                        in which input tensor names are string and input tensors are NumPy array, should
                        be saved in the file.
                        The cosine similarity and mean errors between outputs from optimizer forwarding
                        with float IR or quantized IR and outputs from runtime of the model's framework
                        will be shown if the file path is provided.
    ''')
    args.add_argument('-c', '--cfg', metavar='<net.cfg>',
                      type=str, required=True,
                      help='graph configure file that only consists of the Common section')
    args.add_argument('-l', '--log', metavar='<net.log>',
                      type=str, required=False, default=None, help='redirect parser output to log file')
    args.add_argument('-v', '--verbose',
                      required=False, default=False, action='store_true', help='verbose output')

    options = args.parse_args(sys.argv[1:])
    logfile = options.log
    verbose = options.verbose
    init_logging(verbose, logfile)

    exit_code = 0

    if options.cfg and len(options.cfg) != 0:
        config = configparser.ConfigParser()
        try:
            config.read(options.cfg)
        except configparser.MissingSectionHeaderError as e:
            FATAL('Config file error: %s' % (str(e)))

        if 'Common' in config:
            common = config['Common']
            model_type = 'tensorflow'
            if 'model_type' in common:
                model_type = common['model_type']
                if model_type.upper() not in ('ONNX', 'TFLITE', 'CAFFE', 'TENSORFLOW', 'TF', 'TORCH', 'PYTORCH'):
                    ERROR('Unsupport model type!')
                    return -1

            # Disable AIPU backtrace so that won't show errors when using multiprocessing on some hosts
            original_aipu_disable_bt = os.environ.get('AIPU_DISABLE_BACKTRACE', None)
            os.environ['AIPU_DISABLE_BACKTRACE'] = 'True'

            model_type = model_type.lower()
            common['model_type'] = model_type

            INFO('Begin to parse %s model %s...' %
                 (model_type, common.get('model_name', '')))

            from .univ_parser import univ_parser
            param = dict(common)
            meta_ret = univ_parser(param)
            if not meta_ret:
                exit_code = -1
                ERROR('Universal parser meets error!')

            if get_error_count() > 0:
                exit_code = -1
                ERROR('Parser failed!')
            else:
                INFO('Parser done!')

            # Reset AIPU_DISABLE_BACKTRACE back to original value
            if original_aipu_disable_bt is None:
                os.environ.pop('AIPU_DISABLE_BACKTRACE')
            else:
                os.environ['AIPU_DISABLE_BACKTRACE'] = original_aipu_disable_bt
        else:
            exit_code = -1
            ERROR('Common section is required in config file.')

    return exit_code
