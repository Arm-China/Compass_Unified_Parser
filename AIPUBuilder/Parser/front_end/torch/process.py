# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import os
import onnx
import torch
import torch.onnx.symbolic_helper as helper
from multiprocessing import Process
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL
from ...common.utils import get_version


def convert_torch_to_onnx(model_path, params):
    def _export_to_onnx(model,
                        input_tensors,
                        onnx_model_path,
                        input_names,
                        opset_version=None):
        # Note: Use operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        # or torch.onnx.OperatorExportTypes.ONNX_ATEN for debug if export fails.
        # The failure could be caused by unexpected input shapes.
        torch.onnx.export(model,
                          input_tensors,
                          onnx_model_path,
                          input_names=input_names,
                          opset_version=onnx_opset_version)
        return

    # Check whether inputs and shapes are provided. They must be provided because we cannot get input
    # shapes info from the provided model.
    if not params['input_shapes']:
        FATAL('[Parser]: Input names and shapes must be provided in config file for TorchScript model!')

    # Load TorchScript model
    try:
        model = torch.jit.load(model_path)
    except Exception as e:
        FATAL('[Parser]: Fail to load model (%s) because %s! Only TorchScript format is supported.' %
              (model_path, str(e)))

    # Get onnx opset version to target
    # From https://onnxruntime.ai/docs/reference/compatibility.html,
    # for onnx version 1.x, onnx opset version=x+5
    onnx_version = str(get_version(onnx)).split('.')
    onnx_opset_version = (int(onnx_version[-1]) + 5) if int(onnx_version[0]) == 1 else None
    if onnx_opset_version is not None:
        default_onnx_main_opset = None
        default_onnx_stable_opsets = []
        try:
            torch_version = str(torch.onnx.producer_version)
            if torch_version.startswith('1.11'):
                default_onnx_main_opset = helper._onnx_main_opset
                default_onnx_stable_opsets = helper._onnx_stable_opsets
            elif torch_version >= '1.12.0':
                import torch.onnx._constants as Constant
                default_onnx_main_opset = Constant.onnx_main_opset
                default_onnx_stable_opsets = Constant.onnx_stable_opsets
        except Exception as e:
            DEBUG('[Parser]: Fail to get default onnx opset version because %s' % str(e))
        if default_onnx_main_opset is None:
            onnx_opset_version = None
        elif onnx_opset_version >= default_onnx_main_opset or onnx_opset_version not in default_onnx_stable_opsets:
            onnx_opset_version = default_onnx_main_opset
    DEBUG('[Parser]: Will convert to onnx opset version (%s)!' % str(onnx_opset_version))

    # Get input_tensors and input_names
    input_names = []
    input_tensors = ()
    input_info_dict = copy.deepcopy(params['input_shapes'])
    for input_name, input_shape in input_info_dict.items():
        if len(input_name) >= 1 and input_name[0].isdigit():  # Starting with numbers is not legal in pytorch
            new_input_name = 'input_' + input_name
            WARN('[Parser]: Input name %s is invalid; rename it to %s!' % (input_name, new_input_name))
            params['input_shapes'].pop(input_name)
            params['input_shapes'][new_input_name] = input_shape
            input_name = new_input_name
        input_names.append(input_name)
        # FIXME: dtype is set to float32 but user may want other dtype for inputs
        tensor = torch.randn(input_shape, dtype=torch.float32)
        input_tensors += (tensor, )

    # Get the file name of the onnx model to be exported
    onnx_model_path = os.path.join(params.get('output_dir', './'),
                                   os.path.basename(model_path) + '.onnx')
    INFO('[Parser]: Convert TorchScript (%s) to onnx model...' % model_path)

    # Call torch.onnx.export to convert TorchScript model to onnx model
    exit_code = 1
    try:
        process = Process(target=_export_to_onnx, args=(model,
                                                        input_tensors,
                                                        onnx_model_path,
                                                        input_names,
                                                        onnx_opset_version))
        process.start()
        process.join()
        exit_code = process.exitcode
        try:
            process.close()
        except Exception as e:
            DEBUG('[Parser]: Fail to close process because %s' % str(e))
    except Exception as e:
        FATAL('[Parser]: Fail to convert model (%s) to onnx because %s' % (model_path, str(e)))

    if exit_code != 0:
        FATAL('[Parser]: Fail to convert model (%s) to onnx! Suggest to set env var PYTORCH_JIT_LOG_LEVEL=onnx for debug!' % model_path)

    INFO('[Parser]: Torch model has been converted to onnx model (%s) with opset version (%d)!' %
         (onnx_model_path, 'default' if onnx_opset_version is None else onnx_opset_version))

    # Update params
    updated_params = copy.deepcopy(params)
    updated_params.update({'input_model': onnx_model_path,
                           'input_names': [],
                           'input_shapes': {},
                           'output_names': [],
                           'model_type': 'torch'})

    # Return onnx model path and updated params
    return onnx_model_path, updated_params
