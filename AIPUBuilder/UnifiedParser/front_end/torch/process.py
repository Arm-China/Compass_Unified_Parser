# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import os
import onnx
import torch
import torch.onnx.symbolic_helper as helper
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL
from ...common.utils import get_version


def convert_torch_to_onnx(model_path, params):
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
        if onnx_opset_version >= helper._onnx_main_opset:
            onnx_opset_version = helper._onnx_main_opset
        elif onnx_opset_version not in helper._onnx_stable_opsets:
            onnx_opset_version = None

    # Get input_tensors and input_names
    input_names = []
    input_tensors = ()
    for input_name, input_shape in params['input_shapes'].items():
        input_names.append(input_name)
        # FIXME: dtype is set to float32 but user may want other dtype for inputs
        tensor = torch.randn(input_shape, dtype=torch.float32)
        input_tensors += (tensor, )

    # Get the file name of the onnx model to be exported
    onnx_model_path = os.path.join(params.get('output_dir', './'),
                                   os.path.basename(model_path) + '.onnx')
    INFO('[Parser]: Convert TorchScript (%s) to onnx model...' % model_path)

    # Call torch.onnx.export to convert TorchScript model to onnx model
    try:
        # Note: Use operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        # or torch.onnx.OperatorExportTypes.ONNX_ATEN for debug if export fails.
        # The failure could be caused by unexpected input shapes.
        torch.onnx.export(model,
                          input_tensors,
                          onnx_model_path,
                          input_names=input_names,
                          opset_version=onnx_opset_version)
    except Exception as e:
        FATAL('[Parser]: Fail to convert model (%s) to onnx because %s' % (model_path, str(e)))
    INFO('[Parser]: Torch model has been converted to onnx model (%s) with opset version %d!' %
         (onnx_model_path, helper._export_onnx_opset_version))

    # Update params
    updated_params = copy.deepcopy(params)
    updated_params.update({'input_model': onnx_model_path,
                           'input_names': [],
                           'input_shapes': {},
                           'output_names': [],
                           'model_type': 'torch'})

    # Return onnx model path and updated params
    return onnx_model_path, updated_params
