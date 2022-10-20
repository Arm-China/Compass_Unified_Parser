# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import subprocess

from UnifiedParser.logger import INFO, WARN, DEBUG
from .common import get_model_type, check_float_ir
from .compare import compare_data_dict
from .forward import opt_forward, rt_forward
from .model import read_model


def generate_ir(cfg_path, verbose=False):
    '''Call Parser to generate float IR'''
    entry_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main.py')
    DEBUG('Trigger script: %s' % entry_path)
    run_process = subprocess.Popen(['python3', entry_path, '-c', cfg_path] + (['-v'] if verbose else []),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
    log_file = cfg_path + '.log'
    tee = subprocess.Popen(['tee', log_file], stdin=run_process.stdout)
    run_process.stdout.close()
    tee.communicate()
    # TODO: Currently exit code of parser is not accurate
    with open(log_file, 'r') as f:
        run_pass = 'Parser done' in f.read()
    return run_pass


def run_parser(model_path, feed_dict, output_names=None, model_type=None, save_output=True,
               proto_path=None, verify=True, expected_keywords=[], unexpected_keywords=[]):
    ''' Generate config file for parser and call parser to run tests. Return True if
    test is successfully run, otherwise return False.
    If verify is set, using opt_forward to get parser's output and compare the output
    with the original model's runtime output. Return True if the outputs reach the
    requirements of similarity and mean, otherwise return False.
    '''
    if model_type is None:
        model_type = get_model_type(model_path)

    # Generate config file for parser
    _, cfg_path = read_model(model_path, save_cfg=True,
                             model_type=model_type, proto_path=proto_path)
    INFO('Config file %s is generated' % cfg_path)

    # Run parser to get float IR
    run_pass = generate_ir(cfg_path)

    if not run_pass:
        WARN('Fail in running parser!')
        return run_pass

    output_dir = 'output_dir'
    model_name = os.path.basename(model_path).rsplit('.', 1)[0]
    float_IR_txt = '/'.join([output_dir, model_name + '.txt'])
    float_IR_bin = '/'.join([output_dir, model_name + '.bin'])

    if expected_keywords or unexpected_keywords:
        run_pass = check_float_ir(
            float_IR_txt, expected_keywords, unexpected_keywords)

    if not verify:
        return run_pass

    # Get model output using runtime of original framework(tf, onnx, caffe and etc)
    rt_output_dict = rt_forward(model_path, feed_dict, output_names=output_names,
                                save_output=save_output, proto_path=proto_path)

    # Get model output using opt forward
    opt_output_dict = opt_forward(
        float_IR_txt, float_IR_bin, feed_dict, save_output=save_output)

    # Compare outputs
    INFO('first runtime, second opt:')
    run_pass &= compare_data_dict(rt_output_dict, opt_output_dict)

    # Report result
    if run_pass:
        INFO('Test %s is passed!' % cfg_path)
        run_pass = True
    else:
        INFO('Test %s is failed!' % cfg_path)
        run_pass = False
    return run_pass
