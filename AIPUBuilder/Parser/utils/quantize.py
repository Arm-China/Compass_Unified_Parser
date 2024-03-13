# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import subprocess

from ..logger import INFO, WARN, DEBUG


def generate_symm_quant_cfg(model_name, txt_path, bin_path):
    '''Generate cfg for qtlib'''
    cfg_file = model_name + '-opt.cfg'
    with open(cfg_file, 'w') as f:
        f.write('[Common]\n')
        f.write('graph = ' + txt_path + '\n')
        f.write('bin = ' + bin_path + '\n')
        f.write('model_name = ' + model_name + '\n')
        f.write('dataset = NumpyDataset\n')
        f.write('output_dir = ' + os.path.dirname(txt_path) + '\n')
        f.write('dump = False\n')
        f.write('weight_bits = 8\n')
        f.write('bias_bits = 32\n')
        f.write('activation_bits = 8\n')
        f.write('out_ir_name = ' + model_name + '_opt\n')
        f.write('scaling_bits = {softmax:[20,-1]}\n')
    return cfg_file


def generate_symm_quant_ir(cfg_path):
    '''Call qtlib to generate symm quant IR; return quant txt file and quant bin file'''
    entry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              'Optimizer', 'tools', 'optimizer_main.py')
    INFO('Trigger script to generate symm quant IR: %s' % entry_path)
    run_process = subprocess.Popen(['python3', entry_path, '-c', cfg_path],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
    log_file = cfg_path + '.log'
    tee = subprocess.Popen(['tee', log_file], stdin=run_process.stdout)
    run_process.stdout.close()
    tee.communicate()
    with open(log_file, 'r') as f:
        run_pass = '[Done]' in f.read()
    return run_pass
