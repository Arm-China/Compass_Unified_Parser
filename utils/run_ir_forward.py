"""
//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
"""

# you should install AIPUBuilder first
from AIPUBuilder.Optimizer.tools.optimizer_forward import OptForward


def run_ir_forward(txt_path, bin_path, input_data, forward_type='float'):
    # init a forward instance
    opt_forward = OptForward(txt_path, bin_path, forward_type)

    # input_data should be a dict and data order should be same with input tensors
    try:
        outputs = opt_forward.forward(input_data)
    except Exception as e:
        print(e)
        raise RuntimeError("Error happened during IR forward...")
    return outputs
