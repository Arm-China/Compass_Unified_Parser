# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


# cython: language_level=3
#!/usr/bin/python3
# -*- coding: UTF-8 -*-

__VERSION__ = '1.3'
__build_number__ = None
if __build_number__ is not None:
    __VERSION__ = __VERSION__ + '.' + str(__build_number__)

__release__ = False
