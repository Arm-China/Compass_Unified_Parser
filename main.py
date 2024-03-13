# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


# cython: language_level=3
# !/usr/bin/python
# -*- coding: UTF-8 -*-


from AIPUBuilder.Parser.univ_main import main
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))


if __name__ == '__main__':
    sys.exit(main())
