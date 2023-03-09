# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ..tf_ops.image_ops import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class Tfnon_max_suppressionOp(TfNonMaxSuppressionV3Op, Tf2Op):
    pass
