# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from ..op import *
from ..tf_ops.math_ops import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class TfabsOp(TfAbsOp, Tf2Op):
    pass


class TfacosOp(TfAcosOp, Tf2Op):
    pass


class TfacoshOp(TfAcoshOp, Tf2Op):
    pass


class TfaddOp(TfAddOp, Tf2Op):
    pass


class Tfadd_nOp(TfAddNOp, Tf2Op):
    pass


class TfargmaxOp(OpHasAxis, OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(TfargmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 3, 'TfargmaxOp expects 3 inputs, but got %d.' % len(inputs)
        self.axis = inputs[1].item(0)
        output_type = inputs[2].item(0)
        out_tensor = tf.argmax(inputs[0],
                               axis=self.axis,
                               output_type=tf.int32
                               if output_type == 'int32'
                               else tf.int64).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ArgMax', 'version': 13}


class TfargminOp(OpHasAxis, OpHasOneOutPort, Tf2Op):
    def infer_shape(self):
        super(TfargminOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 3, 'TfargminOp expects 3 inputs, but got %d.' % len(inputs)
        self.axis = inputs[1].item(0)
        output_type = inputs[2].item(0)
        out_tensor = tf.argmin(inputs[0],
                               axis=self.axis,
                               output_type=tf.int32
                               if output_type == 'int32'
                               else tf.int64).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ArgMin', 'version': 13}


class TfasinOp(TfAsinOp, Tf2Op):
    pass


class TfasinhOp(TfAsinhOp, Tf2Op):
    pass


class TfatanOp(TfAtanOp, Tf2Op):
    pass


class TfatanhOp(TfAtanhOp, Tf2Op):
    pass


class TfceilOp(TfCeilOp, Tf2Op):
    pass


class TfcosOp(TfCosOp, Tf2Op):
    pass


class TfcoshOp(TfCoshOp, Tf2Op):
    pass


class TfcumprodOp(OpHasOneOutPort, OpHasAxis, Tf2Op):
    def infer_shape(self):
        super(TfcumprodOp, self).infer_shape()
        inputs = self.get_input_tensors()
        for idx in range(1, len(inputs)):
            assert inputs[idx].size == 1, 'Expect inputs at index %d of TfcumprodOp (%s) to be scalar, but got size %d' % (
                idx, self.name, inputs[idx].size)
        self.axis = inputs[1].item()
        self.exclusive, self.reverse = [int(inp.item()) for inp in inputs[2:4]]
        out_tensor = tf.math.cumprod(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'CumProd', 'version': 1}


class TfcumsumOp(OpHasOneOutPort, OpHasAxis, Tf2Op):
    def infer_shape(self):
        super(TfcumsumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        for idx in range(1, len(inputs)):
            assert inputs[idx].size == 1, 'Expect inputs at index %d of TfcumsumOp (%s) to be scalar, but got size %d' % (
                idx, self.name, inputs[idx].size)
        self.axis = inputs[1].item()
        self.exclusive, self.reverse = [int(inp.item()) for inp in inputs[2:4]]
        out_tensor = tf.math.cumsum(*inputs).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'CumSum', 'version': 14}


class TfequalOp(TfEqualOp, Tf2Op):
    pass


class TferfOp(TfErfOp, Tf2Op):
    pass


class TfexpOp(TfExpOp, Tf2Op):
    pass


class TffloorOp(TfFloorOp, Tf2Op):
    pass


class TffloormodOp(TfFloorModOp, Tf2Op):
    pass


class TfgreaterOp(TfGreaterOp, Tf2Op):
    pass


class Tfgreater_equalOp(TfGreaterEqualOp, Tf2Op):
    pass


class TflessOp(TfLessOp, Tf2Op):
    pass


class Tfless_equalOp(TfLessEqualOp, Tf2Op):
    pass


class TflogOp(TfLogOp, Tf2Op):
    pass


class Tflogical_andOp(TfLogicalAndOp, Tf2Op):
    pass


class Tflogical_notOp(TfLogicalNotOp, Tf2Op):
    pass


class Tflogical_orOp(TfLogicalOrOp, Tf2Op):
    pass


class TfmaximumOp(TfMaximumOp, Tf2Op):
    pass


class TfminimumOp(TfMinimumOp, Tf2Op):
    pass


class TfmultiplyOp(TfMulOp, Tf2Op):
    pass


class TfnegativeOp(TfNegOp, Tf2Op):
    pass


class TfpowOp(TfPowOp, Tf2Op):
    pass


class TfreciprocalOp(TfReciprocalOp, Tf2Op):
    pass


class TfroundOp(TfRoundOp, Tf2Op):
    pass


class Tfsegment_sumOp(TfSegmentSumOp, Tf2Op):
    pass


class TfsignOp(TfSignOp, Tf2Op):
    pass


class TfsigmoidOp(TfSigmoidOp, Tf2Op):
    pass


class TfsinOp(TfSinOp, Tf2Op):
    pass


class TfsinhOp(TfSinhOp, Tf2Op):
    pass


class TfsoftplusOp(TfSoftplusOp, Tf2Op):
    pass


class TfsqrtOp(TfSqrtOp, Tf2Op):
    pass


class TfsubtractOp(TfSubOp, Tf2Op):
    pass


class TftanOp(TfTanOp, Tf2Op):
    pass


class TftanhOp(TfTanhOp, Tf2Op):
    pass
