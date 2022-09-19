# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Conv3DOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsConv3DOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Conv3DOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def Conv3DOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Conv3DOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Conv3DOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideD(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def DilationDFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Conv3DOptions
    def DilationWFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Conv3DOptions
    def DilationHFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

def Conv3DOptionsStart(builder): builder.StartObject(8)
def Conv3DOptionsAddPadding(builder, padding): builder.PrependInt8Slot(0, padding, 0)
def Conv3DOptionsAddStrideD(builder, strideD): builder.PrependInt32Slot(1, strideD, 0)
def Conv3DOptionsAddStrideW(builder, strideW): builder.PrependInt32Slot(2, strideW, 0)
def Conv3DOptionsAddStrideH(builder, strideH): builder.PrependInt32Slot(3, strideH, 0)
def Conv3DOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(4, fusedActivationFunction, 0)
def Conv3DOptionsAddDilationDFactor(builder, dilationDFactor): builder.PrependInt32Slot(5, dilationDFactor, 1)
def Conv3DOptionsAddDilationWFactor(builder, dilationWFactor): builder.PrependInt32Slot(6, dilationWFactor, 1)
def Conv3DOptionsAddDilationHFactor(builder, dilationHFactor): builder.PrependInt32Slot(7, dilationHFactor, 1)
def Conv3DOptionsEnd(builder): return builder.EndObject()
