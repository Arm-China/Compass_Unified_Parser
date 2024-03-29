# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class CastOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsCastOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CastOptions()
        x.Init(buf, n + offset)
        return x

    # CastOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CastOptions
    def InDataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # CastOptions
    def OutDataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def CastOptionsStart(builder): builder.StartObject(2)
def CastOptionsAddInDataType(builder, inDataType): builder.PrependInt8Slot(0, inDataType, 0)
def CastOptionsAddOutDataType(builder, outDataType): builder.PrependInt8Slot(1, outDataType, 0)
def CastOptionsEnd(builder): return builder.EndObject()
