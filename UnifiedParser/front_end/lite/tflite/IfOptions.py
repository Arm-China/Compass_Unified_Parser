# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class IfOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsIfOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = IfOptions()
        x.Init(buf, n + offset)
        return x

    # IfOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # IfOptions
    def ThenSubgraphIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # IfOptions
    def ElseSubgraphIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def IfOptionsStart(builder): builder.StartObject(2)


def IfOptionsAddThenSubgraphIndex(
    builder, thenSubgraphIndex): builder.PrependInt32Slot(0, thenSubgraphIndex, 0)
def IfOptionsAddElseSubgraphIndex(
    builder, elseSubgraphIndex): builder.PrependInt32Slot(1, elseSubgraphIndex, 0)


def IfOptionsEnd(builder): return builder.EndObject()
