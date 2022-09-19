# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class GatherOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsGatherOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GatherOptions()
        x.Init(buf, n + offset)
        return x

    # GatherOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # GatherOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # GatherOptions
    def BatchDims(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def GatherOptionsStart(builder): builder.StartObject(2)
def GatherOptionsAddAxis(builder, axis): builder.PrependInt32Slot(0, axis, 0)
def GatherOptionsAddBatchDims(
    builder, batchDims): builder.PrependInt32Slot(1, batchDims, 0)


def GatherOptionsEnd(builder): return builder.EndObject()
