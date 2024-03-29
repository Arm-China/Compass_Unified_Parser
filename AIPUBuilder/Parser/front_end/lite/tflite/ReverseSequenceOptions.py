# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class ReverseSequenceOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsReverseSequenceOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReverseSequenceOptions()
        x.Init(buf, n + offset)
        return x

    # ReverseSequenceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReverseSequenceOptions
    def SeqDim(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ReverseSequenceOptions
    def BatchDim(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def ReverseSequenceOptionsStart(builder): builder.StartObject(2)
def ReverseSequenceOptionsAddSeqDim(builder, seqDim): builder.PrependInt32Slot(0, seqDim, 0)
def ReverseSequenceOptionsAddBatchDim(builder, batchDim): builder.PrependInt32Slot(1, batchDim, 0)
def ReverseSequenceOptionsEnd(builder): return builder.EndObject()
