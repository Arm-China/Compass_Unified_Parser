# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class SegmentSumOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSegmentSumOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SegmentSumOptions()
        x.Init(buf, n + offset)
        return x

    # SegmentSumOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def SegmentSumOptionsStart(builder): builder.StartObject(0)
def SegmentSumOptionsEnd(builder): return builder.EndObject()