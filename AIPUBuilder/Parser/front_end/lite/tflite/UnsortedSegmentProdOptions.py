# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()


class UnsortedSegmentProdOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsUnsortedSegmentProdOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = UnsortedSegmentProdOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def UnsortedSegmentProdOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # UnsortedSegmentProdOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def UnsortedSegmentProdOptionsStart(builder): builder.StartObject(0)
def UnsortedSegmentProdOptionsEnd(builder): return builder.EndObject()
