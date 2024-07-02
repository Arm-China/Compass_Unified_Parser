# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()


class Rfft2dOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsRfft2dOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Rfft2dOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def Rfft2dOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Rfft2dOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def Rfft2dOptionsStart(builder): builder.StartObject(0)
def Rfft2dOptionsEnd(builder): return builder.EndObject()
