# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class DynamicUpdateSliceOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsDynamicUpdateSliceOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DynamicUpdateSliceOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def DynamicUpdateSliceOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # DynamicUpdateSliceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def DynamicUpdateSliceOptionsStart(builder): builder.StartObject(0)
def DynamicUpdateSliceOptionsEnd(builder): return builder.EndObject()
