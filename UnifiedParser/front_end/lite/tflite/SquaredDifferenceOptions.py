# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class SquaredDifferenceOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSquaredDifferenceOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SquaredDifferenceOptions()
        x.Init(buf, n + offset)
        return x

    # SquaredDifferenceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def SquaredDifferenceOptionsStart(builder): builder.StartObject(0)
def SquaredDifferenceOptionsEnd(builder): return builder.EndObject()
