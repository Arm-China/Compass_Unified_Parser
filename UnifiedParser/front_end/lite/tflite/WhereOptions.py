# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class WhereOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsWhereOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = WhereOptions()
        x.Init(buf, n + offset)
        return x

    # WhereOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def WhereOptionsStart(builder): builder.StartObject(0)
def WhereOptionsEnd(builder): return builder.EndObject()
