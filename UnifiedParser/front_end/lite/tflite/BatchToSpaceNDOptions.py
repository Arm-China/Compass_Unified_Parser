# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class BatchToSpaceNDOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsBatchToSpaceNDOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BatchToSpaceNDOptions()
        x.Init(buf, n + offset)
        return x

    # BatchToSpaceNDOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def BatchToSpaceNDOptionsStart(builder): builder.StartObject(0)
def BatchToSpaceNDOptionsEnd(builder): return builder.EndObject()
