# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class TopKV2Options(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsTopKV2Options(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TopKV2Options()
        x.Init(buf, n + offset)
        return x

    # TopKV2Options
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def TopKV2OptionsStart(builder): builder.StartObject(0)
def TopKV2OptionsEnd(builder): return builder.EndObject()
