# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class MatrixSetDiagOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsMatrixSetDiagOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MatrixSetDiagOptions()
        x.Init(buf, n + offset)
        return x

    # MatrixSetDiagOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def MatrixSetDiagOptionsStart(builder): builder.StartObject(0)
def MatrixSetDiagOptionsEnd(builder): return builder.EndObject()
