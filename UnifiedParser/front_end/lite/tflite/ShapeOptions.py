# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class ShapeOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsShapeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ShapeOptions()
        x.Init(buf, n + offset)
        return x

    # ShapeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ShapeOptions
    def OutType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def ShapeOptionsStart(builder): builder.StartObject(1)
def ShapeOptionsAddOutType(
    builder, outType): builder.PrependInt8Slot(0, outType, 0)


def ShapeOptionsEnd(builder): return builder.EndObject()
