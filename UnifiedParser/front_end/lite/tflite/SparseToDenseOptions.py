# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class SparseToDenseOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSparseToDenseOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SparseToDenseOptions()
        x.Init(buf, n + offset)
        return x

    # SparseToDenseOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SparseToDenseOptions
    def ValidateIndices(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False


def SparseToDenseOptionsStart(builder): builder.StartObject(1)
def SparseToDenseOptionsAddValidateIndices(
    builder, validateIndices): builder.PrependBoolSlot(0, validateIndices, 0)


def SparseToDenseOptionsEnd(builder): return builder.EndObject()
