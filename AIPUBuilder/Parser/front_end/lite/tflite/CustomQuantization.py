# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class CustomQuantization(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsCustomQuantization(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CustomQuantization()
        x.Init(buf, n + offset)
        return x

    # CustomQuantization
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CustomQuantization
    def Custom(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # CustomQuantization
    def CustomAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # CustomQuantization
    def CustomLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0


def CustomQuantizationStart(builder): builder.StartObject(1)
def CustomQuantizationAddCustom(builder, custom): builder.PrependUOffsetTRelativeSlot(
    0, flatbuffers.number_types.UOffsetTFlags.py_type(custom), 0)


def CustomQuantizationStartCustomVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def CustomQuantizationEnd(builder): return builder.EndObject()
