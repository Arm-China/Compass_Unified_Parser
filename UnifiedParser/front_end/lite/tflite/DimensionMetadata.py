# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class DimensionMetadata(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsDimensionMetadata(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DimensionMetadata()
        x.Init(buf, n + offset)
        return x

    # DimensionMetadata
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DimensionMetadata
    def Format(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def DenseSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def ArraySegmentsType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def ArraySegments(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(
            self._tab.Offset(10))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    # DimensionMetadata
    def ArrayIndicesType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(
            self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def ArrayIndices(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(
            self._tab.Offset(14))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None


def DimensionMetadataStart(builder): builder.StartObject(6)


def DimensionMetadataAddFormat(
    builder, format): builder.PrependInt8Slot(0, format, 0)


def DimensionMetadataAddDenseSize(
    builder, denseSize): builder.PrependInt32Slot(1, denseSize, 0)


def DimensionMetadataAddArraySegmentsType(
    builder, arraySegmentsType): builder.PrependUint8Slot(2, arraySegmentsType, 0)


def DimensionMetadataAddArraySegments(builder, arraySegments): builder.PrependUOffsetTRelativeSlot(
    3, flatbuffers.number_types.UOffsetTFlags.py_type(arraySegments), 0)


def DimensionMetadataAddArrayIndicesType(
    builder, arrayIndicesType): builder.PrependUint8Slot(4, arrayIndicesType, 0)
def DimensionMetadataAddArrayIndices(builder, arrayIndices): builder.PrependUOffsetTRelativeSlot(
    5, flatbuffers.number_types.UOffsetTFlags.py_type(arrayIndices), 0)


def DimensionMetadataEnd(builder): return builder.EndObject()
