# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.
import torch

from ..op import *
from ...common.utils import unpack_4bit
import numpy as np


class GroupQueryAttentionMsOp(OpHasMultipleOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'do_rotary': {
                    'type': AttrType.INT, 'default': 0
                },
                'kv_num_heads': {
                    'type': AttrType.INT, 'required': True
                },
                'local_window_size': {
                    'type': AttrType.INT, 'default': -1
                },
                'num_heads': {
                    'type': AttrType.INT, 'required': True
                },
                'rotary_interleaved': {
                    'type': AttrType.INT, 'default': 0
                },
                'scale': {
                    'type': AttrType.FLOAT, 'default': None
                },
                'smooth_softmax': {
                    'type': AttrType.INT, 'default': 0
                },
                'softcap': {
                    'type': AttrType.INT, 'default': 0
                },
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(GroupQueryAttentionMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(GroupQueryAttentionMsOp, attr_dict)
        assert self.check_required(), 'GroupQueryAttentionMsOp is missing a required parameter.'

    def infer_shape(self):
        super(GroupQueryAttentionMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert 3 <= len(inputs) <= 12, f'GroupQueryAttentionMsOp should have 3~12 inputs, but got {len(inputs)}'

        def _softcap(inp, softcap):
            if softcap > 0:
                out = inp / softcap
                out = np.tanh(out)
                return out * softcap
            else:
                return inp

        def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
            x_max = np.max(x, axis=axis, keepdims=True)
            tmp = np.exp(x - x_max)
            s = np.sum(tmp, axis=axis, keepdims=True)
            return tmp / s

        if self.local_window_size != -1 or self.smooth_softmax != 0:
            raise NotImplementedError

        query = inputs[0]
        batch_size = query.shape[0]
        if inputs[1] is None and inputs[2] is None:
            # packed QKV
            head_size = query.shape[-1] // (self.num_heads + 2 * self.kv_num_heads)
            split_size = [
                self.num_heads * head_size,
                self.num_heads * head_size + self.kv_num_heads * head_size,
            ]
            query, key, value = np.split(inputs[0], split_size, axis=-1)
            head_size_q = head_size_k = head_size_v = head_size
        else:
            key = inputs[1]
            value = inputs[2]
            head_size_q = query.shape[-1] // self.num_heads
            head_size_k = key.shape[-1] // self.kv_num_heads
            head_size_v = value.shape[-1] // self.kv_num_heads

        if self.do_rotary != 0:
            assert len(inputs) >= 10 and inputs[7] is not None and inputs[8] is not None and inputs[
                9] is not None, 'cos_cache & sin_cache & position ids must be provided if do_rotary is True.'
            from .llm_ops import RotaryEmbeddingOp
            q_list = [query, inputs[7], inputs[8], inputs[9]]
            query = RotaryEmbeddingOp.rope_infer(q_list, self.num_heads, self.rotary_interleaved)
            k_list = [key, inputs[7], inputs[8], inputs[9]]
            key = RotaryEmbeddingOp.rope_infer(k_list, self.num_heads, self.rotary_interleaved)

        new_shape_q = [batch_size, query.shape[1], self.num_heads, head_size_q]
        query = np.reshape(query, new_shape_q)
        query = query.transpose(0, 2, 1, 3)

        new_shape_k = [batch_size, key.shape[1], self.kv_num_heads, head_size_k]
        key = np.reshape(key, new_shape_k)
        key = key.transpose(0, 2, 1, 3)

        new_shape_v = [batch_size, value.shape[1], self.kv_num_heads, head_size_v]
        value = np.reshape(value, new_shape_v)
        value = value.transpose(0, 2, 1, 3)

        # Calculate Scaling Factor if not provided
        if self.scale is None:
            scale = (1 / np.sqrt(head_size_q)).astype(query.dtype)
        else:
            scale = np.array(self.scale, dtype=query.dtype)

        in_ports = self.get_in_ports()
        # Update key and value cache
        if 3 in in_ports and len(inputs) >= 4 and inputs[3] is not None:
            past_key = inputs[3]
            present_key = np.concatenate((past_key, key), axis=2)
        else:
            present_key = key
        if 4 in in_ports and len(inputs) >= 5 and inputs[4] is not None:
            past_value = inputs[4]
            present_value = np.concatenate((past_value, value), axis=2)
        else:
            present_value = value
        key = present_key
        value = present_value

        # Create attn_bias
        if 10 in in_ports and len(inputs) >= 11 and inputs[10] is not None:
            attn_bias = inputs[10]
        else:
            attn_bias = None

        # Group Query Attention is applied if the following are satisfied
        # 1) q_num_heads != kv_num_heads
        # 2) q_num_heads % kv_num_heads == 0
        # 3) kv_num_heads == k_num_heads == v_num_heads

        if self.num_heads != self.kv_num_heads and self.num_heads % self.kv_num_heads == 0:
            seq_reps = self.num_heads // self.kv_num_heads
            key = key.repeat(seq_reps, axis=1)
            value = value.repeat(seq_reps, axis=1)

        # The following pattern is applied
        #      Q          K          V
        #      |          |          |
        #      |       Transpose     |
        #      |          |          |
        #      ---MatMul * scale---  |
        #            |               |
        # at_mask---Add              |
        #            |               |
        #  softcap (if provided)     |
        #            |               |
        #         Softmax            |
        #            |               |
        #            -----MatMul------
        #                    |
        #                    Y
        k_transpose = np.transpose(key, (0, 1, 3, 2))
        qk_matmul_output = np.matmul(query, k_transpose) * scale
        if attn_bias is not None:
            qk_with_bias = qk_matmul_output + attn_bias
        else:
            qk_with_bias = qk_matmul_output

        # Apply softcap
        if self.softcap != 0:
            qk_with_bias = _softcap(qk_with_bias, self.softcap)

        qk_softmax = _softmax(qk_with_bias)
        output = np.matmul(qk_softmax, value).astype(query.dtype)

        output = np.transpose(output, (0, 2, 1, 3))
        output = np.reshape(output, (output.shape[0], output.shape[1], -1))
        out_tensors = [output, present_key, present_value]
        self.set_out_tensor(out_tensors)


class MatMulNBitsMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'K': {
                    'type': AttrType.INT, 'required': True
                },
                'N': {
                    'type': AttrType.INT, 'required': True
                },
                'accuracy_level': {
                    'type': AttrType.INT, 'default': 0
                },
                'bits': {
                    'type': AttrType.INT, 'required': True
                },
                'block_size': {
                    'type': AttrType.INT, 'required': True
                },
            }
        }

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            if ret is None:
                input_names = ['scales', 'zero_points', 'g_idx', 'bias']
                if item in input_names:
                    item_idx = input_names.index(item) + 2
                    inputs = self.get_input_tensors()
                    if len(inputs) > item_idx and inputs[item_idx] is not None:
                        ret = inputs[item_idx]
                        if 'scale' in item:
                            ret = np.array(ret)
                        if 'zero_points' in item:
                            k_blocks = int(np.ceil(self.K / self.block_size))
                            if ret.dtype == 'uint8':
                                packed_shape = (self.N, int(np.ceil(k_blocks * self.bits / 8)))
                                assert ret.shape == packed_shape
                                ret = (unpack_4bit(ret, [self.N, k_blocks]).astype(np.int32) - 2 **
                                       (self.bits - 1)).astype(inputs[0].dtype)
                            else:
                                assert ret.dtype == inputs[0].dtype, 'Unpacked zp should be same dtype with input.'
                                assert ret.shape == (self.N, k_blocks)
                                ret = (np.array(ret) - 2 ** (self.bits - 1)).astype(inputs[0].dtype)
                        self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
                    if ret is None:
                        if item == 'zero_points':
                            fill_value = 0          # unsigned saved default, - 2**(self.bits - 1) if signed
                            ret = np.full(self.scales.shape, fill_value, dtype=inputs[0].dtype)
                        if item == 'bias':
                            ret = np.zeros([self.N], dtype=inputs[0].dtype)
                        self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(MatMulNBitsMsOp, self).__getattr__(item)
        return ret

    def __init__(self, graph, attr_dict=None):
        super(MatMulNBitsMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(MatMulNBitsMsOp, attr_dict)
        assert self.check_required(), 'MatMulNBitsMsOp is missing a required parameter.'

    def infer_shape(self):
        super(MatMulNBitsMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert 3 <= len(inputs) <= 6, f'MatMulNBitsMsOp should have 3~6 inputs, but got {len(inputs)}'
        assert self.bits in (4, 8), f'MatMulNBitsMsOp only support 4/8 bit, but got {self.bits}'
        A = inputs[0]
        B = inputs[1]
        if self.bits == 4:
            # unpack N bits from quantized weights
            unpack_shape = [self.N, B.size * 2 // self.N]
            # high_nbits = (B >> self.bits) & 0x0F
            # low_nbits = B & 0x0F
            # quant_weights = np.concatenate([high_nbits, low_nbits], axis=-1).reshape(self.N, -1)
            quant_weights = unpack_4bit(B.flatten(), unpack_shape) - 2**(self.bits - 1)
        else:
            quant_weights = B

        if self.block_size < 16 or (self.block_size & (self.block_size - 1) != 0):
            ERROR(f'Node({self.name}) block_size must be a power of 2, and >= 16. Got {self.block_size}.')

        scale = np.repeat(self.scales.reshape(self.N, -1), self.block_size, axis=-1)
        zp = np.repeat(self.zero_points.reshape(self.N, -1), self.block_size, axis=-1)
        if self.is_all_inputs_const():
            # dequantize B tensor according scale
            deq_B = ((quant_weights - zp) * scale).transpose()
            out_tensor = np.matmul(A, deq_B).astype(A.dtype) + self.bias
        else:
            out_shape = list(A.shape).copy()
            out_shape[-1] = self.N
            out_tensor = np.random.ranf(tuple(out_shape)).astype(A.dtype) + self.bias
        out_symbol = self.cal_output_symbol()
        self.set_out_tensor(out_tensor, out_symbol)

    def cal_output_symbol(self):
        if not self._graph._attr['enable_ds']:
            return None
        a_symbol, b_symbol = self.get_input_symbols(local=True)[:2]
        a_dim = len(a_symbol)
        b_dim = len(b_symbol)
        max_dim = max(a_dim, b_dim)
        out_symbol = []
        if max_dim != 1:
            if a_dim == 1:
                out_symbol = b_symbol[:]
                del out_symbol[-2]
            elif b_dim == 1:
                out_symbol = a_symbol[:-1]
            else:
                if b_dim == 2:
                    out_symbol = a_symbol[:-1] + b_symbol[-1:]
                else:
                    if a_dim < max_dim:
                        for i in range(max_dim - a_dim):
                            a_symbol.insert(0, 1)
                    if b_dim < max_dim:
                        for i in range(max_dim - b_dim):
                            b_symbol.insert(0, 1)

                    for i in range(max_dim):
                        if i < max_dim - 2:
                            if a_symbol[i] == 1:
                                out_symbol.append(b_symbol[i])
                            elif b_symbol[i] == 1:
                                out_symbol.append(a_symbol[i])
                            else:
                                input_shapes = self.get_input_shapes()
                                sym_shape_map = {}
                                idx_add = 0
                                for shape in input_shapes:
                                    for idx, s in enumerate(shape):
                                        sym_shape_map[idx + idx_add] = s
                                    idx_add += len(shape)
                                sym_idx = int(re.findall(r'\d+', str(a_symbol[i]))[0])
                                if sym_shape_map[sym_idx] != 1:
                                    out_symbol.append(a_symbol[i])
                                else:
                                    b_sym_idx = int(re.findall(r'\d+', str(b_symbol[i]))[0])
                                    if sym_shape_map[b_sym_idx] == 1:
                                        out_symbol.append(Max(a_symbol[i], b_symbol[i]))
                                    else:
                                        out_symbol.append(b_symbol[i])
                        elif i == max_dim - 2:
                            out_symbol.append(a_symbol[i])
                        else:
                            out_symbol.append(b_symbol[i])
        return out_symbol


class MultiHeadAttentionMsOp(OpHasVariableOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'mask_filter_value': {
                    'type': AttrType.FLOAT, 'default': -10000.0
                },
                'num_heads': {
                    'type': AttrType.INT, 'required': True
                },
                'scale': {
                    'type': AttrType.FLOAT,
                },
                'unidirectional': {
                    'type': AttrType.INT, 'default': 0
                },
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(MultiHeadAttentionMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(MultiHeadAttentionMsOp, attr_dict)
        assert self.check_required(), 'MultiHeadAttentionMsOp is missing a required parameter.'

    def infer_shape(self):
        super(MultiHeadAttentionMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) <= 10, f'MultiHeadAttentionMsOp should have 1~10 inputs, but got {len(inputs)}'

        query = inputs[0]
        bs, seq_len = inputs[0].shape[:2]
        self.head_dim = query.shape[-1] // self.num_heads
        if self.is_all_inputs_const():
            key = inputs[1] if len(inputs) > 1 else None
            value = inputs[2] if len(inputs) > 2 else None
            bias = inputs[3] if len(inputs) > 3 else None
            key_padding_mask = inputs[4] if len(inputs) > 4 else None
            attention_bias = inputs[5] if len(inputs) > 5 else None
            past_key = inputs[6] if len(inputs) > 6 else None
            past_value = inputs[7] if len(inputs) > 7 else None

            # TODO, add infer if bias/key_padding_mask/past_key/past_value is not None
            # TODO, add present_key/value output
            def split_heads(x):
                x = x.reshape(x.shape[0], -1, self.num_heads, self.head_dim)
                return x.transpose(0, 2, 1, 3)

            def concat_heads(x):
                x = np.transpose(x, [0, 2, 1, 3])
                return x.reshape(x.shape[0], -1, self.num_heads * self.head_dim)

            q_split = split_heads(query)
            k_split = split_heads(key)
            qk_matmul = np.matmul(q_split, np.transpose(k_split, [0, 1, 3, 2]))
            scores = qk_matmul * self.scale
            attention_out = scores + attention_bias
            attention = torch.softmax(torch.tensor(attention_out), dim=-1)
            v_split = split_heads(value)
            output = np.matmul(attention.cpu().numpy(), v_split)

            # concat heads
            out_tensors = [concat_heads(output)]
        else:
            out_tensor = np.random.ranf((bs, seq_len, self.num_heads * self.head_dim)).astype(inputs[0].dtype)
            out_tensors = [out_tensor]
        out_symbols = [self.get_input_symbols(local=True)[0]]
        out_ports = self.get_out_ports()
        # if 1 in out_ports:
        #     out_tensors.append(np.array(std_var, np.float32))
        # if 2 in out_ports:
        #     out_tensors.append(np.array(std_var, np.float32))
        self.set_out_tensor(out_tensors, symbols=out_symbols)


class QGemmMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 1.0},
                    'transA': {'type': AttrType.BOOL, 'default': False},
                    'transB': {'type': AttrType.BOOL, 'default': False}}}

    def __init__(self, graph, attr_dict=None):
        super(QGemmMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QGemmMsOp, attr_dict)
        assert self.check_required(), 'QGemmMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            input_names = ['A', 'a_scale', 'a_zero_point', 'B',
                           'b_scale', 'b_zero_point', 'C', 'y_scale', 'y_zero_point']
            if item in input_names:
                item_idx = input_names.index(item)
                inputs = self.get_input_tensors()
                if len(inputs) > item_idx:
                    ret = inputs[item_idx]
                    if ret is not None:
                        ret = np.array(ret, dtype=np.float32) if 'scale' in item else np.array(ret)
                self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QGemmMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QGemmMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 6, 'Meets invalid inputs length of QGemmMsOp(%s)' % self.name
        A = self.A if not self.transA else np.transpose(self.A)
        float_A = (A.astype(np.int32) - self.a_zero_point) * self.a_scale
        B = self.B if not self.transB else np.transpose(self.B)
        float_B = (B.astype(np.int32) - self.b_zero_point) * self.b_scale
        C = np.array(0, dtype=np.int32) if self.C is None else self.C
        c_scale = self.alpha * self.a_scale * self.b_scale
        float_C = C.astype(np.int32) * c_scale
        # alpha * A' * B' + C
        out_tensor = self.alpha * np.matmul(float_A, float_B) + float_C
        if self.y_scale is not None:
            assert self.y_zero_point is not None, \
                'y_zero_point of QGemmMsOp(%s) must not be null if y_scale is not null' % self.name
            out_min = np.iinfo(self.y_zero_point.dtype).min
            out_max = np.iinfo(self.y_zero_point.dtype).max
            out_tensor = np.clip(np.around(out_tensor / self.y_scale) + self.y_zero_point,
                                 out_min, out_max).astype(self.y_zero_point.dtype)
        else:
            assert self.y_zero_point is None, \
                'y_zero_point of QGemmMsOp(%s) must be null if y_scale is null' % self.name
        self.set_out_tensor(out_tensor)


class QLinearAddMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearAddMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearAddMsOp, attr_dict)
        assert self.check_required(), 'QLinearAddMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            if ret is None:
                input_names = ['A', 'A_scale', 'A_zero_point', 'B',
                               'B_scale', 'B_zero_point', 'C_scale', 'C_zero_point']
                if item in input_names:
                    item_idx = input_names.index(item)
                    inputs = self.get_input_tensors()
                    if len(inputs) > item_idx:
                        ret = inputs[item_idx]
                        if 'scale' in item:
                            ret = np.array(ret).astype(np.float32)
                        self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
                if ret is None and item in ('A_zero_point', 'B_zero_point', 'C_zero_point') and self.A is not None:
                    ret = np.array(0, dtype=self.A.dtype)
                    self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QLinearAddMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearAddMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 7, 'Meets invalid inputs length of QLinearAddMs op (%s)' % self.name
        float_a = (self.A.astype(np.int32) - self.A_zero_point) * self.A_scale
        float_b = (self.B.astype(np.int32) - self.B_zero_point) * self.B_scale
        float_y = np.add(float_a, float_b)
        out_min = np.iinfo(self.C_zero_point.dtype).min
        out_max = np.iinfo(self.C_zero_point.dtype).max
        out_tensor = np.clip(np.around(float_y / self.C_scale) + self.C_zero_point,
                             out_min, out_max).astype(self.C_zero_point.dtype)
        self.set_out_tensor(out_tensor)


class QLinearAveragePoolMsOp(BaseOnnxPoolOp, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'channels_last': {'type': AttrType.INT, 'default': 0},
                    'kernel_shape': {'type': AttrType.INTS, 'required': True},
                    'auto_pad': {'type': AttrType.STRING, 'default': 'NOTSET'},
                    'ceil_mode': {'type': AttrType.INT, 'default': 0},
                    'count_include_pad': {'type': AttrType.INT, 'default': 0}}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearAveragePoolMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearAveragePoolMsOp, attr_dict)
        assert self.check_required(), 'QLinearAveragePoolMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            input_names = ['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point']
            if item in input_names:
                item_idx = input_names.index(item)
                inputs = self.get_input_tensors()
                if len(inputs) > item_idx:
                    ret = inputs[item_idx]
                    if 'scale' in item:
                        ret = np.array(ret).astype(np.float32)
                    self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
            if item == 'data_format':
                if self.channels_last == 1:
                    self.__dict__['_attr'][item] = 'NHWC'
                    ret = 'NHWC'
                else:
                    self.__dict__['_attr'][item] = 'NCHW'
                    ret = 'NCHW'
        except:
            ret = None
        if ret is None:
            ret = super(QLinearAveragePoolMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearAveragePoolMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 5, 'Meets invalid inputs length of QLinearAveragePoolMsOp(%s)' % self.name
        pading_value = np.nan if self.count_include_pad == 0 else 0
        in_shape = inputs[0].shape[1:-1] if self.data_format == 'NHWC' else inputs[0].shape[2:]
        spatial_dim = len(inputs[0].shape) - 2
        out_shape = BaseOnnxPoolOp.cal_out_shape(
            in_shape, self.pads, self.strides, self.kernel_shape, self.auto_pad, dilations=self.dilations,
            ceil_mode=self.ceil_mode)
        if self.auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            # re-calculate pads
            self.pads, _ = OpHasPaddingStrides.cal_pads(
                in_shape,
                out_shape,
                self.strides,
                self.kernel_shape,
                self.auto_pad,
                dilations=self.dilations,
                is_transpose=False,
            )
            self.auto_pad = 'NOTSET'
        float_x = (self.x.astype(np.int32) - self.x_zero_point) * self.x_scale
        n_dims = len(self.pads) // 2
        pads_np = [(self.pads[i], self.pads[i + n_dims]) for i in range(n_dims)]
        if self.data_format == 'NHWC':
            perm = [0, spatial_dim + 1] + list(range(1, spatial_dim + 1))
            float_x = np.transpose(float_x, perm)
        padded_input = np.pad(
            float_x,
            ((0, 0), (0, 0), *pads_np),
            mode="constant",
            constant_values=pading_value,
        )
        float_y = BaseOnnxPoolOp.pool(
            padded_input,
            float_x.shape,
            self.kernel_shape,
            self.strides,
            out_shape,
            'AVG',
            self.pads,
            self.dilations,
            self.count_include_pad
        )
        if self.data_format == 'NHWC':
            perm = [0] + list(range(2, spatial_dim + 2)) + [1]
            float_y = np.transpose(float_y, perm)
        if self.y_scale is not None:
            assert self.y_zero_point is not None, \
                'y_zero_point of QLinearAveragePoolMsOp(%s) must not be null if y_scale is not null' % self.name
            out_min = np.iinfo(self.y_zero_point.dtype).min
            out_max = np.iinfo(self.y_zero_point.dtype).max
            out_tensor = np.clip(np.around(float_y / self.y_scale) + self.y_zero_point,
                                 out_min, out_max).astype(self.y_zero_point.dtype)
        else:
            assert self.y_zero_point is None, \
                'y_zero_point of QLinearAveragePoolMsOp(%s) must be null if y_scale is null' % self.name
        self.set_out_tensor(out_tensor)


class QLinearConcatMsOp(OpHasAxis, OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            1: {'axis': {'required': True}}
        }

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            input_names = ['y_scale', 'y_zero_point']
            if item in input_names:
                item_idx = input_names.index(item)
                inputs = self.get_input_tensors()
                if len(inputs) > item_idx:
                    ret = inputs[item_idx]
                    if 'scale' in item:
                        ret = np.array(ret).astype(np.float32)
                    self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QLinearConcatMsOp, self).__getattr__(item)
        return ret

    def __init__(self, graph, attr_dict=None):
        super(QLinearConcatMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearConcatMsOp, attr_dict)
        assert self.check_required(), 'QLinearConcatMsOp is missing a required parameter.'

    def infer_shape(self):
        super(QLinearConcatMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 5, 'Meets invalid inputs length of QLinearConcatMsOp op (%s)' % self.name
        fp_inputs = []
        for i in range(2, len(inputs), 3):
            inp_t = inputs[i]
            inp_scale = inputs[i + 1]
            inp_zp = inputs[i + 2]
            fp_inp = (inp_t.astype(np.int32) - inp_zp) * inp_scale
            fp_inputs.append(fp_inp)
        float_y = np.concatenate(fp_inputs, axis=self.axis)
        out_min = np.iinfo(self.y_zero_point.dtype).min
        out_max = np.iinfo(self.y_zero_point.dtype).max
        out_tensor = np.clip(np.around(float_y / self.y_scale) + self.y_zero_point,
                             out_min, out_max).astype(self.y_zero_point.dtype)
        self.set_out_tensor(out_tensor)


class QLinearGlobalAveragePoolMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'channels_last': {'type': AttrType.INT, 'required': True}}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearGlobalAveragePoolMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearGlobalAveragePoolMsOp, attr_dict)
        assert self.check_required(), 'QLinearGlobalAveragePoolMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            input_names = ['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point']
            if item in input_names:
                item_idx = input_names.index(item)
                inputs = self.get_input_tensors()
                if len(inputs) > item_idx:
                    ret = inputs[item_idx]
                    if 'scale' in item:
                        ret = np.array(ret).astype(np.float32)
                    self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QLinearGlobalAveragePoolMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearGlobalAveragePoolMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 5, 'Meets invalid inputs length of QLinearGlobalAveragePoolMsOp(%s)' % self.name
        if self.channels_last == 0:
            axes = list(range(2, len(inputs[0].shape)))
        else:
            axes = list(range(1, len(inputs[0].shape) - 1))
        float_x = (self.x.astype(np.int32) - self.x_zero_point) * self.x_scale
        float_y = np.mean(float_x, axis=tuple(axes), keepdims=True)
        out_min = np.iinfo(self.y_zero_point.dtype).min
        out_max = np.iinfo(self.y_zero_point.dtype).max
        out_tensor = np.clip(np.around(float_y / self.y_scale) + self.y_zero_point,
                             out_min, out_max).astype(self.y_zero_point.dtype)
        self.set_out_tensor(out_tensor)


class QLinearLeakyReluMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {'alpha': {'type': AttrType.FLOAT, 'default': 0.01}}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearLeakyReluMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearLeakyReluMsOp, attr_dict)
        assert self.check_required(), 'QLinearLeakyReluMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            if ret is None:
                input_names = ['X', 'X_scale', 'X_zero_point', 'Y_scale', 'Y_zero_point']
                if item in input_names:
                    item_idx = input_names.index(item)
                    inputs = self.get_input_tensors()
                    if len(inputs) > item_idx:
                        ret = inputs[item_idx]
                        if 'scale' in item:
                            ret = np.array(ret).astype(np.float32)
                        self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QLinearLeakyReluMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearLeakyReluMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 4, 'Meets invalid inputs length of QLinearLeakyReluMsOp op (%s)' % self.name
        float_x = (self.X.astype(np.int32) - self.X_zero_point) * self.X_scale
        float_y = tf.nn.leaky_relu(float_x.astype(np.float32), alpha=self.alpha).numpy()
        out_min = np.iinfo(self.Y_zero_point.dtype).min
        out_max = np.iinfo(self.Y_zero_point.dtype).max
        out_tensor = np.clip(np.around(float_y / self.Y_scale) + self.Y_zero_point,
                             out_min, out_max).astype(self.Y_zero_point.dtype)
        self.set_out_tensor(out_tensor)


class QLinearSigmoidMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {1: {}}

    def __init__(self, graph, attr_dict=None):
        super(QLinearSigmoidMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(QLinearSigmoidMsOp, attr_dict)
        assert self.check_required(), 'QLinearSigmoidMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            if ret is None:
                input_names = ['X', 'X_scale', 'X_zero_point', 'Y_scale', 'Y_zero_point']
                if item in input_names:
                    item_idx = input_names.index(item)
                    inputs = self.get_input_tensors()
                    if len(inputs) > item_idx:
                        ret = inputs[item_idx]
                        if 'scale' in item:
                            ret = np.array(ret).astype(np.float32)
                        self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(QLinearSigmoidMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(QLinearSigmoidMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 4, 'Meets invalid inputs length of QLinearSigmoidMsOp op (%s)' % self.name
        float_x = (self.X.astype(np.int32) - self.X_zero_point) * self.X_scale
        float_y = tf.sigmoid(float_x.astype(np.float32)).numpy()
        out_min = np.iinfo(self.Y_zero_point.dtype).min
        out_max = np.iinfo(self.Y_zero_point.dtype).max
        out_tensor = np.clip(np.around(float_y / self.Y_scale) + self.Y_zero_point,
                             out_min, out_max).astype(self.Y_zero_point.dtype)
        self.set_out_tensor(out_tensor)


class RotaryEmbeddingMsOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'interleaved': {
                    'type': AttrType.INT, 'default': 0
                },
                'is_packed_batching': {
                    'type': AttrType.INT, 'default': 0
                },
                'num_heads': {
                    'type': AttrType.INT, 'default': 0
                },
                'rotary_embedding_dim': {
                    'type': AttrType.INT, 'default': 0
                },
                'scale': {
                    'type': AttrType.FLOAT, 'default': 1.0
                },
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(RotaryEmbeddingMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(RotaryEmbeddingMsOp, attr_dict)
        assert self.check_required(), 'RotaryEmbeddingMsOp is missing a required parameter.'

    def __getattr__(self, item):
        try:
            ret = self.__dict__['_attr'][item].value
        except:
            ret = None
        try:
            if ret is None:
                input_names = ['input', 'position_ids', 'cos_cache', 'sin_cache']
                if item in input_names:
                    item_idx = input_names.index(item)
                    inputs = self.get_input_tensors()
                    if len(inputs) > item_idx:
                        ret = inputs[item_idx]
                        if 'cache' in item:
                            ret = np.array(ret)
                        self.__dict__['_attr'][item] = Attribute(item, {'type': AttrType.TENSOR, 'value': ret})
        except:
            ret = None
        if ret is None:
            ret = super(RotaryEmbeddingMsOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(RotaryEmbeddingMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 4, f'Expected 4 inputs of RotaryEmbeddingMsOp op ({self.name}), but got {len(inputs)} inputs'
        assert len(inputs[0].shape) in (
            3, 4), f'Input x is expected to be 3 or 4 dimensional, but got {len(inputs[0].shape)}'
        assert len(inputs[1].shape) in (
            0, 1, 2), f'Input position_ids is expected to be 0, 1 or 2 dimensional, but got {len(inputs[1].shape)}'
        assert len(inputs[2].shape) == 2 and len(inputs[3].shape) == 2, \
            f'Input cos_cache/sin_cache is expected to be 2 dimensional, but got {len(inputs[2].shape)}/{len(inputs[3].shape)}'
        if self.rotary_embedding_dim > 0 and self.num_heads == 0:
            ERROR('[Parser]: num_heads must be provided if rotary_embedding_dim > 0 in (%s).' % self.name)

        bs = inputs[0].shape[0]
        seq_len = inputs[0].shape[1]
        hidden_size = inputs[0].shape[2]

        if len(inputs[0].shape) == 4:
            seq_len = inputs[0].shape[2]
            hidden_size = inputs[0].shape[1] * inputs[0].shape[3]
        max_seq_len = self.cos_cache.shape[0]
        head_size = self.cos_cache.shape[1] * 2 if self.rotary_embedding_dim == 0 else hidden_size // self.num_heads
        if self.rotary_embedding_dim > 0 and self.rotary_embedding_dim > head_size:
            ERROR('[Parser]: rotary_embedding_dim must be less than or equal to head_size in %s.' % self.name)

        if self.is_packed_batching == 0 and seq_len > max_seq_len:
            ERROR('[Parser]: Updating cos_cache and sin_cache in RotaryEmbedding is not currently supported')
        if self.interleaved == 1:
            raise NotImplementedError('interleaved still not supported yet.')
        postion_ids = inputs[1]
        input = inputs[0].reshape(bs, seq_len, -1, head_size)
        cos_cache = np.repeat(self.cos_cache, 2, axis=-1)
        sin_cache = np.repeat(self.sin_cache, 2, axis=-1)
        cos_emb = np.expand_dims(np.take(cos_cache, np.array(
            postion_ids, np.int32), axis=0), axis=2)
        cos_mul = input * np.tile(cos_emb, reps=(1, 1, int(input.shape[2]), 1))

        rotate_input = np.concatenate([np.negative(input[..., head_size // 2:]),
                                       input[..., : head_size // 2]], axis=-1)

        sin_emb = np.expand_dims(np.take(sin_cache, np.array(postion_ids, np.int32), axis=0), axis=2)
        sin_mul = rotate_input * np.tile(sin_emb, reps=(1, 1, int(input.shape[2]), 1))
        out_tensor = np.reshape((cos_mul + sin_mul), (bs, seq_len, -1)).astype(input.dtype)
        out_symbol = self.get_input_symbols(local=True)[0]
        self.set_out_tensor(out_tensor, out_symbol)


class SkipSimplifiedLayerNormalizationMsOp(OpHasVariableOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'epsilon': {
                    'type': AttrType.FLOAT, 'required': True
                },
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(SkipSimplifiedLayerNormalizationMsOp, self).__init__(graph, attr_dict)
        self.update_attributes(SkipSimplifiedLayerNormalizationMsOp, attr_dict)
        assert self.check_required(), 'SkipSimplifiedLayerNormalizationMsOp is missing a required parameter.'

    def infer_shape(self):
        super(SkipSimplifiedLayerNormalizationMsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) in (
            3, 4), f'SkipSimplifiedLayerNormalizationMsOp should have 3 or 4 inputs, but got {len(inputs)}'

        input, skip, gamma = inputs[:3]
        inp_skip_bias_sum = input + skip
        if len(inputs) == 4:
            bias = inputs[3]
            inp_skip_bias_sum += bias

        # RMSNorm
        inp_skip_bias_sum = inp_skip_bias_sum.astype(np.float32)
        square = np.power(inp_skip_bias_sum, 2)
        mean = np.mean(square, axis=(-1, ), keepdims=True)
        variance = np.var(square, axis=(-1, ), keepdims=True)
        normalized = inp_skip_bias_sum / np.sqrt(mean + self.epsilon)
        normalized = normalized.astype(input.dtype)
        axes = OpHasAxis.make_axes_non_negative(
            [-1], len(inp_skip_bias_sum.shape))
        weights = OpHasAxis.expand_to(gamma, axes, len(inp_skip_bias_sum.shape))
        rms_output = (normalized * weights).astype(input.dtype)
        out_ports = self.get_out_ports()
        output_tensors = [rms_output]
        if 1 in out_ports:
            output_tensors.append(mean.astype(np.float32))
        if 2 in out_ports:
            output_tensors.append(variance.astype(np.float32))
        if 3 in out_ports:
            output_tensors.append(inp_skip_bias_sum.astype(input.dtype))
        out_symbols = self.cal_output_symbol()
        self.set_out_tensor(output_tensors, symbols=out_symbols)

    def cal_output_symbol(self):
        if self._graph._attr['enable_ds']:
            input_symbols = self.get_input_symbols(local=True)
            out_ports = self.get_out_ports()
            out_symbols = [input_symbols[0]]
            mean_var_symbol = input_symbols[0]
            mean_var_symbol[-1] = 1
            if 1 in out_ports:
                out_symbols.append(mean_var_symbol)
            if 2 in out_ports:
                out_symbols.append(mean_var_symbol)
            if 3 in out_ports:
                out_symbols.append(input_symbols[0])
        else:
            out_symbols = []
        return out_symbols
