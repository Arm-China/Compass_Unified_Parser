# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from ..op import *
import onnx
import numpy as np


class AttentionOp(OpHasVariableOutPorts, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            23: {
                'is_causal': {'type': AttrType.INT, 'default': 0},
                'kv_num_heads': {'type': AttrType.INT, 'default': 0},
                'q_num_heads': {'type': AttrType.INT, 'default': 0},
                'qk_matmul_output_mode': {'type': AttrType.INT, 'default': 0},
                'scale': {'type': AttrType.FLOAT, 'default': None},
                'softcap': {'type': AttrType.FLOAT, 'default': 0.0},
                'softmax_precision': {'type': AttrType.INT, 'default': -1},
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(AttentionOp, self).__init__(graph, attr_dict)
        self.update_attributes(AttentionOp, attr_dict)
        assert self.check_required(), 'AttentionOp is missing a required parameter.'

    def infer_shape(self):
        super(AttentionOp, self).infer_shape()

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

        inputs = self.get_input_tensors()
        Q, K, V = inputs[0], inputs[1], inputs[2]
        assert len(Q.shape) == len(K.shape) == len(V.shape)
        # Set input tensors (Q, K, V) to the correct shape if input shape is 3D
        # NewShapeQ (batch_size, q_num_heads, q_sequence_length, head_size)
        # NewShapeK  (batch_size, kv_num_heads, kv_sequence_length, head_size)
        # NewShapeV (value) has shape (batch_size, kv_num_heads, kv_sequence_length, v_head_size)
        input_shape_len = len(Q.shape)
        batch_size = Q.shape[0]
        if len(Q.shape) == 3:
            hidden_size_q = Q.shape[2]
            hidden_size_k = K.shape[2]
            hidden_size_v = V.shape[2]
            assert self.q_num_heads != 0 and self.kv_num_heads != 0

            head_size_q = int(hidden_size_q / self.q_num_heads)
            new_shape_q = [batch_size, Q.shape[1], self.q_num_heads, head_size_q]
            Q = np.reshape(Q, new_shape_q)
            Q = Q.transpose(0, 2, 1, 3)

            head_size_k = int(hidden_size_k / self.kv_num_heads)
            new_shape_k = [batch_size, K.shape[1], self.kv_num_heads, head_size_k]
            K = np.reshape(K, new_shape_k)
            K = K.transpose(0, 2, 1, 3)

            head_size_v = int(hidden_size_v / self.kv_num_heads)
            new_shape_v = [batch_size, V.shape[1], self.kv_num_heads, head_size_v]
            V = np.reshape(V, new_shape_v)
            V = V.transpose(0, 2, 1, 3)

        assert len(Q.shape) == 4 and len(K.shape) == 4 and len(V.shape) == 4

        # Calculate Scaling Factor if not provided
        if self.scale is None:
            q_head_size = Q.shape[3]
            scale = 1 / np.sqrt(q_head_size)
        else:
            scale = self.scale

        in_ports = self.get_in_ports()
        # Update key and value cache
        if 4 in in_ports and len(inputs) >= 5 and inputs[4] is not None:
            past_key = inputs[4]
            present_key = np.concatenate((past_key, K), axis=2)
        else:
            present_key = K
        if 5 in in_ports and len(inputs) >= 6 and inputs[5] is not None:
            past_value = inputs[5]
            present_value = np.concatenate((past_value, V), axis=2)
        else:
            present_value = V
        K = present_key
        V = present_value

        # Create attn_bias
        q_sequence_length = Q.shape[2]
        kv_sequence_length = K.shape[2]
        attn_bias = np.zeros((batch_size, Q.shape[1], q_sequence_length, kv_sequence_length), dtype=Q.dtype)
        # First case: If is_causal is provided
        # If set to true, the attention masking is a lower triangular matrix when the mask
        # is a square matrix. The attention masking has the form of the upper left causal
        # bias due to the alignment when the mask is a non-square matrix.
        if self.is_causal == 1:
            assert 3 not in in_ports or inputs[3] is None
            temp_mask = np.ones((q_sequence_length, kv_sequence_length), dtype=bool)
            temp_mask = np.tril(temp_mask, k=0)
            temp_mask = np.logical_not(temp_mask)
            attn_bias_ma = np.ma.array(attn_bias, mask=temp_mask)
            attn_bias = attn_bias_ma.filled(fill_value=float("-inf"))
        if 3 in in_ports and len(inputs) >= 4 and inputs[3] is not None:
            attn_mask = inputs[3]
            assert self.is_causal != 1
            if attn_mask.dtype == bool:
                attn_mask = (1 - attn_mask).astype(Q.dtype)
                attn_mask[attn_mask == 1] = -np.inf
            attn_bias += attn_mask

        # Group Query Attention is applied if the following are satisfied
        # 1) q_num_heads != kv_num_heads
        # 2) q_num_heads % kv_num_heads == 0
        # 3) kv_num_heads == k_num_heads == v_num_heads
        q_num_heads = Q.shape[1]

        if self.kv_num_heads == 0:
            k_num_heads = K.shape[1]
            v_num_heads = K.shape[1]
        else:
            k_num_heads = self.kv_num_heads
            v_num_heads = self.kv_num_heads
        if (
                (q_num_heads != k_num_heads)
                and (q_num_heads % k_num_heads == 0)
                and (k_num_heads == v_num_heads)
        ):
            seq_reps = int(q_num_heads / k_num_heads)
            reps = [1, seq_reps, 1, 1]
            K = np.tile(K, reps)
            V = np.tile(V, reps)

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
        k_transpose = np.transpose(K, (0, 1, 3, 2))
        qk_matmul_output = np.matmul(Q, k_transpose) * scale
        qk_with_bias = qk_matmul_output + attn_bias
        if self.qk_matmul_output_mode == 1:
            qk_matmul_output = qk_matmul_output + attn_bias

        # Apply softcap
        if self.softcap != 0:
            qk_with_bias = _softcap(qk_with_bias, self.softcap)
            if self.qk_matmul_output_mode == 2:
                qk_matmul_output = qk_with_bias

        if self.softmax_precision != -1:
            qk_with_bias = qk_with_bias.astype(
                onnx.helper.tensor_dtype_to_np_dtype(self.softmax_precision)
            )
        qk_softmax = _softmax(qk_with_bias)
        if self.qk_matmul_output_mode == 3:
            qk_matmul_output = qk_softmax
        qk_matmul_output = qk_matmul_output.astype(Q.dtype)

        output = np.matmul(qk_softmax, V).astype(Q.dtype)
        if input_shape_len == 3:
            output = np.transpose(output, (0, 2, 1, 3))
            output = np.reshape(output, (output.shape[0], output.shape[1], -1))
        out_tensors = [output]
        out_ports = self.get_out_ports()
        if 1 in out_ports:
            out_tensors.append(present_key)
        if 2 in out_ports:
            out_tensors.append(present_value)
        if 3 in out_ports:
            out_tensors.append(qk_matmul_output)
        self.set_out_tensor(out_tensors)

    def convert_version(self):
        # from ...front_end.onnx.passes.common_passes import insert_constant
        max_ver = type(self).max_ver()
        cur_ver = self.cur_version
        if cur_ver < max_ver:
            # TODO
            self.cur_version = max_ver


class RotaryEmbeddingOp(OpHasOneOutPort, OnnxOp):
    @classmethod
    def attributes(cls):
        return {
            23: {
                'interleaved': {'type': AttrType.INT, 'default': 0},
                'num_heads': {'type': AttrType.INT, 'default': 0},
                'rotary_embedding_dim': {'type': AttrType.INT, 'default': 0},
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(RotaryEmbeddingOp, self).__init__(graph, attr_dict)
        self.update_attributes(RotaryEmbeddingOp, attr_dict)
        assert self.check_required(), 'RotaryEmbeddingOp is missing a required parameter.'

    def infer_shape(self):
        super(RotaryEmbeddingOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 3, 'At least 3 inputs are needed in RotaryEmbeddingOp.'
        original_input_shape = inputs[0].shape
        # First ensure input to be processed has shape [batch_size, seq_len, num_heads, head_size]
        if len(inputs[0].shape) == 4:
            input = np.transpose(inputs[0], (0, 2, 1, 3))
        else:
            input = inputs[0]
        batch_size = input.shape[0]
        sequence_length = input.shape[1]
        if len(input.shape) == 3:
            hidden_size = input.shape[2]
            assert self.num_heads != 0
            head_size = int(hidden_size / self.num_heads)
            new_shape = [batch_size, sequence_length, self.num_heads, head_size]
            input = np.reshape(input, new_shape)
        assert len(input.shape) == 4
        head_size = input.shape[3]

        # Fully or partially perform rotation on input based on rotary_embedding_dim attribute
        if self.rotary_embedding_dim is None or self.rotary_embedding_dim == 0:
            # If rotary_embedding_dim not provided, perform full rotation by using head_size
            rotary_embedding_dim = head_size
        else:
            rotary_embedding_dim = self.rotary_embedding_dim
        x_rotate = input[:, :, :, :rotary_embedding_dim]
        x_not_rotate = input[:, :, :, rotary_embedding_dim:]
        rotary_embedding_dim_half = rotary_embedding_dim // 2

        cos_cache = inputs[1]
        sin_cache = inputs[2]
        # Retrieve sin and cos caches using position ids
        if len(inputs) > 3:
            position_ids = inputs[3]
            cos = cos_cache[position_ids]  # Shape: [batch_size, sequence_length, head_size/2]
            sin = sin_cache[position_ids]  # Shape: [batch_size, sequence_length, head_size/2]
        else:
            cos = cos_cache
            sin = sin_cache
        cos = cos[:, :, :rotary_embedding_dim_half]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
        sin = sin[:, :, :rotary_embedding_dim_half]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
        cos = np.expand_dims(cos, axis=2)  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]
        sin = np.expand_dims(sin, axis=2)  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]

        # Either divide the input in halves or interleave (based on interleaved attribute)
        if self.interleaved:
            x1 = x_rotate[:, :, :, 0::2]
            x2 = x_rotate[:, :, :, 1::2]
        else:
            x1, x2 = np.split(x_rotate, 2, axis=-1)

        # Calculate real and imaginary values
        real = (cos * x1) - (sin * x2)
        imag = (sin * x1) + (cos * x2)

        # Inserted rotated embeddings back to the original input
        if self.interleaved:
            # x_rotate[:, :, :, 0::2] = real
            # x_rotate[:, :, :, 1::2] = imag
            real = np.expand_dims(real, axis=-1)
            imag = np.expand_dims(imag, axis=-1)
            x_rotate_concat = np.concatenate((real, imag), axis=-1)
            x_rotate = np.reshape(x_rotate_concat, x_rotate.shape)
        else:
            x_rotate = np.concatenate((real, imag), axis=-1)
        output = np.concatenate((x_rotate, x_not_rotate), axis=-1)
        if len(original_input_shape) == 3:
            output = np.reshape(output, original_input_shape)
        else:
            output = np.transpose(output, (0, 2, 1, 3))
        self.set_out_tensor(output)
