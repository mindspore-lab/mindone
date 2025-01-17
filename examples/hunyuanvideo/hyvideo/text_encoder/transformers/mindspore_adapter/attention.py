import numpy as np

import mindspore as ms
from mindspore import nn, ops
from mindspore.ops.operations.nn_ops import FlashAttentionScore as _FlashAttention

DTYPE_FP16_MIN = float(np.finfo(np.float16).min)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dtype=None):
    # force dtype(fp16 or bf16) precision calculation
    ori_dtype = query.dtype
    if dtype is not None:
        query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)

    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_mask = attn_mask.to(ms.float32)
            attn_mask = attn_mask.masked_fill((1 - attn_mask).to(ms.bool_), DTYPE_FP16_MIN)
        attn_mask = attn_mask.to(query.dtype)

        attn_weight = ops.softmax(
            ops.matmul(query, key.swapaxes(-2, -1)) / (query.shape[-1] ** 0.5) + attn_mask, axis=-1, dtype=ms.float32
        ).astype(query.dtype)
    else:
        attn_weight = ops.softmax(
            ops.matmul(query, key.swapaxes(-2, -1)) / (query.shape[-1] ** 0.5), axis=-1, dtype=ms.float32
        ).astype(query.dtype)

    out = ops.matmul(attn_weight, value)
    out = out.astype(ori_dtype)

    return out


class FlashAttention2(nn.Cell):
    def __init__(
        self,
        head_dim: int,
        head_num: int,
        attention_dropout: float = 0.0,
        input_layout: str = "BNSD",
        dtype: ms.dtype = ms.float16,
    ):
        super().__init__()
        self.input_layout = input_layout
        if input_layout not in ["BSH", "BNSD"]:
            raise ValueError(f"input_layout must be in ['BSH', 'BNSD'], but get {input_layout}.")
        self.head_dim = head_dim

        self.flash_attention = _FlashAttention(
            scale_value=head_dim**-0.5,
            head_num=head_num,
            input_layout=input_layout,
            keep_prob=1 - attention_dropout,
        )

        self.dtype = dtype
        cand_d_list = [64, 80, 96, 120, 128, 256]
        self.d_pad = 0
        for d in cand_d_list:
            if head_dim == d:
                self.d_pad = 0
                break
            elif head_dim < d:
                self.d_pad = d - head_dim
                break
        if head_dim > 256:
            raise ValueError("head_dim must <= 256!")
        self.need_pad = self.d_pad != 0

    def _rearange_input(self, x):
        x = x.to(self.dtype)
        if self.need_pad:
            if self.input_layout == "BNSD":
                B, N, S, D = x.shape
                pad = ops.zeros((B, N, S, self.d_pad), x.dtype)
            else:
                B, S = x.shape[:2]
                x = x.reshape(B, S, -1, self.head_dim)
                pad = ops.zeros((B, S, x.shape[2], self.d_pad), x.dtype)
            x = ops.concat((x, pad), axis=-1)
        if self.input_layout == "BSH":
            B, S = x.shape[:2]
            x = x.reshape(B, S, -1)
        return x

    def _rearange_output(self, x, dtype):
        if self.input_layout == "BSH":
            B, S = x.shape[:2]
            x = x.reshape(B, S, -1, self.head_dim + self.d_pad)
        if self.need_pad:
            x = x[:, :, :, : self.head_dim]
        return x.to(dtype)

    def construct(self, q, k, v, mask=None):
        q_dtype = q.dtype
        q = self._rearange_input(q)
        k = self._rearange_input(k)
        v = self._rearange_input(v)
        if mask is not None:
            mask = mask.to(ms.uint8)
        out = self.flash_attention(q, k, v, None, None, None, mask)[3]
        out = self._rearange_output(out, q_dtype)
        return out
