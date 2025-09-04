# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Any, Optional

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import mindspore.nn as nn
import mindspore.ops as ops

MEMORY_LAYOUT = {
    "flash": (lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]), lambda x: x),
    "vanilla": (lambda x: x.transpose(1, 2), lambda x: x.transpose(1, 2)),
}


def attention(
    q: ms.Tensor,
    k: ms.Tensor,
    v: ms.Tensor,
    mode: str = "flash",
    drop_rate: float = 0,
    attn_mask: Optional[ms.Tensor] = None,
    causal: bool = False,
    max_seqlen_q: Optional[int] = None,
    batch_size: int = 1,
) -> ms.Tensor:
    """
    Perform QKV self attention.

    Args:
        q (ms.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (ms.Tensor): Key tensor with shape [b, s1, a, d]
        v (ms.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'flash' and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (ms.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (ms.Tensor): dtype ms.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (ms.Tensor): dtype ms.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        ms.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    if mode == "flash":
        x = ops.flash_attention_score(
            q,
            k,
            v,
            q.shape[-2],
            scalar_value=1 / math.sqrt(q.shape[-1]),
            keep_prob=1.0 - drop_rate,
            input_layout="BSND",
        )
        # x with shape [(bxs), a, d]
        x = x.view(batch_size, max_seqlen_q, x.shape[-2], x.shape[-1])  # reshape x to [b, s, a, d]
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.shape[-1])

        b, a, s, _ = q.shape
        s1 = k.shape[2]
        attn_bias = mint.zeros((b, a, s, s1), dtype=q.dtype)
        if causal:
            # Only applied to self attention
            assert attn_mask is None, "Causal mask and attn_mask cannot be used together"
            temp_mask = mint.ones((b, a, s, s), dtype=ms.bool_).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == ms.bool_:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # TODO: Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=drop_rate, training=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


class CausalConv1d(nn.Cell):
    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "replicate",
        dtype: Any = ms.float32,
        **kwargs,
    ):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T
        self.time_causal_padding = padding

        self.conv = mint.nn.Conv1d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, dtype=dtype, **kwargs
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class MotionEncoder_tc(nn.Cell):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, need_global: bool = True, dtype: Any = ms.float32):
        factory_kwargs = {"dtype": dtype}
        super().__init__()

        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1, dtype=dtype)
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1, dtype=dtype)
        self.norm1 = mint.nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.act = mint.nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2, dtype=dtype)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2, dtype=dtype)

        if need_global:
            self.final_linear = mint.nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)

        self.norm1 = mint.nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.norm2 = mint.nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.norm3 = mint.nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.padding_tokens = ms.Parameter(mint.zeros((1, 1, 1, hidden_dim), dtype=dtype))

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # b t c -> b c t
        x = x.transpose(0, 2, 1)
        x_ori = x.clone()
        b, c, t = x.shape
        x = self.conv1_local(x)
        # b (n c) t -> (b n) t c
        x = x.reshape(x.shape[0], self.num_heads, -1, x.shape[2])
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(-1, *x.shape[2:])
        x = self.norm1(x)
        x = self.act(x)
        # b t c -> b c t
        x = x.transpose(0, 2, 1)
        x = self.conv2(x)
        # b c t -> b t c
        x = x.transpose(0, 2, 1)
        x = self.norm2(x)
        x = self.act(x)
        # b t c -> b c t
        x = x.transpose(0, 2, 1)
        x = self.conv3(x)
        # b c t -> b t c
        x = x.transpose(0, 2, 1)
        x = self.norm3(x)
        x = self.act(x)
        # (b n) t c -> b t n c
        x = x.reshape(b, -1, *x.shape[1:])
        x = x.transpose(0, 2, 1, 3)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1)
        x = mint.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(x_ori)
        # b c t -> b t c
        x = x.transpose(0, 2, 1)
        x = self.norm1(x)
        x = self.act(x)
        # b t c -> b c t
        x = x.transpose(0, 2, 1)
        x = self.conv2(x)
        # b c t -> b t c
        x = x.transpose(0, 2, 1)
        x = self.norm2(x)
        x = self.act(x)
        # b t c -> b c t
        x = x.transpose(0, 2, 1)
        x = self.conv3(x)
        # b c t -> b t c
        x = x.transpose(0, 2, 1)
        x = self.norm3(x)
        x = self.act(x)
        x = self.final_linear(x)
        # (b n) t c -> b t n c
        x = x.reshape(b, -1, *x.shape[1:])
        x = x.transpose(0, 2, 1, 3)

        return x, x_local
