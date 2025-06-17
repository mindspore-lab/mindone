import math
from typing import Optional

from packaging.version import parse
from transformers.utils import logging

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.ops.operations.nn_ops import FlashAttentionScore as _FlashAttention

from .utils import dtype_to_min

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def math_attention_op(query, key, value, attn_mask=None, dtype=None):
    # force dtype(fp16 or bf16) precision calculation
    ori_dtype = query.dtype
    if dtype is not None:
        query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)

    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_mask = mint.logical_not(attn_mask).to(query.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask.to(ms.bool_), dtype_to_min(query.dtype))

        attn_weight = mint.softmax(
            mint.matmul(query, key.swapaxes(-2, -1)) / (query.shape[-1] ** 0.5) + attn_mask, dim=-1, dtype=ms.float32
        ).astype(query.dtype)
    else:
        attn_weight = mint.softmax(
            mint.matmul(query, key.swapaxes(-2, -1)) / (query.shape[-1] ** 0.5), dim=-1, dtype=ms.float32
        ).astype(query.dtype)

    out = mint.matmul(attn_weight, value)
    out = out.astype(ori_dtype)

    return out


def flash_attention_op(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    keep_prob: float = 1.0,
    scale: Optional[float] = None,
):
    # For most scenarios, qkv has been processed into a BNSD layout before sdp
    input_layout = "BNSD"
    head_num = query.shape[1]
    scale = scale if scale else float(1 / math.sqrt(query.shape[-1]))

    # In case qkv is 3-dim after `head_to_batch_dim`
    if query.ndim == 3:
        input_layout = "BSH"
        head_num = 1

    # process `attn_mask` as logic is different between PyTorch and Mindspore
    # In MindSpore, False indicates retention and True indicates discard, in PyTorch it is the opposite
    if attn_mask is not None:
        attn_mask = mint.logical_not(attn_mask) if attn_mask.dtype == ms.bool_ else attn_mask.bool()
        attn_mask = mint.broadcast_to(
            attn_mask, (attn_mask.shape[0], attn_mask.shape[1], query.shape[-2], key.shape[-2])
        )[:, :1, :, :]

    return ops.operations.nn_ops.FlashAttentionScore(
        head_num=head_num, keep_prob=keep_prob, scale_value=scale, input_layout=input_layout
    )(query, key, value, None, None, None, attn_mask)[3]


def scaled_dot_product_attention(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    r"""
    Perform scaled dot-product attention using either the flash attention operator or the mathematical
    formula-based attention, depending on the availability of the flash attention operator and the
    data-types of the inputs.

    Parameters:
        query (ms.Tensor): The query tensor.
        key (ms.Tensor): The key tensor.
        value (ms.Tensor): The value tensor.
        attn_mask (Optional[ms.Tensor], optional): The attention mask tensor. Defaults to None.
        dropout_p (float, optional): The dropout probability. Defaults to 0.0.
        is_causal (bool): Un-used. Aligned with Torch
        scale (float, optional): scaled value

    Returns:
        ms.Tensor: The result of the scaled dot-product attention.

    Notes:
        - If the flash attention operator is not available (`self.fa_op_available` is False),
          the function falls back to the mathematical formula-based attention.
        - If the data types of `query`, `key`, and `value` are either `float16` or `bfloat16`, the
          flash-attention operator is used directly.
        - If `self.fa_force_dtype` is set to `float16` or `bfloat16`, the input tensors are cast to
          this data-type, the flash attention operator is applied, and the result is cast back to the
          original data type of `query`.
        - Otherwise, the function falls back to the mathematical formula-based attention.
    """
    fa_op_available = parse(ms.__version__) >= parse("2.3.0") and ms.get_context("device_target").lower() == "ascend"
    fa_force_dtype = ms.float16

    head_dim = query.shape[-1]

    if not fa_op_available:
        return math_attention_op(query, key, value, attn_mask)
    elif head_dim > 512:
        logger.warning("Flash attention requires that the head dimension must <= 512")
        return math_attention_op(query, key, value, attn_mask)
    elif query.dtype in (ms.float16, ms.bfloat16):
        return flash_attention_op(query, key, value, attn_mask, keep_prob=1 - dropout_p, scale=scale)
    elif query.dtype == ms.float32:
        return flash_attention_op(
            query.to(fa_force_dtype),
            key.to(fa_force_dtype),
            value.to(fa_force_dtype),
            attn_mask,
            keep_prob=1 - dropout_p,
            scale=scale,
        ).to(query.dtype)
    else:
        return math_attention_op(query, key, value, attn_mask)


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
                pad = mint.zeros((B, N, S, self.d_pad), x.dtype)
            else:
                B, S = x.shape[:2]
                x = x.reshape(B, S, -1, self.head_dim)
                pad = mint.zeros((B, S, x.shape[2], self.d_pad), x.dtype)
            x = mint.concat((x, pad), axis=-1)
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
