import math
from typing import Optional

import mindspore as ms
from mindspore import mint, ops


def dispatch_attention_fn(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    # Note: PyTorch's SDPA and MindSpore's FA handle `attention_mask` slightly differently.
    # In PyTorch, if the mask is not boolean (e.g., float32 with 0/1 values), it is interpreted
    # as an additive bias: `attn_bias = attn_mask + attn_bias`.
    # This implicit branch may lead to issues if the pipeline mistakenly provides
    # a 0/1 float mask instead of a boolean mask.
    # While this behavior is consistent with HF Diffusers for now,
    # it may still be a potential bug source worth validating.
    if attn_mask is not None and attn_mask.dtype != ms.bool_ and 1.0 in attn_mask:
        L, S = query.shape[-2], key.shape[-2]
        scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
        attn_bias = mint.zeros((L, S), dtype=query.dtype)
        if is_causal:
            if attn_mask is not None:
                if attn_mask.dtype == ms.bool_:
                    attn_mask = mint.logical_and(attn_mask, mint.ones((L, S), dtype=ms.bool_).tril(diagonal=0))
                else:
                    attn_mask = attn_mask + mint.triu(
                        mint.full((L, S), float("-inf"), dtype=attn_mask.dtype), diagonal=1
                    )
            else:
                temp_mask = mint.ones((L, S), dtype=ms.bool_).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias = attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == ms.bool_:
                attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        attn_weight = mint.matmul(query, key.swapaxes(-2, -1)) * scale_factor
        attn_weight += attn_bias
        attn_weight = mint.softmax(attn_weight, dim=-1)
        attn_weight = ops.dropout(attn_weight, dropout_p, training=True)
        return mint.matmul(attn_weight, value).permute(0, 2, 1, 3)

    if query.dtype in (ms.float16, ms.bfloat16):
        out = flash_attention_op(query, key, value, attn_mask, keep_prob=1 - dropout_p, scale=scale)
    else:
        out = flash_attention_op(
            query.to(ms.float16),
            key.to(ms.float16),
            value.to(ms.float16),
            attn_mask,
            keep_prob=1 - dropout_p,
            scale=scale,
        ).to(query.dtype)
    return out.permute(0, 2, 1, 3)


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
    if scale is None:
        scale = query.shape[-1] ** (-0.5)

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
