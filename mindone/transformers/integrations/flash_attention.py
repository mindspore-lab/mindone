"""Adapted from https://github.com/huggingface/transformers/tree/main/src/transformers/integrations/flash_attention.py."""

from typing import Optional

from transformers.utils import logging

import mindspore as ms
from mindspore import mint, nn

from ..modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)


def flash_attention_forward(
    module: nn.Cell,
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    dropout: float = 0.0,
    scaling: float = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[ms.Tensor, None]:
    """
    Flash attention forward function. This function is a wrapper for `flash_attention_score` in MindSpore. It is used
    to calculate the attention score of the query and key, and then apply the attention score to the value.
    Args:
        module (`ms.Cell``):
            The attention module to be applied to the attention score.
        query (`ms.Tensor`):
            The query tensor of shape `(batch_size, num_head, seq_length, head_dim)`.
        key (`ms.Tensor`):
            The key tensor of shape `(batch_size, num_head, seq_length, head_dim)`.
        value (`ms.Tensor`):
            The value tensor of shape `(batch_size, num_head, seq_length, head_dim)`.
        attention_mask (`ms.Tensor`):
            The attention mask tensor of bool or uint8. For each element, 0/False indicates discard and 1/True
            indicates retention.The shape is `(batch_size, num_head, seq_length_q, seq_length_k)`,
            `(batch_size, 1, seq_length_q, seq_length_k)` or `(seq_length_q, seq_length_k)`.
            Default to `None`, which means no attention mask is applied.
        scaling (`float`, *required*):
            The scaling factor for the attention score.
        sliding_window (`int`):
            The sliding window size of self-attention. Default to `None`.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2. Default to `None`.

    """
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    if (
        kwargs.get("position_ids", None) is not None
        and query.shape[0] == 1
        and (
            kwargs.get("max_length_q", None) is not None
            or (query.shape[2] != 1 and not (mint.diff(kwargs["position_ids"], dim=-1) >= 0).all())
        )
    ):
        raise RuntimeError("FlashAttention's variable-length attention is not available.")
    if all(
        kwarg is not None
        for kwarg in (
            kwargs.get("cu_seq_lens_q", None),
            kwargs.get("cu_seq_lens_k", None),
            kwargs.get("max_length_q", None),
            kwargs.get("max_length_k", None),
        )
    ):
        raise RuntimeError("FlashAttention's variable-length attention is not available.")
    if sliding_window is not None:
        raise NotImplementedError("Sliding window is not supported in Mindspore yet. Please set `sliding_window=None`.")
    if softcap is not None:
        raise NotImplementedError("Softcap is not supported in Mindspore yet. Please set `softcap=None`.")
    if scaling is None:
        # `flash_attention_score` does not support `None`
        # and the value can't be set in jit mode, thus must be set in advance
        raise ValueError("`scaling` must be provided.")

    # BNSD -> BSND
    query = query.swapaxes(1, 2)
    key = key.swapaxes(1, 2)
    value = value.swapaxes(1, 2)
    input_layout = "BSND"

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        input_layout=input_layout,
        **kwargs,
    )

    return attn_output, None
