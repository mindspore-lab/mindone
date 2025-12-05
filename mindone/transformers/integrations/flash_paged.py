from typing import Optional

import mindspore as ms


def paged_attention_forward(
    module: ms.nn.Cell,
    q: ms.Tensor,
    k: ms.Tensor,
    v: ms.Tensor,
    attention_mask: Optional[ms.Tensor] = None,
    **kwargs,
) -> ms.Tensor:
    r"""Perform the forward pass of attention with paged key-value cache.

    This function handles the cache updates and performs the attention computation
    using the flash_attn_varlen_func for efficient processing.

    Args:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.  but if there is a block table it can be the full k
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.  but if there is a block table it can be the full v
        kwargs:
        batch_valid_length, block_tables, slot_mapping are needed to do mapping between cache getting/saving and physical blocks
    """
    if not hasattr(module, "infer_attention"):
        raise NotImplementedError(
            "This model has not supported PagedAttenion yet. Please refer to qwen3 setting and supplement infer_attention as well."
        )

    attn_output = module.infer_attention(
        q,
        k,
        v,
        kwargs["batch_valid_length"],
        kwargs["block_tables"],
        kwargs["slot_mapping"],
        kwargs["freqs_cis"],
        kwargs["mask"],
        q_seq_lens=None,
    )

    return attn_output, None
