# Adapted from https://github.com/hpcaitech/Open-Sora/blob/main/opensora/models/mmdit/math.py

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import mint


def liger_rope(pos: Tensor, dim: int, theta: int) -> tuple[Tensor, Tensor]:
    assert dim % 2 == 0
    scale = mint.arange(0, dim, 2, dtype=mstype.float32) / dim
    omega = 1.0 / (theta**scale)
    out = mint.einsum("...n,d->...nd", pos, omega)  # (b, seq, dim//2)
    cos = out.cos()
    sin = out.sin()

    return cos, sin


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = mint.arange(0, dim, 2, dtype=mstype.float64) / dim
    omega = 1.0 / (theta**scale)
    out = mint.einsum("...n,d->...nd", pos, omega)
    out = mint.stack([mint.cos(out), -mint.sin(out), mint.sin(out), mint.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:3], 2, 2)  # b n d (i j) -> b n d i j
    return out.to(mstype.float32)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.to(mstype.float32).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(mstype.float32).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mint.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim=1) -> tuple[Tensor, Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`Tensor`): The query tensor.
        k (`Tensor`): The key tensor.
        cos (`Tensor`): The cosine part of the rotary embedding.
        sin (`Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.type_as(q), k_embed.type_as(k)


def rearrange_tensor(tensor: Tensor) -> Tensor:
    """
    Rearranges the last dimension (D) of the input tensor based on the specified mapping:
    2d -> d, 2d+1 -> D/2 + d.

    Args:
        tensor (Tensor): Input tensor of shape [B, H, L, D], where D is even.

    Returns:
        Tensor: Tensor with rearranged last dimension, same shape as input.
    """
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")

    half_D = D // 2
    indices = mint.empty(D, dtype=mstype.int64)

    # Fill the indices based on the mapping rule
    indices[:half_D] = mint.arange(0, D, 2)
    indices[half_D:] = mint.arange(1, D, 2)

    # Rearrange the tensor based on the computed indices
    return tensor.index_select(dim=-1, index=indices)


def reverse_rearrange_tensor(tensor: Tensor) -> Tensor:
    """
    Restores the original order of the last dimension (D) of the input tensor based on the reverse mapping:
    d -> 2d, D/2 + d -> 2d + 1.

    Args:
        tensor (Tensor): Input tensor of shape [B, H, L, D], where D is even.

    Returns:
        Tensor: Tensor with restored original last dimension order, same shape as input.
    """
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")

    half_D = D // 2
    reverse_indices = mint.empty(D, dtype=mstype.int64)

    # Fill the reverse indices to restore the original order
    reverse_indices[::2] = mint.arange(half_D)
    reverse_indices[1::2] = mint.arange(half_D, D)

    # Rearrange the tensor based on the reverse indices
    return tensor.index_select(dim=-1, index=reverse_indices)
