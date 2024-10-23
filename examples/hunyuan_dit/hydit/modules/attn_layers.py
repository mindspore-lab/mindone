import math
from typing import Optional, Tuple, Union

import mindspore as ms
from mindspore import nn, ops

from .norm_layers import LayerNorm

try:
    from mindspore.ops.operations.nn_ops import FlashAttentionScore as FlashAttention
except Exception as e:
    print(f"flash_attn import failed: {e}")


def reshape_for_broadcast(freqs_cis: Union[ms.Tensor, Tuple[ms.Tensor]], x: ms.Tensor, head_first=False):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (Union[ms.Tensor, Tuple[ms.Tensor]]): Frequency tensor to be reshaped.
        x (ms.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        ms.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            shape = tuple([d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)])
        else:
            shape = tuple([d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)])
        return freqs_cis[0].view(shape), freqs_cis[1].view(shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            shape = tuple([d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)])
        else:
            shape = tuple([d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)])
        return freqs_cis.view(shape)


def rotate_half(x):
    x_real, x_imag = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)  # [B, S, H, D//2]
    return ops.stack([-x_imag, x_real], axis=-1).flatten(start_dim=3)


def apply_rotary_emb(
    xq: ms.Tensor,
    xk: Optional[ms.Tensor],
    freqs_cis: Union[ms.Tensor, Tuple[ms.Tensor]],
    head_first: bool = False,
) -> Tuple[ms.Tensor, ms.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (ms.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (ms.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (Union[ms.Tensor, Tuple[ms.Tensor]]): Precomputed frequency tensor for complex exponentials.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[ms.Tensor, ms.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        if xk is not None:
            xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        xq_ = xq.float().reshape(xq.shape[:-1] + (-1, 2))  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_[:, :, :, :, 0], head_first)  # [S, D//2] --> [1, S, 1, D//2]
        xq_out = (xq_ * freqs_cis).flatten(3).type_as(xq)
        if xk is not None:
            xk_ = xk.float().reshape(xk.shape[:-1] + (-1, 2))  # [B, S, H, D//2]
            xk_out = (xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


class FlashSelfMHAModified(nn.Cell):
    """
    Use QK Normalization.
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        dtype=None,
        norm_layer=LayerNorm,
    ):
        factory_kwargs = {"dtype": dtype} if dtype else {}
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Dense(dim, 3 * dim, has_bias=qkv_bias, **factory_kwargs)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.inner_attn = FlashAttention(
            head_num=self.num_heads,
            keep_prob=1 - attn_drop,
            scale_value=1.0 / math.sqrt(self.head_dim),
            input_layout="BSND",
        )
        self.out_proj = nn.Dense(dim, dim, has_bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x, freqs_cis_img=None):
        """
        Parameters
        ----------
        x: ms.Tensor
            (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        freqs_cis_img: ms.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        dtype = x.dtype
        b, s, d = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)  # [b, s, 3, h, d]
        q, k, v = qkv.unbind(dim=2)  # [b, s, h, d]
        q = self.q_norm(q).half()  # [b, s, h, d]
        k = self.k_norm(k).half()

        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, kk = apply_rotary_emb(q, k, freqs_cis_img)
            q, k = qq, kk

        context = self.inner_attn(q, k, v.half())[-1].to(dtype)
        out = self.out_proj(context.view(b, s, d))
        out = self.proj_drop(out)

        out_tuple = (out,)

        return out_tuple


class FlashCrossMHAModified(nn.Cell):
    """
    Use QK Normalization.
    """

    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        dtype=None,
        norm_layer=LayerNorm,
    ):
        factory_kwargs = {"dtype": dtype} if dtype else {}
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Dense(qdim, qdim, has_bias=qkv_bias, **factory_kwargs)
        self.kv_proj = nn.Dense(kdim, 2 * qdim, has_bias=qkv_bias, **factory_kwargs)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

        self.inner_attn = FlashAttention(
            head_num=num_heads, keep_prob=1 - attn_drop, scale_value=self.scale, input_layout="BSND"
        )
        self.out_proj = nn.Dense(qdim, qdim, has_bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x, y, freqs_cis_img=None):
        """
        Parameters
        ----------
        x: ms.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num_heads * head_dim)
        y: ms.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: ms.Tensor
            (batch, hidden_dim // num_heads), RoPE for image
        """
        dtype = x.dtype
        b, s1, _ = x.shape  # [b, s1, D]
        _, s2, _ = y.shape  # [b, s2, 1024]

        q = self.q_proj(x).view(b, s1, self.num_heads, self.head_dim)  # [b, s1, h, d]
        kv = self.kv_proj(y).view(b, s2, 2, self.num_heads, self.head_dim)  # [b, s2, 2, h, d]
        k, v = kv.unbind(dim=2)  # [b, s2, h, d]
        q = self.q_norm(q).half()  # [b, s1, h, d]
        k = self.k_norm(k).half()  # [b, s2, h, d]
        v = v.half()  # [b, s2, h, d]

        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, _ = apply_rotary_emb(q, None, freqs_cis_img)
            q = qq  # [b, s1, h, d]

        context = self.inner_attn(q, k, v)[-1].to(dtype)  # [b, s1, h, d]
        context = context.view(b, s1, -1)  # [b, s1, D]

        out = self.out_proj(context)
        out = self.proj_drop(out)

        out_tuple = (out,)

        return out_tuple


class CrossAttention(nn.Cell):
    """
    Use QK Normalization.
    """

    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        dtype=None,
        norm_layer=LayerNorm,
    ):
        factory_kwargs = {"dtype": dtype} if dtype else {}
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Dense(qdim, qdim, has_bias=qkv_bias, **factory_kwargs)
        self.kv_proj = nn.Dense(kdim, 2 * qdim, has_bias=qkv_bias, **factory_kwargs)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.out_proj = nn.Dense(qdim, qdim, has_bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x, y, freqs_cis_img=None):
        """
        Parameters
        ----------
        x: ms.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: ms.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: ms.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s1, c = x.shape  # [b, s1, D]
        _, s2, c = y.shape  # [b, s2, 1024]

        q = self.q_proj(x).view(b, s1, self.num_heads, self.head_dim)  # [b, s1, h, d]
        kv = self.kv_proj(y).view(b, s2, 2, self.num_heads, self.head_dim)  # [b, s2, 2, h, d]
        k, v = kv.unbind(dim=2)  # [b, s, h, d]
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, _ = apply_rotary_emb(q, None, freqs_cis_img)
            q = qq

        q = q * self.scale
        q = q.swapaxes(-2, -3)  # q ->  B, L1, H, C - B, H, L1, C
        k = k.permute(0, 2, 3, 1)  # k ->  B, L2, H, C - B, H, C, L2
        attn = ops.matmul(q, k)  # attn -> B, H, L1, L2
        attn = ops.softmax(attn, axis=-1)  # attn -> B, H, L1, L2
        attn = self.attn_drop(attn)
        x = ops.matmul(attn, v.swapaxes(-2, -3))  # v -> B, L2, H, C - B, H, L2, C    x-> B, H, L1, C
        context = x.swapaxes(1, 2)  # context -> B, H, L1, C - B, L1, H, C

        context = context.view(b, s1, -1)

        out = self.out_proj(context)  # context.reshape - B, L1, -1
        out = self.proj_drop(out)

        out_tuple = (out,)

        return out_tuple


class Attention(nn.Cell):
    """
    We rename some layer names to align with flash attention
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "dim should be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim**-0.5

        # qkv --> Wqkv
        self.Wqkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.out_proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x, freqs_cis_img=None):
        B, N, C = x.shape
        qkv = self.Wqkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, b, h, s, d]
        q, k, v = qkv.unbind(0)  # [b, h, s, d]
        q = self.q_norm(q)  # [b, h, s, d]
        k = self.k_norm(k)  # [b, h, s, d]

        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, kk = apply_rotary_emb(q, k, freqs_cis_img, head_first=True)
            q, k = qq, kk

        q = q * self.scale
        attn = ops.matmul(q, k.swapaxes(-2, -1))  # [b, h, s, d] @ [b, h, d, s]
        attn = ops.softmax(attn, axis=-1)  # [b, h, s, s]
        attn = self.attn_drop(attn)
        x = ops.matmul(attn, v)  # [b, h, s, d]

        x = x.swapaxes(1, 2).reshape(B, N, C)  # [b, s, h, d]
        x = self.out_proj(x)
        x = self.proj_drop(x)

        out_tuple = (x,)

        return out_tuple
