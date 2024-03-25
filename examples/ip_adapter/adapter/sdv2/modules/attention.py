from ldm.modules.attention import (
    BasicTransformerBlock,
    CrossAttention,
    FeedForward,
    Normalize,
    SpatialTransformer,
    default,
    zero_module,
)

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class IPAdapterCrossAttention(CrossAttention):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=1.0,
        dtype=ms.float32,
        enable_flash_attention=False,
        upcast=False,
        fa_max_head_dim=256,
        ip_scale=1.0,
        num_tokens=4,
    ):
        super().__init__(
            query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            dtype=dtype,
            enable_flash_attention=enable_flash_attention,
            upcast=upcast,
            fa_max_head_dim=fa_max_head_dim,
        )
        self.ip_scale = ip_scale
        self.num_tokens = num_tokens
        inner_dim = dim_head * heads
        self.to_k_ip = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_v_ip = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)

    def _cal_z(self, q, context, mask=None, ip_branch=False):
        h = self.heads

        if ip_branch:
            k = self.to_k_ip(context)
            v = self.to_v_ip(context)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        q_b, q_n, _ = q.shape  # (b n h*d)
        k_b, k_n, _ = k.shape
        v_b, v_n, _ = v.shape

        head_dim = q.shape[-1] // self.heads

        if (
            self.enable_flash_attention and q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= self.fa_max_head_dim
        ):  # restrict head_dim to avoid UB oom. Reduce fa_max_head_dim value in case of OOM.
            # reshape qkv shape ((b n h*d) -> (b h n d)) and mask dtype for FA input format
            q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
            k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
            v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)
            if mask is None:
                mask = ops.zeros((q_b, q_n, q_n), self.fa_mask_dtype)

            out = self.flash_attention(
                q.to(ms.float16), k.to(ms.float16), v.to(ms.float16), mask.to(self.fa_mask_dtype)
            )

            b, h, n, _ = out.shape
            # reshape FA output to original attn input format, (b h n d) -> (b n h*d)
            out = out.transpose(0, 2, 1, 3).view(b, n, -1)
        else:
            # (b, n, h*d) -> (b*h, n, d)
            q = self._rearange_in(q, h)
            k = self._rearange_in(k, h)
            v = self._rearange_in(v, h)

            out = self.attention(q, k, v, mask)
            # (b*h, n, d) -> (b, n, h*d)
            out = self._rearange_out(out, h)

        return out

    def construct(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)

        end_pos = context.shape[1] - self.num_tokens
        context, ip_context = (
            context[:, :end_pos, :],
            context[:, end_pos:, :],
        )

        z = self._cal_z(q, context, mask, False)
        z_ip = self._cal_z(q, ip_context, mask, True)
        out = z + self.ip_scale * z_ip

        return self.to_out(out)


class IPAdapterBasicTransformerBlock(BasicTransformerBlock):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=1.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        dtype=ms.float32,
        enable_flash_attention=False,
        upcast_attn=False,
        fa_max_head_dim=256,
        ip_scale=1.0,
        num_tokens=4,
    ):
        super(BasicTransformerBlock, self).__init__()
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
            enable_flash_attention=enable_flash_attention,
            upcast=upcast_attn,
            fa_max_head_dim=fa_max_head_dim,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype)
        self.attn2 = IPAdapterCrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
            enable_flash_attention=enable_flash_attention,
            upcast=upcast_attn,
            fa_max_head_dim=fa_max_head_dim,
            ip_scale=ip_scale,
            num_tokens=num_tokens,
        )  # is self-attn if context is none
        self.norm1 = (
            nn.LayerNorm([dim], epsilon=1e-05).to_float(ms.float32)
            if upcast_attn
            else nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        )
        self.norm2 = (
            nn.LayerNorm([dim], epsilon=1e-05).to_float(ms.float32)
            if upcast_attn
            else nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        )
        self.norm3 = (
            nn.LayerNorm([dim], epsilon=1e-05).to_float(ms.float32)
            if upcast_attn
            else nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        )
        self.checkpoint = checkpoint


class IPAdapterSpatialTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=1.0,
        context_dim=None,
        use_checkpoint=True,
        use_linear=False,
        dtype=ms.float32,
        enable_flash_attention=False,
        fa_max_head_dim=256,
        upcast_attn=False,
        ip_scale=1.0,
        num_tokens=4,
    ):
        super(SpatialTransformer, self).__init__()
        self.in_channels = in_channels
        self.dtype = dtype
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="pad"
            ).to_float(dtype)
        else:
            self.proj_in = nn.Dense(in_channels, inner_dim).to_float(dtype)

        self.transformer_blocks = nn.CellList(
            [
                IPAdapterBasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    checkpoint=use_checkpoint,
                    dtype=self.dtype,
                    enable_flash_attention=enable_flash_attention,
                    upcast_attn=upcast_attn,
                    fa_max_head_dim=fa_max_head_dim,
                    ip_scale=ip_scale,
                    num_tokens=num_tokens,
                )
                for d in range(depth)
            ]
        )

        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(
                    inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="pad"
                ).to_float(self.dtype)
            )
        else:
            self.proj_out = zero_module(nn.Dense(in_channels, inner_dim).to_float(dtype))

        self.use_linear = use_linear
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
