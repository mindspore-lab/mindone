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
        dropout=1,
        dtype=ms.float32,
        enable_flash_attention=False,
        upcast=False,
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
        )
        self.ip_scale = ip_scale
        self.num_tokens = num_tokens
        inner_dim = dim_head * heads
        self.to_k_ip = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_v_ip = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)

    def _rearange_in(self, x):
        # (b, n, h*d) -> (b*h, n, d)
        h = self.heads
        b, n, d = x.shape
        d = d // h

        x = self.reshape(x, (b, n, h, d))
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (b * h, n, d))
        return x

    def _rearange_out(self, x):
        # (b*h, n, d) -> (b, n, h*d)
        h = self.heads
        b, n, d = x.shape
        b = b // h

        x = self.reshape(x, (b, h, n, d))
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (b, n, h * d))
        return x

    def _cal_z(self, q, context, mask=None, ip_branch=False):
        if ip_branch:
            k = self.to_k_ip(context)
            v = self.to_v_ip(context)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        q = self._rearange_in(q)
        k = self._rearange_in(k)
        v = self._rearange_in(v)

        if self.use_flash_attention and q.shape[1] % 16 == 0 and k.shape[1] % 16 == 0:
            out = self.flash_attention(q, k, v)
        else:
            out = self.attention(q, k, v, mask)

        out = self._rearange_out(out)
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
