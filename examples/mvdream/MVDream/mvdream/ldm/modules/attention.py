from sv3d.sgm.modules.attention import BasicTransformerBlock, SpatialTransformer

from mindspore import nn

try:
    from mindone.models.modules.flash_attention import FLASH_IS_AVAILABLE

    print("flash attention is available under mvdream.")
except ImportError:
    FLASH_IS_AVAILABLE = False
    print("flash attention is unavailable under mvdream.")


def exists(val):
    return val is not None


class BasicTransformerBlock3D(BasicTransformerBlock):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0,
        context_dim=None,
        gated_ff=True,
        disable_self_attn=False,
        attn_mode="softmax",
        use_recompute=False,
    ):
        attn_mode = "flash-attention" if FLASH_IS_AVAILABLE else "vanilla"
        super().__init__(dim, n_heads, d_head, dropout, context_dim, gated_ff, disable_self_attn, attn_mode)
        if use_recompute:
            self.recompute()

    def construct(self, x, context=None, num_frames=1):
        # x = rearrange(x, "(b f) l c -> b (f l) c", f=num_frames).contiguous()
        x = x.view(x.shape[0] // num_frames, num_frames * x.shape[1], x.shape[2])
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        # x = rearrange(x, "b (f l) c -> (b f) l c", f=num_frames).contiguous()
        x = x.view(int(x.shape[0] * num_frames), x.shape[1] // num_frames, x.shape[2])
        x = self.attn2(self.norm2(x), context=context) + x
        a = self.ff(self.norm3(x))
        x = a + x
        return x


class SpatialTransformer3D(SpatialTransformer):
    """3D self-attention"""

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_recompute=False,
        attn_type="vanilla",
    ):
        super().__init__(
            in_channels, n_heads, d_head, depth, dropout, context_dim, disable_self_attn, use_linear, attn_type
        )
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        inner_dim = n_heads * d_head
        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock3D(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    use_recompute=use_recompute,
                    attn_mode=attn_type,
                )
                for d in range(depth)
            ]
        )

    def construct(self, x, context=None, num_frames=1):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        # x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = x.view(*x.shape[:2], -1).transpose(0, 2, 1)
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], num_frames=num_frames)
        if self.use_linear:
            x = self.proj_out(x)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = x.transpose(0, 2, 1)
        x = x.view(*x.shape[:2], h, w)
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
