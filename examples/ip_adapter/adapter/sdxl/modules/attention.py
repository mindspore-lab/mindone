from gm.modules.attention import (
    FLASH_IS_AVAILABLE,
    BasicTransformerBlock,
    CrossAttention,
    FeedForward,
    MemoryEfficientCrossAttention,
    SpatialTransformer,
)
from gm.modules.diffusionmodules.util import normalization, zero_module
from gm.modules.transformers import scaled_dot_product_attention
from gm.util import default, exists

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class IPAdapterMemoryEfficientCrossAttention(MemoryEfficientCrossAttention):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, ip_scale=1.0):
        super().__init__(query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ip_scale = ip_scale
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.to_k_ip = nn.Dense(context_dim, inner_dim, has_bias=False)
        self.to_v_ip = nn.Dense(context_dim, inner_dim, has_bias=False)

    def _cal_z(self, q, h, context, mask=None, additional_tokens=None, n_tokens_to_mask=0, ip_branch=False):
        if ip_branch:
            k = self.to_k_ip(context)
            v = self.to_v_ip(context)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        # rearange_in, "b n (h d) -> b h n d"
        q_b, q_n, _ = q.shape
        q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
        k_b, k_n, _ = k.shape
        k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
        v_b, v_n, _ = v.shape
        v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)

        head_dim = q.shape[-1]
        if q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= 256:
            if mask is None:
                mask = ops.zeros((q_b, q_n, q_n), q.dtype)
            out = self.flash_attention(q, k, v, mask)
        else:
            out = scaled_dot_product_attention(q, k, v, attn_mask=mask)  # scale is dim_head ** -0.5 per default

        # rearange_out, "b h n d -> b n (h d)"
        b, h, n, _ = out.shape
        out = out.transpose(0, 2, 1, 3).view(b, n, -1)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return out

    def construct(self, x, context=None, mask=None, additional_tokens=None):
        h = self.heads

        n_tokens_to_mask = 0
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = ops.concat((additional_tokens, x), axis=1)

        q = self.to_q(x)

        if context is None:
            context = x
            ip_context = x
        else:
            end_pos = context.shape[1] - self.num_tokens
            context, ip_context = (
                context[:, :end_pos, :],
                context[:, end_pos:, :],
            )

        z = self._cal_z(q, h, context, mask, additional_tokens, n_tokens_to_mask, False)
        z_ip = self._cal_z(q, h, ip_context, mask, additional_tokens, n_tokens_to_mask, True)
        out = z + self.ip_scale * z_ip

        return self.to_out(out)


class IPAdapterCrossAttention(CrossAttention):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, ip_scale=1.0, num_tokens=4):
        super().__init__(query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ip_scale = ip_scale
        self.num_tokens = num_tokens
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.to_k_ip = nn.Dense(context_dim, inner_dim, has_bias=False)
        self.to_v_ip = nn.Dense(context_dim, inner_dim, has_bias=False)

    def _cal_z(self, q, h, context, mask=None, additional_tokens=None, n_tokens_to_mask=0, ip_branch=False):
        if ip_branch:
            k = self.to_k_ip(context)
            v = self.to_v_ip(context)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        # b n (h d) -> b h n d
        q_b, q_n, _ = q.shape
        q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
        k_b, k_n, _ = k.shape
        k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
        v_b, v_n, _ = v.shape
        v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)

        out = scaled_dot_product_attention(q, k, v, attn_mask=mask)  # scale is dim_head ** -0.5 per default

        # b h n d -> b n (h d)
        b, h, n, _ = out.shape
        out = out.transpose(0, 2, 1, 3).view(b, n, -1)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return out

    def construct(self, x, context=None, mask=None, additional_tokens=None):
        h = self.heads

        n_tokens_to_mask = 0
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = ops.concat((additional_tokens, x), axis=1)

        q = self.to_q(x)

        if context is None:
            context = x
            ip_context = x
        else:
            end_pos = context.shape[1] - self.num_tokens
            context, ip_context = (
                context[:, :end_pos, :],
                context[:, end_pos:, :],
            )

        z = self._cal_z(q, h, context, mask, additional_tokens, n_tokens_to_mask, False)
        z_ip = self._cal_z(q, h, ip_context, mask, additional_tokens, n_tokens_to_mask, True)

        out = z + self.ip_scale * z_ip

        return self.to_out(out)


class IPAdapterBasicTransformerBlock(BasicTransformerBlock):
    # overwrite the original attention modes
    IPADAPTER_ATTENTION_MODES = {
        "vanilla": IPAdapterCrossAttention,  # vanilla attention
        "flash-attention": IPAdapterMemoryEfficientCrossAttention,  # flash attention
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        disable_self_attn=False,
        attn_mode="vanilla",  # ["vanilla", "flash-attention"]
        ip_scale=1.0,
    ):
        super(BasicTransformerBlock, self).__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "vanilla" and not FLASH_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in MindSpore >= 2.0.1 on Ascend devices; "
                f"FYI, you are running with MindSpore version {ms.__version__}"
            )
            attn_mode = "vanilla"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        ip_adapter_attn_cls = self.IPADAPTER_ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # apply IPAdapter to attn2 only
        self.attn2 = ip_adapter_attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            ip_scale=ip_scale,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm([dim], epsilon=1e-5)
        self.norm2 = nn.LayerNorm([dim], epsilon=1e-5)
        self.norm3 = nn.LayerNorm([dim], epsilon=1e-5)


class IPAdapterSpatialTransformer(SpatialTransformer):
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
        attn_type="vanilla",
        ip_scale=1.0,
    ):
        super(SpatialTransformer, self).__init__()
        print(f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads")
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels, eps=1e-6)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="valid"
            )
        else:
            self.proj_in = nn.Dense(in_channels, inner_dim)

        self.transformer_blocks = nn.CellList(
            [
                IPAdapterBasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    ip_scale=ip_scale,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="valid")
            )
        else:
            self.proj_out = zero_module(nn.Dense(inner_dim, in_channels))
        self.use_linear = use_linear
