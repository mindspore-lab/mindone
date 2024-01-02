import math
from typing import Optional

import numpy as np

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import initializer

from .util import GroupNorm32


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    weight = initializer("zeros", module.weight.shape)
    bias_weight = initializer("zeros", module.bias.shape)
    module.weight.set_data(weight)
    module.bias.set_data(bias_weight)
    return module


def get_motion_module(in_channels, motion_module_type: str, motion_module_kwargs: dict):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            **motion_module_kwargs,
        )
    else:
        raise ValueError


class VanillaTemporalModule(nn.Cell):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()

        # print("D---: zero_initialize in MM", zero_initialize)
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def construct(
        self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None, video_length=None
    ):
        """
        Args:
            hidden_states: x, (b*f c h w)
            temb: time embedding, not used
            encoder_hidden_states: prompt embedding, not used
        Returns:
            (b*f c h w)
        """
        dtype = input_tensor.dtype

        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(
            hidden_states, encoder_hidden_states, attention_mask, video_length=video_length
        )

        output = hidden_states

        output = output.to(dtype)  # TODO: need debug for fp16 or fp32
        return output


class FeedForward(nn.Cell):
    def __init__(
        self,
        dim: int,
        dim_out=None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        dtype=ms.float32,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, approximate=False, dtype=dtype)
        else:
            raise NotImplementedError

        self.net = nn.CellList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(p=dropout))
        # project out
        self.net.append(nn.Dense(inner_dim, dim_out).to_float(dtype))

    def construct(self, hidden_states: ms.Tensor, scale: float = 1.0) -> ms.Tensor:
        # TODO: simple use self.net(x)?
        for module in self.net:
            hidden_states = module(hidden_states)

        return hidden_states


# TODO: check
class GEGLU(nn.Cell):
    def __init__(self, dim_in, dim_out, approximate=False, dtype=ms.float32):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2).to_float(dtype)
        self.split = ops.Split(-1, 2)
        # self.gelu = ops.GeLU()
        self.gelu = nn.GELU(approximate=approximate)  # better precision

    def construct(self, x):
        x, gate = self.split(self.proj(x))

        return x * self.gelu(gate.to(ms.float32)).to(x.dtype)  # compute gelu in fp32 to align with torch


# TODO: check correctness
class LayerNorm32(nn.LayerNorm):
    def construct(self, x, dtype=ms.float32):
        ori_dtype = x.dtype
        out = super().construct(x.astype(dtype))

        return out.astype(ori_dtype)


class TemporalTransformer3DModel(ms.nn.Cell):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        dtype=ms.float32,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = GroupNorm32(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )  # TODO: any diff in beta gamma init?
        self.proj_in = nn.Dense(in_channels, inner_dim).to_float(dtype)
        # cur
        self.transformer_blocks = nn.CellList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    dtype=dtype,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Dense(inner_dim, in_channels).to_float(dtype)

    def construct(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        """
        Args:
            hidden_states: x, (b*f c h w)
        Returns:
            (b*f c h w)
        """
        # assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        # video_length = hidden_states.shape[2]
        # hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        assert video_length is not None, "video_length is needed for motion module,but got None"

        batch, channel, height, weight = hidden_states.shape  # batch = b * f
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length)

        # output
        hidden_states = self.proj_out(hidden_states)
        # TODO: ms don't have contiguous API, which is for memory-access efficiency
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2)  # .contiguous()

        output = hidden_states + residual

        # output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output


class TemporalTransformerBlock(nn.Cell):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        dtype=ms.float32,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    dtype=dtype,
                )
            )
            norms.append(LayerNorm32([dim], epsilon=1.0e-5))  # TODO: need fp32?

        self.attention_blocks = nn.CellList(attention_blocks)
        self.norms = nn.CellList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        # TODO: check correctness
        self.ff_norm = LayerNorm32([dim], epsilon=1.0e-5)  # TODO: need fp32?

    def construct(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            # print("D--: attention_block.is_cross_attention ", attention_block.is_cross_attention)
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                    video_length=video_length,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(ms.nn.Cell):
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = ms.nn.Dropout(p=dropout)
        position = np.expand_dims(np.arange(max_len), 1)

        div_term = np.exp((np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).astype(np.float32))
        pe = np.zeros((1, max_len, d_model), dtype=np.float32)

        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)

        self.pe = ms.Parameter(ms.Tensor(pe, dtype=ms.float32), name="pe", requires_grad=False)

    def construct(self, x):
        # x: (b*d f c)
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class VersatileAttention(ms.nn.Cell):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        dtype=ms.float32,
    ):
        super().__init__()
        assert attention_mode == "Temporal"

        inner_dim = dim_head * heads

        self.is_cross_attention = cross_attention_dim is not None  # MUST be placed before
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            # TODO:
            self.group_norm = GroupNorm32(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = ms.nn.Dense(query_dim, inner_dim, has_bias=bias).to_float(dtype)
        self.to_k = ms.nn.Dense(cross_attention_dim, inner_dim, has_bias=bias).to_float(dtype)
        self.to_v = ms.nn.Dense(cross_attention_dim, inner_dim, has_bias=bias).to_float(dtype)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Dense(added_kv_proj_dim, cross_attention_dim).to_float(dtype)
            self.add_v_proj = nn.Dense(added_kv_proj_dim, cross_attention_dim).to_float(dtype)

        # self.to_out = ms.nn.CellList([])
        # self.to_out.append(nn.Dense(inner_dim, query_dim).to_float(dtype))
        # self.to_out.append(nn.Dropout(p=dropout))

        self.to_out = ms.nn.SequentialCell(
            nn.Dense(inner_dim, query_dim).to_float(dtype),
            nn.Dropout(p=dropout),
        )
        self.attention_mode = attention_mode

        # TODO: adapt
        self.pos_encoder = (
            PositionalEncoding(query_dim, dropout=0.0, max_len=temporal_position_encoding_max_len)
            if (temporal_position_encoding and attention_mode == "Temporal")
            else None
        )

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def construct(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        """
        Args:
            hidden_states: (b*f h*w c),
        Returns:
        """
        bs, sequence_length, ftr_dim = hidden_states.shape

        def _rearrange_in(x, f=None):
            # (b*f d c) -> (b f d c) -> (b d f c) -> (b*d f c)
            bf, d, c = x.shape
            b = bf // f
            x = x.reshape(b, f, d, c).permute(0, 2, 1, 3).reshape(b * d, f, c)
            return x

        def _rearrange_out(x, d=None):
            #  (b*d f c) -> (b d f c) -> (b f d c) -> (b*f d c)
            bd, f, c = x.shape
            b = bd // d
            x = x.reshape(b, d, f, c).permute(0, 2, 1, 3).reshape(b * f, d, c)
            return x

        d = hidden_states.shape[1]
        # hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        hidden_states = _rearrange_in(hidden_states, f=video_length)

        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        # encoder_hidden_states = encoder_hidden_states
        # encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        # dim = query.shape[-1]

        # (b*d f attn_dim*num_heads) -> (b*d*num_heads f attn_dim), where c=attn_dim*num_heads
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                # TODO:
                attention_mask = ops.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        hidden_states = self._attention(query, key, value, attention_mask)

        # linear proj
        # TODO: merge with dropout using Sequential?
        hidden_states = self.to_out(hidden_states)

        # dropout
        # hidden_states = self.to_out[1](hidden_states)

        # hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
        hidden_states = _rearrange_out(hidden_states, d=d)

        return hidden_states

    def _attention(self, query, key, value, attention_mask=None):
        """
        Args:
            q, k, v shape: (b*d*num_heads f attn_dim)
        Returns:
            out: (b*d f attn_dim*num_heads) = (b*d f c)
        """
        if self.upcast_attention:
            query = query.to_float(ms.float32)
            key = key.to_float(ms.float32)

        attention_scores = ms.ops.matmul(query, ms.ops.transpose(key, (0, 2, 1))) * self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.to_float(ms.float32)

        # TODO: compute in fp32?
        attention_probs = ms.ops.softmax(attention_scores, axis=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = ms.ops.matmul(attention_probs, value)  # TODO: torch use bmm

        # reshape hidden_states
        # (b*d*num_heads f attn_dim) -> (b*d f attn_dim*num_heads) = (b*d f c)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        return hidden_states
