# Diffusersのコードをベースとした sd_xl_baseのU-Net
# state dictの形式をSDXLに合わせてある

"""
      target: sgm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        adm_in_channels: 2816
        num_classes: sequential
        use_checkpoint: True
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [4, 2]
        num_res_blocks: 2
        channel_mult: [1, 2, 4]
        num_head_channels: 64
        use_spatial_transformer: True
        use_linear_in_transformer: True
        transformer_depth: [1, 2, 10]  # note: the first is unused (due to attn_res starting at 2) 32, 16, 8 --> 64, 32, 16
        context_dim: 2048
        spatial_transformer_attn_type: softmax-xformers
        legacy: False
"""

import logging
import math
from types import SimpleNamespace
from typing import Optional

import mindspore as ms
from mindspore import Tensor, nn, ops

from mindone.models.modules.flash_attention import MSFlashAttention

logger = logging.getLogger(__name__)

IN_CHANNELS: int = 4
OUT_CHANNELS: int = 4
ADM_IN_CHANNELS: int = 2816
CONTEXT_DIM: int = 2048
MODEL_CHANNELS: int = 320
TIME_EMBED_DIM = 320 * 4

USE_REENTRANT = True

# region memory efficient attention

# FlashAttentionを使うCrossAttention

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def get_parameter_dtype(parameter: nn.Cell):
    return next(parameter.parameters()).dtype


def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * ops.arange(start=0, end=half_dim, dtype=ms.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = ops.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings: flipped from Diffusers original ver because always flip_sin_to_cos=True
    emb = ops.cat([ops.cos(emb), ops.sin(emb)], axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = ops.pad(emb, (0, 1, 0, 0))
    return emb


# Deep Shrink: We do not common this function, because minimize dependencies.
def resize_like(x, target, mode="bicubic", align_corners=False):
    org_dtype = x.dtype
    if org_dtype == ms.bfloat16:
        x = x.to(ms.float32)

    if x.shape[-2:] != target.shape[-2:]:
        if mode == "nearest":
            x = ops.interpolate(x, size=target.shape[-2:], mode=mode)
        else:
            x = ops.interpolate(x, size=target.shape[-2:], mode=mode, align_corners=align_corners)

    if org_dtype == ms.bfloat16:
        x = x.to(org_dtype)
    return x


class GroupNorm32(nn.GroupNorm):
    def construct(self, x):
        # if self.weight.dtype != ms.float32:
        #     return super().construct(x)
        return super().construct(x.float()).type(x.dtype)


class ResnetBlock2D(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_layers = nn.SequentialCell(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
        )

        self.emb_layers = nn.SequentialCell(nn.SiLU(), nn.Dense(TIME_EMBED_DIM, out_channels))

        self.out_layers = nn.SequentialCell(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Identity(),  # to make state_dict compatible with original model
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
        )

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
            )
        else:
            self.skip_connection = nn.Identity()

        self._gradient_checkpointing = False

    @property
    def gradient_checkpointing(self):
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._gradient_checkpointing = value
        self.recompute()

    def construct(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        x = self.skip_connection(x)
        return x + h


class Downsample2D(nn.Cell):
    def __init__(self, channels, out_channels):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels

        self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1, pad_mode="pad", has_bias=True)

        self._gradient_checkpointing = False

    @property
    def gradient_checkpointing(self):
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._gradient_checkpointing = value
        self.recompute()

    def construct(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.op(hidden_states)

        return hidden_states


class CrossAttention(nn.Cell):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False)
        self.to_k = nn.Dense(cross_attention_dim, inner_dim, has_bias=False)
        self.to_v = nn.Dense(cross_attention_dim, inner_dim, has_bias=False)

        self.to_out = nn.CellList([])
        self.to_out.append(nn.Dense(inner_dim, query_dim))
        # no dropout here

        self.use_memory_efficient_attention_xformers = False
        self.use_memory_efficient_attention_mem_eff = False
        self.use_sdpa = False
        self.flash_attention = MSFlashAttention(
            head_dim=self.dim_head,
            head_num=self.heads,
            fix_head_dims=[72],
            attention_dropout=0.0,
        )

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        self.use_memory_efficient_attention_xformers = xformers
        self.use_memory_efficient_attention_mem_eff = mem_eff

    def set_use_sdpa(self, sdpa):
        self.use_sdpa = sdpa

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

    def construct(self, hidden_states, context=None, mask=None):
        if self.use_memory_efficient_attention_xformers:
            return self.forward_memory_efficient_fa(hidden_states, context, mask)
        if self.use_memory_efficient_attention_mem_eff:
            return self.forward_memory_efficient_mem_eff(hidden_states, context, mask)
        if self.use_sdpa:
            return self.forward_sdpa(hidden_states, context, mask)

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        hidden_states = self._attention(query, key, value)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # hidden_states = self.to_out[1](hidden_states)     # no dropout
        return hidden_states

    def _attention(self, query, key, value):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = ops.baddbmm(
            ms.numpy.empty((query.shape[0], query.shape[1], key.shape[1]), dtype=query.dtype),
            query,
            ops.transpose(key, (0, 2, 1)),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = ops.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    # TODO support Hypernetworks
    def forward_memory_efficient_fa(self, x, context=None, mask=None):
        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)

        b, n, _ = q_in.shape
        q = q_in.reshape(b, n, h, -1).swapaxes(1, 2)
        b, n, _ = k_in.shape
        k = k_in.reshape(b, n, h, -1).swapaxes(1, 2)
        b, n, _ = v_in.shape
        v = v_in.reshape(b, n, h, -1).swapaxes(1, 2)

        out = self.flash_attention(q, k, v, mask=mask)  # 最適なのを選んでくれる
        out = out.swapaxes(1, 2)
        b, n, _, _ = out.shape
        out = out.reshape(b, n, -1)

        out = self.to_out[0](out)
        return out

    # def construct_memory_efficient_mem_eff(self, x, context=None, mask=None):
    #     flash_func = FlashAttentionFunction

    #     q_bucket_size = 512
    #     k_bucket_size = 1024

    #     h = self.heads
    #     q = self.to_q(x)
    #     context = context if context is not None else x
    #     context = context.to(x.dtype)
    #     k = self.to_k(context)
    #     v = self.to_v(context)
    #     del context, x

    #     q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

    #     out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

    #     out = rearrange(out, "b h n d -> b n (h d)")

    #     out = self.to_out[0](out)
    #     return out

    # def construct_sdpa(self, x, context=None, mask=None):
    #     h = self.heads
    #     q_in = self.to_q(x)
    #     context = context if context is not None else x
    #     context = context.to(x.dtype)
    #     k_in = self.to_k(context)
    #     v_in = self.to_v(context)

    #     q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q_in, k_in, v_in))
    #     del q_in, k_in, v_in

    #     out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

    #     out = rearrange(out, "b h n d -> b n (h d)", h=h)

    #     out = self.to_out[0](out)
    #     return out


# feedforward
class GEGLU(nn.Cell):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2)

    def gelu(self, gate):
        # if gate.device.type != "mps":
        #     return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return ops.gelu(gate.to(dtype=ms.float32)).to(dtype=gate.dtype)

    def construct(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, axis=-1)
        return hidden_states * self.gelu(gate)


class FeedForward(nn.Cell):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        inner_dim = int(dim * 4)  # mult is always 4

        self.net = nn.CellList([])
        # project in
        self.net.append(GEGLU(dim, inner_dim))
        # project dropout
        self.net.append(nn.Identity())  # nn.Dropout(0)) # dummy for dropout with 0
        # project out
        self.net.append(nn.Dense(inner_dim, dim))

    def construct(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        upcast_attention: bool = False,
    ):
        super().__init__()

        self.gradient_checkpointing = False

        # 1. Self-Attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )
        self.ff = FeedForward(dim)

        # 2. Cross-Attn
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )

        self.norm1 = nn.LayerNorm((dim,))
        self.norm2 = nn.LayerNorm((dim,))

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm((dim,))

    def set_use_memory_efficient_attention(self, xformers: bool, mem_eff: bool):
        self.attn1.set_use_memory_efficient_attention(xformers, mem_eff)
        self.attn2.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa: bool):
        self.attn1.set_use_sdpa(sdpa)
        self.attn2.set_use_sdpa(sdpa)

    @property
    def gradient_checkpointing(self):
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._gradient_checkpointing = value
        self.recompute()

    def construct(self, hidden_states, context=None, timestep=None):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(norm_hidden_states, context=context) + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


class Transformer2DModel(nn.Cell):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        num_transformer_layers: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.use_linear_projection = use_linear_projection

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        # self.norm = GroupNorm32(32, in_channels, eps=1e-6, affine=True)

        if use_linear_projection:
            self.proj_in = nn.Dense(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
            )

        blocks = []
        for _ in range(num_transformer_layers):
            blocks.append(
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                )
            )

        self.transformer_blocks = nn.CellList(blocks)

        if use_linear_projection:
            self.proj_out = nn.Dense(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
            )

        self.gradient_checkpointing = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        for transformer in self.transformer_blocks:
            transformer.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa):
        for transformer in self.transformer_blocks:
            transformer.set_use_sdpa(sdpa)

    def construct(self, hidden_states, encoder_hidden_states=None, timestep=None):
        # 1. Input
        batch, _, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=encoder_hidden_states, timestep=timestep)

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2)
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2)

        output = hidden_states + residual

        return output


class Upsample2D(nn.Cell):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1, pad_mode="pad", has_bias=True)

        self._gradient_checkpointing = False

    @property
    def gradient_checkpointing(self):
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._gradient_checkpointing = value
        self.recompute()

    def construct(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == ms.bfloat16:
            hidden_states = hidden_states.to(ms.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = ops.interpolate(
                hidden_states, scale_factor=2.0, mode="nearest", recompute_scale_factor=True
            )
        else:
            hidden_states = ops.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == ms.bfloat16:
            hidden_states = hidden_states.to(dtype)

        hidden_states = self.conv(hidden_states)

        return hidden_states


class SdxlUNet2DConditionModel(nn.Cell):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = IN_CHANNELS
        self.out_channels = OUT_CHANNELS
        self.model_channels = MODEL_CHANNELS
        self.time_embed_dim = TIME_EMBED_DIM
        self.adm_in_channels = ADM_IN_CHANNELS

        self.gradient_checkpointing = False
        # self.sample_size = sample_size

        # time embedding
        self.time_embed = nn.SequentialCell(
            nn.Dense(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Dense(self.time_embed_dim, self.time_embed_dim),
        )

        # label embedding
        self.label_emb = nn.SequentialCell(
            nn.SequentialCell(
                nn.Dense(self.adm_in_channels, self.time_embed_dim),
                nn.SiLU(),
                nn.Dense(self.time_embed_dim, self.time_embed_dim),
            )
        )

        # input
        input_blocks = [
            nn.SequentialCell(
                nn.Conv2d(
                    self.in_channels, self.model_channels, kernel_size=3, padding=1, pad_mode="pad", has_bias=True
                ),
            )
        ]

        # level 0
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=1 * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            ]
            input_blocks.append(nn.CellList(layers))

        input_blocks.append(
            nn.SequentialCell(
                Downsample2D(
                    channels=1 * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            )
        )

        # level 1
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=(1 if i == 0 else 2) * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            input_blocks.append(nn.CellList(layers))

        input_blocks.append(
            nn.SequentialCell(
                Downsample2D(
                    channels=2 * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
            )
        )

        # level 2
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=(2 if i == 0 else 4) * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            input_blocks.append(nn.CellList(layers))
        self.input_blocks = nn.CellList(input_blocks)

        # mid
        self.middle_block = nn.CellList(
            [
                ResnetBlock2D(
                    in_channels=4 * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
                ResnetBlock2D(
                    in_channels=4 * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
            ]
        )

        # output
        output_blocks = []

        # level 2
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=4 * self.model_channels + (4 if i <= 1 else 2) * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            if i == 2:
                layers.append(
                    Upsample2D(
                        channels=4 * self.model_channels,
                        out_channels=4 * self.model_channels,
                    )
                )

            output_blocks.append(nn.CellList(layers))

        # level 1
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=2 * self.model_channels + (4 if i == 0 else (2 if i == 1 else 1)) * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            if i == 2:
                layers.append(
                    Upsample2D(
                        channels=2 * self.model_channels,
                        out_channels=2 * self.model_channels,
                    )
                )

            output_blocks.append(nn.CellList(layers))

        # level 0
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=1 * self.model_channels + (2 if i == 0 else 1) * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            ]

            output_blocks.append(nn.CellList(layers))
        self.output_blocks = nn.CellList(output_blocks)

        # output
        self.out = nn.CellList(
            [
                GroupNorm32(32, self.model_channels),
                nn.SiLU(),
                nn.Conv2d(self.model_channels, self.out_channels, 3, padding=1, pad_mode="pad", has_bias=True),
            ]
        )

    # region diffusers compatibility
    def prepare_config(self):
        self.config = SimpleNamespace()

    @property
    def dtype(self) -> ms.dtype:
        # `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        return get_parameter_dtype(self)

    def set_attention_slice(self, slice_size):
        raise NotImplementedError("Attention slicing is not supported for this model.")

    def is_gradient_checkpointing(self) -> bool:
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        # self.set_gradient_checkpointing(value=True)

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.set_gradient_checkpointing(value=False)

    def set_use_memory_efficient_attention(self, xformers: bool, mem_eff: bool) -> None:
        # blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in self.input_blocks:
            for module in block:
                if hasattr(module, "set_use_memory_efficient_attention"):
                    # logger.info(module.__class__.__name__)
                    module.set_use_memory_efficient_attention(xformers, mem_eff)
        for block in [self.middle_block]:
            for module in block:
                if hasattr(module, "set_use_memory_efficient_attention"):
                    # logger.info(module.__class__.__name__)
                    module.set_use_memory_efficient_attention(xformers, mem_eff)
        for block in self.output_blocks:
            for module in block:
                if hasattr(module, "set_use_memory_efficient_attention"):
                    # logger.info(module.__class__.__name__)
                    module.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa: bool) -> None:
        blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in blocks:
            for module in block:
                if hasattr(module, "set_use_sdpa"):
                    module.set_use_sdpa(sdpa)

    def set_gradient_checkpointing(self, value=False):
        # blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in self.input_blocks:
            for module in block.cells():
                if hasattr(module, "gradient_checkpointing"):
                    # logger.info(f{module.__class__.__name__} {module.gradient_checkpointing} -> {value}")
                    module.gradient_checkpointing = value
        for block in [self.middle_block]:
            for module in block.cells():
                if hasattr(module, "gradient_checkpointing"):
                    # logger.info(f{module.__class__.__name__} {module.gradient_checkpointing} -> {value}")
                    module.gradient_checkpointing = value
        for block in self.output_blocks:
            for module in block.cells():
                if hasattr(module, "gradient_checkpointing"):
                    # logger.info(f{module.__class__.__name__} {module.gradient_checkpointing} -> {value}")
                    module.gradient_checkpointing = value

    def to(self, dtype: Optional[ms.Type] = None):
        for p in self.get_parameters():
            p.set_dtype(dtype)
        return self

    # endregion

    def construct(self, x, timesteps=None, context=None, y=None, **kwargs):
        # broadcast timesteps to batch dimension
        if len(timesteps.shape) == 0 or timesteps.shape[0] != x.shape[0]:
            timesteps = timesteps.tile((x.shape[0],))

        hs = []
        t_emb = get_timestep_embedding(timesteps, self.model_channels, downscale_freq_shift=0)  # , repeat_only=False)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        # assert x.dtype == self.dtype
        emb = emb + self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                # logger.info(layer.__class__.__name__, x.dtype, emb.dtype, context.dtype if context is not None else None)
                if isinstance(layer, ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        # h = x.type(self.dtype)
        h = x

        for module in self.input_blocks:
            h = call_module(module, h, emb, context)
            hs.append(h)

        h = call_module(self.middle_block, h, emb, context)

        for module in self.output_blocks:
            tmp = hs.pop()
            h = ops.cat([h, tmp], axis=1)
            h = call_module(module, h, emb, context)

        h = h.type(x.dtype)
        h = call_module(self.out, h, emb, context)

        return h


class InferSdxlUNet2DConditionModel:
    def __init__(self, original_unet: SdxlUNet2DConditionModel, **kwargs):
        self.delegate = original_unet

        # override original model's forward method: because forward is not called by `__call__`
        # overriding `__call__` is not enough, because nn.Cell.forward has a special handling
        self.delegate.forward = self.forward

        # Deep Shrink
        self.ds_depth_1 = None
        self.ds_depth_2 = None
        self.ds_timesteps_1 = None
        self.ds_timesteps_2 = None
        self.ds_ratio = None

    # call original model's methods
    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def __call__(self, *args, **kwargs):
        return self.delegate(*args, **kwargs)

    def set_deep_shrink(self, ds_depth_1, ds_timesteps_1=650, ds_depth_2=None, ds_timesteps_2=None, ds_ratio=0.5):
        if ds_depth_1 is None:
            logger.info("Deep Shrink is disabled.")
            self.ds_depth_1 = None
            self.ds_timesteps_1 = None
            self.ds_depth_2 = None
            self.ds_timesteps_2 = None
            self.ds_ratio = None
        else:
            logger.info(
                f"Deep Shrink is enabled: [depth={ds_depth_1}/{ds_depth_2}, timesteps={ds_timesteps_1}/{ds_timesteps_2}, ratio={ds_ratio}]"
            )
            self.ds_depth_1 = ds_depth_1
            self.ds_timesteps_1 = ds_timesteps_1
            self.ds_depth_2 = ds_depth_2 if ds_depth_2 is not None else -1
            self.ds_timesteps_2 = ds_timesteps_2 if ds_timesteps_2 is not None else 1000
            self.ds_ratio = ds_ratio

    def construct(self, x, timesteps=None, context=None, y=None, **kwargs):
        r"""
        current implementation is a copy of `SdxlUNet2DConditionModel.forward()` with Deep Shrink.
        """
        _self = self.delegate

        # broadcast timesteps to batch dimension
        timesteps = timesteps.expand(x.shape[0])

        hs = []
        t_emb = get_timestep_embedding(timesteps, _self.model_channels, downscale_freq_shift=0)  # , repeat_only=False)
        t_emb = t_emb.to(x.dtype)
        emb = _self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        # assert x.dtype == _self.dtype
        emb = emb + _self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                # print(layer.__class__.__name__, x.dtype, emb.dtype, context.dtype if context is not None else None)
                if isinstance(layer, ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        # h = x.type(self.dtype)
        h = x

        for depth, module in enumerate(_self.input_blocks):
            # Deep Shrink
            if self.ds_depth_1 is not None:
                if (depth == self.ds_depth_1 and timesteps[0] >= self.ds_timesteps_1) or (
                    self.ds_depth_2 is not None
                    and depth == self.ds_depth_2
                    and timesteps[0] < self.ds_timesteps_1
                    and timesteps[0] >= self.ds_timesteps_2
                ):
                    # print("downsample", h.shape, self.ds_ratio)
                    org_dtype = h.dtype
                    if org_dtype == ms.bfloat16:
                        h = h.to(ms.float32)
                    h = ops.interpolate(h, scale_factor=self.ds_ratio, mode="bicubic", align_corners=False).to(
                        org_dtype
                    )

            h = call_module(module, h, emb, context)
            hs.append(h)

        h = call_module(_self.middle_block, h, emb, context)

        for module in _self.output_blocks:
            # Deep Shrink
            if self.ds_depth_1 is not None:
                if hs[-1].shape[-2:] != h.shape[-2:]:
                    # print("upsample", h.shape, hs[-1].shape)
                    h = resize_like(h, hs[-1])
            tmp = hs.pop()
            h = ops.cat([h, tmp], axis=1)
            h = call_module(module, h, emb, context)

        # Deep Shrink: in case of depth 0
        if self.ds_depth_1 == 0 and h.shape[-2:] != x.shape[-2:]:
            # print("upsample", h.shape, x.shape)
            h = resize_like(h, x)

        h = h.type(x.dtype)
        h = call_module(_self.out, h, emb, context)

        return h


if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    logger.info("create unet")
    unet = SdxlUNet2DConditionModel()
    noisy_latents = ops.randn(1, 4, 52, 52).to(ms.float32)
    timesteps = ms.Tensor([566]).to(ms.float32)
    text_embedding = ops.randn(1, 77, 2048).to(ms.float32)
    vector_embedding = ops.randn(1, 2816).to(ms.float32)
    unet_results = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
