import logging
import numbers
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
from opensora.acceleration.communications import AllToAll_SBH
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info

import mindspore as ms
from mindspore import Parameter, mint, nn, ops
from mindspore.common.initializer import initializer

from mindone.diffusers.models.attention import FeedForward, GatedSelfAttentionDense
from mindone.diffusers.models.attention_processor import Attention as Attention_
from mindone.diffusers.models.embeddings import SinusoidalPositionalEmbedding
from mindone.diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero
from mindone.utils.version_control import check_valid_flash_attention, choose_flash_attention_dtype

from .rope import PositionGetter3D, RoPE3D

logger = logging.getLogger(__name__)


# Positional Embeddings
def get_3d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)
    grid_t = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / interpolation_scale[0]
    grid_h = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / interpolation_scale[1]
    grid_w = np.arange(grid_size[2], dtype=np.float32) / (grid_size[2] / base_size[2]) / interpolation_scale[2]
    grid = np.meshgrid(grid_w, grid_h, grid_t)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[2], grid_size[1], grid_size[0]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be divisible by 3")

    # use 1/3 of dimensions to encode grid_t/h/w
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (T*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (T*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (T*H*W, D/3)

    emb = np.concatenate([emb_t, emb_h, emb_w], axis=1)  # (T*H*W, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / interpolation_scale[0]
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / interpolation_scale[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use 1/3 of dimensions to encode grid_t/h/w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
):
    """
    grid_size: int of the grid return: pos_embed: [grid_size, embed_dim] or
    [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)

    grid = np.arange(grid_size, dtype=np.float32) / (grid_size / base_size) / interpolation_scale
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D/2)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Attention(Attention_):
    def __init__(self, downsampler, attention_mode, use_rope, interpolation_scale_thw, **kwags):
        FA_dtype = kwags.pop("FA_dtype", ms.bfloat16)
        processor = AttnProcessor2_0(
            attention_mode=attention_mode,
            use_rope=use_rope,
            interpolation_scale_thw=interpolation_scale_thw,
            FA_dtype=FA_dtype,
            dim_head=kwags["dim_head"],
        )
        kwags["processor"] = processor
        super().__init__(**kwags)
        if attention_mode == "xformers":
            self.set_use_memory_efficient_attention_xformers(True)
        self.processor = processor
        self.downsampler = None
        if downsampler:  # downsampler  k155_s122
            downsampler_ker_size = list(re.search(r"k(\d{2,3})", downsampler).group(1))  # 122
            down_factor = list(re.search(r"s(\d{2,3})", downsampler).group(1))
            downsampler_ker_size = [int(i) for i in downsampler_ker_size]
            downsampler_padding = [(i - 1) // 2 for i in downsampler_ker_size]
            down_factor = [int(i) for i in down_factor]

            if len(downsampler_ker_size) == 2:
                self.downsampler = DownSampler2d(
                    kwags["query_dim"],
                    kwags["query_dim"],
                    kernel_size=downsampler_ker_size,
                    stride=1,
                    padding=downsampler_padding,
                    groups=kwags["query_dim"],
                    down_factor=down_factor,
                    down_shortcut=True,
                )
            elif len(downsampler_ker_size) == 3:
                self.downsampler = DownSampler3d(
                    kwags["query_dim"],
                    kwags["query_dim"],
                    kernel_size=downsampler_ker_size,
                    stride=1,
                    padding=downsampler_padding,
                    groups=kwags["query_dim"],
                    down_factor=down_factor,
                    down_shortcut=True,
                )

    def prepare_attention_mask(
        self, attention_mask: ms.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> ms.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`ms.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `ms.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if get_sequence_parallel_state():
            head_size = head_size // hccl_info.world_size
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            attention_mask = ops.pad(attention_mask, (0, target_length), mode="constant", value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, 0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, 1)

        return attention_mask


@ms.jit_class
class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention or xFormers-like memory efficient attention.
    """

    def __init__(
        self,
        attention_mode="xformers",
        use_rope=False,
        interpolation_scale_thw=(1, 1, 1),
        FA_dtype=ms.bfloat16,
        dim_head=64,
    ):
        self.use_rope = use_rope
        self.interpolation_scale_thw = interpolation_scale_thw
        if self.use_rope:
            self._init_rope(interpolation_scale_thw, dim_head=dim_head)
        self.attention_mode = attention_mode
        # Currently we only support setting attention_mode to `flash` or `math`
        assert self.attention_mode in [
            "xformers",
            "math",
        ], f"Unsupported attention mode {self.attention_mode}. Currently we only support ['xformers', 'math']!"
        self.enable_FA = attention_mode == "xformers"
        self.FA_dtype = FA_dtype
        assert self.FA_dtype in [ms.float16, ms.bfloat16], f"Unsupported flash-attention dtype: {self.FA_dtype}"
        if self.enable_FA:
            FLASH_IS_AVAILABLE = check_valid_flash_attention()
            self.enable_FA = FLASH_IS_AVAILABLE and self.enable_FA

        self.fa_mask_dtype = choose_flash_attention_dtype()
        if get_sequence_parallel_state():
            self.sp_size = hccl_info.world_size
            self.alltoall_sbh_q = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_k = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_v = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_out = AllToAll_SBH(scatter_dim=1, gather_dim=0)
        else:
            self.sp_size = 1
            self.alltoall_sbh_q = None
            self.alltoall_sbh_k = None
            self.alltoall_sbh_v = None
            self.alltoall_sbh_out = None

    def _init_rope(self, interpolation_scale_thw, dim_head):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw, dim_head=dim_head)
        self.position_getter = PositionGetter3D()

    def run_ms_flash_attention(
        self,
        attn,
        query,
        key,
        value,
        attention_mask,
        input_layout="BSH",
        attention_dropout: float = 0.0,
    ):
        # Memory efficient attention on mindspore uses flash attention under the hoods.
        # Flash attention implementation is called `FlashAttentionScore`
        # which is an experimental api with the following limitations:
        # 1. Sequence length of query must be divisible by 16 and in range of [1, 32768].
        # 2. Head dimensions must be one of [64, 80, 96, 120, 128, 256].
        # 3. The input dtype must be float16 or bfloat16.
        # Sequence length of query must be checked in runtime.
        if input_layout not in ["BSH", "BNSD"]:
            raise ValueError(f"input_layout must be in ['BSH', 'BNSD'], but get {input_layout}.")
        Bs, query_tokens, _ = query.shape
        assert query_tokens % 16 == 0, f"Sequence length of query must be divisible by 16, but got {query_tokens=}."
        key_tokens = key.shape[1]
        heads = attn.heads if not get_sequence_parallel_state() else attn.heads // hccl_info.world_size
        query = query.view(Bs, query_tokens, heads, -1)
        key = key.view(Bs, key_tokens, heads, -1)
        value = value.view(Bs, key_tokens, heads, -1)
        # Head dimension is checked in Attention.set_use_memory_efficient_attention_xformers. We maybe pad on head_dim.
        if attn.head_dim_padding > 0:
            query_padded = ops.pad(query, (0, attn.head_dim_padding), mode="constant", value=0.0)
            key_padded = ops.pad(key, (0, attn.head_dim_padding), mode="constant", value=0.0)
            value_padded = ops.pad(value, (0, attn.head_dim_padding), mode="constant", value=0.0)
        else:
            query_padded, key_padded, value_padded = query, key, value
        flash_attn = ops.operations.nn_ops.FlashAttentionScore(
            scale_value=attn.scale, head_num=heads, input_layout=input_layout, keep_prob=1 - attention_dropout
        )
        if attention_mask is not None:
            # flip mask, since ms FA treats 1 as discard, 0 as retain.
            attention_mask = ~attention_mask if attention_mask.dtype == ms.bool_ else 1 - attention_mask
            # (b, 1, 1, k_n) - > (b, 1, q_n, k_n), manual broadcast
            if attention_mask.shape[-2] == 1:
                attention_mask = mint.tile(attention_mask.bool(), (1, 1, query_tokens, 1))
            attention_mask = attention_mask.to(self.fa_mask_dtype)

        if input_layout == "BNSD":
            # (b s n d) -> (b n s d)
            query_padded = query_padded.swapaxes(1, 2)
            key_padded = key_padded.swapaxes(1, 2)
            value_padded = value_padded.swapaxes(1, 2)
        elif input_layout == "BSH":
            query_padded = query_padded.view(Bs, query_tokens, -1)
            key_padded = key_padded.view(Bs, key_tokens, -1)
            value_padded = value_padded.view(Bs, key_tokens, -1)
        hidden_states_padded = flash_attn(
            query_padded.to(self.FA_dtype),
            key_padded.to(self.FA_dtype),
            value_padded.to(self.FA_dtype),
            None,
            None,
            None,
            attention_mask,
        )[3]
        # If we did padding before calculate attention, undo it!
        if attn.head_dim_padding > 0:
            if input_layout == "BNSD":
                hidden_states = hidden_states_padded[..., : attn.head_dim]
            else:
                hidden_states = hidden_states_padded.view(Bs, query_tokens, heads, -1)[..., : attn.head_dim]
                hidden_states = hidden_states.view(Bs, query_tokens, -1)
        else:
            hidden_states = hidden_states_padded
        if input_layout == "BNSD":
            # b n s d -> b s n d
            hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = hidden_states.reshape(Bs, query_tokens, -1)
        hidden_states = hidden_states.to(query.dtype)
        return hidden_states

    def run_math_attention(self, attn, query, key, value, attention_mask):
        _head_size = attn.heads if not get_sequence_parallel_state() else attn.heads // hccl_info.world_size
        query = self._head_to_batch_dim(_head_size, query)
        key = self._head_to_batch_dim(_head_size, key)
        value = self._head_to_batch_dim(_head_size, value)

        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqeeuze(1)
            assert attention_mask.shape[1] == 1
            attention_mask = attention_mask.repeat_interleave(_head_size, 1)
            attention_mask = attention_mask.reshape(-1, attention_mask.shape[-2], attention_mask.shape[-1])
            attention_mask = ops.zeros(attention_mask.shape).masked_fill(attention_mask.to(ms.bool_), -10000.0)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = ops.bmm(attention_probs, value)
        hidden_states = self._batch_to_head_dim(_head_size, hidden_states)
        return hidden_states

    def _batch_to_head_dim(self, head_size, tensor: ms.Tensor) -> ms.Tensor:
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def _head_to_batch_dim(self, head_size, tensor: ms.Tensor, out_dim: int = 3) -> ms.Tensor:
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        frame: int = 8,
        height: int = 16,
        width: int = 16,
    ) -> ms.Tensor:
        if attn.downsampler is not None:
            hidden_states, attention_mask = attn.downsampler(hidden_states, attention_mask, t=frame, h=height, w=width)
            frame, height, width = attn.downsampler.t, attn.downsampler.h, attn.downsampler.w

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            channel = None

        if get_sequence_parallel_state():
            sequence_length, batch_size, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

        else:
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        if get_sequence_parallel_state():
            query = query.view(-1, attn.heads, head_dim)  # [s // sp, b, h * d] -> [s // sp * b, h, d]
            key = key.view(-1, attn.heads, head_dim)
            value = value.view(-1, attn.heads, head_dim)
            # query = attn.q_norm(query)
            # key = attn.k_norm(key)
            h_size = attn.heads * head_dim
            sp_size = hccl_info.world_size
            h_size_sp = h_size // sp_size
            # apply all_to_all to gather sequence and split attention heads [s // sp * b, h, d] -> [s * b, h // sp, d]
            query = self.alltoall_sbh_q(query).view(-1, batch_size, h_size_sp)
            key = self.alltoall_sbh_k(key).view(-1, batch_size, h_size_sp)
            value = self.alltoall_sbh_v(value).view(-1, batch_size, h_size_sp)

            if self.use_rope:
                query = query.view(-1, batch_size, attn.heads // sp_size, head_dim)
                key = key.view(-1, batch_size, attn.heads // sp_size, head_dim)
                # require the shape of (batch_size x nheads x ntokens x dim)
                pos_thw = self.position_getter(batch_size, t=frame * sp_size, h=height, w=width)
                query = self.rope(query, pos_thw)
                key = self.rope(key, pos_thw)
            query = query.view(-1, batch_size, h_size_sp).swapaxes(0, 1)  # SBH to BSH
            key = key.view(-1, batch_size, h_size_sp).swapaxes(0, 1)
            value = value.view(-1, batch_size, h_size_sp).swapaxes(0, 1)
            if self.attention_mode == "math":
                # FIXME: shape error
                hidden_states = self.run_math_attention(attn, query, key, value, attention_mask)
            elif self.attention_mode == "xformers":
                hidden_states = self.run_ms_flash_attention(attn, query, key, value, attention_mask)
            # [s * b, h // sp, d] -> [s // sp * b, h, d] -> [s // sp, b, h * d]
            hidden_states = hidden_states.view(batch_size, -1, attn.heads // sp_size, head_dim).transpose(2, 1, 0, 3)
            hidden_states = self.alltoall_sbh_out(hidden_states).transpose(1, 2, 0, 3).view(-1, batch_size, h_size)
        else:
            query = query.view(batch_size, -1, attn.heads, head_dim)
            key = key.view(batch_size, -1, attn.heads, head_dim)
            # query = attn.q_norm(query)
            # key = attn.k_norm(key)
            if self.use_rope:
                # require the shape of (batch_size x ntokens x nheads x dim)
                pos_thw = self.position_getter(batch_size, t=frame, h=height, w=width)
                query = self.rope(query, pos_thw)
                key = self.rope(key, pos_thw)
            query = query.view(batch_size, -1, attn.heads * head_dim)
            key = key.view(batch_size, -1, attn.heads * head_dim)

            if self.attention_mode == "math":
                hidden_states = self.run_math_attention(attn, query, key, value, attention_mask)
            elif self.attention_mode == "xformers":
                hidden_states = self.run_ms_flash_attention(attn, query, key, value, attention_mask)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if attn.downsampler is not None:
            hidden_states = attn.downsampler.reverse(hidden_states, t=frame, h=height, w=width)
        return hidden_states


class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, dtype=ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gamma = Parameter(initializer("ones", normalized_shape, dtype=dtype))
            self.beta = Parameter(initializer("zeros", normalized_shape, dtype=dtype))
        else:
            self.gamma = ops.ones(normalized_shape, dtype=dtype)
            self.beta = ops.zeros(normalized_shape, dtype=dtype)
        self.layer_norm = ops.LayerNorm(-1, -1, epsilon=eps)

    def construct(self, x: ms.Tensor):
        x, _, _ = self.layer_norm(x, self.gamma, self.beta)
        return x


class PatchEmbed2D(nn.Cell):
    """2D Image to Patch Embedding but with 3D positional embedding"""

    def __init__(
        self,
        num_frames=1,
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=True,
    ):
        super().__init__()
        # assert num_frames == 1
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), has_bias=bias
        )
        if layer_norm:
            self.norm = LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None
        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.height, self.width), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.pos_embed = ms.Tensor(pos_embed).float().unsqueeze(0)
        self.num_frames = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.base_size_t = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.interpolation_scale_t = interpolation_scale_t

        if get_sequence_parallel_state():
            self.sp_size = hccl_info.world_size
            rank_offset = hccl_info.rank % hccl_info.world_size
            num_frames = (self.num_frames + self.sp_size - 1) // self.sp_size * self.sp_size
            temp_pos_embed = get_1d_sincos_pos_embed(
                embed_dim, num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t
            )
            num_frames //= self.sp_size
            self.temp_pos_st = rank_offset * num_frames
            self.temp_pos_ed = (rank_offset + 1) * num_frames
        else:
            temp_pos_embed = get_1d_sincos_pos_embed(
                embed_dim, self.num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t
            )

        self.temp_pos_embed = ms.Tensor(temp_pos_embed).float().unsqueeze(0)

    def construct(self, latent, num_frames):
        b, c, t, h, w = latent.shape
        video_latent, image_latent = None, None
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        # b c t h w -> (b t) c h w
        latent = latent.swapaxes(1, 2).reshape(b * t, c, h, w)

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(start_dim=2).permute(0, 2, 1)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            # Interpolate positional embeddings if needed.
            # (For PixArt-Alpha: https://github.com/PixArt-alpha/\
            # PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
            if self.height != height or self.width != width:
                # raise NotImplementedError
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = ms.Tensor(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0)
            else:
                pos_embed = self.pos_embed

            if self.num_frames != num_frames:
                if get_sequence_parallel_state():
                    # f, h -> f, 1, h
                    temp_pos_embed = self.temp_pos_embed[self.temp_pos_st : self.temp_pos_ed].unsqueeze(1)
                else:
                    temp_pos_embed = get_1d_sincos_pos_embed(
                        embed_dim=self.temp_pos_embed.shape[-1],
                        grid_size=num_frames,
                        base_size=self.base_size_t,
                        interpolation_scale=self.interpolation_scale_t,
                    )
                temp_pos_embed = ms.Tensor(temp_pos_embed)
                temp_pos_embed = temp_pos_embed.float().unsqueeze(0)
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)

        # (b t) n c -> b t n c
        latent = latent.reshape(b, t, -1, self.embed_dim)
        video_latent, image_latent = latent[:, :num_frames], latent[:, num_frames:]

        if self.use_abs_pos:
            # temp_pos_embed = temp_pos_embed.unsqueeze(2) * self.temp_embed_gate.tanh()
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            video_latent = (
                (video_latent + temp_pos_embed).to(video_latent.dtype)
                if video_latent is not None and video_latent.numel() > 0
                else None
            )
            image_latent = (
                (image_latent + temp_pos_embed[:, :1]).to(image_latent.dtype)
                if image_latent is not None and image_latent.numel() > 0
                else None
            )
        # 'b t n c -> b (t n) c'
        video_latent = (
            video_latent.reshape(b, -1, self.embed_dim)
            if video_latent is not None and video_latent.numel() > 0
            else None
        )
        # 'b t n c -> (b t) n c'
        image_latent = (
            image_latent.reshape(b * t, -1, self.embed_dim)
            if image_latent is not None and image_latent.numel() > 0
            else None
        )

        if num_frames == 1 and image_latent is None and not get_sequence_parallel_state():
            image_latent = video_latent
            video_latent = None

        return video_latent, image_latent


class OverlapPatchEmbed3D(nn.Cell):
    """2D Image to Patch Embedding but with 3D positional embedding"""

    def __init__(
        self,
        num_frames=1,
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=True,
    ):
        super().__init__()
        # assert num_frames == 1
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size_t, patch_size, patch_size),
            stride=(patch_size_t, patch_size, patch_size),
            has_bias=bias,
        )
        if layer_norm:
            self.norm = LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None
        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.height, self.width), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.pos_embed = ms.Tensor(pos_embed).float().unsqueeze(0)
        self.num_frames = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.base_size_t = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.interpolation_scale_t = interpolation_scale_t

        if get_sequence_parallel_state():
            self.sp_size = hccl_info.world_size
            rank_offset = hccl_info.rank % hccl_info.world_size
            num_frames = (self.num_frames + self.sp_size - 1) // self.sp_size * self.sp_size
            temp_pos_embed = get_1d_sincos_pos_embed(
                embed_dim, num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t
            )
            num_frames //= self.sp_size
            self.temp_pos_st = rank_offset * num_frames
            self.temp_pos_ed = (rank_offset + 1) * num_frames
        else:
            temp_pos_embed = get_1d_sincos_pos_embed(
                embed_dim, self.num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t
            )

        self.temp_pos_embed = ms.Tensor(temp_pos_embed).float().unsqueeze(0)

    def construct(self, latent, num_frames):
        b, c, t, h, w = latent.shape
        video_latent, image_latent = None, None
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        latent = self.proj(latent)
        if self.flatten:
            # b c t h w -> (b t) (h w) c
            latent = latent.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            # Interpolate positional embeddings if needed.
            # (For PixArt-Alpha: https://github.com/PixArt-alpha/\
            # PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
            if self.height != height or self.width != width:
                # raise NotImplementedError
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = ms.Tensor(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0)
            else:
                pos_embed = self.pos_embed

            if self.num_frames != num_frames:
                if get_sequence_parallel_state():
                    # f, h -> f, 1, h
                    temp_pos_embed = self.temp_pos_embed[self.temp_pos_st : self.temp_pos_ed].unsqueeze(1)
                else:
                    temp_pos_embed = get_1d_sincos_pos_embed(
                        embed_dim=self.temp_pos_embed.shape[-1],
                        grid_size=num_frames,
                        base_size=self.base_size_t,
                        interpolation_scale=self.interpolation_scale_t,
                    )
                temp_pos_embed = ms.Tensor(temp_pos_embed)
                temp_pos_embed = temp_pos_embed.float().unsqueeze(0)
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)

        # (b t) n c -> b t n c
        latent = latent.reshape(b, t, -1, self.embed_dim)
        video_latent, image_latent = latent[:, :num_frames], latent[:, num_frames:]

        if self.use_abs_pos:
            # temp_pos_embed = temp_pos_embed.unsqueeze(2) * self.temp_embed_gate.tanh()
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            video_latent = (
                (video_latent + temp_pos_embed).to(video_latent.dtype)
                if video_latent is not None and video_latent.numel() > 0
                else None
            )
            image_latent = (
                (image_latent + temp_pos_embed[:, :1]).to(image_latent.dtype)
                if image_latent is not None and image_latent.numel() > 0
                else None
            )
        # 'b t n c -> b (t n) c'
        video_latent = (
            video_latent.reshape(b, -1, self.embed_dim)
            if video_latent is not None and video_latent.numel() > 0
            else None
        )
        # 'b t n c -> (b t) n c'
        image_latent = (
            image_latent.reshape(b * t, -1, self.embed_dim)
            if image_latent is not None and image_latent.numel() > 0
            else None
        )

        if num_frames == 1 and image_latent is None:
            image_latent = video_latent
            video_latent = None

        return video_latent, image_latent


class OverlapPatchEmbed2D(nn.Cell):
    """2D Image to Patch Embedding but with 3D positional embedding"""

    def __init__(
        self,
        num_frames=1,
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=True,
    ):
        super().__init__()
        assert patch_size_t == 1
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), has_bias=bias
        )
        if layer_norm:
            self.norm = LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None
        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.height, self.width), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.pos_embed = ms.Tensor(pos_embed).float().unsqueeze(0)
        self.num_frames = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.base_size_t = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.interpolation_scale_t = interpolation_scale_t

        if get_sequence_parallel_state():
            self.sp_size = hccl_info.world_size
            rank_offset = hccl_info.rank % hccl_info.world_size
            num_frames = (self.num_frames + self.sp_size - 1) // self.sp_size * self.sp_size
            temp_pos_embed = get_1d_sincos_pos_embed(
                embed_dim, num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t
            )
            num_frames //= self.sp_size
            self.temp_pos_st = rank_offset * num_frames
            self.temp_pos_ed = (rank_offset + 1) * num_frames
        else:
            temp_pos_embed = get_1d_sincos_pos_embed(
                embed_dim, self.num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t
            )

        self.temp_pos_embed = ms.Tensor(temp_pos_embed).float().unsqueeze(0)

    def construct(self, latent, num_frames):
        b, c, t, h, w = latent.shape
        video_latent, image_latent = None, None
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        # b c t h w -> (bt) c h w
        latent = latent.swapaxes(1, 2).reshape(b * t, c, h, w)
        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(start_dim=2).permute(0, 2, 1)  # BT C H W -> BT N C
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            # Interpolate positional embeddings if needed.
            # (For PixArt-Alpha: https://github.com/PixArt-alpha/\
            # PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
            if self.height != height or self.width != width:
                # raise NotImplementedError
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = ms.Tensor(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0)
            else:
                pos_embed = self.pos_embed

            if self.num_frames != num_frames:
                if get_sequence_parallel_state():
                    # f, h -> f, 1, h
                    temp_pos_embed = self.temp_pos_embed[self.temp_pos_st : self.temp_pos_ed].unsqueeze(1)
                else:
                    temp_pos_embed = get_1d_sincos_pos_embed(
                        embed_dim=self.temp_pos_embed.shape[-1],
                        grid_size=num_frames,
                        base_size=self.base_size_t,
                        interpolation_scale=self.interpolation_scale_t,
                    )
                temp_pos_embed = ms.Tensor(temp_pos_embed)
                temp_pos_embed = temp_pos_embed.float().unsqueeze(0)
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)

        # (b t) n c -> b t n c
        latent = latent.reshape(b, t, -1, self.embed_dim)
        video_latent, image_latent = latent[:, :num_frames], latent[:, num_frames:]

        if self.use_abs_pos:
            # temp_pos_embed = temp_pos_embed.unsqueeze(2) * self.temp_embed_gate.tanh()
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            video_latent = (
                (video_latent + temp_pos_embed).to(video_latent.dtype)
                if video_latent is not None and video_latent.numel() > 0
                else None
            )
            image_latent = (
                (image_latent + temp_pos_embed[:, :1]).to(image_latent.dtype)
                if image_latent is not None and image_latent.numel() > 0
                else None
            )
        # 'b t n c -> b (t n) c'
        video_latent = (
            video_latent.reshape(b, -1, self.embed_dim)
            if video_latent is not None and video_latent.numel() > 0
            else None
        )
        # 'b t n c -> (b t) n c'
        image_latent = (
            image_latent.reshape(b * t, -1, self.embed_dim)
            if image_latent is not None and image_latent.numel() > 0
            else None
        )

        if num_frames == 1 and image_latent is None:
            image_latent = video_latent
            video_latent = None

        return video_latent, image_latent


class DownSampler3d(nn.Cell):
    def __init__(self, *args, **kwargs):
        """Required kwargs: down_factor, downsampler"""
        super().__init__()
        self.down_factor = kwargs.pop("down_factor")
        self.down_shortcut = kwargs.pop("down_shortcut")
        self.layer = nn.Conv3d(*args, **kwargs)

    def construct(self, x, attention_mask, t, h, w):
        b = x.shape[0]
        # b (t h w) d -> b d t h w
        x = x.reshape(b, t, h, w, -1).permute(0, 4, 1, 2, 3)

        x_dtype = x.dtype
        x = self.layer(x).to(x_dtype) + (x if self.down_shortcut else 0)

        # b d (t dt) (h dh) (w dw) -> (b dt dh dw) (t h w) d
        dt, dh, dw = self.down_factor
        x = x.reshape(b, -1, t // dt, dt, h // dh, dh, w // dw, dw)
        x = x.permute(0, 3, 5, 7, 2, 4, 6, 1).reshape(b * dt * dw * dh, -1, x.shape[1])
        # b 1 (t h w) -> b 1 t h w
        attention_mask = attention_mask.reshape(b, 1, t, h, w)
        # b 1 (t dt) (h dh) (w dw) -> (b dt dh dw) 1 (t h w)
        attention_mask = attention_mask.reshape(b, 1, t // dt, dt, h // dh, dh, w // dw, dw)
        attention_mask = attention_mask.permute(0, 3, 5, 7, 1, 2, 4, 6).reshape(b * dt * dh * dw, 1, -1)

        return x, attention_mask

    def reverse(self, x, t, h, w):
        d = x.shape[2]
        dt, dh, dw = self.down_factor
        # (b dt dh dw) (t h w) d -> b (t dt h dh w dw) d
        x = x.reshape(-1, dt, dh, dw, t, h, w, d)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(-1, t * dt * h * dt * w * dw, d)
        return x


class DownSampler2d(nn.Cell):
    def __init__(self, *args, **kwargs):
        """Required kwargs: down_factor, downsampler"""
        super().__init__()
        self.down_factor = kwargs.pop("down_factor")
        self.down_shortcut = kwargs.pop("down_shortcut")
        self.layer = nn.Conv2d(*args, **kwargs)

    def construct(self, x, attention_mask, t, h, w):
        b = x.shape[0]
        d = x.shape[-1]
        # b (t h w) d -> (b t) d h w
        x = x.reshape(b, t, h, w, -1).permute(0, 1, 4, 2, 3).reshape(b * t, d, h, w)
        x = self.layer(x) + (x if self.down_shortcut else 0)

        dh, dw = self.down_factor
        # b d (h dh) (w dw) -> (b dh dw) (h w) d
        x = x.reshape(b, d, h // dh, dh, w // dw, dw)
        x = x.permute(0, 3, 5, 2, 4, 1).reshape(b * dh * dw, -1, d)
        # b 1 (t h w) -> (b t) 1 h w
        attention_mask = attention_mask.reshape(b, 1, t, h, w).swapaxes(1, 2).reshape(b * t, 1, h, w)
        # b 1 (h dh) (w dw) -> (b dh dw) 1 (h w)
        attention_mask = attention_mask.reshape(b, 1, h // dh, dh, w // dw, dw)
        attention_mask = attention_mask.permute(0, 3, 5, 1, 2, 4).reshape(b * dh * dw, 1, -1)

        return x, attention_mask

    def reverse(self, x, t, h, w):
        # (b t dh dw) (h w) d -> b (t h dh w dw) d
        d = x.shape[-1]
        dh, dw = self.down_factor
        x = x.reshape(-1, t, dh, dw, h, w, d)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(-1, t * h * dh * w * dw, d)
        return x


class FeedForward_Conv2d(nn.Cell):
    def __init__(self, downsampler, dim, hidden_features, bias=True):
        super(FeedForward_Conv2d, self).__init__()

        self.bias = bias

        self.project_in = nn.Dense(dim, hidden_features, has_bias=bias)

        self.dwconv = nn.CellList(
            [
                nn.Conv2d(
                    hidden_features,
                    hidden_features,
                    kernel_size=(5, 5),
                    stride=1,
                    padding=(2, 2),
                    dilation=1,
                    groups=hidden_features,
                    has_bias=bias,
                    pad_mode="pad",
                ),
                nn.Conv2d(
                    hidden_features,
                    hidden_features,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(1, 1),
                    dilation=1,
                    groups=hidden_features,
                    has_bias=bias,
                    pad_mode="pad",
                ),
                nn.Conv2d(
                    hidden_features,
                    hidden_features,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=(0, 0),
                    dilation=1,
                    groups=hidden_features,
                    has_bias=bias,
                    pad_mode="pad",
                ),
            ]
        )

        self.project_out = nn.Dense(hidden_features, dim, has_bias=bias)
        self.gelu = nn.GELU(approximate=False)

    def construct(self, x, t, h, w):
        x = self.project_in(x)
        b, _, d = x.shape
        # b (t h w) d -> (b t) d h w
        x = x.reshape(b, t, h, w, d).permute(0, 1, 4, 2, 3).reshape(b * t, d, h, w)
        x = self.gelu(x)
        out = x
        for module in self.dwconv:
            out = out + module(x)
        # (b t) d h w -> b (t h w) d
        d = out.shape[1]
        out = out.reshape(b, t, d, h, w).permute(0, 1, 3, 4, 2).reshape(b, -1, d)
        x = self.project_out(out)
        return x


class BasicTransformerBlock(nn.Cell):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        downsampler: str = None,
        interpolation_scale_thw: Tuple[int] = (1, 1, 1),
        attention_mode: str = "xformers",
        use_rope: bool = False,
        FA_dtype=ms.bfloat16,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.downsampler = downsampler

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"
        self.FA_dtype = FA_dtype

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )
        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm
        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1_ada = AdaLayerNorm(dim, num_embeds_ada_norm)
            self.norm1_ada.norm = LayerNorm(dim, elementwise_affine=False)
        elif norm_type == "ada_norm_zero":
            self.norm1_ada_zero = AdaLayerNormZero(dim, num_embeds_ada_norm)
            self.norm1_ada_zero.norm = LayerNorm(dim, elementwise_affine=False)
        elif norm_type == "ada_norm_continuous":
            self.norm1_ada_con = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1_ln = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            attention_mode=attention_mode,
            out_bias=attention_out_bias,
            downsampler=downsampler,
            use_rope=use_rope,
            interpolation_scale_thw=interpolation_scale_thw,
            FA_dtype=self.FA_dtype,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if norm_type == "ada_norm":
                self.norm2_ada = AdaLayerNorm(dim, num_embeds_ada_norm)
                self.norm2_ada.norm = LayerNorm(dim, elementwise_affine=False)
            elif norm_type == "ada_norm_continuous":
                self.norm2_ada_con = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2_ln = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                downsampler=False,
                use_rope=False,  # do not position in cross attention
                attention_mode=attention_mode,
                FA_dtype=self.FA_dtype,
                interpolation_scale_thw=interpolation_scale_thw,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )
        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
            self.norm3 = LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None
        if downsampler:
            self.ff = FeedForward_Conv2d(
                downsampler,
                dim,
                2 * dim,
                bias=ff_bias,
            )
        else:
            self.ff = FeedForward(
                dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = ms.Parameter(ops.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        timestep: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[ms.Tensor] = None,
        frame: int = None,
        height: int = None,
        width: int = None,
        added_cond_kwargs: Optional[Dict[str, ms.Tensor]] = None,
    ) -> ms.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]
        gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1_ada(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1_ada_zero(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1_ln(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1_ada_con(hidden_states, added_cond_kwargs["pooled_text_emb"])

        elif self.norm_type == "ada_norm_single":
            if get_sequence_parallel_state():
                batch_size = hidden_states.shape[1]  # S B H
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mint.chunk(
                    self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1), 6, dim=0
                )
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mint.chunk(
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1), 6, dim=1
                )
            norm_hidden_states = self.norm1_ln(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            # norm_hidden_states = norm_hidden_states.squeeze(1)  # error message
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        if "gligen" in cross_attention_kwargs:
            gligen_kwargs = cross_attention_kwargs["gligen"]
        else:
            gligen_kwargs = None
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            frame=frame,
            height=height,
            width=width,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2_ada(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2_ln(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2_ada_con(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2_ln(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self.downsampler:
            ff_output = self.ff(norm_hidden_states, t=frame, h=height, w=width)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states
