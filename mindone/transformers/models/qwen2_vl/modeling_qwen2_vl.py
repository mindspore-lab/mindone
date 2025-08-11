# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MindSpore Qwen2-VL model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from transformers.models.qwen2_vl import Qwen2VLConfig
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import mindspore as ms
from mindspore import _no_grad, jit_class, mint, nn, ops
from mindspore.common.initializer import Initializer, Normal
from mindspore.nn import CrossEntropyLoss, LayerNorm

from mindone.transformers.activations import ACT2FN
from mindone.transformers.cache_utils import (  # TODO: SlidingWindowCache
    Cache,
    DynamicCache,
    get_max_length,
    get_seq_length,
    update,
)
from mindone.transformers.mindspore_adapter.utils import _MIN_FP16
from mindone.transformers.modeling_attn_mask_utils import dtype_to_min
from mindone.transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from mindone.transformers.modeling_utils import MSPreTrainedModel

from ...processing_utils import Unpack
from ...utils import TransformersKwargs

try:
    from transformers.models.qwen2_vl import Qwen2VLTextConfig, Qwen2VLVisionConfig  # transformers >= 4.48.0
except ImportError:
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLTextConfig, Qwen2VLVisionConfig

from mindone.transformers.utils import is_flash_attn_2_available
from mindone.utils.version_control import check_valid_flash_attention

FLASH_IS_AVAILABLE = is_flash_attn_2_available and check_valid_flash_attention()
logger = logging.get_logger(__name__)
if FLASH_IS_AVAILABLE:
    from mindone.models.modules.flash_attention import MSFlashAttention

_CONFIG_FOR_DOC = "Qwen2VLConfig"


@jit_class
class no_grad(_no_grad):
    """
    A context manager that suppresses gradient memory allocation in PyNative mode.
    """

    def __init__(self):
        super().__init__()
        self._pynative = ms.get_context("mode") == ms.PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)


@dataclass
class Qwen2VLModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(ms.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`ms.Tensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: ms.Tensor = None
    past_key_values: Optional[list[ms.Tensor]] = None
    hidden_states: Optional[tuple[ms.Tensor]] = None
    attentions: Optional[tuple[ms.Tensor]] = None
    rope_deltas: Optional[ms.Tensor] = None


@dataclass
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2VL causal language model (or autoregressive) outputs.

    Args:
        loss (`ms.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`ms.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(ms.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(ms.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`ms.Tensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[ms.Tensor] = None
    logits: ms.Tensor = None
    past_key_values: Optional[List[ms.Tensor]] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None
    rope_deltas: Optional[ms.Tensor] = None  # integer


class Qwen2VLRotaryEmbedding(nn.Cell):
    def __init__(self, config: Qwen2VLTextConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        # Use "default" rope type
        # Compute the inverse frequencies
        base = config.rope_theta  # Qwen2-VL used 1000,000.0
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)
        self.inv_freq = 1.0 / (base ** (ops.arange(0, dim, 2, dtype=ms.int32).float() / dim))
        # self.inv_freq = ms.Parameter(inv_freq, requires_grad=False, name="inv_freq_buffer")
        self.original_inv_freq = self.inv_freq
        self.attention_scaling = 1.0  # Unused in this type of RoPE

    def _dynamic_frequency_update(self, position_ids):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = ops.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            self.inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, seq_len=seq_len
            )  # TODO: add it when use dynamic rope
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.inv_freq = self.original_inv_freq
            self.max_seq_len_cached = self.original_max_seq_len

    def construct(self, x, position_ids):
        with no_grad():
            # if "dynamic" in self.rope_type: # NOTE: No Use, uncomment it in future feature
            #     self._dynamic_frequency_update(position_ids)

            # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
            # So we expand the inv_freq to shape (3, ...)
            inv_freq_expanded = (
                self.inv_freq[None, None, :, None].float().broadcast_to((3, position_ids.shape[1], -1, 1))
            )
            position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
            # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).swapaxes(2, 3)
            emb = ops.cat((freqs, freqs), axis=-1)
            cos = emb.cos()
            sin = emb.sin()

            # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
            cos = cos * self.attention_scaling
            sin = sin * self.attention_scaling

            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), axis=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`ms.Tensor`): The query tensor.
        k (`ms.Tensor`): The key tensor.
        cos (`ms.Tensor`): The cosine part of the rotary embedding.
        sin (`ms.Tensor`): The sine part of the rotary embedding.
        position_ids (`ms.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`): e.g., [2, 1, 1]
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(ms.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = ops.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))], axis=-1).unsqueeze(unsqueeze_dim)
    sin = ops.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, axis=-1))], axis=-1).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(tensor: ms.Tensor, freqs: ms.Tensor) -> ms.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).tile((1, 1, 2)).unsqueeze(0).float()
    sin = sin.unsqueeze(1).tile((1, 1, 2)).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionRotaryEmbedding(nn.Cell):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (theta ** (ops.arange(0, dim, 2, dtype=ms.float32) / dim))

    def construct(self, seqlen: int) -> ms.Tensor:
        seq = ops.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = ops.outer(seq, self.inv_freq)
        return freqs


class PatchEmbed(nn.Cell):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = (
            temporal_patch_size,
            patch_size,
            patch_size,
        )  # For 'Conv3d', the type of 'kernel_size' should be one of '['int', 'tuple']'

        if ms.__version__ >= "2.5.0":
            self.proj = mint.nn.Conv3d(
                in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False
            ).to_float(ms.bfloat16)
        else:
            self.proj = nn.Conv3d(
                in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, has_bias=False
            ).to_float(ms.bfloat16)
        # nn.Conv3d does not support float32

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            (-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        )
        hidden_states = self.proj(hidden_states).view((-1, self.embed_dim)).to(dtype=target_dtype)
        return hidden_states


class PatchMerger(nn.Cell):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm([context_dim], epsilon=1e-6).to_float(ms.float32)
        self.mlp = nn.SequentialCell(
            nn.Dense(self.hidden_size, self.hidden_size, has_bias=True),
            nn.GELU(approximate=False),
            nn.Dense(self.hidden_size, dim, has_bias=True),
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        target_dtype = x.dtype  # LayerNorm used fp32
        x = self.mlp(self.ln_q(x).view((-1, self.hidden_size)).to(target_dtype))
        return x


class VisionMlp(nn.Cell):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Dense(dim, hidden_dim, has_bias=True)
        self.act = ACT2FN[hidden_act]  # QuickGELUActivation()
        self.fc2 = nn.Dense(hidden_dim, dim, has_bias=True)

    def construct(self, x) -> ms.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Cell):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Dense(dim, dim * 3, has_bias=True)
        self.proj = nn.Dense(dim, dim, has_bias=True)

    def construct(self, hidden_states: ms.Tensor, cu_seqlens: ms.Tensor, rotary_pos_emb: ms.Tensor = None) -> ms.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = ops.full([1, seq_length, seq_length], dtype_to_min(q.dtype), dtype=q.dtype)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.swapaxes(0, 1)
        k = k.swapaxes(0, 1)
        v = v.swapaxes(0, 1)
        attn_weights = ops.matmul(q, k.swapaxes(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = ops.softmax(attn_weights, axis=-1).to(q.dtype)
        attn_output = ops.matmul(attn_weights, v)
        attn_output = attn_output.swapaxes(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class VisionFlashAttention2(nn.Cell):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Dense(dim, dim * 3, has_bias=True)
        self.proj = nn.Dense(dim, dim, has_bias=True)
        self.flash_attn_func = MSFlashAttention(
            head_dim=self.head_dim,
            head_num=self.num_heads,
            input_layout="BNSD",
            dtype=ms.float16,
        )

    def construct(
        self, hidden_states: ms.Tensor, cu_seqlens: ms.Tensor = None, rotary_pos_emb: ms.Tensor = None
    ) -> ms.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute((1, 0, 2, 3)).unbind(0)
        )  # [3, seq_len, heads, dim]
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).swapaxes(
            1, 2
        )  # BSND=>BNSD [batch=1, heads, seq_len, dimension]
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).swapaxes(1, 2)  # BSND=>BNSD

        # max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = self.flash_attn_func(q, k, v)  # BNSD
        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class VisionSdpaAttention(nn.Cell):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        raise NotImplementedError


QWEN2_VL_VISION_ATTENTION_CLASSES = {
    "eager": VisionAttention,
    "flash_attention_2": VisionFlashAttention2,
    "sdpa": VisionSdpaAttention,  # TODO：VisionSdpaAttention, not support yet
}


class Qwen2VLVisionBlock(nn.Cell):
    def __init__(self, config, attn_implementation: str = "eager") -> None:
        super().__init__()
        self.norm1 = LayerNorm([config.embed_dim], epsilon=1e-6)
        self.norm2 = LayerNorm([config.embed_dim], epsilon=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = QWEN2_VL_VISION_ATTENTION_CLASSES[attn_implementation](config.embed_dim, num_heads=config.num_heads)
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def construct(self, hidden_states, cu_seqlens, rotary_pos_emb) -> ms.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
class Qwen2RMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = ms.Parameter(ops.ones(hidden_size, dtype=ms.float32), name="weight")
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2MLP
class Qwen2MLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # SiLU

    def construct(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: ms.Tensor, n_rep: int) -> ms.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].broadcast_to((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2VLAttention(nn.Cell):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2VLTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=True)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=True)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=True)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False)

        self.rotary_emb = Qwen2VLRotaryEmbedding(config)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,  # long
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[ms.Tensor] = None,  # long
        position_embeddings: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view((bsz, q_len, self.num_heads, self.head_dim)).swapaxes(1, 2)
        key_states = key_states.view((bsz, q_len, self.num_key_value_heads, self.head_dim)).swapaxes(1, 2)
        value_states = value_states.view((bsz, q_len, self.num_key_value_heads, self.head_dim)).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0] + 1

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            if isinstance(past_key_value, Cache):
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:  # tuple static cache
                key_states, value_states = update(past_key_value, key_states, value_states, cache_position)
                past_key_value = (key_states, value_states)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == ms.float16:
            attn_weights = ops.where(ops.isinf(attn_weights), ops.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights.to(ms.float32), axis=-1).to(query_states.dtype)
        attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2VLFlashAttention2(Qwen2VLAttention):
    """
    Qwen2VL flash attention module, following Qwen2VL attention module. This module inherits from `Qwen2VLAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dropout_rate = 0.0 if not self.training else self.attention_dropout
        self.flash_attention = MSFlashAttention(
            head_dim=self.head_dim,
            head_num=self.num_heads,
            attention_dropout=dropout_rate,
            input_layout="BNSD",
            dtype=ms.float16,
        )

    def convert_mask_to_fa_format(self, attention_mask):
        if attention_mask is not None:
            if attention_mask.dtype == ms.bool_:
                # flip mask, since ms FA treats 1 as discard, 0 as retain.
                attention_mask = 1 - attention_mask
                attention_mask = attention_mask.to(ms.uint8)
            else:
                attention_mask = attention_mask.to(ms.float16)
                attention_mask = ops.select(
                    ops.equal(attention_mask, _MIN_FP16),
                    ops.ones((), ms.uint8),
                    ops.zeros((), ms.uint8),
                )

        return attention_mask

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[ms.Tensor] = None,
        position_embeddings: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,  # will become mandatory in v4.46
    ):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view((bsz, q_len, self.num_heads, self.head_dim)).swapaxes(1, 2)  # BNSD
        key_states = key_states.view((bsz, q_len, self.num_key_value_heads, self.head_dim)).swapaxes(1, 2)
        value_states = value_states.view((bsz, q_len, self.num_key_value_heads, self.head_dim)).swapaxes(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            if isinstance(past_key_value, Cache):
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:  # tuple static cache
                key_states, value_states = update(past_key_value, key_states, value_states, cache_position)
                past_key_value = (key_states, value_states)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Flash Attention
        if attention_mask is not None:  # no matter the length, we just slice it
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attention_mask = self.convert_mask_to_fa_format(attention_mask)

        attn_output = self.flash_attention(
            query_states, key_states, value_states, attention_mask
        )  # BNSD (bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2).reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        attn_weights = None  # FA always does not output attn_weights
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2VLSdpaAttention(Qwen2VLAttention):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


QWEN2_VL_ATTENTION_CLASSES = {
    "eager": Qwen2VLAttention,
    "flash_attention_2": Qwen2VLFlashAttention2,
    "sdpa": Qwen2VLSdpaAttention,  # TODO: Qwen2VLSdpaAttention, Not support yet
}


class Qwen2VLDecoderLayer(nn.Cell):
    def __init__(self, config: Qwen2VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QWEN2_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[ms.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[ms.Tensor] = None,
        position_embeddings: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[Tuple[ms.Tensor, ms.Tensor]]]:
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(ms.Tensor)`, *optional*): cached past key and value projection states
            cache_position (`ms.Tensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[ms.Tensor, ms.Tensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


QWEN2VL_START_DOCSTRING = r"""
    This model inherits from [`MSPreTrainedModel`](https://github.com/mindspore-lab/mindone/blob/master/mindone/transformers/modeling_utils.py#L513).
    Check the superclass documentation for the generic methods the library implements for all its model
    (such as downloading or saving, resizing the input embeddings, pruning heads, etc.)

    This model is also a MindSpore [mindspore.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) subclass.
    Use it as a regular MindSpore network and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2VLConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2VL Model outputting raw hidden-states without any specific head on top.",
    QWEN2VL_START_DOCSTRING,
)
class Qwen2VLPreTrainedModel(MSPreTrainedModel):
    config_class = Qwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True  # default Cache class: DynamicCache
    _supports_static_cache = False  # StaticCache, not used

    def _init_weights(self, module):
        if self.training:
            std = self.config.initializer_range
            if isinstance(module, (nn.Dense, nn.Conv3d)):
                weight = Initializer(Normal(sigma=std, mean=0.0), shape=module.weight.shape)
                module.weight.set_data(weight)
                if module.bias is not None:
                    bias_weight = Initializer("zeros", module.bias.shape)
                    module.bias.set_data(bias_weight)
            elif isinstance(module, nn.Embedding):
                embedding_table = Initializer(Normal(sigma=std, mean=0.0), shape=module.embedding_table.shape)
                if module.padding_idx is not None:
                    embedding_table[module.padding_idx] = 0
                module.embedding_table.set_data(embedding_table)


class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    config_class = Qwen2VLVisionConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.CellList(
            [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )

    def get_dtype(self) -> ms.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            t = t.item()
            h = h.item()
            w = w.item()
            hpos_ids = ops.arange(h).unsqueeze(1).broadcast_to((-1, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten(start_dim=0)

            wpos_ids = ops.arange(w).unsqueeze(0).broadcast_to((h, -1))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten(start_dim=0)
            pos_ids.append(ops.stack([hpos_ids, wpos_ids], axis=-1).tile((t, 1)))
        pos_ids = ops.cat(pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_dim=1)
        return rotary_pos_emb

    def construct(self, hidden_states: ms.Tensor, grid_thw: ms.Tensor) -> ms.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = (
            ops.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).int().cumsum(axis=0, dtype=ms.int32)
        )
        cu_seqlens = ops.pad(cu_seqlens, (1, 0), value=None)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)


@add_start_docstrings(
    "The bare Qwen2VL Model outputting raw hidden-states without any specific head on top.",
    QWEN2VL_START_DOCSTRING,
)
class Qwen2VLTextModel(Qwen2VLPreTrainedModel):
    config: Qwen2VLTextConfig

    def __init__(self, config: Qwen2VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Note: parameters and order are different from Pytorch!
        self.embed_tokens = nn.Embedding(
            vocab_size=config.vocab_size, embedding_size=config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = nn.CellList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[ms.Tensor] = None,  # Long
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is not None) and (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            if self._supports_cache_class:
                past_key_values = DynamicCache()
            else:
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = 0
            if past_key_values is not None and (isinstance(past_key_values, tuple)):
                past_seen_tokens = get_seq_length(past_key_values)
            else:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = ops.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view((1, 1, -1)).broadcast_to((3, inputs_embeds.shape[0], -1))
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].broadcast_to((3, position_ids.shape[0], -1))

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[layer_idx] if isinstance(past_key_values, tuple) else past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: ms.Tensor,
        input_tensor: ms.Tensor,
        cache_position: ms.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        past_seen_tokens = 0
        if past_key_values is not None:
            past_seen_tokens = (
                get_seq_length(past_key_values)
                if isinstance(past_key_values, tuple)
                else past_key_values.get_seq_length()
            )
        using_static_cache = isinstance(past_key_values, tuple)

        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = get_max_length(past_key_values)
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, ms.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

    @staticmethod
    # Copied from transformers.models.mistral.modeling_mistral.MistralModel._prepare_4d_causal_attention_mask_with_cache_position with Mistral->Qwen2VL
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: ms.Tensor,
        sequence_length: int,
        target_length: int,
        cache_position: ms.Tensor,
        batch_size: int,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`ms.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding,
                the part of the cache that is not filled yet.                .
            cache_position (`ms.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`ms.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            # dtype to use for the 4D attention mask, note FlashAttention supports mask in uint8 or fp16.
            dtype = ms.float16
            min_dtype = dtype_to_min(dtype)
            causal_mask = ops.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype)
            diagonal_attend_mask = ops.arange(target_length) > cache_position.reshape(-1, 1)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].broadcast_to((batch_size, 1, -1, -1))
            if attention_mask is not None:
                causal_mask = causal_mask[:]
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


QWEN2_VL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(ms.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(ms.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        pixel_values (`ms.Tensor` of shape `(seq_length, num_channels * image_size * image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing images.
        pixel_values_videos (`ms.Tensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing videos.
        image_grid_thw (`ms.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`ms.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`ms.Tensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
"""


class Qwen2VLModel(Qwen2VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}

    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = Qwen2VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[ms.Tensor] = None,
        image_grid_thw: Optional[ms.Tensor] = None,
        video_grid_thw: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`ms.Tensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`ms.Tensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`ms.Tensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`ms.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = mint.ones_like(total_input_ids)
            position_ids = mint.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype)
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = ops.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(mint.arange(text_len).view(1, -1).expand((3, -1)) + st_idx)

                    t_index = mint.arange(llm_grid_t).view(-1, 1).expand((-1, llm_grid_h * llm_grid_w)).flatten()
                    h_index = mint.arange(llm_grid_h).view(1, -1, 1).expand((llm_grid_t, -1, llm_grid_w)).flatten()
                    w_index = mint.arange(llm_grid_w).view(1, 1, -1).expand((llm_grid_t, llm_grid_h, -1)).flatten()
                    llm_pos_ids_list.append(mint.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(mint.arange(text_len).view(1, -1).expand((3, -1)) + st_idx)

                llm_positions = mint.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = ms.Tensor(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand((3, -1, -1))
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = mint.arange(input_ids.shape[1]).view(1, 1, -1).expand((3, input_ids.shape[0], -1))
                mrope_position_deltas = mint.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(self, pixel_values_videos: ms.Tensor, video_grid_thw: Optional[ms.Tensor] = None):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`ms.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`ms.Tensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = mint.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(self, pixel_values: ms.Tensor, image_grid_thw: Optional[ms.Tensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`ms.Tensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = mint.split(image_embeds, split_sizes)
        return image_embeds

    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[ms.Tensor] = None,
        pixel_values_videos: Optional[ms.Tensor] = None,
        image_grid_thw: Optional[ms.Tensor] = None,
        video_grid_thw: Optional[ms.Tensor] = None,
        rope_deltas: Optional[ms.Tensor] = None,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2VLModelOutputWithPast]:
        r"""
        image_grid_thw (`ms.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`ms.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`ms.Tensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = mint.cat(image_embeds, dim=0)

            if input_ids is None:
                image_mask = inputs_embeds == self.get_input_embeddings()(
                    ms.Tensor(self.config.image_token_id, dtype=ms.int64)
                )
                image_mask = image_mask.all(-1)
            else:
                image_mask = input_ids == self.config.image_token_id

            n_image_tokens = image_mask.sum()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            n_image_features = image_embeds.shape[0]

            image_embeds = image_embeds.to(inputs_embeds.dtype)
            inputs_embeds = (
                inputs_embeds.float().masked_scatter(image_mask, image_embeds.float()).to(inputs_embeds.dtype)
            )

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = mint.cat(video_embeds, dim=0)

            if input_ids is None:
                video_mask = inputs_embeds == self.get_input_embeddings()(
                    ms.Tensor(self.config.image_token_id, dtype=ms.int64)
                )
                n_video_tokens = (video_mask).sum(dim=1).sum(dim=0)[0]
            else:
                video_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                video_mask = video_mask.expand_as(inputs_embeds)
                n_video_tokens = (input_ids == self.config.image_token_id).sum()

            n_video_features = video_embeds.shape[0]

            video_embeds = video_embeds.to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.float().masked_scatter(video_mask, video_embeds.float()).to(inputs_embeds)

        if position_ids is None:
            if self.rope_deltas is None or cache_position is None or cache_position[0] == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = mint.arange(seq_length)
                position_ids = position_ids.view(1, 1, -1).expand((3, batch_size, -1))
                if cache_position is not None:
                    delta = cache_position[0] + self.rope_deltas
                else:
                    delta = mint.zeros((batch_size, seq_length))
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids += delta

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel(config.vision_config)
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: ms.Tensor,
        image_grid_thw: Optional[ms.Tensor] = None,
        video_grid_thw: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`): Long
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`ms.Tensor` of shape `(num_images, 3)`, *optional*): Long
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`ms.Tensor` of shape `(num_videos, 3)`, *optional*): Long
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`ms.Tensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`ms.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = ops.ones_like(total_input_ids)
            position_ids = ops.ones((3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype)
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = ops.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(ops.arange(text_len).view(1, -1).broadcast_to((3, -1)) + st_idx)

                    t_index = (
                        ops.arange(llm_grid_t)
                        .view((-1, 1))
                        .broadcast_to((-1, llm_grid_h * llm_grid_w))
                        .flatten(start_dim=0)
                    )
                    h_index = (
                        ops.arange(llm_grid_h)
                        .view((1, -1, 1))
                        .broadcast_to((llm_grid_t, -1, llm_grid_w))
                        .flatten(start_dim=0)
                    )
                    w_index = (
                        ops.arange(llm_grid_w)
                        .view((1, 1, -1))
                        .broadcast_to((llm_grid_t, llm_grid_h, -1))
                        .flatten(start_dim=0)
                    )
                    llm_pos_ids_list.append(ops.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(ops.arange(text_len).view((1, -1)).broadcast_to((3, -1)) + st_idx)

                llm_positions = ops.cat(llm_pos_ids_list, axis=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(input_ids.dtype)
                mrope_position_deltas.append(llm_positions.max().item() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = ms.Tensor(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.int().cumsum(-1) - 1
                position_ids = ops.masked_fill(position_ids.long(), attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).broadcast_to((3, -1, -1))
                max_position_ids = position_ids.max(0, keepdims=False)[0].max(-1, keepdims=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = ops.arange(input_ids.shape[1]).view((1, 1, -1)).broadcast_to((3, input_ids.shape[0], -1))
                mrope_position_deltas = ops.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    @add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[ms.Tensor] = None,
        pixel_values_videos: Optional[ms.Tensor] = None,
        image_grid_thw: Optional[ms.Tensor] = None,
        video_grid_thw: Optional[ms.Tensor] = None,
        rope_deltas: Optional[ms.Tensor] = None,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        >>> from mindspore import Tensor

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(Tensor(inputs.input_ids), max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = (
                    inputs_embeds.float().masked_scatter(image_mask, image_embeds.float()).to(inputs_embeds.dtype)
                )
                # masked_scatter does not support bf16

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.to(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = (
                    inputs_embeds.float().masked_scatter(video_mask, video_embeds.float()).to(inputs_embeds.dtype)
                )
                # masked_scatter does not support bf16

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            past_seen_tokens = 0
            if past_key_values is not None:
                past_seen_tokens = (
                    get_seq_length(past_key_values)
                    if isinstance(past_key_values, tuple)
                    else past_key_values.get_seq_length()
                )
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_seen_tokens == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = ops.arange(seq_length)
                position_ids = position_ids.view(1, -1).broadcast_to((batch_size, -1))
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).broadcast_to((3, -1, -1))

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()  # some logit warper operations need fp32

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view((-1, self.config.vocab_size))
            shift_labels = shift_labels.view((-1))
            # Enable model parallelism
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        # Exception 4: If input_embeds are passed then slice it through `cache_position`, to keep only the unprocessed tokens and
        # generate the first token for each sequence. Later use the generated Input ids for continuation.
        if past_key_values is not None and isinstance(past_key_values, Cache):
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or (cache_position[-1] >= input_ids.shape[1]):  # Exception 1  # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        past_length = 0
        if past_key_values is not None and isinstance(past_key_values, tuple):
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else get_seq_length(past_key_values)
            max_cache_length = get_max_length(past_key_values) if get_max_length(past_key_values) is not None else None
            cache_length = past_length if max_cache_length is None else ops.minimum(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and int(attention_mask.sum(-1).max()) > input_ids.shape[1]:
                input_ids = input_ids[:, -(int(attention_mask.sum(-1).max()) - int(past_length)) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, int(past_length) :]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, tuple) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
            else:
                batch_size, sequence_length = input_ids.shape

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=get_max_length(past_key_values),
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": ms.mutable(past_key_values)
                if isinstance(past_key_values, tuple)
                else past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
            }
        )
        return model_inputs


if __name__ == "__main__":
    # Debug and testing use only

    import time

    from PIL import Image

    ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)
    # ms.set_context(mode = ms.GRAPH_MODE) # NOT SUPPORTED YET

    # Qwen2-VL-7B-Instruct config.json
    config_json = {
        "architectures": ["Qwen2VLForConditionalGeneration"],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "vision_token_id": 151654,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "hidden_act": "silu",
        "hidden_size": 3584,
        "initializer_range": 0.02,
        "intermediate_size": 18944,
        "max_position_embeddings": 32768,
        "max_window_layers": 28,
        "model_type": "qwen2_vl",
        "num_attention_heads": 28,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "sliding_window": 32768,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.41.2",
        "use_cache": True,
        "use_sliding_window": False,
        "vision_config": {
            "depth": 32,
            "embed_dim": 1280,
            "mlp_ratio": 4,
            "num_heads": 16,
            "in_chans": 3,
            "hidden_size": 3584,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "spatial_patch_size": 14,
            "temporal_patch_size": 2,
        },
        "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
        "vocab_size": 152064,
        "attn_implementation": "flash_attention_2",
    }

    # TEST: loading model
    start_time = time.time()
    config = Qwen2VLConfig(**config_json)
    try:
        model = Qwen2VLForConditionalGeneration(config)
        print("*" * 100)
        print("Test passed: Sucessfully loaded Qwen2VLForConditionalGeneration")
        print("Time elapsed: %.4fs" % (time.time() - start_time))
        print("*" * 100)
    except RuntimeError:
        raise RuntimeError("Load Qwen2VLForConditionalGeneration Error.")

    # TEST: load processor
    start_time = time.time()
    from transformers import Qwen2VLProcessor  # Qwen2VLImageProcessor, Qwen2TokenizerFast

    try:
        # processor = Qwen2VLProcessor(image_processor=Qwen2VLImageProcessor(), tokenizer=Qwen2TokenizerFast(), chat_template=None)
        processor = Qwen2VLProcessor.from_pretrained("Qwen2-VL/Qwen2-VL-7B-Instruct")
        print("*" * 100)
        print("Test passed: Sucessfully loaded Qwen2VLProcessor")
        print("Time elapsed: %.4fs" % (time.time() - start_time))
        print("*" * 100)
    except RuntimeError:
        raise RuntimeError("Load Qwen2VLProcessor Error.")

    # TEST: process input
    start_time = time.time()

    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg", # REPLACE_WITH_YOUR_IMAGE_PATH
    #             },
    #             {"type": "text", "text": "Describe this image."},
    #         ],
    #     }
    # ]
    # # prepare text inuput
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # prepare vision input
    # image_inputs, video_inputs = process_vision_info(messages) # a list of PIL Images
    w, h = 1024, 512
    text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
        <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant"
    image_path = "demo.jpeg"  # REPLACE with your image
    image_inputs = [Image.open(image_path).convert("RGB").resize((w, h))]
    # image = np.uint8(np.random.rand(h, w, 3) * 255)
    # image_inputs = [Image.fromarray(image).convert("RGB")]
    video_inputs = None
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="np",
    )
    # convert input to Tensor
    for key, value in inputs.items():  # by default input numpy array or list
        if isinstance(value, np.ndarray):
            inputs[key] = ms.Tensor(value)
        elif isinstance(value, list):
            inputs[key] = ms.Tensor(value)
        if inputs[key].dtype == ms.int64:
            inputs[key] = inputs[key].to(ms.int32)  # "input_ids", "attention_mask", "image_grid_thw"

    print("*" * 100)
    print("Test passed: Sucessfully processed input data using Qwen2VLProcessor")
    print("Time elapsed: %.4fs" % (time.time() - start_time))
    print("*" * 100)

    # TEST: dummy inference
    start_time = time.time()
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        print("*" * 100)
        print("Test passed: Sucessfully generated tokens using Qwen2VLForConditionalGeneration")
        print(f"generated_ids length / #steps: {len(generated_ids[0])}")
        elapsed = time.time() - start_time
        print("Time elapsed: %.4fs" % (elapsed))
        print("Average speed %.4fs/step" % (elapsed / len(generated_ids[0])))
        print("*" * 100)
    except RuntimeError:
        raise RuntimeError("Run generate() Error.")

    start_time = time.time()
    try:
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print("*" * 100)
        print("Test passed: Sucessfully detokenize generated tokens")
        print("Time elapsed: %.4fs" % (time.time() - start_time))
        print("*" * 100)
    except RuntimeError:
        raise RuntimeError("Run Qwen2VLProcessor.decode() Error.")
