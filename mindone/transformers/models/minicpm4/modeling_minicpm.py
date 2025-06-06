""" MindSpore MiniCPM model."""
import math
import re
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from mindone.transformers.activations import ACT2FN
from mindone.transformers.cache_utils import Cache, get_max_length, get_seq_length, update
from mindone.transformers.mindspore_adapter import str_to_dtype
from mindone.transformers.mindspore_adapter.paged_attention_block_tables import BlockTables
from mindone.transformers.mindspore_adapter.paged_attention_freqs import FreqsMgr
from mindone.transformers.mindspore_adapter.paged_attention_infer_attention_block import InferAttention
from mindone.transformers.mindspore_adapter.paged_attention_mask import LowerTriangularMaskWithDynamic
from mindone.transformers.mindspore_utils import ALL_LAYERNORM_LAYERS
from mindone.transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from mindone.transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from mindone.transformers.modeling_utils import PreTrainedModel

from .configuration_minicpm import MiniCPMConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MiniCPMConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=ms.int32)
    indices = mint.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(mint.cumsum(seqlens_in_batch, dim=0, dtype=ms.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: ms.Tensor, dtype: ms.Type, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.minicpm.modeling_minicpm._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. "
        "Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(input_ids_shape, dtype: ms.Type, past_key_values_length: int = 0):
    warnings.warn(
        "Calling `transformers.models.minicpm.modeling_minicpm._make_causal_mask` is deprecated and will be removed in v4.37. "
        "Use `transformers.models.minicpm.modeling_minicpm.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, past_key_values_length=past_key_values_length
    )


# @torch.jit.script  # type: ignore
def rms_layernorm(hidden: ms.Tensor, weight: ms.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(ms.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * mint.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class MiniCPMRMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniCPMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(mint.ones(hidden_size))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(MiniCPMRMSNorm)


class MiniCPMRotaryEmbedding(nn.Cell):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (ops.arange(0, self.dim, 2).float() / self.dim))
        self.inv_freq = inv_freq

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=ms.float32)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)

        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def construct(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # if seq_len > self.max_seq_len_cached:
        #     self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MiniCPMLinearScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


class MiniCPMDynamicNTKScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (ops.arange(0, self.dim, 2).float() / self.dim))
            self.inv_freq = inv_freq

        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)

        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


class MiniCPMLongRoPE(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        short_factor=None,
        long_factor=None,
        original_max_position_embeddings=None,
    ):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        if seq_len > self.original_max_position_embeddings:
            ext_factors = ms.tensor(self.long_factor, dtype=ms.float32)
        else:
            ext_factors = ms.tensor(self.short_factor, dtype=ms.float32)

        freqs = mint.mul(ops.outer(t, 1.0 / ext_factors), self.inv_freq.to(dtype))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().to(dtype) * self.scaling_factor
        self.sin_cached = emb.sin().to(dtype) * self.scaling_factor


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=ms.float32)
    k_fp32 = k.to(dtype=ms.float32)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


class MiniCPMMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = ops.cat(
                [mint.matmul(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], axis=-1
            )
            up_proj = ops.cat([mint.matmul(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], axis=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                mint.matmul(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


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


class MiniCPMAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MiniCPMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisiable by num_heads (got 'hidden_size': {self.hidden_size}"
                f" and 'num_heads': {self.num_heads}"
            )

        self.q_proj = nn.Dense(
            self.hidden_size,
            self.num_heads * self.head_dim,
            has_bias=config.attention_bias,
        )
        self.k_proj = nn.Dense(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            has_bias=config.attention_bias,
        )
        self.v_proj = nn.Dense(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            has_bias=config.attention_bias,
        )
        self.o_proj = nn.Dense(
            self.num_heads * self.head_dim,
            self.hidden_size,
            has_bias=config.attention_bias,
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["rope_type"]
            scaling_factor = self.config.rope_scaling.get("factor", None)
            if scaling_type == "linear":
                self.rotary_emb = MiniCPMLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MiniCPMDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "longrope":
                self.rotary_emb = MiniCPMLongRoPE(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    short_factor=self.config.rope_scaling["short_factor"],
                    long_factor=self.config.rope_scaling["long_factor"],
                    base=self.rope_theta,
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"],
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: ms.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2).contiguous()

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.shape

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, axis=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, axis=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, axis=0)

            query_states = [ops.dense(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = ops.cat(query_states, axis=-1)

            key_states = [ops.dense(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = ops.cat(key_states, axis=-1)

            value_states = [ops.dense(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = ops.cat(value_states, axis=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            # if self.layer_idx is None:
            #     raise ValueError(
            #         f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
            #         "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
            #         "with a layer index."
            #     )
            # kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            kv_seq_len = past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states.to(ms.float32), seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None and use_cache:
            key_states, value_states = update(past_key_value, key_states, value_states, cache_position)
            past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = mint.matmul(query_states, key_states.swapaxes(2, 3)) / (self.head_dim**0.5)

        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = mint.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query_states.dtype)
        attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = mint.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, axis=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, axis=1)
            attn_output = sum([ops.dense(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MiniCPMFlashAttention2(MiniCPMAttention):
    """
    MiniCPM flash attention module. This module inherits from `MiniCPMAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, config: MiniCPMConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        scale_factor = 1 / math.sqrt(self.head_dim)
        self.flash_attention = FlashAttentionScore(
            self.num_heads, keep_prob=1 - self.attention_dropout, scale_value=scale_factor, input_layout="BNSD"
        )

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        # MiniCPMFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len = past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states.to(ms.float32), seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states, value_states = update(past_key_value, key_states, value_states, cache_position)
            past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == ms.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.head_dim ** (-0.5),
        )

        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # 1. flash attention
        if attention_mask is not None:  # no matter the length, we just slice it
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # flip mask to ms FA format, 1 - drop, 0 - retain
        attention_mask = (-attention_mask).to(ms.bool_)
        _, _, _, attn_output = self.flash_attention(
            query_states, key_states, value_states, None, None, None, attention_mask
        )

        return attn_output


class MiniCPMPagedAttention(MiniCPMAttention):
    """Paged Attention"""

    def __init__(self, config: MiniCPMConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        compute_dtype = str_to_dtype(config.mindspore_dtype)

        self.infer_attention = InferAttention(
            config.num_attention_heads,
            self.head_dim,
            config.num_key_value_heads,
            seq_length=config.max_position_embeddings,
            pa_n_head_split=config.num_attention_heads,
            pa_n_kv_head_split=self.head_dim,
            scale_value=1.0 / (math.sqrt(self.head_dim)),
            pre_tokens=2147483647,
            next_tokens=0,
            block_size=32,
            num_blocks=1024,
            is_dynamic=True,
            use_flash_attention=True,
            use_rope_rotary_emb=False,
            compute_dtype=compute_dtype,
        )

        self.is_first_iteration = True

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[ms.Tensor] = None,
        block_tables: Optional[ms.Tensor] = None,
        slot_mapping: Optional[ms.Tensor] = None,
        freqs_cis: Optional[ms.Tensor] = None,
        mask: Optional[ms.Tensor] = None,
        batch_valid_length: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = batch_valid_length[0].to(ms.int64)

        cos, sin = self.rotary_emb(value_states.to(ms.float32), seq_len=kv_seq_len)

        if not self.is_first_iteration:
            length = position_ids.shape[1]
            position_ids = position_ids[:, length - 1 : length]

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        query_states = query_states.swapaxes(1, 2).reshape(bsz, q_len, -1)
        key_states = key_states.swapaxes(1, 2).reshape(bsz, q_len, -1)
        value_states = value_states.swapaxes(1, 2).reshape(bsz, q_len, -1)

        if not self.is_first_iteration:
            query_states = query_states[:, -1, :].reshape(bsz, 1, -1)
            key_states = key_states[:, -1, :].reshape(bsz, 1, -1)
            value_states = value_states[:, -1, :].reshape(bsz, 1, -1)

        attn_output = self.infer_attention(
            query_states,
            key_states,
            value_states,
            batch_valid_length,
            block_tables,
            slot_mapping,
            freqs_cis,
            mask,
            q_seq_lens=None,
        )

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


MINICPM_ATTENTION_CLASSES = {
    "eager": MiniCPMAttention,
    "flash_attention_2": MiniCPMFlashAttention2,
    "paged_attention": MiniCPMPagedAttention,
}


class MiniCPMDecoderLayer(nn.Cell):
    def __init__(self, config: MiniCPMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MINICPM_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

        if config._attn_implementation == "paged_attention":
            self.is_first_iteration = True

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[ms.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[ms.Tensor] = None,
        block_tables: Optional[ms.Tensor] = None,
        slot_mapping: Optional[ms.Tensor] = None,
        freqs_cis: Optional[ms.Tensor] = None,
        mask: Optional[ms.Tensor] = None,
        batch_valid_length: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[Tuple[ms.Tensor, ms.Tensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

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
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            freqs_cis=freqs_cis,
            mask=mask,
            batch_valid_length=batch_valid_length,
            **kwargs,
        )

        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


MINICPM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Cell](https://pytorch.org/docs/stable/nn.html#torch.nn.Cell) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`MiniCPMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPMPreTrainedModel(PreTrainedModel):
    config_class = MiniCPMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniCPMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = False

    def _init_weights(self, module):
        pass


MINICPM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
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
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPMModel(MiniCPMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiniCPMDecoderLayer`]
    Args:
        config: MiniCPMConfig
    """

    def __init__(self, config: MiniCPMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = mint.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.CellList(
            [MiniCPMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        if self.config._attn_implementation == "paged_attention":
            self.is_first_iteration = True

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
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
        cache_position: Optional[ms.Tensor] = None,
        block_tables: Optional[ms.Tensor] = None,
        slot_mapping: Optional[ms.Tensor] = None,
        freqs_cis: Optional[ms.Tensor] = None,
        mask: Optional[ms.Tensor] = None,
        batch_valid_length: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            # use_legacy_cache = not isinstance(past_key_values, Cache)
            # if use_legacy_cache:
            #     past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_length = past_key_values.get_usable_length(seq_length)
            past_key_values_length = get_seq_length(past_key_values)

        if position_ids is None:
            if block_tables is None:
                position_ids = ops.arange(past_key_values_length, seq_length + past_key_values_length, dtype=ms.int64)
                position_ids = position_ids.unsqueeze(0)
            else:
                position_ids = ops.arange(
                    past_key_values_length, batch_valid_length[0] + past_key_values_length, dtype=ms.int64
                )
                position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb

        if block_tables is None:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_caches = () if use_cache else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values[layer_idx] if past_key_values is not None else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    freqs_cis=freqs_cis,
                    mask=mask,
                    batch_valid_length=batch_valid_length,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_caches += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_caches, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_caches,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MiniCPMForCausalLM(MiniCPMPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        if self.config._attn_implementation == "paged_attention":
            compute_dtype = str_to_dtype(config.mindspore_dtype)

            self.freqs_mgr = FreqsMgr(
                head_dim=config.hidden_size // config.num_attention_heads,
                seq_length=config.max_position_embeddings,
                max_position_embedding=config.max_position_embeddings,
                rotary_dtype=compute_dtype,
                theta=config.rope_theta,
                is_dynamic=True,
            )

            self.casual_mask = LowerTriangularMaskWithDynamic(
                seq_length=config.max_position_embeddings,
                batch_size=1,
                compute_type=compute_dtype,
                is_dynamic=True,
                pad_token_id=config.pad_token_id,
                use_flash_attention=True,
                use_attn_mask_compression=False,
                use_past=True,
                seq_split_num=1,
                chunk_prefill=False,
            )

            self.is_first_iteration = True

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

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attn.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attn.infer_attention.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attn.infer_attention.paged_attention_mgr.add_flags(is_first_iteration=is_first_iteration)

    def enable_dynamic_shape(self):
        input_ids = Tensor(shape=[None, None], dtype=ms.int32)
        position_ids = Tensor(shape=[None, None], dtype=ms.int32)
        attention_mask = None
        past_key_values = None
        inputs_embeds = None
        labels = None
        use_cache = False
        output_attentions = False
        output_hidden_states = False
        return_dict = False
        cache_position = Tensor(shape=[None], dtype=ms.int32)
        block_tables = Tensor(shape=[None, None], dtype=ms.int32)
        slot_mapping = Tensor(shape=[None], dtype=ms.int32)
        batch_valid_length = ms.mutable(Tensor(shape=[None], dtype=ms.int32))

        self.set_inputs(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            cache_position,
            block_tables,
            slot_mapping,
            batch_valid_length,
        )

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ms.Tensor, ms.Tensor]]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[ms.Tensor] = None,
        block_tables: Optional[ms.Tensor] = None,
        slot_mapping: Optional[ms.Tensor] = None,
        batch_valid_length: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        # >>> from transformers import AutoTokenizer, MiniCPMForCausalLM
        # >>> model = MiniCPMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        # >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        # >>> prompt = "Hey, are you conscious? Can you talk to me?"
        # >>> inputs = tokenizer(prompt, return_tensors="pt")
        # >>> # Generate
        # >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        # >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if block_tables is not None:
            bs, seq_len = input_ids.shape
            mask = None
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
                mask = self.casual_mask.prefill()
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        else:
            freqs_cis = None
            mask = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            freqs_cis=freqs_cis,
            mask=mask,
            batch_valid_length=batch_valid_length,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [mint.matmul(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = ops.cat(logits, axis=-1)
        else:
            logits = self.lm_head(hidden_states / (self.config.hidden_size / self.config.dim_model_base))
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=False,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
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

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.to(ms.int32).cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values and past_length > 0:
                cur_len = attention_mask.sum(-1).max()
                position_ids = position_ids[:, cur_len - input_ids.shape[1] : cur_len]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # TODO: use `next_tokens` directly instead.
            if not isinstance(input_ids, Tensor):
                input_ids = Tensor(input_ids, dtype=ms.int32)

            # Padding to max_len when no cache
            if past_key_values is None:
                pad_len = max(0, attention_mask.shape[1] - input_ids.shape[1])
                input_ids = ops.pad(input_ids, (0, pad_len), value=0)

            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]

        if cache_position is None:
            cache_position = ops.arange(past_length, past_length + input_length)
        elif use_cache:
            if input_length < cache_position.shape[0]:
                assert cache_position.shape[0] == attention_mask.shape[-1]
                cur_len = int(attention_mask.sum(-1).max())
                cache_position = cache_position[cur_len - input_length : cur_len]
            else:
                cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": ms.mutable(past_key_values) if past_key_values is not None else None,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )

        # Paged Attention
        if self.config._attn_implementation == "paged_attention":
            bs, seq_len = input_ids.shape
            step = kwargs["step"]
            if step == 0:
                self.enable_dynamic_shape()

                # init block tables
                self.block_mgr = BlockTables(1024, 32, self.config.max_position_embeddings)
                self.block_mgr.init_cache_engine(bs)

                # get slot mapping and block tables
                max_input_length = self.config.max_position_embeddings
                self.valid_length_each_example = ms.tensor(seq_len).reshape(bs)
                block_tables, slot_mapping = self.block_mgr.assemble_pa_full_inputs(
                    max_input_length, self.valid_length_each_example, [False]
                )
                slot_mapping = np.delete(slot_mapping, np.where(slot_mapping == -1))

                # set batch valid length
                self.batch_valid_length = ms.tensor(seq_len).to(ms.int32).reshape(bs)

                self.phase = "prefill"
                self.add_flags_custom(True)
            else:
                model_inputs.update({"input_ids": input_ids[:, -1].reshape(bs, 1)})

                # get slot mapping and block tables
                self.valid_length_each_example += 1
                block_tables, slot_mapping = self.block_mgr.assemble_pa_inc_inputs(
                    self.valid_length_each_example, [False]
                )

                # set batch valid length
                self.batch_valid_length += 1

                if step == 1:
                    self.phase = "increment"
                    self.add_flags_custom(False)
            slot_mapping = ms.tensor(slot_mapping)
            block_tables = ms.tensor(block_tables)
            model_inputs.update(
                {
                    "attention_mask": None,
                    "block_tables": block_tables,
                    "slot_mapping": slot_mapping,
                    "batch_valid_length": self.batch_valid_length,
                }
            )
            model_inputs.pop("step", None)
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def chat(
        self,
        tokenizer,
        query: str,
        history: List[Dict] = None,
        role: str = "user",
        max_length: int = 4096,
        num_beams=1,
        do_sample=True,
        top_p=0.8,
        temperature=0.3,
        logits_processor=None,
        **kwargs,
    ):
        if history is None:
            history = []
        if logits_processor:
            gen_kwargs = {
                "max_length": max_length,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "top_p": top_p,
                "temperature": temperature,
                "logits_processor": logits_processor,
                **kwargs,
            }
        else:
            gen_kwargs = {
                "max_length": max_length,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "top_p": top_p,
                "temperature": temperature,
                "logits_processor": logits_processor,
                **kwargs,
            }

        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(history_str, return_tensors="np")
        for key in inputs.keys():
            inputs[key] = ms.tensor(inputs[key])
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) : -1]
        response = tokenizer.decode(outputs)
        pattern = re.compile(r".*?(?=<AI>|<用户>)", re.DOTALL)
        matches = pattern.findall(response)
        if len(matches) > 0:
            response = matches[0]
        history.append({"role": "assistant", "content": response})
        return response, history


@add_start_docstrings(
    """
    The MiniCPM Model transformer with a sequence classification head on top (linear layer).
    [`MiniCPMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.
    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MINICPM_START_DOCSTRING,
)
class MiniCPMForSequenceClassification(MiniCPMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MiniCPMModel(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
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
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (mint.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[ops.arange(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == ms.int64 or labels.dtype == ms.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
