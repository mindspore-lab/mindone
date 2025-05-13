# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import mindspore as ms
from mindspore import Tensor, mint, nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache, get_seq_length, update
from ...generation import GenerationMixin
from ...mindspore_adapter import dtype_to_min, str_to_dtype
from ...mindspore_adapter.paged_attention_freqs import FreqsMgr
from ...mindspore_adapter.paged_attention_infer_attention_block import InferAttention
from ...mindspore_adapter.paged_attention_mask import LowerTriangularMaskWithDynamic
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-8B"
_CONFIG_FOR_DOC = "Qwen3Config"


class Qwen3RMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = ms.Parameter(ms.tensor(np.ones(hidden_size), ms.float32), name="weight")
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * mint.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Cell):
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
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mint.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`ms.Tensor`): The query tensor.
        k (`ms.Tensor`): The key tensor.
        cos (`ms.Tensor`): The cosine part of the rotary embedding.
        sin (`ms.Tensor`): The sine part of the rotary embedding.
        position_ids (`ms.Tensor`, *optional*):
            Deprecated and unused.
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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: ms.Tensor, n_rep: int) -> ms.Tensor:
    """
    This is the equivalent of mint.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].broadcast_to((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Cell,
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = mint.matmul(query, key_states.swapaxes(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = mint.nn.functional.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query.dtype)
    attn_weights = mint.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = mint.matmul(attn_weights, value_states)
    attn_output = attn_output.swapaxes(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3Attention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Dense(
            config.hidden_size, config.num_attention_heads * self.head_dim, has_bias=config.attention_bias
        )
        self.k_proj = nn.Dense(
            config.hidden_size, config.num_key_value_heads * self.head_dim, has_bias=config.attention_bias
        )
        self.v_proj = nn.Dense(
            config.hidden_size, config.num_key_value_heads * self.head_dim, has_bias=config.attention_bias
        )
        self.o_proj = nn.Dense(
            config.num_attention_heads * self.head_dim, config.hidden_size, has_bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None
        else:
            # TODO to add sliding window in cache utils
            raise NotImplementedError("Sliding window is not supported yet.")

    def construct(
        self,
        hidden_states: ms.Tensor,
        position_embeddings: Tuple[ms.Tensor, ms.Tensor],
        attention_mask: Optional[ms.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).swapaxes(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).swapaxes(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).swapaxes(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            if isinstance(past_key_value, Cache):
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            elif isinstance(past_key_value, tuple):
                key_states, value_states = update(past_key_value, key_states, value_states, cache_position)
                past_key_value = (key_states, value_states)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa":
                logger.warning_once(
                    "MindSpore dose not support `scaled_dot_product_attention` yet. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class Qwen3PageAttention(Qwen3Attention):
    """
    Qwen3 page attention module
    """

    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        compute_dtype = str_to_dtype(config.mindspore_dtype)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

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
            rotary_cos_format=2,
            compute_dtype=compute_dtype,
        )

        self.is_first_iteration = True

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[tuple[ms.Tensor, ms.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[ms.Tensor] = None,
        block_tables: Optional[ms.Tensor] = None,
        slot_mapping: Optional[ms.Tensor] = None,
        freqs_cis: Optional[ms.Tensor] = None,
        mask: Optional[ms.Tensor] = None,
        batch_valid_length: Optional[ms.Tensor] = None,
        **kwargs,
    ):
        bs, seq_len, hidden_dim = hidden_states.shape
        q_bsnd_shape = (bs, seq_len, self.num_attention_heads, self.head_dim)
        q_bsh_shape = (bs, seq_len, self.num_attention_heads * self.head_dim)

        kv_bsnd_shape = (bs, seq_len, self.num_key_value_heads, self.head_dim)
        kv_bsh_shape = (bs, seq_len, self.num_key_value_heads * self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(q_bsnd_shape)).view(q_bsh_shape)
        key_states = self.k_norm(self.k_proj(hidden_states).view(kv_bsnd_shape)).view(kv_bsh_shape)
        value_states = self.v_proj(hidden_states)

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


QWEN3_ATTENTION_CLASSES = {
    "eager": Qwen3Attention,
    "flash_attention_2": Qwen3Attention,  # fa and eager share project, norm and rope
    "paged_attention": Qwen3PageAttention,
}


class Qwen3DecoderLayer(nn.Cell):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QWEN3_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if (
            config.sliding_window and config._attn_implementation != "flash_attention_2"
        ):  # diff with Llama is this warning
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

        self.is_first_iteration = True

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[ms.Tensor] = None,
        position_embeddings: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,  # necessary, but kept here for BC
        block_tables: Optional[ms.Tensor] = None,
        slot_mapping: Optional[ms.Tensor] = None,
        freqs_cis: Optional[ms.Tensor] = None,
        mask: Optional[ms.Tensor] = None,
        batch_valid_length: Optional[ms.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[ms.Tensor, Optional[Tuple[ms.Tensor, ms.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        is_page_attention = block_tables is not None
        if is_page_attention:
            kwargs.update(
                {
                    "block_tables": block_tables,
                    "slot_mapping": slot_mapping,
                    "freqs_cis": freqs_cis,
                    "mask": mask,
                    "batch_valid_length": batch_valid_length,
                }
            )
        hidden_states, self_attn_weights, next_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
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
            outputs += (next_cache,)

        return outputs


class Qwen3RotaryEmbedding(nn.Cell):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        self.inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self.original_inv_freq = self.inv_freq

    # fixme currently graph mode dose not dynamic_rope_update. To write a graceful if-else branch.
    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def construct(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().broadcast_to((position_ids.shape[0], -1, 1))
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 start
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).swapaxes(1, 2)
        emb = mint.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        # Force float32 end

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


QWEN3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a MindSpore
    [mindspore.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) subclass.
    Use it as a regular MindSpore Module and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = False  # SDPA, not support yet
    _supports_flex_attn = False  # FlexAttention, not support yet
    _supports_cache_class = False  # this will affect the padding behavior of input id
    _supports_quantized_cache = True
    _supports_static_cache = False  # StaticCache, not used
    _supports_attention_backend = True

    def _init_weights(self, module):
        if not self.training:
            return

        std = self.config.initializer_range
        if isinstance(module, nn.Dense):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN3_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
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
        cache_position (`ms.Tensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class Qwen3Model(Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = mint.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.CellList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.is_first_iteration = True

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # mindspore graph not support @can_return_tuple decorator
    # @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[ms.Tensor] = None,
        block_tables: Optional[ms.Tensor] = None,
        slot_mapping: Optional[ms.Tensor] = None,
        freqs_cis: Optional[ms.Tensor] = None,
        mask: Optional[ms.Tensor] = None,
        batch_valid_length: Optional[ms.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) and (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = get_seq_length(past_key_values) if past_key_values is not None else 0
            cache_position = mint.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        is_page_attention = block_tables is not None
        # for paged attention, casual_mask is processed in the attention layer
        causal_mask = (
            self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
            if not is_page_attention
            else attention_mask
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_caches = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    freqs_cis=freqs_cis,
                    mask=mask,
                    batch_valid_length=batch_valid_length,
                    **flash_attn_kwargs,
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

        return hidden_states, next_caches, all_hidden_states, all_self_attns

    def _update_causal_mask(
        self,
        attention_mask: ms.Tensor,
        input_tensor: ms.Tensor,
        cache_position: ms.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        # if self.config._attn_implementation == "flash_attention_2":
        #     if attention_mask is not None and past_key_values is not None:
        #         is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
        #         if is_padding_right:
        #             raise ValueError(
        #                 "You are attempting to perform batched generation with padding_side='right'"
        #                 " this may lead to unexpected behaviour for Flash Attention version of Qwen3. Make sure to "
        #                 " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
        #             )
        #     if attention_mask is not None and 0.0 in attention_mask:
        #         return attention_mask
        #     return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = get_seq_length(past_key_values) if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        # using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)
        using_sliding_window_cache = False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        min_dtype = dtype_to_min(dtype)
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
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
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if self.config._attn_implementation == "sdpa" and attention_mask is not None and not output_attentions:
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: ms.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: ms.dtype,
        cache_position: ms.Tensor,
        batch_size: int,
        config: Qwen3Config,
        past_key_values: Cache,
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
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`ms.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`ms.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`ms.Tensor`):
                Batch size.
            config (`Qwen3Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = dtype_to_min(dtype)
            causal_mask = mint.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype)
            diagonal_attend_mask = mint.arange(target_length) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = mint.arange(target_length) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].broadcast_to((batch_size, 1, -1, -1))
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    ...


class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        if self.config._attn_implementation == "paged_attention":
            compute_dtype = str_to_dtype(config.mindspore_dtype)

            self.freqs_mgr = FreqsMgr(
                head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
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

    def enable_dynamic_shape(self):
        input_ids = Tensor(shape=[None, None], dtype=ms.int32)
        position_ids = Tensor(shape=[None, None], dtype=ms.int32)
        attention_mask = Tensor(shape=[None, None], dtype=ms.int32)
        past_key_values = None
        inputs_embeds = None
        labels = None
        use_cache = False
        output_attentions = False
        output_hidden_states = False
        cache_position = None
        block_tables = Tensor(shape=[None, None], dtype=ms.int32)
        slot_mapping = Tensor(shape=[None], dtype=ms.int32)
        batch_valid_length = ms.mutable(Tensor(shape=[None], dtype=ms.int32))
        logits_to_keep = 0

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
            cache_position,
            block_tables,
            slot_mapping,
            batch_valid_length,
            logits_to_keep,
        )

    # @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[ms.Tensor] = None,
        block_tables: Optional[ms.Tensor] = None,
        slot_mapping: Optional[ms.Tensor] = None,
        batch_valid_length: Optional[ms.Tensor] = None,
        logits_to_keep: Union[int, ms.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `ms.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `ms.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        is_page_attention = block_tables is not None
        if is_page_attention:
            bs, seq_len = input_ids.shape
            mask = None
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
                mask = self.casual_mask.prefill()
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
            kwargs.update(
                {
                    "block_tables": block_tables,
                    "slot_mapping": slot_mapping,
                    "freqs_cis": freqs_cis,
                    "mask": mask,
                    "batch_valid_length": batch_valid_length,
                }
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        result = (loss, logits) + outputs[1:]
        result = tuple(v for v in result if v is not None)
        return result


@add_start_docstrings(
    """
    The Qwen3 Model transformer with a sequence classification head on top (linear layer).

    [`Qwen3ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    QWEN3_START_DOCSTRING,
)
class Qwen3ForSequenceClassification(Qwen3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3Model(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, ms.int32)
            token_indices = mint.arange(input_ids.shape[-1], dtype=ms.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[mint.arange(batch_size), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    The Qwen3 Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    QWEN3_START_DOCSTRING,
)
class Qwen3ForTokenClassification(Qwen3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3Model(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
The Qwen3 Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    QWEN3_START_DOCSTRING,
)
class Qwen3ForQuestionAnswering(Qwen3PreTrainedModel):
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = Qwen3Model(config)
        self.qa_outputs = nn.Dense(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    # @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        start_positions: Optional[ms.Tensor] = None,
        end_positions: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        r"""
        start_positions (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3Model",
    "Qwen3PreTrainedModel",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]
