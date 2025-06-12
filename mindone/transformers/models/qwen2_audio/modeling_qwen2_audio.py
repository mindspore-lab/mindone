# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""MindSpore Qwen2Audio model."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from transformers import Qwen2AudioConfig, Qwen2AudioEncoderConfig
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import mindspore as ms
from mindspore import Tensor, mint, nn, ops
from mindspore.common.initializer import Normal, initializer

from ...activations import ACT2FN
from ...cache_utils import Cache, StaticCache, get_seq_length
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import MSPreTrainedModel
from ...utils import is_flash_attn_2_available

if is_flash_attn_2_available:
    from mindspore.ops.operations.nn_ops import FlashAttentionScore as MSFlashAttention

import inspect

from mindone.transformers.mindspore_adapter.paged_attention_block_tables import BlockTables
from mindone.transformers.mindspore_adapter.utils import _MIN_FP16, dtype_to_min

from ...mindspore_adapter import dtype_to_max
from ..qwen2 import Qwen2ForCausalLM, modeling_qwen2

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen2AudioConfig"


@dataclass
class Qwen2AudioCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2Audio causal language model (or autoregressive) outputs.

    Args:
        loss (`ms.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`ms.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Pre-computed hidden-states that can be used to speed up auto-regressive (sequential) decoding. There are
            two sets of pre-computed hidden-states: key and values states in the self-attention blocks.
            The `past_key_values` are returned when `use_cache=True` is passed or when `config.use_cache=True`.
            It is a [`~cache_utils.Cache`] instance.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those
            that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
            all `input_ids` of shape `(batch_size, sequence_length)`.
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        attention_mask (`ms.Tensor`, *optional*):
            Attentions mask, used to update attention mask and position_ids.
    """

    loss: Optional[ms.Tensor] = None
    logits: ms.Tensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None
    attention_mask: Optional[ms.Tensor] = None


class Qwen2AudioAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.whisper.modeling_whisper.WhisperAttention.__init__ with Whisper->Qwen2Audio
    def __init__(
        self,
        config: Optional[Qwen2AudioConfig] = None,
        layer_idx: Optional[int] = None,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.layer_idx = layer_idx

        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=bias)

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    def _shape(self, tensor: ms.Tensor, seq_len: int, bsz: int):
        return tensor.view((bsz, seq_len, self.num_heads, self.head_dim)).swapaxes(1, 2).contiguous()

    def construct(
        self,
        hidden_states: ms.Tensor,
        key_value_states: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[ms.Tensor] = None,
        layer_head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        attn_weights = mint.matmul(query_states, key_states.swapaxes(2, 3))

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = mint.nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.shape != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.shape}"
                )
            attn_weights = layer_head_mask.view((1, -1, 1, 1)) * attn_weights

        attn_probs = mint.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = mint.matmul(attn_probs, value_states)

        if attn_output.shape != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, None


class Qwen2AudioFlashAttention2(Qwen2AudioAttention):
    """
    Qwen2Audio flash attention module. This module inherits from `Qwen2AudioAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.whisper.modeling_whisper.WhisperFlashAttention2.__init__ with Whisper->Qwen2Audio
    def __init__(self, config: Qwen2AudioConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.fa_dtype = ms.float16
        dropout_rate = 0.0 if not self.training else self.dropout
        self.flash_attention = MSFlashAttention(
            scale_value=self.head_dim**-0.5,
            head_num=self.num_heads,
            keep_prob=1 - dropout_rate,
            input_layout="BNSD",
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
        key_value_states: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[ms.Tensor] = None,
        layer_head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        # Qwen2AudioFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, tgt_len, _ = hidden_states.shape

        # get query, key, value proj => BNSD [batch_size, num_heads, sequence_length, head_dim]
        query_states = self.q_proj(hidden_states)
        query_states = query_states.reshape((tgt_len, self.num_heads, -1))
        key_states = self.k_proj(hidden_states)
        key_states = key_states.reshape((tgt_len, self.num_heads, -1))
        value_states = self.v_proj(hidden_states)
        value_states = value_states.reshape((tgt_len, self.num_heads, -1))

        query_states = query_states.unsqueeze(0).swapaxes(1, 2)
        value_states = value_states.unsqueeze(0).swapaxes(1, 2)
        key_states = key_states.unsqueeze(0).swapaxes(1, 2)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, : key_states.shape[-2]]
        causal_mask = self.convert_mask_to_fa_format(causal_mask)

        attn_output = self.flash_attention(
            query_states.to(self.fa_dtype),
            key_states.to(self.fa_dtype),
            value_states.to(self.fa_dtype),
            attn_mask=causal_mask,
        )[
            3
        ]  # BNSD (bsz, self.num_heads, q_len, self.head_dim)

        attn_output = attn_output.swapaxes(1, 2).reshape(bsz, tgt_len, -1)
        attn_output = self.out_proj(attn_output.to(hidden_states.dtype))

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, None


QWEN2AUDIO_ATTENTION_CLASSES = {
    "eager": Qwen2AudioAttention,
    "flash_attention_2": Qwen2AudioFlashAttention2,
}


# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoderLayer with Whisper->Qwen2Audio, WHISPER->QWEN2AUDIO
class Qwen2AudioEncoderLayer(nn.Cell):
    def __init__(self, config: Qwen2AudioConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = QWEN2AUDIO_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.self_attn_layer_norm = mint.nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Dense(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Dense(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = mint.nn.LayerNorm(self.embed_dim)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
        layer_head_mask: ms.Tensor,
        output_attentions: bool = False,
    ) -> ms.Tensor:
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`ms.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == ms.float16 and (ops.isinf(hidden_states).any() or ops.isnan(hidden_states).any()):
            clamp_value = dtype_to_max(hidden_states.dtype) - 1000
            hidden_states = mint.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if not output_attentions:
            attn_weights = None
        outputs += (attn_weights,)

        return outputs


QWEN2AUDIO_START_DOCSTRING = r"""
    This model inherits from [`MSPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a MindSpore [mindspore.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) subclass.
    Use it as a regular MindSpore Module and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2AudioConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~MSPreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2Audio Model outputting raw hidden-states without any specific head on top.",
    QWEN2AUDIO_START_DOCSTRING,
)
class Qwen2AudioPreTrainedModel(MSPreTrainedModel):
    config_class = Qwen2AudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2AudioAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = False

    def _init_weights(self, module):
        # important: this ported version of Qwen2Audio isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = self.config.init_std if hasattr(self.config, "init_std") else self.config.audio_config.init_std

        if isinstance(module, (mint.nn.Linear, nn.Conv1d)):
            weight = initializer(Normal(sigma=std, mean=0.0), shape=module.weight.shape)
            module.weight.set_data(weight)
            if module.bias is not None:
                bias_weight = initializer("zeros", module.bias.shape)
                module.bias.set_data(bias_weight)
        elif isinstance(module, mint.nn.Embedding):
            module.weight.set_data(
                initializer(
                    Normal(sigma=std, mean=0.0),
                    module.weight.shape,
                    module.weight.dtype,
                )
            )
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0


QWEN2AUDIOENCODER_START_DOCSTRING = r"""
    This model inherits from [`MSPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a MindSpore [mindspore.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) subclass.
    Use it as a regular MindSpore Module and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2AudioEncoderConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~MSPreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    """The audio model from Qwen2Audio without any head or projection on top.""",
    QWEN2AUDIOENCODER_START_DOCSTRING,
)
# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoder with Whisper->Qwen2Audio
class Qwen2AudioEncoder(Qwen2AudioPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen2AudioEncoderLayer`].

    Args:
        config: Qwen2AudioEncoderConfig
    """

    # Ignore copy
    config_class = Qwen2AudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["Qwen2AudioEncoderLayer"]

    def __init__(self, config: Qwen2AudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1, pad_mode="pad", has_bias=True)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=True)

        self.embed_positions = mint.nn.Embedding(self.max_source_positions, embed_dim, _freeze=True)
        self.embed_positions.requires_grad = False

        self.layers = nn.CellList(
            [Qwen2AudioEncoderLayer(config, layer_idx) for layer_idx in range(config.encoder_layers)]
        )
        self.layer_norm = mint.nn.LayerNorm(config.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Cell:
        return self.conv1

    def set_input_embeddings(self, value: nn.Cell):
        self.conv1 = value

    def construct(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`ms.Tensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `ms.Tensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`ms.Tensor`)`, *optional*):
                Qwen2Audio does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`ms.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        expected_seq_length = self.config.max_source_positions * self.conv1.stride[1] * self.conv2.stride[1]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Qwen2Audio expects the mel input features to be of "
                f"length {expected_seq_length}, but found {input_features.shape[-1]}. "
                f"Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype)

        inputs_embeds = mint.nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = mint.nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.shape[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.shape[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = mint.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Ignore copy
        hidden_states = hidden_states.permute((0, 2, 1)).float()
        hidden_states = self.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute((0, 2, 1)).to(input_features.dtype)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: ms.Tensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class Qwen2AudioMultiModalProjector(nn.Cell):
    def __init__(self, config: Qwen2AudioConfig):
        super().__init__()
        self.linear = nn.Dense(config.audio_config.d_model, config.text_config.hidden_size, has_bias=True)

    def construct(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


QWEN2AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_features (`ms.Tensor` of shape `(batch_size, feature_size, feature_sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `ms.Tensor`. See [`~WhisperFeatureExtractor.__call__`]
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
        feature_attention_mask (`ms.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        position_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Pre-computed hidden-states that can be used to speed up auto-regressive (sequential) decoding. There are
            two sets of pre-computed hidden-states: key and values states in the self-attention blocks.
            The `past_key_values` are returned when `use_cache=True` is passed or when `config.use_cache=True`.
            It is a [`~cache_utils.Cache`] instance.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those
            that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
            all `input_ids` of shape `(batch_size, sequence_length)`.shape `(batch_size, 1)` instead of all
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
"""


@add_start_docstrings(
    """The QWEN2AUDIO model which consists of a audio backbone and a language model.""",
    QWEN2AUDIO_START_DOCSTRING,
)
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: ms.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: ms.dtype,
    device: None,
    min_dtype: float,
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
            The target length: when generating with static cache, the mask should be as long as the static cache,
            to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`ms.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`ms.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = ops.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype)
        if sequence_length != 1:
            causal_mask = ops.triu(causal_mask, diagonal=1)
        causal_mask *= ops.arange(target_length) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].broadcast_to((batch_size, 1, -1, -1))
        if attention_mask is not None:
            # causal_mask = causal_mask  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            # padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = ops.narrow(causal_mask, -1, 0, mask_length) + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            # causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            #     padding_mask, min_dtype
            # )
            if mask_length >= causal_mask.shape[-1]:
                causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)
            else:
                causal_mask = ops.cat(
                    [
                        ops.narrow(causal_mask, -1, 0, mask_length).masked_fill(padding_mask, min_dtype),
                        ops.narrow(causal_mask, -1, mask_length, causal_mask.shape[-1] - mask_length),
                    ],
                    axis=-1,
                )

    return causal_mask


class Qwen2AudioForConditionalGeneration(Qwen2AudioPreTrainedModel, GenerationMixin):
    def __init__(self, config: Qwen2AudioConfig):
        super().__init__(config)
        print("Qwen2AudoConfig:", config._attn_implementation)
        if config._attn_implementation in modeling_qwen2.QWEN2_ATTENTION_CLASSES:
            config.text_config._attn_implementation = config._attn_implementation
        if config._attn_implementation in QWEN2AUDIO_ATTENTION_CLASSES:
            config.audio_config._attn_implementation = config._attn_implementation
        else:
            config.audio_config._attn_implementation = "flash_attention_2"
        print("audio_config_attention:", config.audio_config._attn_implementation)
        print("text_config_attention:", config.text_config._attn_implementation)
        self.audio_tower = Qwen2AudioEncoder(config.audio_config)  # Usually a `Qwen2AudioEncoder` instance
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = Qwen2ForCausalLM(config.text_config)  # e.g. Qwen2Model` instance
        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        self.post_init()

    @property
    def padding_side(self):
        return self._padding_side

    @padding_side.setter
    def padding_side(self, padding_side: str):
        if padding_side not in ["left", "right"]:
            raise ValueError(f"{padding_side} is not `left` or `right`.")
        self._padding_side = padding_side

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _merge_input_ids_with_audio_features(
        self, audio_features, num_audio_tokens, inputs_embeds, input_ids, attention_mask, labels
    ):
        """
        Merge input_ids with with audio features into final embeddings

        Args:
            audio_features (`ms.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
                All audio vectors of all audios in the batch
            num_audio_tokens (`ms.Tensor` of shape `(num_audios)`):
                The length of audio embeddings of each audio as stacked in `audio_features`
            inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with audio embeddings
            input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with audio token
            attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                labels need to be recalculated to support training (if provided)
        Returns:
            final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

        Explanation:
            each audio has variable length embeddings, with length specified by num_audio_tokens
            audio_features is concatenation of all audio embed vectors
            task: fill each <|AUDIO|> with the correct number of audio embeddings
            Example:
                X (5 tokens), Y (3 tokens), Z (8 tokens)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but audio token sizes are different, then cannot infer left or right padding
                ```python
                url1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
                audio1, _ = librosa.load(BytesIO(urlopen(url1).read()), sr=processor.feature_extractor.sampling_rate)
                url2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
                audio2, _ = librosa.load(BytesIO(urlopen(url2).read()), sr=processor.feature_extractor.sampling_rate)
                prompts = [
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                ]
                inputs = processor(text=prompts, audios=[audio1, audio2], return_tensors='pt', padding=True).to("cuda")
                    audio1 has 101 tokens, while audio2 has 72 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = mint.arange(max_audio_tokens).broadcast_to(
            (num_audios, max_audio_tokens)
        ) < num_audio_tokens.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].view((-1, embed_dim))
        batch_size, sequence_length = input_ids.shape
        _left_padding = mint.any(attention_mask[:, 0] == 0)
        _right_padding = mint.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                # both side is 1, so cannot tell
                left_padding = self.padding_side == "left"
            else:
                # invalid attention_mask
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.config.audio_token_index
        num_special_audio_tokens = mint.sum(special_audio_token_mask, dim=-1)

        batch_indices, non_audio_indices = mint.where(
            (input_ids != self.config.audio_token_index) & (attention_mask == 1)
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
        # `mint.cumsum` computes how each audio token shifts subsequent text token positions.
        token_placeholder_num = mint.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = mint.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = mint.zeros(batch_size, max_token_num, embed_dim, dtype=inputs_embeds.dtype)
        final_attention_mask = mint.zeros(batch_size, max_token_num, dtype=attention_mask.dtype)
        final_input_ids = mint.full((batch_size, max_token_num), self.pad_token_id, dtype=input_ids.dtype)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            final_labels = mint.full_like(final_attention_mask, self.config.ignore_index).to(ms.int32)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = mint.full((batch_size, max_token_num), True, dtype=ms.bool_)
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = mint.arange(max_token_num).unsqueeze(0)
        seq_indices = seq_indices.broadcast_to((batch_size, max_token_num))

        if left_padding:
            # exclude padding on the left
            max_token_num = max_token_num
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            # exclude padding on the right
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = masked_audio_features.contiguous().reshape(-1, embed_dim)
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values: Union[Cache, Tuple] = None,
        attention_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = get_seq_length(past_key_values) if past_key_values is not None else 0
            cache_position = ops.arange(past_length, input_ids.shape[1], dtype=ms.int32)

        if kwargs["use_cache"]:
            model_inputs["cache_position"] = cache_position

        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        # Excpetion 4: If input_embeds are passed then slice it through `cache_position`, to keep only the unprocessed tokens and
        # generate the first token for each sequence. Later use the generated Input ids for continuation.
        if past_key_values is not None:
            # Make sure `past_key_values` is mutable (required for graph mode to work faster)
            model_inputs["past_key_values"] = (
                ms.mutable(past_key_values) if isinstance(past_key_values, tuple) else past_key_values
            )
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:  # Exception 1  # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # Padding input_id to max_len when no cache
                if past_key_values is None:
                    pad_len = max(0, attention_mask.shape[1] - input_ids.shape[1])
                    input_ids = mint.nn.functional.pad(input_ids, (0, pad_len), value=0)
                model_inputs[input_ids_key] = input_ids
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone()

        # 4. Create missing `position_ids` on the fly
        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        )
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.construct).parameters.keys())
        ):
            position_ids = attention_mask.to(ms.int32).cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                _past_key_values = past_key_values
                if isinstance(past_key_values, (tuple, list)) and get_seq_length(past_key_values) == 0:
                    _past_key_values = None

                if _past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    if attention_mask is not None:
                        # since attention_mask maybe padded,
                        # it's safer to use the valid length instead of the total length
                        cur_len = attention_mask.sum(-1).max()
                        model_input = model_input[:, cur_len - current_input_length : cur_len]
                    else:
                        model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone()
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class).
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(
                    self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                logger.warning_once(
                    f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                    "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                    "writing code, see Llama for an example implementation. If you're a user, please report this "
                    "issue on GitHub."
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    cache_position=cache_position,
                    batch_size=batch_size,
                )
        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask

        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)

        # Paged Attention
        if self.config._attn_implementation == "paged_attention":
            bs, seq_len = input_ids.shape
            step = kwargs["step"]
            if step == 0:
                self.enable_dynamic_shape()

                # init block tables
                self.block_mgr = BlockTables(1024, 32, self.config.text_config.max_position_embeddings)
                self.block_mgr.init_cache_engine(bs)

                # get slot mapping and block tables
                max_input_length = self.config.text_config.max_position_embeddings
                self.valid_length_each_example = ms.tensor(seq_len).reshape(bs)
                block_tables, slot_mapping = self.block_mgr.assemble_pa_full_inputs(
                    max_input_length, self.valid_length_each_example, [False]
                )
                slot_mapping = np.delete(slot_mapping, np.where(slot_mapping == -1))

                # set batch valid length
                self.batch_valid_length = ms.tensor(seq_len).to(ms.int32).reshape(bs)

                self.phase = "prefill"
                self._add_flags_custom(True)
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
                    self._add_flags_custom(False)
            slot_mapping = ms.tensor(slot_mapping)
            block_tables = ms.tensor(block_tables)
            model_inputs.update(
                {
                    "block_tables": block_tables,
                    "slot_mapping": slot_mapping,
                    "batch_valid_length": self.batch_valid_length,
                }
            )
            model_inputs.pop("step", None)
        return model_inputs

    def _add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.language_model.add_flags(is_first_iteration=is_first_iteration)
        self.language_model.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.language_model.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attn.infer_attention.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attn.infer_attention.paged_attention_mgr.add_flags(is_first_iteration=is_first_iteration)

    def enable_dynamic_shape(self):
        input_ids = ms.mutable(Tensor(shape=[None, None], dtype=ms.int32))
        input_feature = None
        position_ids = ms.mutable(Tensor(shape=[None, None], dtype=ms.int32))
        attention_mask = ms.mutable(Tensor(shape=[None, None], dtype=ms.int32))
        feature_attention_mask = None
        past_key_values = None
        inputs_embeds = None
        labels = None
        use_cache = False
        output_attentions = False
        output_hidden_states = False
        return_dict = False
        cache_position = None
        block_tables = Tensor(shape=[None, None], dtype=ms.int32)
        slot_mapping = ms.mutable(Tensor(shape=[None], dtype=ms.int32))
        batch_valid_length = Tensor(shape=[None], dtype=ms.int32)

        self.set_inputs(
            input_ids,
            input_feature,
            attention_mask,
            feature_attention_mask,
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

    @add_start_docstrings_to_model_forward(QWEN2AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2AudioCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: ms.Tensor = None,
        input_features: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        feature_attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
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
    ) -> Union[Tuple, Qwen2AudioCausalLMOutputWithPast]:
        r"""
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from mindspore import Tensor
        >>> from transformers import AutoProcessor
        >>> from mindone.transformers import Qwen2AudioForConditionalGeneration

        >>> model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")

        >>> prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        >>> url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        >>> audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

        >>> inputs = processor(text=prompt, audios=audio, return_tensors="np")

        >>> # Generate
        >>> generate_ids = model.generate(Tensor(inputs.input_ids, dtype=ms.int32), max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Generate the caption in English: Glass is breaking."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_features is not None:
            input_features = input_features
            feature_attention_mask = feature_attention_mask

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and audios
            if input_features is not None and input_ids.shape[1] != 1:
                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    feature_attention_mask.float().sum(-1)
                )
                batch_size, _, max_mel_seq_len = input_features.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                # Create a sequence tensor of shape (batch_size, max_seq_len)
                seq_range = (
                    mint.arange(0, max_seq_len, dtype=feature_attention_mask.dtype)
                    .unsqueeze(0)
                    .broadcast_to((batch_size, max_seq_len))
                )
                lengths_expand = audio_feat_lengths.unsqueeze(1).broadcast_to((batch_size, max_seq_len))
                # Create mask
                padding_mask = seq_range >= lengths_expand

                audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).broadcast_to(
                    (batch_size, 1, max_seq_len, max_seq_len)
                )
                audio_attention_mask = audio_attention_mask_.to(input_features.dtype)
                audio_attention_mask[audio_attention_mask_] = dtype_to_min(input_features.dtype)

                audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask)
                selected_audio_feature = audio_outputs.last_hidden_state
                audio_features = self.multi_modal_projector(selected_audio_feature)

                # if we have consecutive audio tokens, then it means we expanded input_ids in processing
                audio_tokens = input_ids == self.config.audio_token_index
                legacy_processing = (audio_tokens[:, :-1] & audio_tokens[:, 1:]).sum() == 0

                if legacy_processing:
                    logger.warning_once("Expanding inputs for audio tokens in Qwen2Audio should be done in processing.")
                    inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                        audio_features, audio_output_lengths, inputs_embeds, input_ids, attention_mask, labels
                    )
                else:
                    num_audios, max_audio_tokens, embed_dim = audio_features.shape
                    audio_features_mask = mint.arange(max_audio_tokens)[None, :]
                    audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
                    audio_features = audio_features[audio_features_mask]

                    n_audio_tokens = (input_ids == self.config.audio_token_index).sum().item()
                    n_audio_features = audio_features.shape[0]

                    if n_audio_tokens != n_audio_features:
                        raise ValueError(
                            f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                        )
                    special_audio_mask = input_ids == self.config.audio_token_index
                    special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    audio_features = audio_features.to(inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.float()
                    inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features.float())
                    inputs_embeds = inputs_embeds.to(input_features.dtype)

        outputs = self.language_model(
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
            batch_valid_length=batch_valid_length,
        )

        logits = outputs[0].float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2AudioCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )


__all__ = ["Qwen2AudioForConditionalGeneration", "Qwen2AudioPreTrainedModel", "Qwen2AudioEncoder"]
