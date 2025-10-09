# coding=utf-8
# Copyright 2025 The OpenAI Team Authors, The HuggingFace Team,
# The BAAI Team Authors and The Huawei MindSpore Team Authors. All rights reserved.
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
"""MindSpore EvaCLIP model."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from transformers.utils import ModelOutput, logging

import mindspore as ms
from mindspore import mint, nn

from mindone.diffusers.utils.mindspore_utils import dtype_to_min
from mindone.transformers.activations import ACT2FN
from mindone.transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from mindone.transformers.modeling_utils import MSPreTrainedModel

from .configuration_evaclip import EvaCLIPConfig, EvaCLIPTextConfig, EvaCLIPVisionConfig

logger = logging.get_logger(__name__)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: ms.Tensor, dtype: ms.Type, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].broadcast_to((bsz, 1, tgt_len, src_len)).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(ms.bool_), dtype_to_min(dtype))


def contrastive_loss(logits: ms.Tensor) -> ms.Tensor:
    return mint.functional.cross_entropy(logits, mint.arange(len(logits)))


def clip_loss(similarity: ms.Tensor) -> ms.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


@dataclass
class EvaCLIPVisionModelOutput(ModelOutput):
    image_embeds: Optional[ms.Tensor] = None
    last_hidden_state: ms.Tensor = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None


@dataclass
class EvaCLIPTextModelOutput(ModelOutput):
    text_embeds: Optional[ms.Tensor] = None
    last_hidden_state: ms.Tensor = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None


@dataclass
class EvaCLIPOutput(ModelOutput):
    loss: Optional[ms.Tensor] = None
    logits_per_image: ms.Tensor = None
    logits_per_text: ms.Tensor = None
    text_embeds: ms.Tensor = None
    image_embeds: ms.Tensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class RMSNorm(nn.Cell):
    """
    adepted from transformers T5LayerNorm
    """

    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = ms.Parameter(mint.ones(hidden_size))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(ms.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * mint.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [ms.float16, ms.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class EvaCLIPAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = mint.nn.Linear(self.embed_dim, self.embed_dim, bias=config.k_bias)
        self.v_proj = mint.nn.Linear(self.embed_dim, self.embed_dim, bias=config.v_bias)
        self.q_proj = mint.nn.Linear(self.embed_dim, self.embed_dim, bias=config.q_bias)
        self.out_proj = mint.nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        if config.use_rope:
            self.rope = VisionRotaryEmbedding(config)
        else:
            self.rope = None

        if not config.use_sub_ln:
            self.inner_attn_ln = mint.nn.Identity()
        elif config.use_rms_norm:
            self.inner_attn_ln = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        else:
            self.inner_attn_ln = mint.nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def _shape(self, tensor: ms.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2).contiguous()

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        causal_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.shape
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)

        query_states = self._shape(self.q_proj(hidden_states), -1, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # RoPE
        if self.rope:
            query_states_cls, query_states_rest = query_states[:, :, :1], query_states[:, :, 1:]
            key_states_cls, key_states_rest = key_states[:, :, :1], key_states[:, :, 1:]

            query_states = mint.cat([query_states_cls, self.rope(query_states_rest)], dim=-2).type_as(value_states)
            key_states = mint.cat([key_states_cls, self.rope(key_states_rest)], dim=-2).type_as(value_states)

        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = mint.bmm(query_states * self.scale, key_states.swapaxes(1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = mint.nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = mint.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = mint.bmm(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.inner_attn_ln(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class EvaCLIPVisionEmbeddings(nn.Cell):
    def __init__(self, config: EvaCLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = ms.Parameter(mint.randn(self.embed_dim))

        self.patch_embedding = mint.nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = mint.nn.Embedding(self.num_positions, self.embed_dim)
        self.position_ids = mint.arange(self.num_positions).broadcast_to((1, -1))

    def construct(self, pixel_values: ms.Tensor) -> ms.Tensor:
        # TODO: This is a temporary limitations and we will figure out how to handle this in a more elegant way later
        if pixel_values.shape[-1] != self.image_size or pixel_values.shape[-2] != self.image_size:
            raise ValueError(
                f"Input pixel_values should have height and width ({self.image_size}, {self.image_size}),"
                f" but got {pixel_values.shape}."
            )

        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(start_dim=2).swapaxes(1, 2)

        class_embeds = self.class_embedding.broadcast_to((batch_size, 1, -1))
        embeddings = mint.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class VisionRotaryEmbedding(nn.Cell):
    def __init__(self, config):
        super().__init__()
        seq_len = config.image_size // config.patch_size
        dim = config.hidden_size // config.num_attention_heads // 2

        t = mint.arange(seq_len) / seq_len * config.pretrained_seq_len
        freqs = 1.0 / (config.rope_theta ** (mint.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        freqs = t.unsqueeze(-1) * freqs.unsqueeze(0)
        freqs = freqs.repeat_interleave(2, dim=-1)  # [seq_len, dim]
        freqs = mint.cat(
            [freqs.unsqueeze(1).broadcast_to((-1, seq_len, -1)), freqs.unsqueeze(0).broadcast_to((seq_len, -1, -1))],
            dim=-1,
        )

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = mint.stack((-x2, x1), dim=-1)
        return x.flatten(start_dim=-2)

    def construct(self, x):
        return x * self.freqs_cos.to(x.dtype) + self.rotate_half(x) * self.freqs_sin.to(x.dtype)


class EvaCLIPTextEmbeddings(nn.Cell):
    def __init__(self, config: EvaCLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = mint.nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = mint.nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = mint.arange(config.max_position_embeddings).broadcast_to((1, -1))

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class EvaCLIPMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = mint.nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = mint.nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_ln = mint.nn.LayerNorm(config.intermediate_size) if config.use_sub_ln else mint.nn.Identity()

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class EvaCLIPSwiGLUMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = mint.nn.SiLU()
        self.fc1 = mint.nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = mint.nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc3 = mint.nn.Linear(config.intermediate_size, config.hidden_size)

        if not config.use_sub_ln:
            self.ffn_ln = mint.nn.Identity()
        elif config.use_rms_norm:
            self.ffn_ln = RMSNorm(config.intermediate_size, eps=config.layer_norm_eps)
        else:
            self.ffn_ln = mint.nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states1 = self.fc1(hidden_states)
        hidden_states2 = self.fc2(hidden_states)
        hidden_states = self.activation_fn(hidden_states1) * hidden_states2
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.fc3(hidden_states)
        return hidden_states


class EvaCLIPEncoderLayer(nn.Cell):
    def __init__(self, config: EvaCLIPConfig):
        super().__init__()
        norm_layer = RMSNorm if config.use_rms_norm else mint.nn.LayerNorm

        self.config = config
        self.embed_dim = config.hidden_size
        self.post_layernorm = config.post_layernorm if config.post_layernorm is not None else False
        self.self_attn = EvaCLIPAttention(config)
        self.layer_norm1 = norm_layer(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = EvaCLIPSwiGLUMLP(config) if config.use_swiglu_mlp else EvaCLIPMLP(config)
        self.layer_norm2 = norm_layer(self.embed_dim, eps=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
        causal_attention_mask: ms.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor]:
        residual = hidden_states

        if not self.post_layernorm:
            hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        if self.post_layernorm:
            hidden_states = self.layer_norm1(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        if not self.post_layernorm:
            hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_layernorm:
            hidden_states = self.layer_norm2(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class EvaCLIPEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`EvaCLIPEncoderLayer`].

    Args:
        config: EvaCLIPConfig
    """

    def __init__(self, config: EvaCLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.CellList([EvaCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        inputs_embeds,
        attention_mask: Optional[ms.Tensor] = None,
        causal_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class EvaCLIPPreTrainedModel(MSPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EvaCLIPConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        # """Initialize the weights"""
        # factor = self.config.initializer_factor
        # if isinstance(module, EvaCLIPTextEmbeddings):
        #     module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        #     module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        # elif isinstance(module, EvaCLIPVisionEmbeddings):
        #     factor = self.config.initializer_factor
        #     nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
        #     nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
        #     nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        # elif isinstance(module, EvaCLIPAttention):
        #     factor = self.config.initializer_factor
        #     in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
        #     out_proj_std = (module.embed_dim**-0.5) * factor
        #     nn.init.normal_(module.q_proj.weight, std=in_proj_std)
        #     nn.init.normal_(module.k_proj.weight, std=in_proj_std)
        #     nn.init.normal_(module.v_proj.weight, std=in_proj_std)
        #     nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # elif isinstance(module, EvaCLIPMLP):
        #     factor = self.config.initializer_factor
        #     in_proj_std = (
        #         (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
        #     )
        #     fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
        #     nn.init.normal_(module.fc1.weight, std=fc_std)
        #     nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # elif isinstance(module, EvaCLIPModel):
        #     nn.init.normal_(
        #         module.text_projection.weight,
        #         std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
        #     )
        #     nn.init.normal_(
        #         module.visual_projection.weight,
        #         std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
        #     )
        # elif isinstance(module, EvaCLIPVisionModelWithProjection):
        #     nn.init.normal_(
        #         module.visual_projection.weight,
        #         std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
        #     )
        # elif isinstance(module, EvaCLIPTextModelWithProjection):
        #     nn.init.normal_(
        #         module.text_projection.weight,
        #         std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
        #     )

        # if isinstance(module, mint.nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        # if isinstance(module, mint.nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()
        ...

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, EvaCLIPEncoder):
            module.gradient_checkpointing = value


class EvaCLIPVisionTransformer(nn.Cell):
    def __init__(self, config: EvaCLIPVisionConfig):
        super().__init__()
        self.config = config

        self.embeddings = EvaCLIPVisionEmbeddings(config)
        self.encoder = EvaCLIPEncoder(config)
        self.post_layernorm = (
            RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.use_rms_norm
            else mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )

    def gradient_checkpointing_enable(self):
        self.encoder.gradient_checkpointing = True

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class EvaCLIPTextTransformer(nn.Cell):
    def __init__(self, config: EvaCLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        norm_layer = RMSNorm if config.use_rms_norm else mint.nn.LayerNorm
        self.embeddings = EvaCLIPTextEmbeddings(config)
        self.encoder = EvaCLIPEncoder(config)
        self.final_layer_norm = norm_layer(embed_dim, eps=config.layer_norm_eps)

    def gradient_checkpointing_enable(self):
        self.encoder.gradient_checkpointing = True

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            mint.arange(last_hidden_state.shape[0]),
            input_ids.to(dtype=ms.int64).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = mint.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask = mask.fill(ms.Tensor(dtype_to_min(dtype)))
        mask = mask.triu(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


class EvaCLIPVisionModel(EvaCLIPPreTrainedModel):
    config_class = EvaCLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: EvaCLIPVisionConfig):
        super().__init__(config)

        self.vision_model = EvaCLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class EvaCLIPTextModel(EvaCLIPPreTrainedModel):
    config_class = EvaCLIPTextConfig

    _no_split_modules = ["EvaCLIPEncoderLayer"]

    def __init__(self, config: EvaCLIPTextConfig):
        super().__init__(config)
        self.text_model = EvaCLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class EvaCLIPModel(EvaCLIPPreTrainedModel):
    config_class = EvaCLIPConfig

    def __init__(self, config: EvaCLIPConfig):
        super().__init__(config)

        if not (type(config.text_config).__name__ == "EvaCLIPTextConfig"):
            raise ValueError(
                "config.text_config is expected to be of type EvaCLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not (type(config.vision_config).__name__ == "EvaCLIPVisionConfig"):
            raise ValueError(
                "config.vision_config is expected to be of type EvaCLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = EvaCLIPTextTransformer(text_config)
        self.vision_model = EvaCLIPVisionTransformer(vision_config)

        self.visual_projection = mint.nn.Linear(self.vision_embed_dim, self.projection_dim)
        self.text_projection = mint.nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = ms.Parameter(ms.Tensor(config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    def encode_text(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ms.Tensor:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def encode_image(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ms.Tensor:
        # Use EvaCLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EvaCLIPOutput]:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = mint.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return EvaCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class EvaCLIPVisionModelWithProjection(EvaCLIPPreTrainedModel):
    config_class = EvaCLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: EvaCLIPVisionConfig):
        super().__init__(config)

        vision_model = EvaCLIPVisionModel._from_config(config)
        self.vision_model = vision_model.vision_model

        self.visual_projection = mint.nn.Linear(config.hidden_size, config.projection_dim)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EvaCLIPVisionModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output

        image_embeds = self.visual_projection(pooled_output)

        if not return_dict:
            outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return EvaCLIPVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )


class EvaCLIPTextModelWithProjection(EvaCLIPPreTrainedModel):
    config_class = EvaCLIPTextConfig

    _no_split_modules = ["EvaCLIPEncoderLayer"]

    def __init__(self, config: EvaCLIPTextConfig):
        super().__init__(config)

        self.text_model = EvaCLIPTextTransformer(config)

        self.text_projection = mint.nn.Linear(config.hidden_size, config.projection_dim, bias=False)

    def get_input_embeddings(self) -> nn.Cell:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EvaCLIPTextModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]

        text_embeds = self.text_projection(pooled_output)

        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return EvaCLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )
