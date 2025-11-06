#
# This code is adapted from https://github.com/Tencent-Hunyuan/HunyuanImage-3.0
# with modifications to run diffusers on mindspore.
#
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Copyright 2025 The HuggingFace Inc. team.
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
# ==============================================================================

from typing import Optional, Tuple, Union

from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings

import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, mint

from mindone.transformers.activations import ACT2FN
from mindone.transformers.mindspore_adapter.nn.MultiheadAttention import MultiheadAttention
from mindone.transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from mindone.transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling


class Config(object):
    def __init__(self, config):
        if config is not None:
            for key, value in config.items():
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        return setattr(self, key, value)


# Copied from mindone.transformers.models.siglip2.modeling_siglip2.Siglip2VisionEmbeddings
class Siglip2VisionEmbeddings(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = mint.nn.Linear(
            config.num_channels * self.patch_size * self.patch_size,
            self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = mint.nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: ms.Tensor,
        spatial_shapes: ms.Tensor,
        max_length: int,
    ) -> ms.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`ms.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`ms.Tensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `ms.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = mint.zeros(
            (batch_size, max_length, embed_dim),
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            # TODO mint/ops interpolate has precision bugs because there is not antialias interface
            # Gussian Blur is tried to simulate the antialias process but precision fp32 could not be less than threshold
            resized_embeddings = mint.nn.functional.interpolate(
                positional_embeddings,
                size=(int(height), int(width)),
                mode="bilinear",
                align_corners=False,
            )
            # resized_embeddings = torch.nn.functional.interpolate(
            #     ms.Tensor(positional_embeddings.numpy()),
            #     size=(int(height), int(width)),
            #     mode="bilinear",
            #     align_corners=False,
            #     antialias=True,
            # )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, int(height * width)).swapaxes(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def construct(self, pixel_values: ms.Tensor, spatial_shapes: ms.Tensor) -> ms.Tensor:
        """
        Args:
            pixel_values (`ms.Tensor`):
                Pixel values of shape (batch_size, max_num_patches, num_channels * patch_size * patch_size)
            spatial_shapes (`List[Tuple[int, int]]`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
        """

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Get positional resized and padded positional embeddings
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


# Copied from mindone.transformers.models.siglip2.modeling_siglip2.Siglip2Attention
class Siglip2Attention(nn.Cell):
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

        self.k_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = mint.matmul(query_states, key_states.swapaxes(2, 3)) * self.scale

        if attn_weights.shape != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = mint.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query_states.dtype)
        attn_weights = mint.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = mint.matmul(attn_weights, value_states)

        if attn_output.shape != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Siglip2SdpaAttention(Siglip2Attention):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


# Copied from mindone.transformers.models.siglip2.modeling_siglip2.Siglip2MLP
class Siglip2MLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = mint.nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = mint.nn.Linear(config.intermediate_size, config.hidden_size)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


SIGLIP2_ATTENTION_CLASSES = {
    "eager": Siglip2Attention,
    "sdpa": Siglip2SdpaAttention,  # Not supported
}


# Copied from mindone.transformers.models.siglip2.modeling_siglip2.Siglip2EncoderLayer
class Siglip2EncoderLayer(nn.Module):
    def __init__(self, config: Siglip2Config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SIGLIP2_ATTENTION_CLASSES[config._attn_implementation](config=config)
        self.layer_norm1 = mint.nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)
        self.layer_norm2 = mint.nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor]:
        """
        Args:
            hidden_states (`ms.Tensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`ms.Tensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from mindone.transformers.models.siglip2.modeling_siglip2.Siglip2Encoder
class Siglip2Encoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Siglip2EncoderLayer`].

    Args:
        config: Siglip2Config
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.CellList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def construct(
        self,
        inputs_embeds,
        attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
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


# Copied from mindone.transformers.models.siglip2.modeling_siglip2.Siglip2MultiheadAttentionPoolingHead
class Siglip2MultiheadAttentionPoolingHead(nn.Cell):
    """Multihead Attention Pooling."""

    def __init__(self, config):
        super().__init__()

        self.probe = Parameter(mint.randn(1, 1, config.hidden_size))
        self.attention = MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)
        self.num_heads = config.num_attention_heads

    def construct(self, hidden_state: ms.Tensor, attention_mask: Optional[ms.Tensor] = None):
        batch_size = hidden_state.shape[0]
        probe = self.probe.tile((batch_size, 1, 1))

        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.tile((1, self.num_heads, target_len, 1))
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


SIGLIP2_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*, defaults to `False`):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from mindone.transformers.models.siglip2.modeling_siglip2.Siglip2VisionTransformer
class Siglip2VisionTransformer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = Siglip2MultiheadAttentionPoolingHead(config)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    @add_start_docstrings_to_model_forward(SIGLIP2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Siglip2VisionConfig)
    def construct(
        self,
        pixel_values: ms.Tensor,
        attention_mask: ms.Tensor,
        spatial_shapes: ms.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values, spatial_shapes)

        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        else:
            encoder_attention_mask = attention_mask

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state, attention_mask) if self.use_head else None
        if not return_dict:
            return (last_hidden_state, pooler_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class LightProjector(nn.Cell):
    def __init__(self, config):
        config = Config(config)
        super().__init__()

        if config.projector_type == "linear":
            modules = mint.nn.Linear(config.input_dim, config.n_embed)

        elif config.projector_type == "mlp_gelu":
            modules = [mint.nn.Linear(config.input_dim, config.n_embed)]
            for _ in range(1, config.depth):
                modules.append(mint.nn.GELU())
                modules.append(mint.nn.Linear(config.n_embed, config.n_embed))
            modules = nn.SequentialCell(*modules)

        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

        self.layers = modules

    def construct(self, x):
        return self.layers(x)
