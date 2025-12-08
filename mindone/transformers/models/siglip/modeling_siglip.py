# coding=utf-8
# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
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
"""Mindspore Siglip model."""

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from transformers.models.siglip.configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from transformers.utils import logging

import mindspore as ms
from mindspore import mint, nn

from ...activations import ACT2FN
from ...mindspore_adapter.nn.MultiheadAttention import MultiheadAttention
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, can_return_tuple, filter_out_non_signature_kwargs
from ...utils.generic import check_model_inputs

logger = logging.get_logger(__name__)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->Siglip
class SiglipVisionModelOutput(ModelOutput):
    r"""
    image_embeds (`ms.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The image embeddings obtained by applying the projection layer to the pooler_output.
    """

    image_embeds: Optional[ms.Tensor] = None
    last_hidden_state: Optional[ms.Tensor] = None
    hidden_states: Optional[tuple[ms.Tensor, ...]] = None
    attentions: Optional[tuple[ms.Tensor, ...]] = None


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPTextModelOutput with CLIP->Siglip
class SiglipTextModelOutput(ModelOutput):
    r"""
    text_embeds (`ms.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The text embeddings obtained by applying the projection layer to the pooler_output.
    """

    text_embeds: Optional[ms.Tensor] = None
    last_hidden_state: ms.Tensor = None
    hidden_states: Optional[tuple[ms.Tensor, ...]] = None
    attentions: Optional[tuple[ms.Tensor, ...]] = None


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->Siglip
class SiglipOutput(ModelOutput):
    """
    Args:
        loss (`ms.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`ms.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`ms.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds (`ms.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`SiglipTextModel`].
        image_embeds (`ms.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`SiglipVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`SiglipTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`SiglipVisionModel`].
    """

    loss: Optional[ms.Tensor] = None
    logits_per_image: ms.Tensor = None
    logits_per_text: ms.Tensor = None
    text_embeds: ms.Tensor = None
    image_embeds: ms.Tensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class SiglipVisionEmbeddings(nn.Cell):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = mint.nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = mint.nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", mint.arange(self.num_positions).broadcast_to((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: ms.Tensor, height: int, width: int) -> ms.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support mindspore.jit tracing and no class embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = ms.Tensor(num_positions**0.5, dtype=ms.int32)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = mint.functional.interpolate(
            patch_pos_embed,
            scale_factor=(new_height / math.sqrt(num_positions), new_width / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=True,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def construct(self, pixel_values: ms.Tensor, interpolate_pos_encoding=False) -> ms.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).swapaxes(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->Siglip
class SiglipTextEmbeddings(nn.Cell):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = mint.nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = mint.nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", mint.arange(config.max_position_embeddings).broadcast_to((1, -1)), persistent=False
        )

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


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
    attn_weights = mint.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = mint.functional.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query.dtype)
    attn_weights = mint.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = mint.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class SiglipAttention(nn.Cell):
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
        self.is_causal = False

        self.k_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> tuple[ms.Tensor, Optional[ms.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Siglip
class SiglipMLP(nn.Cell):
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


class SiglipEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Union[SiglipVisionConfig, SiglipTextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = mint.nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = mint.nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    # Ignore copy
    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[ms.Tensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipPreTrainedModel(PreTrainedModel):
    config: SiglipConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "SiglipTextEmbeddings",
        "SiglipVisionEmbeddings",
        "SiglipEncoderLayer",
        "SiglipMultiheadAttentionPoolingHead",
    ]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": SiglipEncoderLayer,
        "attentions": SiglipAttention,
    }

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


# Copied from transformers.models.altclip.modeling_altclip.AltCLIPEncoder with AltCLIP->Siglip
class SiglipEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.layers = nn.CellList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def construct(
        self,
        inputs_embeds,
        attention_mask: Optional[ms.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                **kwargs,
            )

        return BaseModelOutput(last_hidden_state=hidden_states)


class SiglipTextTransformer(nn.Cell):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.final_layer_norm = mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.head = mint.nn.Linear(embed_dim, config.projection_size)

    @can_return_tuple
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # note: SigLIP's text model does not use a causal mask, unlike the original CLIP model.
        # expand attention_mask
        uses_flash_attention = "flash" in self.config._attn_implementation
        if uses_flash_attention:
            attention_mask = None
        elif attention_mask is not None and not uses_flash_attention:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Assuming "sticky" EOS tokenization, last token is always EOS.
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class SiglipTextModel(SiglipPreTrainedModel):
    config: SiglipTextConfig

    def __init__(self, config: SiglipTextConfig):
        super().__init__(config)
        self.text_model = SiglipTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @check_model_inputs
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import SiglipTextModel
        >>> import mindspore as ms
        >>> import numpy as np

        >>> model = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

        >>> # important: make sure to set padding="max_length" as that's how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="np")
        >>> for key, value in inputs.items():
        >>>     if isinstance(value, np.ndarray):
        >>>         inputs[key] = ms.tensor(value)
        >>>     elif isinstance(value, list):
        >>>         inputs[key] = ms.tensor(value)

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


class SiglipVisionTransformer(nn.Cell):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    @can_return_tuple
    def construct(
        self,
        pixel_values,
        interpolate_pos_encoding: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
        )


class SiglipMultiheadAttentionPoolingHead(nn.Cell):
    """Multihead Attention Pooling."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.probe = ms.Parameter(mint.randn(1, 1, config.hidden_size))
        self.attention = MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def construct(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionModel(SiglipPreTrainedModel):
    config: SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)

        self.vision_model = SiglipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        return self.vision_model.embeddings.patch_embedding

    @check_model_inputs
    def construct(
        self,
        pixel_values,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor
        >>> from mindone.transformers import SiglipVisionModel
        >>> import mindspore as ms
        >>> import numpy as np

        >>> model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="np")
        >>> for key, value in inputs.items():
        >>>     if isinstance(value, np.ndarray):
        >>>         inputs[key] = ms.tensor(value)
        >>>     elif isinstance(value, list):
        >>>         inputs[key] = ms.tensor(value)

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""

        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )


class SiglipModel(SiglipPreTrainedModel):
    config: SiglipConfig

    def __init__(self, config: SiglipConfig):
        super().__init__(config)

        if not isinstance(config.text_config, SiglipTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type SiglipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, SiglipVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type SiglipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        # First, initialize the text and vision models with proper attention implementation
        text_model = SiglipTextModel._from_config(text_config)
        vision_model = SiglipVisionModel._from_config(vision_config)

        # Second, get the text and vision submodules (for backward compatibility)
        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model

        self.logit_scale = ms.Parameter(mint.randn(1))
        self.logit_bias = ms.Parameter(mint.randn(1))

        # Initialize weights and apply final processing
        self.post_init()

    @filter_out_non_signature_kwargs()
    def get_text_features(
        self,
        input_ids: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        r"""
        Returns:
            text_features (`ms.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import AutoModel
        >>> import mindspore as ms
        >>> import numpy as np

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

        >>> # important: make sure to set padding="max_length" as that's how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="np")
        >>> for key, value in inputs.items():
        >>>     if isinstance(value, np.ndarray):
        >>>         inputs[key] = ms.tensor(value)
        >>>     elif isinstance(value, list):
        >>>         inputs[key] = ms.tensor(value)
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        pooled_output = text_outputs.pooler_output

        return pooled_output

    @filter_out_non_signature_kwargs()
    def get_image_features(
        self,
        pixel_values: ms.Tensor,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ms.Tensor:
        r"""
        Returns:
            image_features (`ms.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor
        >>> from mindone.transformers import AutoModel
        >>> import mindspore as ms
        >>> import numpy as np

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="np")
        >>> for key, value in inputs.items():
        >>>     if isinstance(value, np.ndarray):
        >>>         inputs[key] = ms.tensor(value)
        >>>     elif isinstance(value, list):
        >>>         inputs[key] = ms.tensor(value)

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )
        pooled_output = vision_outputs.pooler_output

        return pooled_output

    # NOTE: SiglipModel uses Pretrained backbones, so we don't need to add `check_model_inputs` here
    @can_return_tuple
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        return_loss: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SiglipOutput:
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor
        >>> from mindone.transformers import AutoModel
        >>> import mindspore as ms
        >>> import numpy as np

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> texts = ["a photo of 2 cats", "a photo of 2 dogs"]
        >>> # important: we pass `padding=max_length` since the model was trained with this
        >>> inputs = processor(text=texts, images=image, padding="max_length", return_tensors="np")
        >>> for key, value in inputs.items():
        >>>     if isinstance(value, np.ndarray):
        >>>         inputs[key] = ms.tensor(value)
        >>>     elif isinstance(value, list):
        >>>         inputs[key] = ms.tensor(value)

        >>> outputs = model(**inputs)

        >>> logits_per_image = outputs.logits_per_image
        >>> probs = mint.sigmoid(logits_per_image) # these are the probabilities
        >>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
        31.9% that image 0 is 'a photo of 2 cats'
        ```"""
        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output
        compute_dtype = image_embeds.dtype
        image_embeds = image_embeds.float()
        text_embeds = text_embeds.float()

        # normalized features
        image_embeds = (image_embeds / image_embeds.norm(ord=2, dim=-1, keepdim=True)).to(compute_dtype)
        text_embeds = (text_embeds / text_embeds.norm(ord=2, dim=-1, keepdim=True)).to(compute_dtype)

        # cosine similarity as logits
        logits_per_text = mint.matmul(text_embeds, image_embeds.t())

        logit_scale, logit_bias = self.logit_scale, self.logit_bias
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            eye = mint.eye(logits_per_text.size(0))
            m1_diag1 = -mint.ones_like(logits_per_text) + 2 * eye
            loglik = mint.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -mint.sum(loglik, dim=-1)
            loss = nll.mean()

        return SiglipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class SiglipForImageClassification(SiglipPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels

        # Create the vision model with proper attention
        # and take only vision_model submodule (for backward compatibility)
        vision_model = SiglipVisionModel._from_config(config.vision_config)
        self.vision_model = vision_model.vision_model

        # Classifier head
        self.classifier = (
            mint.nn.Linear(config.vision_config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor
        >>> from mindone.transformers import SiglipForImageClassification
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # note: we are loading a `SiglipModel` from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random if seed is not set above.
        >>> image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        >>> model = SiglipForImageClassification.from_pretrained("google/siglip-base-patch16-224")

        >>> inputs = image_processor(images=image, return_tensors="np")
        >>> for key, value in inputs.items():
        >>>     if isinstance(value, np.ndarray):
        >>>         inputs[key] = ms.tensor(value)
        >>>     elif isinstance(value, list):
        >>>         inputs[key] = ms.tensor(value)
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the two classes
        >>> predicted_class_idx = logits.argmax(-1).item()
        >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        Predicted class: LABEL_1
        ```"""
        outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state

        # average pool the patch tokens
        sequence_output = mint.mean(sequence_output, dim=1)
        # apply classifier
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )


__all__ = [
    "SiglipModel",
    "SiglipPreTrainedModel",
    "SiglipTextModel",
    "SiglipVisionModel",
    "SiglipForImageClassification",
]
