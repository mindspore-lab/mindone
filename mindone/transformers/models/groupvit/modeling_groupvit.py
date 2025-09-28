# coding=utf-8
# Copyright 2022 NVIDIA and The HuggingFace Team. All rights reserved.
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
"""MindSpore GroupViT model."""

import collections.abc
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
from transformers.models.groupvit.configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
from transformers.utils import ModelOutput

import mindspore as ms
from mindspore import mint, nn, ops

from ...activations import ACT2FN
from ...mindspore_adapter import ReLU
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import logging

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "nvidia/groupvit-gcc-yfcc"


# modified from mindone.transformers.utils.generic.mindspore_int for adapting graph mode
def mindspore_int(x):
    """
    Casts an input to a mindspore int64 tensor if we are in a tracing context, otherwise to a Python int.
    """

    return x.to(ms.int64) if isinstance(x, ms.Tensor) else int(x)


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: ms.Tensor) -> ms.Tensor:
    return mint.nn.functional.cross_entropy(logits, mint.arange(len(logits)))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->groupvit
def groupvit_loss(similarity: ms.Tensor) -> ms.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def hard_softmax(logits: ms.Tensor, dim: int):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = mint.zeros_like(logits).scatter_(dim, index, 1.0)
    ret = y_hard - ops.stop_gradient(y_soft) + y_soft

    return ret


def gumbel_softmax(logits: ms.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> ms.Tensor:
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = nn.probability.distribution.Gumbel(
        ms.tensor(0.0, dtype=logits.dtype), ms.tensor(1.0, dtype=logits.dtype)
    )
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = mint.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - ops.stop_gradient(y_soft) + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def resize_attention_map(attentions, height, width, align_corners=False):
    """
    Args:
        attentions (`ms.Tensor`): attention map of shape [batch_size, groups, feat_height*feat_width]
        height (`int`): height of the output attention map
        width (`int`): width of the output attention map
        align_corners (`bool`, *optional*): the `align_corner` argument for `mint.nn.functional.interpolate`.

    Returns:
        `ms.Tensor`: resized attention map of shape [batch_size, groups, height, width]
    """

    scale = (height * width // attentions.shape[2]) ** 0.5
    if height > width:
        feat_width = int(np.round(width / scale))
        feat_height = attentions.shape[2] // feat_width
    else:
        feat_height = int(np.round(height / scale))
        feat_width = attentions.shape[2] // feat_height

    batch_size = attentions.shape[0]
    groups = attentions.shape[1]  # number of group token
    # [batch_size, groups, height*width, groups] -> [batch_size, groups, height, width]
    attentions = attentions.reshape(batch_size, groups, feat_height, feat_width)
    attentions = mint.nn.functional.interpolate(
        attentions, size=(height, width), mode="bilinear", align_corners=align_corners
    )
    return attentions


def get_grouping_from_attentions(attentions, hw_shape):
    """
    Args:
        attentions (`tuple(ms.Tensor)`: tuple of attention maps returned by `GroupViTVisionTransformer`
        hw_shape (`tuple(int)`): height and width of the output attention map
    Returns:
        `ms.Tensor`: the attention map of shape [batch_size, groups, height, width]
    """

    attn_maps = []

    prev_attn_masks = None
    for attn_masks in attentions:
        # [batch_size, num_groups, height x width] -> [batch_size, height x width, num_groups]
        attn_masks = attn_masks.permute(0, 2, 1).contiguous()
        if prev_attn_masks is None:
            prev_attn_masks = attn_masks
        else:
            prev_attn_masks = prev_attn_masks @ attn_masks
        # [batch_size, heightxwidth, num_groups] -> [batch_size, num_groups, heightxwidth] -> [batch_size, num_groups, height, width]
        cur_attn_map = resize_attention_map(prev_attn_masks.permute(0, 2, 1).contiguous(), *hw_shape)
        attn_maps.append(cur_attn_map)

    # [batch_size, num_groups, height, width]
    final_grouping = attn_maps[-1]

    return final_grouping


class GroupViTCrossAttentionLayer(nn.Cell):
    def __init__(self, config: GroupViTVisionConfig):
        super().__init__()
        self.attn = GroupViTAttention(config)
        self.norm2 = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GroupViTMLP(config)
        self.norm_post = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def construct(self, query, key):
        x = query
        x = x + self.attn(query, encoder_hidden_states=key)[0]
        x = x + self.mlp(self.norm2(x))
        x = self.norm_post(x)
        return x


class GroupViTAssignAttention(nn.Cell):
    def __init__(self, config: GroupViTVisionConfig):
        super().__init__()
        self.scale = config.hidden_size**-0.5

        self.q_proj = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.proj = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.assign_eps = config.assign_eps

    def get_attn(self, attn, gumbel=True, hard=True):
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=-2, hard=hard)
        else:
            if hard:
                attn = hard_softmax(attn, dim=-2)
            else:
                attn = mint.nn.functional.softmax(attn, dim=-2)

        return attn

    def construct(self, query, key):
        value = key
        # [batch_size, query_length, channels]
        query = self.q_proj(query)

        # [batch_size, key_length, channels]
        key = self.k_proj(key)

        # [batch_size, key_length, channels]
        value = self.v_proj(value)

        # [batch_size, query_length, key_length]
        raw_attn = (query @ key.swapaxes(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)

        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)

        out = attn @ value

        out = self.proj(out)

        return out, soft_attn


class GroupViTTokenAssign(nn.Cell):
    def __init__(self, config: GroupViTVisionConfig, num_group_token, num_output_group):
        super().__init__()
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        assign_mlp_ratio = (
            config.assign_mlp_ratio
            if isinstance(config.assign_mlp_ratio, collections.abc.Iterable)
            else (config.assign_mlp_ratio, config.assign_mlp_ratio)
        )
        tokens_dim, channels_dim = [int(x * config.hidden_size) for x in assign_mlp_ratio]
        self.mlp_inter = GroupViTMixerMLP(config, num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # norm on x
        self.norm_x = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_assign_attn = GroupViTCrossAttentionLayer(config)

        self.assign = GroupViTAssignAttention(config)
        self.norm_new_x = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_channels = GroupViTMLP(config, config.hidden_size, channels_dim, config.hidden_size)

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (ms.Tensor): group tokens, [batch_size, num_group_tokens, channels]

        Returns:
            projected_group_tokens (ms.Tensor): [batch_size, num_output_groups, channels]
        """
        # [B, num_output_groups, C] <- [B, num_group_tokens, C]
        projected_group_tokens = self.mlp_inter(group_tokens)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def construct(self, image_tokens, group_tokens):
        """
        Args:
            image_tokens (`ms.Tensor`): image tokens, of shape [batch_size, input_length, channels]
            group_tokens (`ms.Tensor`): group tokens, [batch_size, num_group_tokens, channels]
        """

        group_tokens = self.norm_tokens(group_tokens)
        image_tokens = self.norm_x(image_tokens)
        # [batch_size, num_output_groups, channels]
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, image_tokens)
        new_image_tokens, attention = self.assign(projected_group_tokens, image_tokens)
        new_image_tokens += projected_group_tokens

        new_image_tokens = new_image_tokens + self.mlp_channels(self.norm_new_x(new_image_tokens))

        return new_image_tokens, attention


@dataclass
class GroupViTModelOutput(ModelOutput):
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
        segmentation_logits (`ms.Tensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        text_embeds (`ms.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`GroupViTTextModel`].
        image_embeds (`ms.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`GroupViTVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`GroupViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`GroupViTVisionModel`].
    """

    loss: Optional[ms.Tensor] = None
    logits_per_image: Optional[ms.Tensor] = None
    logits_per_text: Optional[ms.Tensor] = None
    segmentation_logits: Optional[ms.Tensor] = None
    text_embeds: Optional[ms.Tensor] = None
    image_embeds: Optional[ms.Tensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class GroupViTPatchEmbeddings(nn.Cell):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = mint.nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def construct(self, pixel_values: ms.Tensor, interpolate_pos_encoding: bool = False) -> ms.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        x = self.projection(pixel_values).flatten(2).swapaxes(1, 2)
        return x


class GroupViTVisionEmbeddings(nn.Cell):
    def __init__(self, config: GroupViTVisionConfig):
        super().__init__()

        self.patch_embeddings = GroupViTPatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = ms.Parameter(
            mint.zeros((1, num_patches, config.hidden_size)), name="position_embeddings"
        )
        self.dropout = mint.nn.Dropout(config.dropout)
        self.layernorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: ms.Tensor, height: int, width: int) -> ms.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        patch_pos_embed = self.position_embeddings

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = mindspore_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = mint.nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def construct(self, pixel_values: ms.Tensor, interpolate_pos_encoding: bool = False) -> ms.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        embeddings = self.layernorm(embeddings)

        batch_size, seq_len, _ = embeddings.shape

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->GroupViT
class GroupViTTextEmbeddings(nn.Cell):
    def __init__(self, config: GroupViTTextConfig):
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


class GroupViTStage(nn.Cell):
    """This corresponds to the `GroupingLayer` class in the GroupViT implementation."""

    def __init__(
        self,
        config: GroupViTVisionConfig,
        depth: int,
        num_prev_group_token: int,
        num_group_token: int,
        num_output_group: int,
    ):
        super().__init__()
        self.depth = depth
        self.num_group_token = num_group_token
        if num_group_token > 0:
            self.group_token = ms.Parameter(mint.zeros((1, num_group_token, config.hidden_size)), name="group_token")
        else:
            self.group_token = None
        self.layers = nn.CellList([GroupViTEncoderLayer(config) for _ in range(depth)])

        if num_group_token > 0:
            self.downsample = GroupViTTokenAssign(
                config=config,
                num_group_token=num_group_token,
                num_output_group=num_output_group,
            )
        else:
            self.downsample = None

        if num_prev_group_token > 0 and num_group_token > 0:
            self.group_projector = nn.SequentialCell(
                mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                GroupViTMixerMLP(config, num_prev_group_token, config.hidden_size // 2, num_group_token),
            )
        else:
            self.group_projector = None

    @property
    def with_group_token(self):
        return self.group_token is not None

    def split_x(self, x):
        if self.with_group_token:
            return x[:, : -self.num_group_token], x[:, -self.num_group_token :]
        else:
            return x, None

    def concat_x(self, x: ms.Tensor, group_token: Optional[ms.Tensor] = None) -> ms.Tensor:
        if group_token is None:
            return x
        return mint.cat([x, group_token], dim=1)

    def construct(
        self,
        hidden_states: ms.Tensor,
        prev_group_token: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor]:
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the grouping tensors of Grouping block.
        """
        if self.with_group_token:
            group_token = self.group_token.broadcast_to((hidden_states.shape[0], -1, -1))
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None

        x = hidden_states

        cat_x = self.concat_x(x, group_token)
        for layer in self.layers:
            layer_out = layer(cat_x, attention_mask=None, causal_attention_mask=None)
            cat_x = layer_out[0]

        x, group_token = self.split_x(cat_x)

        attention = None
        if self.downsample is not None:
            x, attention = self.downsample(x, group_token)

        outputs = (x, group_token)
        if output_attentions:
            outputs = outputs + (attention,)

        return outputs


class GroupViTMLP(nn.Cell):
    def __init__(
        self,
        config: GroupViTVisionConfig,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        output_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        output_size = output_size if output_size is not None else hidden_size
        self.fc1 = mint.nn.Linear(hidden_size, intermediate_size)
        self.fc2 = mint.nn.Linear(intermediate_size, output_size)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class GroupViTMixerMLP(GroupViTMLP):
    def construct(self, x):
        x = super().construct(x.swapaxes(1, 2))
        return x.swapaxes(1, 2)


class GroupViTAttention(nn.Cell):
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

    def _shape(self, tensor: ms.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2).contiguous()

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        causal_attention_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.shape
        is_cross_attention = encoder_hidden_states is not None

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        if is_cross_attention:
            key_states = self._shape(self.k_proj(encoder_hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(encoder_hidden_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = mint.bmm(query_states, key_states.swapaxes(1, 2))

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

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Copied from transformers.models.altclip.modeling_altclip.AltCLIPEncoderLayer with AltCLIP->GroupViT
class GroupViTEncoderLayer(nn.Cell):
    def __init__(self, config: GroupViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = GroupViTAttention(config)
        self.layer_norm1 = mint.nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = GroupViTMLP(config)
        self.layer_norm2 = mint.nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
        causal_attention_mask: ms.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor]:
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
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


class GroupViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GroupViTConfig
    base_model_prefix = "groupvit"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


GROUPVIT_START_DOCSTRING = r"""
    This model is a MindSpore
    [ms.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.htm) subclass. Use it
    as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`GroupViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GROUPVIT_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`CLIPTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

GROUPVIT_VISION_INPUTS_DOCSTRING = r"""
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
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

GROUPVIT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`CLIPTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class GroupViTVisionEncoder(nn.Cell):
    def __init__(self, config: GroupViTVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.stages = nn.CellList(
            [
                GroupViTStage(
                    config=config,
                    depth=config.depths[i],
                    num_group_token=config.num_group_tokens[i],
                    num_output_group=config.num_output_groups[i],
                    num_prev_group_token=config.num_output_groups[i - 1] if i > 0 else 0,
                )
                for i in range(len(config.depths))
            ]
        )
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = () if output_hidden_states else None
        all_groupings = () if output_attentions else None

        group_tokens = None

        for i, stage in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = stage(hidden_states, group_tokens, output_attentions)

            hidden_states = layer_outputs[0]
            group_tokens = layer_outputs[1]

            if output_attentions and layer_outputs[2] is not None:
                all_groupings = all_groupings + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_groupings] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_groupings
        )


class GroupViTTextEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self-attention layers. Each layer is a
    [`GroupViTEncoderLayer`].

    Args:
        config: GroupViTTextConfig
    """

    def __init__(self, config: GroupViTTextConfig):
        super().__init__()
        self.config = config
        self.layers = nn.CellList([GroupViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])
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
            causal_attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

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


class GroupViTTextTransformer(nn.Cell):
    def __init__(self, config: GroupViTTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = GroupViTTextEmbeddings(config)
        self.encoder = GroupViTTextEncoder(config)
        self.final_layer_norm = mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
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

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(input_shape, hidden_states.dtype)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

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

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            pooled_output = last_hidden_state[
                mint.arange(last_hidden_state.shape[0]), input_ids.to(dtype=ms.int32).argmax(dim=-1)
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                mint.arange(last_hidden_state.shape[0]),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=ms.int32) == self.eos_token_id).int().argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GroupViTTextModel(GroupViTPreTrainedModel):
    config_class = GroupViTTextConfig

    def __init__(self, config: GroupViTTextConfig):
        super().__init__(config)
        self.text_model = GroupViTTextTransformer(config)
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
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer
        >>> from mindone.transformers import GroupViTTextModel
        >>> import mindspore as ms

        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc", revision="refs/pr/2")
        >>> model = GroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc", revision="refs/pr/2")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class GroupViTVisionTransformer(nn.Cell):
    def __init__(self, config: GroupViTVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = GroupViTVisionEmbeddings(config)
        self.encoder = GroupViTVisionEncoder(config)
        self.layernorm = mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
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

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        # normalize the last hidden state
        last_hidden_state = self.layernorm(last_hidden_state)
        pooled_output = last_hidden_state.mean(dim=1)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GroupViTVisionModel(GroupViTPreTrainedModel):
    config_class = GroupViTVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: GroupViTVisionConfig):
        super().__init__(config)
        self.vision_model = GroupViTVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> GroupViTPatchEmbeddings:
        return self.vision_model.embeddings.patch_embeddings

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor
        >>> from mindone.transformers import GroupViTVisionModel
        >>> import mindspore as ms

        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc", revision="refs/pr/2")
        >>> model = GroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc", revision="refs/pr/2")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class GroupViTModel(GroupViTPreTrainedModel):
    config_class = GroupViTConfig

    def __init__(self, config: GroupViTConfig):
        super().__init__(config)

        if not isinstance(config.text_config, GroupViTTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type GroupViTTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, GroupViTVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type GroupViTVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.projection_intermediate_dim = config.projection_intermediate_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = GroupViTTextTransformer(text_config)
        self.vision_model = GroupViTVisionTransformer(vision_config)

        self.visual_projection = nn.SequentialCell(
            mint.nn.Linear(self.vision_embed_dim, self.projection_intermediate_dim, bias=True),
            mint.nn.BatchNorm1d(self.projection_intermediate_dim),
            ReLU(inplace=True),
            mint.nn.Linear(self.projection_intermediate_dim, self.projection_dim, bias=True),
        )
        self.text_projection = nn.SequentialCell(
            mint.nn.Linear(self.text_embed_dim, self.projection_intermediate_dim, bias=True),
            mint.nn.BatchNorm1d(self.projection_intermediate_dim),
            ReLU(inplace=True),
            mint.nn.Linear(self.projection_intermediate_dim, self.projection_dim, bias=True),
        )
        self.logit_scale = ms.Parameter(ms.tensor(self.config.logit_scale_init_value), name="logit_scale")

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ms.Tensor:
        r"""
        Returns:
            text_features (`ms.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`GroupViTTextModel`].

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer
        >>> from mindone.transformers import GroupViTModel
        >>> import mindspore as ms

        >>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc", revision="refs/pr/2")
        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc", revision="refs/pr/2")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use GROUPVIT model's config for some fields (if specified) instead of those of vision & text components.
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

    def get_image_features(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ms.Tensor:
        r"""
        Returns:
            image_features (`ms.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`GroupViTVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor
        >>> from mindone.transformers import GroupViTModel
        >>> import mindspore as ms

        >>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc", revision="refs/pr/2")
        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc", revision="refs/pr/2")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use GROUPVIT model's config for some fields (if specified) instead of those of vision & text components.
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
        output_segmentation: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, GroupViTModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor
        >>> from mindone.transformers import GroupViTModel
        >>> import mindspore as ms

        >>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc", revision="refs/pr/2")
        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc", revision="refs/pr/2")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="np", padding=True
        ... )
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(axis=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use GROUPVIT model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_segmentation = (
            output_segmentation if output_segmentation is not None else self.config.output_segmentation
        )
        if output_segmentation:
            output_attentions = True
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
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = mint.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        seg_logits = None
        if output_segmentation:
            # grouped features
            # [batch_size_image, num_group, hidden_size]
            image_group_embeds = vision_outputs[0]
            # [batch_size_image*num_group, hidden_size]
            image_group_embeds = self.visual_projection(image_group_embeds.reshape(-1, image_group_embeds.shape[-1]))
            if output_hidden_states:
                attentions = vision_outputs[3]
            else:
                attentions = vision_outputs[2]
            # [batch_size_image, num_group, height, width]
            grouping = get_grouping_from_attentions(attentions, pixel_values.shape[2:])

            # normalized features
            image_group_embeds = image_group_embeds / image_group_embeds.norm(dim=-1, keepdim=True)
            # [batch_size_image x num_group, batch_size_text]
            logits_per_image_group = mint.matmul(image_group_embeds, text_embeds.t()) * logit_scale
            # [batch_size_image, batch_size_text, num_group]
            logits_per_image_group = logits_per_image_group.reshape(
                image_embeds.shape[0], -1, text_embeds.shape[0]
            ).permute(0, 2, 1)

            # [batch_size_image, batch_size_text, height x width]
            flatten_grouping = grouping.reshape(grouping.shape[0], grouping.shape[1], -1)

            # [batch_size_image, batch_size_text, height, width]
            seg_logits = mint.matmul(logits_per_image_group, flatten_grouping) * logit_scale
            seg_logits = seg_logits.reshape(
                seg_logits.shape[0], seg_logits.shape[1], grouping.shape[2], grouping.shape[3]
            )

        loss = None
        if return_loss:
            loss = groupvit_loss(logits_per_text)

        if not return_dict:
            if seg_logits is not None:
                output = (
                    logits_per_image,
                    logits_per_text,
                    seg_logits,
                    text_embeds,
                    image_embeds,
                    text_outputs,
                    vision_outputs,
                )
            else:
                output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return GroupViTModelOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            segmentation_logits=seg_logits,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


__all__ = ["GroupViTModel", "GroupViTPreTrainedModel", "GroupViTTextModel", "GroupViTVisionModel"]
