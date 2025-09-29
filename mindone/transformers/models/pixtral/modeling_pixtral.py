# coding=utf-8
# Copyright 2024 Mistral and the HuggingFace Inc. team. All rights reserved.
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
"""MindSpore Pixtral model."""

from typing import Optional, Tuple, Union

from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig

import mindspore as ms
from mindspore import mint, nn, ops

from ...activations import ACT2FN
from ...mindspore_adapter import dtype_to_min
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging

logger = logging.get_logger(__name__)


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        mesh = mint.meshgrid(mint.arange(height), mint.arange(width), indexing="ij")
        h_grid, v_grid = mint.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid
        positions.append(ids[:, 0])
    return mint.cat(positions)


class PixtralRotaryEmbedding(nn.Cell):
    """
    The key with pixtral embedding is just that you have a frequency for each pixel positions.
    If you have height x width pixels (or embedding pixels), then the frequency used for ROPE
    is given by indexing the pre_computed frequency on the width and height.

    What you output is of dimension (batch, height * width, dim) with dim the embed dim.

    This simply means that for each image hidden state, you are going to add
    a corresponding positional embedding, based on its index in the grid.
    """

    def __init__(self, config):
        super().__init__()
        self.rope_type = "default"
        self.dim = config.head_dim
        self.base = config.rope_theta
        max_patches_per_side = config.image_size // config.patch_size
        freqs = 1.0 / (self.base ** (mint.arange(0, self.dim, 2).float() / self.dim))

        h = mint.arange(max_patches_per_side)
        w = mint.arange(max_patches_per_side)

        freqs_h = mint.outer(h, freqs[::2]).float()
        freqs_w = mint.outer(w, freqs[1::2]).float()
        inv_freq = mint.cat(
            [
                freqs_h[:, None, :].tile((1, max_patches_per_side, 1)),
                freqs_w[None, :, :].tile((max_patches_per_side, 1, 1)),
            ],
            dim=-1,
        ).reshape(
            -1, self.dim // 2
        )  # we reshape to only index on the position indexes, not tuple of indexes
        # Different from paper, but it uses a different permutation in order to obtain the same calculation

        self.inv_freq = ms.Parameter(mint.cat((inv_freq, inv_freq), dim=-1), requires_grad=False, name="inv_freq")

    # @dynamic_rope_update
    def construct(self, x, position_ids):
        freqs = self.inv_freq[position_ids]

        # Force float32
        emb = freqs.float()
        cos = emb.cos()
        sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
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


class PixtralAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = mint.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = mint.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = mint.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.o_proj = mint.nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_embeddings: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).swapaxes(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=0)

        attn_weights = mint.matmul(query_states, key_states.swapaxes(2, 3)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = mint.nn.functional.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query_states.dtype)
        attn_weights = mint.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = mint.matmul(attn_weights, value_states)

        attn_output = attn_output.swapaxes(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, patches, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Pixtral
class PixtralMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = mint.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = mint.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = mint.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Pixtral
class PixtralRMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        """
        PixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = ms.Parameter(mint.ones(hidden_size))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * mint.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class PixtralAttentionLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.feed_forward = PixtralMLP(config)
        self.attention = PixtralAttention(config)
        self.ffn_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
        position_embeddings: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
        output_attentions: Optional[bool] = None,
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

        hidden_states = self.attention_norm(hidden_states)
        hidden_states, attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class PixtralTransformer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = ms.nn.CellList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(PixtralAttentionLayer(config))
        self.gradient_checkpointing = False

    def construct(
        self,
        inputs_embeds,
        attention_mask: Optional[ms.Tensor] = None,
        position_embeddings: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embeddings which serve as input to the Transformer.
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

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
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


PIXTRAL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a MindSpore
    [ms.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) subclass.
    Use it as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PixtralVisionConfig`]):
            Model configuration class with all the parameters of the vision encoder. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


class PixtralPreTrainedModel(PreTrainedModel):
    config_class = PixtralVisionConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PixtralAttentionLayer"]

    def _init_weights(self, module):
        pass


PIXTRAL_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`AutoImageProcessor.__call__`]
            for details.
        image_sizes (`ms.Tensor` of shape `(batch_size, 2)`, *optional*):
            The sizes of the images in the batch, being (height, width) for each image.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def generate_block_attention_mask(patch_embeds_list, tensor):
    dtype = tensor.dtype
    seq_len = tensor.shape[1]
    d_min = dtype_to_min(dtype)
    causal_mask = ops.full((seq_len, seq_len), fill_value=d_min, dtype=dtype)

    block_end_idx = ms.tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = ms.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask[start:end, start:end] = 0

    causal_mask = causal_mask[None, None, :, :].broadcast_to((tensor.shape[0], 1, -1, -1))
    return causal_mask


class PixtralVisionModel(PixtralPreTrainedModel):
    base_model_prefix = "vision_encoder"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.patch_conv = mint.nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.patch_size = config.patch_size
        self.ln_pre = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = PixtralTransformer(config)
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.patch_conv

    def construct(
        self,
        pixel_values: ms.Tensor,
        image_sizes: ms.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Returns:
            pixel_values: tensor of token features for
                all tokens of all images of shape (N_toks, D)
        """
        # pass images through initial convolution independently
        patch_embeds = self.patch_conv(pixel_values)
        patch_embeds_list = [
            embed[..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)]
            for embed, size in zip(patch_embeds, image_sizes)
        ]

        # flatten to a single sequence
        patch_embeds = mint.cat([p.flatten(1).T for p in patch_embeds_list], dim=0).unsqueeze(0)
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list, max_width=self.config.image_size // self.config.patch_size
        )
        position_embeddings = self.patch_positional_embedding(patch_embeds, position_ids)

        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )

        out = self.transformer(
            patch_embeds,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        return out


__all__ = ["PixtralVisionModel", "PixtralPreTrainedModel"]
