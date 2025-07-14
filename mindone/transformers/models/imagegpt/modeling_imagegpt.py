# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and HuggingFace Inc. team.
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
"""MindSpore OpenAI ImageGPT model."""

import warnings
from typing import Any, Optional, Tuple, Union

import numpy as np
from transformers.models.imagegpt.configuration_imagegpt import ImageGPTConfig

import mindspore as ms
from mindspore import mint, nn

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...mindspore_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import MSPreTrainedModel
from ...utils import (  # add_start_docstrings,; add_start_docstrings_to_model_forward,; replace_return_docstrings,
    logging,
)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openai/imagegpt-small"
_CONFIG_FOR_DOC = "ImageGPTConfig"


def dtype_to_min(dtype):
    if dtype == ms.float16:
        return ms.tensor(np.finfo(np.float16).min, dtype=ms.float16)
    if dtype == ms.float32:
        return ms.tensor(np.finfo(np.float32).min, dtype=ms.float32)
    if dtype == ms.float64:
        return ms.tensor(np.finfo(np.float64).min, dtype=ms.float64)
    if dtype == ms.bfloat16:
        return ms.tensor(float.fromhex("-0x1.fe00000000000p+127"), dtype=ms.bfloat16)
    else:
        raise ValueError(f"Only support get minimum value of (float16, ), but got {dtype}")


class ImageGPTLayerNorm(nn.Cell):
    def __init__(self, hidden_size: Tuple[int], eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = ms.Parameter(mint.zeros(hidden_size))

    def construct(self, tensor: ms.Tensor) -> ms.Tensor:
        # input is not mean centered
        tensor = tensor / mint.sqrt(mint.mean(mint.square(tensor), dim=-1, keepdim=True) + self.eps)
        tensor = tensor * self.weight
        return tensor


class ImageGPTAttention(nn.Cell):
    def __init__(self, config, is_cross_attention: Optional[bool] = False, layer_idx: Optional[int] = None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.bias = mint.tril(mint.ones((max_positions, max_positions), dtype=ms.bool_)).view(
            1, 1, max_positions, max_positions
        )
        self.masked_bias = ms.tensor(-1e4)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = mint.nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = mint.nn.Dropout(p=config.resid_pdrop)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prunes heads of the model. heads: list of heads to prune in this layer.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = mint.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = mint.matmul(query, mint.transpose(key, -1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / mint.sqrt(ms.tensor(value.shape[-1], dtype=attn_weights.dtype))

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / ms.tensor(float(self.layer_idx + 1), dtype=attn_weights.dtype)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.shape[-2], key.shape[-2]
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            # Get min value for the dtype
            mask_value = dtype_to_min(attn_weights.dtype)
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            mask_value = ms.tensor(mask_value, dtype=attn_weights.dtype)
            attn_weights = mint.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = mint.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = mint.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `mint.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.shape
        _, _, k_seq_len, _ = key.shape

        # Preallocate attn_weights for `baddbmm`
        attn_weights = mint.empty((bsz * num_heads, q_seq_len, k_seq_len), dtype=mint.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.shape[-1]) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        # TODO:  cast to float32??
        q, k = query.reshape(-1, q_seq_len, dk), mint.transpose(key, -1, -2).reshape(-1, dk, k_seq_len)
        attn_weights = mint.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
        attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.shape[-2], key.shape[-2]
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            # Get min value for the dtype
            mask_value = dtype_to_min(attn_weights.dtype)
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = ms.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = mint.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = mint.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != ms.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype mint.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = mint.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.shape[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.shape[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def construct(
        self,
        hidden_states: ms.Tensor,
        layer_past: Optional[bool] = None,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> tuple:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `ImageGPTAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = mint.cat((past_key, key), dim=-2)
            value = mint.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class ImageGPTMLP(nn.Cell):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = mint.nn.Dropout(p=config.resid_pdrop)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ImageGPTBlock(nn.Cell):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = ImageGPTAttention(config, layer_idx=layer_idx)
        self.ln_2 = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = ImageGPTAttention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = ImageGPTMLP(inner_dim, config)

    def construct(
        self,
        hidden_states: ms.Tensor,
        layer_past: Optional[bool] = None,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,) + (outputs if use_cache else outputs[1:])

        return outputs  # hidden_states, present, (attentions)


class ImageGPTPreTrainedModel(MSPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ImageGPTConfig
    base_model_prefix = "transformer"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ImageGPTBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    # def _init_weights(self, cell):
    #     """Initialize the weights."""
    #     if isinstance(cell, (mint.nn.Linear, Conv1D)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         cell.weight.set_data(initializer(TruncatedNormal(sigma=self.config.initializer_range),
    #                                        cell.weight.shape, cell.weight.dtype))
    #         if cell.bias is not None:
    #             cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
    #     elif isinstance(cell, mint.nn.Embedding):
    #         cell.embedding_table.set_data(initializer(TruncatedNormal(sigma=self.config.initializer_range),
    #                                                 cell.embedding_table.shape, cell.embedding_table.dtype))
    #     elif isinstance(cell, nn.LayerNorm):
    #         cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))
    #         cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))


IMAGEGPT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ImageGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

IMAGEGPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoImageProcessor`]. See [`ImageGPTImageProcessor.__call__`] for details.

        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
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


# @add_start_docstrings(
#     "The bare ImageGPT Model transformer outputting raw hidden-states without any specific head on top.",
#     IMAGEGPT_START_DOCSTRING,
# )
class ImageGPTModel(ImageGPTPreTrainedModel):
    """
    The ImageGPT Model transformer outputting raw hidden-states without any specific head on top.

    This model is a PyTorch [torch.nn.Cell](https://pytorch.org/docs/stable/nn.html#torch.nn.Cell) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    """

    def __init__(self, config: ImageGPTConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = mint.nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = mint.nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = mint.nn.Dropout(p=config.embd_pdrop)
        self.h = nn.CellList([ImageGPTBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = ImageGPTLayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        # self.model_parallel = False
        # self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from mindone.transformers import AutoImageProcessor, ImageGPTModel
        >>> from PIL import Image
        >>> from mindspore import tensor
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
        >>> model = ImageGPTModel.from_pretrained("openai/imagegpt-small")

        >>> inputs = image_processor(images=image, return_tensors="np")
        >>> outputs = model(tensor(inputs))
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""

        if "pixel_values" in kwargs:
            warnings.warn(
                "The `pixel_values` argument is deprecated and will be removed in v4.47, use `input_ids` instead.",
                FutureWarning,
            )

            if input_ids is not None:
                raise ValueError(
                    "You cannot pass both `pixel_values` and `input_ids`. Please make sure to only pass `input_ids`."
                )

            input_ids = kwargs.pop("pixel_values")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        # TODO: delete this
        # if position_ids is not None:
        #     position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = mint.arange(past_length, input_shape[-1] + past_length, dtype=ms.int64)
            position_ids = position_ids.unsqueeze(0)

        # ImageGPTAttention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * dtype_to_min(attention_mask.dtype)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = mint.ones(encoder_hidden_shape)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            inputs_embeds = inputs_embeds + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# @add_start_docstrings(
#     """
#     The ImageGPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
#     embeddings).
#     """,
#     IMAGEGPT_START_DOCSTRING,
# )
class ImageGPTForCausalImageModeling(ImageGPTPreTrainedModel, GenerationMixin):
    """
    The ImageGPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ImageGPTConfig):
        super().__init__(config)
        self.transformer = ImageGPTModel(config)
        self.lm_head = mint.nn.Linear(config.hidden_size, config.vocab_size - 1, bias=False)
        # self.lm_head.weight = self.transformer.wte.embedding_table

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ImageGPTForCausalImageModeling
        >>> import torch
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np

        >>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
        >>> model = ImageGPTForCausalImageModeling.from_pretrained("openai/imagegpt-small")
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> # unconditional generation of 8 images
        >>> batch_size = 4
        >>> context = torch.full((batch_size, 1), model.config.vocab_size - 1)  # initialize with SOS token
        >>> context = context.to(device)
        >>> output = model.generate(
        ...     input_ids=context, max_length=model.config.n_positions + 1, temperature=1.0, do_sample=True, top_k=40
        ... )

        >>> clusters = image_processor.clusters
        >>> height = image_processor.size["height"]
        >>> width = image_processor.size["width"]

        >>> samples = output[:, 1:].cpu().detach().numpy()
        >>> samples_img = [
        ...     np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [height, width, 3]).astype(np.uint8) for s in samples
        ... ]  # convert color cluster tokens back to pixels
        >>> f, axes = plt.subplots(1, batch_size, dpi=300)

        >>> for img, ax in zip(samples_img, axes):  # doctest: +IGNORE_RESULT
        ...     ax.axis("off")
        ...     ax.imshow(img)
        ```"""

        if "pixel_values" in kwargs:
            warnings.warn(
                "The `pixel_values` argument is deprecated and will be removed in v4.47, use `input_ids` instead.",
                FutureWarning,
            )

            if input_ids is not None:
                raise ValueError(
                    "You cannot pass both `pixel_values` and `input_ids`. Please make sure to only pass `input_ids`."
                )

            input_ids = kwargs.pop("pixel_values")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = mint.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values: Tuple[Tuple[ms.Tensor]], beam_idx: ms.Tensor) -> Tuple[Tuple[ms.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past) for layer_past in past_key_values
        )


# @add_start_docstrings(
#     """
#     The ImageGPT Model transformer with an image classification head on top (linear layer).
#     [`ImageGPTForImageClassification`] average-pools the hidden states in order to do the classification.
#     """,
#     IMAGEGPT_START_DOCSTRING,
# )
class ImageGPTForImageClassification(ImageGPTPreTrainedModel):
    """
    The ImageGPT Model transformer with an image classification head on top (linear layer).
    [`ImageGPTForImageClassification`] average-pools the hidden states in order to do the classification.
    """

    def __init__(self, config: ImageGPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = ImageGPTModel(config)
        self.score = mint.nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ImageGPTForImageClassification
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
        >>> model = ImageGPTForImageClassification.from_pretrained("openai/imagegpt-small")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        ```"""

        if "pixel_values" in kwargs:
            warnings.warn(
                "The `pixel_values` argument is deprecated and will be removed in v4.47, use `input_ids` instead.",
                FutureWarning,
            )

            if input_ids is not None:
                raise ValueError(
                    "You cannot pass both `pixel_values` and `input_ids`. Please make sure to only pass `input_ids`."
                )

            input_ids = kwargs.pop("pixel_values")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        # average-pool the hidden states along the sequence dimension
        pooled_hidden_states = hidden_states.mean(dim=1)
        # project from (batch_size, hidden_size) to (batch_size, num_labels)
        logits = self.score(pooled_hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == ms.int64 or labels.dtype == ms.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = mint.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = mint.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = mint.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


__all__ = [
    "ImageGPTForCausalImageModeling",
    "ImageGPTForImageClassification",
    "ImageGPTModel",
    "ImageGPTPreTrainedModel",
]
