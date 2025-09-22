# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
"""MindSpore VipLlava model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from transformers import VipLlavaConfig
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg

import mindspore as ms
import mindspore.mint as mint
from mindspore import nn

from mindone.models.utils import normal_, zeros_

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ..auto import AutoModel, AutoModelForCausalLM

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "VipLlavaConfig"


@dataclass
# Copied from transformers.models.llava.modeling_llava.LlavaCausalLMOutputWithPast with Llava->VipLlava
class VipLlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for VipLlava causal language model (or autoregressive) outputs.

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
        image_hidden_states (`ms.Tensor`, *optional*):
            A `ms.Tensor` of size (batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[ms.Tensor] = None
    logits: ms.Tensor = None
    past_key_values: Optional[List[ms.Tensor]] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None
    image_hidden_states: Optional[ms.Tensor] = None


class VipLlavaMultiModalProjector(nn.Cell):
    def __init__(self, config: VipLlavaConfig):
        super().__init__()
        num_feature_layers = 1 if isinstance(config.vision_feature_layers, int) else len(config.vision_feature_layers)
        self.projector_layernorm = mint.nn.LayerNorm(
            num_feature_layers * config.vision_config.hidden_size, eps=config.projector_layernorm_eps
        )

        self.linear_1 = mint.nn.Linear(
            num_feature_layers * config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = mint.nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def construct(self, hidden_states):
        hidden_states = self.projector_layernorm(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


VIPLLAVA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a MindSpore [mindspore.mint.nn.Cell](https://mindspore.org/docs/stable/nn.html#mindspore.mint.nn.Cell) subclass.
    Use it as a regular MindSpore Module and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VipLlavaConfig`] or [`VipLlavaVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare VipLlava Model outputting raw hidden-states without any specific head on top.",
    VIPLLAVA_START_DOCSTRING,
)
# Copied from transformers.models.llava.modeling_llava.LlavaPreTrainedModel with Llava->VipLlava,llava->vipllava
class VipLlavaPreTrainedModel(PreTrainedModel):
    config_class = VipLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VipLlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = False
    _supports_static_cache = False

    def _init_weights(self, module):
        # important: this ported version of VipLlava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/vipllava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            normal_(module.class_embedding, mean=0.0, std=std)

        if isinstance(module, (mint.nn.Linear, mint.nn.Conv2d)):
            normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, mint.nn.Embedding):
            normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0


VIPLLAVA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details ([]`LlavaProcessor`] uses
            [`CLIPImageProcessor`] for processing images).
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
        cache_position (`ms.Tensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    """The VIPLLAVA model which consists of a vision backbone and a language model.""",
    VIPLLAVA_START_DOCSTRING,
)
# Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration with LLAVA->VIPLLAVA,Llava->VipLlava
class VipLlavaForConditionalGeneration(VipLlavaPreTrainedModel, GenerationMixin):
    def __init__(self, config: VipLlavaConfig):
        super().__init__(config)
        # config.vision_config._attn_implementation = config._attn_implementation # FIXME: CLIP no support FA yet
        config.text_config._attn_implementation = config._attn_implementation

        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = VipLlavaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.post_init()

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

    # Ignore copy
    def get_image_features(self, pixel_values: ms.Tensor, vision_feature_layers: Union[int, List[int]]):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`ms.Tensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layers (`Union[int, List[int]]`):
                The vision feature layer, or the list of indexes of the layers to select
                the vision feature.
        Returns:
            image_features (`ms.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, return_dict=True)

        # If multiple feature layers are provided (which is usually the case)
        # then the image features are concatenated after the CLS is removed.
        if isinstance(vision_feature_layers, int):
            image_features = image_outputs.hidden_states[vision_feature_layers][:, 1:]
        else:
            # Usually, we select the features from index 1: the layers -2, -5, -8, -11 and 6
            image_features = [image_outputs.hidden_states[index][:, 1:] for index in vision_feature_layers]
            image_features = mint.cat(image_features, dim=-1)
        image_features = self.multi_modal_projector(image_features)
        return image_features

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(VIPLLAVA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VipLlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # Ignore copy
    def construct(
        self,
        input_ids: ms.Tensor = None,
        pixel_values: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        vision_feature_layers: Optional[Union[int, List[int]]] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[ms.Tensor] = None,
        logits_to_keep: Union[int, ms.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[Tuple, VipLlavaCausalLMOutputWithPast]:
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
        >>> import mindspore as ms

        >>> from PIL import Image
        >>> import requests
        >>> from mindone.transformers import AutoProcessor, VipLlavaForConditionalGeneration

        >>> model = VipLlavaForConditionalGeneration.from_pretrained("llava-hf/vip-llava-7b-hf", mindspore_dtype=ms.float16)
        >>> processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf")

        >>> prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, \
        and polite answers to the human's questions.###Human: <image>\n{}###Assistant:"
        >>> question = "Can you please describe this image?"
        >>> prompt = prompt.format(question)
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-neg.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="np")
        >>> for k, v in inputs.items():
        ...     inputs[k] = ms.tensor(v)
        ...     if inputs[k].dtype == ms.int64:
        ...         inputs[k] = inputs[k].to(ms.int32)
        ...     else:
        ...         inputs[k] = inputs[k].to(ms.float16)

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=20)
        >>> processor.decode(generate_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        The image features a brown and white cat sitting on a green surface, with a red ball in its
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layers = (
            vision_feature_layers if vision_feature_layers is not None else self.config.vision_feature_layers
        )

        if (input_ids is None) != (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values, vision_feature_layers=vision_feature_layers
            )

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = (input_ids == self.config.image_token_index).sum()
                n_image_features = image_features.shape[0] * image_features.shape[1]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.float()
            inputs_embeds = (
                inputs_embeds.float().masked_scatter(special_image_mask, image_features).to(inputs_embeds.dtype)
            )

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
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :]
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = mint.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VipLlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        if logits_to_keep is None:  # Mindspore-specific
            logits_to_keep = 1

        # Overwritten -- in specific circumstances we don't want to construct image inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


__all__ = ["VipLlavaForConditionalGeneration", "VipLlavaPreTrainedModel"]
