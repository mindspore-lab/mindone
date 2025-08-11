# coding=utf-8
# Copyright 2023 HuggingFace Inc. team. All rights reserved.
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
"""Mindspore Fuyu model."""

from typing import Optional, Union

from transformers.models.fuyu.configuration_fuyu import FuyuConfig
from transformers.utils import logging

import mindspore as ms
from mindspore import mint
from mindspore.common.initializer import Normal, initializer

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.auto.modeling_auto import AutoModel

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "FuyuConfig"


class FuyuPreTrainedModel(PreTrainedModel):
    config: FuyuConfig
    base_model_prefix = "fuyu"
    supports_gradient_checkpointing = True
    _supports_attention_backend = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, mint.nn.Linear):
            module.weight.set_data(initializer(Normal(sigma=std, mean=0.0), module.weight.shape, module.weight.dtype))
            if module.bias is not None:
                module.bias.set_data(initializer("zeros", module.bias.shape, module.bias.dtype))
        elif isinstance(module, mint.nn.Embedding):
            module.weight.set_data(initializer(Normal(sigma=std, mean=0.0), module.weight.shape, module.weight.dtype))
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0


class FuyuModel(FuyuPreTrainedModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

    def __init__(self, config: FuyuConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModel.from_config(config.text_config)
        self.vision_embed_tokens = mint.nn.Linear(
            config.patch_size * config.patch_size * config.num_channels, config.hidden_size
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def gather_continuous_embeddings(
        self,
        word_embeddings: ms.Tensor,
        continuous_embeddings: list[ms.Tensor],
        image_patch_input_indices: ms.Tensor,
    ) -> ms.Tensor:
        """This function places the continuous_embeddings into the word_embeddings at the locations
        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
        embeddings.

        Args:
            word_embeddings (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Tensor of word embeddings.
            continuous_embeddings (`ms.Tensor` of shape `(batch_size, num_patches, hidden_size)`):
                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is shape
                [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
                indices in image_patch_input_indices for that batch element.
            image_patch_input_indices (`ms.Tensor` of shape `(batch_size, sequence_length)`):
                Tensor of indices of the image patches in the input_ids tensor.
        """
        if not (word_embeddings.shape[0] == len(continuous_embeddings)):
            raise ValueError(
                f"Batch sizes must match! Got {len(continuous_embeddings)=} and {word_embeddings.shape[0]=}"
            )

        output_embeddings = word_embeddings.clone()
        for batch_idx in range(word_embeddings.shape[0]):
            # First, find the positions of all the non-negative values in image_patch_input_indices, those are the
            # positions in word_embeddings that we want to replace with content from continuous_embeddings.
            dst_indices = mint.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            # Next look up those indices in image_patch_input_indices to find the indices in continuous_embeddings that we
            # want to use to replace the values in word_embeddings.
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            # Check if we have more indices than embeddings. Note that we could have fewer indices if images got truncated.
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(
                    f"Number of continuous embeddings {continuous_embeddings[batch_idx].shape=} does not match "
                    f"number of continuous token ids {src_indices.shape=} in batch element {batch_idx}."
                )
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices]
        return output_embeddings

    def get_image_features(self, pixel_values: ms.Tensor, **kwargs):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
        """
        patch_embeddings = [
            self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype)).squeeze(0)
            for patch in pixel_values
        ]
        return patch_embeddings

    def construct(
        self,
        input_ids: ms.Tensor = None,
        image_patches: ms.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        image_patches (`ms.Tensor` of shape `(batch_size, num_total_patches, patch_size_ x patch_size x num_channels)`, *optional*):
            Image patches to be used as continuous embeddings. The patches are flattened and then projected to the
            hidden size of the model.
        image_patches_indices (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Tensor of indices of the image patches in the input_ids tensor.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_is or inputs_embeds")

        if position_ids is None:
            past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = mint.arange(past_key_values_length, seq_length + past_key_values_length, dtype=ms.int64)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if image_patches is not None:
            patch_embeddings = self.get_image_features(image_patches)
            patch_embeddings = mint.cat(patch_embeddings, dim=0)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    ms.Tensor(self.config.image_token_id, dtype=ms.int64)
                )
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = input_ids == self.config.image_token_id

            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            patch_embeddings = patch_embeddings.to(inputs_embeds.dtype)
            inputs_embeds = (
                inputs_embeds.float()
                .masked_scatter(special_image_mask, patch_embeddings.float())
                .to(inputs_embeds.dtype)
            )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs,
        )

        return outputs


class FuyuForCausalLM(FuyuPreTrainedModel, GenerationMixin):
    def __init__(self, config: FuyuConfig):
        super().__init__(config)
        self.model = FuyuModel(config)
        self.lm_head = mint.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def construct(
        self,
        input_ids: ms.Tensor = None,
        image_patches: ms.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        image_patches (`ms.Tensor` of shape `(batch_size, num_total_patches, patch_size_ x patch_size x num_channels)`, *optional*):
            Image patches to be used as continuous embeddings. The patches are flattened and then projected to the
            hidden size of the model.
        image_patches_indices (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Tensor of indices of the image patches in the input_ids tensor.
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Examples:

        ```python
        >>> from transformers import FuyuProcessor, FuyuForCausalLM
        >>> from PIL import Image
        >>> import requests

        >>> processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        >>> model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

        >>> url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> prompt = "Generate a coco-style caption.\n"

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=7)
        >>> generation_text = processor.batch_decode(generated_ids[:, -7:], skip_special_tokens=True)
        >>> print(generation_text[0])
        A blue bus parked on the side of a road.
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            image_patches=image_patches,
            image_patches_indices=image_patches_indices,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=True,
            # don't pass kwargs because Persimmon-backbone doesn't accept FA2 kwargs yet, TODO: raushan
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

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
        image_patches=None,
        image_patches_indices=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to construce image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_patches=image_patches,
            image_patches_indices=image_patches_indices,
            **kwargs,
        )

        if past_key_values is not None:
            model_inputs["image_patches_indices"] = None
            model_inputs["image_patches"] = None

        return model_inputs


__all__ = ["FuyuForCausalLM", "FuyuPreTrainedModel", "FuyuModel"]
