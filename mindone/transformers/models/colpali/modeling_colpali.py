# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""MindSpore ColPali model"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from transformers import ColPaliConfig
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

import mindspore as ms
from mindspore import mint

from mindone.models.utils import normal_, zeros_

from ...cache_utils import Cache
from ...modeling_utils import PreTrainedModel
from ..auto import AutoModelForImageTextToText

_CONFIG_FOR_DOC = "ColPaliConfig"

COLPALI_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a MindSpore [nn.Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html) subclass.
    Use it as a regular MindSpore Module and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ColPaliConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare ColPali model outputting raw hidden-states without any specific head on top.",
    COLPALI_START_DOCSTRING,
)
class ColPaliPreTrainedModel(PreTrainedModel):
    config_class = ColPaliConfig
    base_model_prefix = "model"
    _no_split_modules = []

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.vlm_config.text_config.initializer_range
        )

        if isinstance(module, (mint.nn.Linear, mint.nn.Conv2d)):
            normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, mint.nn.Embedding):
            normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0


@dataclass
class ColPaliForRetrievalOutput(ModelOutput):
    """
    Base class for ColPali embeddings output.

    Args:
        loss (`ms.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        embeddings (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The embeddings of the model.
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
            A `ms.Tensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    """

    loss: Optional[ms.Tensor] = None
    embeddings: ms.Tensor = None
    past_key_values: Optional[Union[List[ms.Tensor], Cache]] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None
    image_hidden_states: Optional[ms.Tensor] = None


COLPALI_FOR_RETRIEVAL_INPUT_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SiglipImageProcessor.__call__`] for details ([]`PaliGemmaProcessor`] uses
            [`SiglipImageProcessor`] for processing images). If none, ColPali will only process text (query embeddings).
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
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional key word arguments passed along to the vlm backbone model.
"""


@add_start_docstrings(
    """
    In our proposed ColPali approach, we leverage VLMs to construct efficient multi-vector embeddings directly
    from document images for document retrieval. We train the model to maximize the similarity
    between these document embeddings and the corresponding query embeddings, using the late interaction method
    introduced in ColBERT.

    Using ColPali removes the need for potentially complex and brittle layout recognition and OCR pipelines with a
    single model that can take into account both the textual and visual content (layout, charts, etc.) of a document.
    """
)
class ColPaliForRetrieval(ColPaliPreTrainedModel):
    _checkpoint_conversion_mapping = {
        "vlm.language_model.model": "vlm.model.language_model",
        "vlm.vision_tower": "vlm.model.vision_tower",
        "vlm.multi_modal_projector": "vlm.model.multi_modal_projector",
        "vlm.language_model.lm_head": "vlm.lm_head",
    }

    def __init__(self, config: ColPaliConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vlm_config.text_config.vocab_size

        vlm = AutoModelForImageTextToText.from_config(config.vlm_config)
        if vlm.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"vlm.language_model.{k}" for k in vlm.language_model._tied_weights_keys]
        self.vlm = vlm

        self.embedding_dim = self.config.embedding_dim
        self.embedding_proj_layer = mint.nn.Linear(
            self.config.vlm_config.text_config.hidden_size,
            self.embedding_dim,
        )

        self.post_init()

    @add_start_docstrings_to_model_forward(COLPALI_FOR_RETRIEVAL_INPUT_DOCSTRING)
    @replace_return_docstrings(output_type=ColPaliForRetrievalOutput, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: ms.Tensor = None,
        pixel_values: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ColPaliForRetrievalOutput]:
        r"""
        Returns:

        Example:
        ```python
        >>> import requests
        >>> import mindspore as ms
        >>> from PIL import Image

        >>> from mindone.transformers import ColPaliForRetrieval, ColPaliProcessor


        >>> # Load the model and the processor
        >>> model_name = "vidore/colpali-v1.2-hf"

        >>> model = ColPaliForRetrieval.from_pretrained(
        ...     model_name,
        ...     mindspore_dtype=ms.bfloat16,
        ... )
        >>> processor = ColPaliProcessor.from_pretrained(model_name)

        >>> # The document page screenshots from your corpus
        >>> url1 = "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg"
        >>> url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg"

        >>> images = [
        ...     Image.open(requests.get(url1, stream=True).raw),
        ...     Image.open(requests.get(url2, stream=True).raw),
        ... ]

        >>> # The queries you want to retrieve documents for
        >>> queries = [
        ...     "When was the United States Declaration of Independence proclaimed?",
        ...     "Who printed the edition of Romeo and Juliet?",
        ... ]

        >>> # Process the inputs
        >>> inputs_images = processor(images=images, return_tensors="ms")
        >>> inputs_text = processor(text=queries, return_tensors="ms")
        >>> for k, v in inputs_images.items():
        ...     if v.dtype == ms.int64:
        ...         inputs_images[k] = v.to(ms.int32)
        ...     else:
        ...         inputs_images[k] = v.to(model.dtype)
        >>> for k, v in inputs_text.items():
        ...     if v.dtype == ms.int64:
        ...         inputs_text[k] = v.to(ms.int32)
        ...     else:
        ...         inputs_text[k] = v.to(model.dtype)

        >>> # Forward pass
        >>> image_embeddings = model(**inputs_images).embeddings
        >>> query_embeddings = model(**inputs_text).embeddings

        >>> # Score the queries against the images
        >>> scores = processor.score_retrieval(query_embeddings, image_embeddings)

        >>> print("Retrieval scores (query x image):")
        >>> print(scores)
        ```"""
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vlm_output = self.vlm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            output_attentions=output_attentions,
            **kwargs,
        )
        vlm_hidden_states = vlm_output.hidden_states if output_hidden_states else None
        vlm_image_hidden_states = vlm_output.image_hidden_states if pixel_values is not None else None

        last_hidden_states = vlm_output[0]  # (batch_size, sequence_length, hidden_size)
        embeddings = self.embedding_proj_layer(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return ColPaliForRetrievalOutput(
            embeddings=embeddings,
            past_key_values=vlm_output.past_key_values,
            hidden_states=vlm_hidden_states,
            attentions=vlm_output.attentions,
            image_hidden_states=vlm_image_hidden_states,
        )

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vlm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.vlm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.vlm.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.vlm.set_decoder(decoder)

    def get_decoder(self):
        return self.vlm.get_decoder()

    def tie_weights(self):
        return self.vlm.tie_weights()

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> mint.nn.Embedding:
        model_embeds = self.vlm.resize_token_embeddings(
            new_num_tokens=new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=mean_resizing,
        )

        self.config.vlm_config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vlm_config.vocab_size = model_embeds.num_embeddings
        self.vlm.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings

        return model_embeds


__all__ = [
    "ColPaliForRetrieval",
    "ColPaliForRetrievalOutput",
    "ColPaliPreTrainedModel",
]
