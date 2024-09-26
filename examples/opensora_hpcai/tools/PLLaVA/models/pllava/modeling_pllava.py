""" MindSpore Llava model."""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from functools import reduce
import mindspore as ms
import mindnlp
import mindnlp.core.nn as nn
import mindnlp.core.ops as ops
from mindnlp.transformers import PreTrainedModel
from mindnlp.transformers.activations import ACT2FN
from mindnlp.transformers.cache_utils import Cache
from mindnlp.transformers.modeling_outputs import ModelOutput
from mindnlp.transformers.models.auto import AutoModel, AutoModelForCausalLM
from mindnlp.transformers import logging
from .configuration_pllava import PllavaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlavaConfig"

PLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "",
    "",
    "",
    # See all Llava models at https://huggingface.co/models?filter=llava
]


@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->Llava
class PllavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(mindspore.Tensor)`, *optional*):
            Tuple of `mindspore.Tensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[ms.Tensor] = None
    logits: ms.Tensor = None
    past_key_values: Optional[List[ms.Tensor]] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None
    image_hidden_states: Optional[Tuple[ms.Tensor]] = None

class PllavaMultiModalProjector(nn.Module):
    supported_highres = ['pad_crop_four', 'slide', ]
    def __init__(self, config: PllavaConfig):
        super().__init__()  
        self.use_pooling = config.use_pooling
        self.frame_shape=config.frame_shape
        self.num_frames = config.num_frames
        self.pooling_shape = config.pooling_shape
        
        self.pooling = ms.nn.AdaptiveAvgPool3d(config.pooling_shape)
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def convert_Fembeddings2video(self, input, num_videos, frame_shape):
        num_videos_frames, _, embed_dims = input.shape
        num_frames = num_videos_frames // num_videos
        input = ops.reshape(input, (num_videos, num_frames * frame_shape[0] * frame_shape[1], embed_dims))
        input = ops.swapaxes(input, 1, 2)
        input = ops.reshape(input, (num_videos, embed_dims, num_frames, frame_shape[0], frame_shape[1]))
        return input

    def convert_video2Fembeddings(self, input):
        num_videos, embed_dims, num_frames, h, w = input.shape
        input = ops.reshape(input, (num_videos * num_frames, h * w, embed_dims))
        return input

    def convert_video2MMembeddings(self, input):
        num_videos, embed_dims, num_frames, h, w = input.shape
        input = ops.reshape(input, (num_videos, num_frames * h * w, embed_dims))
        return input

    def forward(self, image_features, media_type, batch_size=None, num_videos=None):
        frame_shape = self.frame_shape
        num_frames = self.num_frames
        assert media_type in ( 'video', 'image'), f'only image or video, but got media_type {media_type}'
        hidden_states = image_features

        if media_type == 'image':
            hidden_states = hidden_states.repeat(num_frames, 1, 1)

        total_frames, spatial_seqlen, embed_dims = hidden_states.shape
        #TODO: temporal code, should ensure num_frames == total frames in data loading later
        if total_frames < num_frames and self.use_pooling: # 
            multiplier = int(num_frames/total_frames)+1
            hidden_states= hidden_states.repeat_interleave(multiplier, dim=0)[:num_frames]
            total_frames, spatial_seqlen, embed_dims = hidden_states.shape

        assert total_frames % num_frames == 0
        assert frame_shape[0] * frame_shape[1] == spatial_seqlen
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states_videos = self.convert_Fembeddings2video(hidden_states, num_videos * batch_size, frame_shape)
        hidden_states_videos = self.pooling(hidden_states_videos)
        batch_size_num_videos, embed_dims, num_frames, h, w = hidden_states_videos.shape
        hidden_states = ops.reshape(hidden_states_videos, (batch_size_num_videos, embed_dims, num_frames * h * w))
        hidden_states = ops.swapaxes(hidden_states, 1, 2)
        return hidden_states



class PllavaPreTrainedModel(PreTrainedModel):
    """
        The bare LLaMA Model outputting raw hidden-states without any specific head on top.

        This model is also a MindSpore nn.Cell subclass. Use it as a regular MindSpore Cell
        and refer to the MindSpore documentation for all matter related to general usage and behavior.

        Parameters:
            config (PllavaConfig or LlavaVisionConfig):
                Model configuration class with all the parameters of the model. Initializing with a config file does not
                load the weights associated with the model, only the configuration.
        """
    config_class = PllavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, PllavaMultiModalProjector):
            # module.register_embed.data.normal_(mean=0.0, std=std)
            if self.config.register:
                module.register_embed.data.zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa

class PllavaForConditionalGeneration(PllavaPreTrainedModel):
    def __init__(self, config: PllavaConfig):
        super().__init__(config)
        self.config = config
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = PllavaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config, ms_dtype=config.ms_dtype, attn_implementation="flash_attention_2")
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.text_config.pad_token_id
        assert self.pad_token_id is not None, 'provide the model with pad_token_id, this would be used to arrange new embedings'
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

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not ops.sum(input_ids[:, -1] == ms.Tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = ops.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)).item() + sequence_length
        batch_indices, non_image_indices = ops.nonzero(input_ids != self.config.image_token_index, as_tuple=True)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = ops.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = ops.zeros(
            (batch_size, max_embed_dim, embed_dim), dtype=inputs_embeds.dtype
        )
        final_attention_mask = ops.zeros(
            (batch_size, max_embed_dim), dtype=attention_mask.dtype
        )
        if labels is not None:
            final_labels = ops.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype
            )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = ops.all(final_embedding == 0, dim=-1) # .astype(ms.int32)
        image_to_overwrite = (image_to_overwrite.int()
                              & (image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None]).int()).bool()

        if image_to_overwrite.sum() != reduce(lambda x, y: x*y, image_features.shape[:-1]):
            raise ValueError(
                f"The inputs provided to the model are wrong. The number of image tokens is "
                f"{ops.sum(special_image_token_mask)} while the number of image given to the model"
                f" is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim)
        final_attention_mask = (final_attention_mask.int() | image_to_overwrite.int()).bool()
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill((final_attention_mask == 0), 1)

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids


    def forward(
        self,
        input_ids: ms.Tensor = None,
        pixel_values: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        media_type: str = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PllavaCausalLMOutputWithPast]:
        r"""

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from mindnlp.transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="ms")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "\nUSER: What's the content of the image?\nASSISTANT: The image features a stop sign on a street corner"
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            no_img_input_ids = ops.where(input_ids!=self.config.image_token_index, input_ids, self.pad_token_id) # some model used up all the embeddings
            inputs_embeds = self.get_input_embeddings()(no_img_input_ids)
            batch_size = inputs_embeds.shape[0]
            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer] #  ( b, img_seqlen, embed_dim)
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    raise ValueError("not implemented")
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                num_videos = pixel_values.shape[0]//self.config.num_frames//batch_size
                # TODO: division by 0 check
                image_features = self.multi_modal_projector(selected_image_feature,
                                                            media_type,
                                                            batch_size=batch_size,
                                                            num_videos=num_videos,)
                    
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    # TODO: originally ms.int64 / long
                    labels = ops.full_like(attention_mask, self.config.ignore_index).to(ms.int32)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = ops.nonzero(first_layer_past_key_value.float().sum(-2) == 0, as_tuple = True)

                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = ops.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.shape[-1]
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    if new_batch_index.size > 0 and new_non_attended_tokens.size > 0:
                        extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = ops.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = ops.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        self.language_model.set_train(False)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

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
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PllavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None,
            attention_mask=None, media_type=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids[0]:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            # TODO: originally .long() (i.e., int64)
            position_ids = attention_mask.astype(ms.int32).cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "media_type": media_type,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
