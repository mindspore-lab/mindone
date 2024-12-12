""" MindSpore PLLaVA model."""
from typing import List, Optional, Tuple
from functools import reduce
import mindspore as ms

import mindspore.nn as nn
import mindspore.ops as ops

from ..activation import ACT2FN
from ..cache_utils import Cache
from ..clip import CLIPVisionModel
from ..llama import LlamaForCausalLM

from transformers.cache_utils import Cache

from .configuration_pllava import PllavaConfig

class PllavaMultiModalProjector(nn.Cell):
    def __init__(self, config: PllavaConfig):
        super().__init__() #TODO: modify this
        self.use_pooling = config.use_pooling
        self.frame_shape=config.frame_shape
        self.num_frames = config.num_frames
        self.pooling_shape = config.pooling_shape
        
        self.pooling = ms.nn.AdaptiveAvgPool3d(config.pooling_shape)
        self.linear_1 = nn.Dense(config.vision_config.hidden_size, config.text_config.hidden_size, has_bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, has_bias=True)

    def convert_Fembeddings2video(self, input, num_videos, frame_shape):
        num_videos_frames, _, embed_dims = input.shape
        num_frames = num_videos_frames // num_videos
        input = ops.reshape(input, (num_videos, num_frames * frame_shape[0] * frame_shape[1], embed_dims))
        input = ops.swapaxes(input, 1, 2)
        input = ops.reshape(input, (num_videos, embed_dims, num_frames, frame_shape[0], frame_shape[1]))
        return input

    def construct(self, image_features, media_type, batch_size=None, num_videos=None):
        frame_shape = self.frame_shape
        num_frames = self.num_frames
        assert media_type in ('video'), f'only video, but got media_type {media_type}'
        hidden_states = image_features

        total_frames, spatial_seqlen, embed_dims = hidden_states.shape
        if total_frames < num_frames and self.use_pooling:
            multiplier = int(num_frames/total_frames)+1
            hidden_states= hidden_states.repeat_interleave(multiplier, axis=0)[:num_frames]
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

class PllavaForConditionalGeneration(nn.Cell):
    def __init__(self, config: PllavaConfig):
        super().__init__(config)
        self.config = config
        self.vision_tower = CLIPVisionModel(**config.vision_config)
        self.multi_modal_projector = PllavaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        self.language_model = LlamaForCausalLM(**config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.text_config.pad_token_id
        assert self.pad_token_id is not None, 'provide the model with pad_token_id, this would be used to arrange new embedings'

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

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
        image_to_overwrite = ops.all(final_embedding == 0, axis=-1) # .astype(ms.int32)
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


    def construct(
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
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[ms.Tensor]]:

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
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer] #  ( b, img_seqlen, embed_dim)
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                num_videos = pixel_values.shape[0]//self.config.num_frames//batch_size

                image_features = self.multi_modal_projector(selected_image_feature,
                                                            media_type,
                                                            batch_size=batch_size,
                                                            num_videos=num_videos,)
                    
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
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

                    attention_mask = ops.cat((attention_mask, extended_attention_mask), axis=1)
                    position_ids = ops.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        self.language_model.set_train(False)
        logits, key_cache_list, value_cache_list = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return logits, key_cache_list, value_cache_list

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
