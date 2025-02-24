""" MindSpore PLLaVA model."""
from functools import reduce
from typing import Dict, Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from ..activation import ACT2FN
from ..clip import CLIPVisionModel
from ..llama import LlamaForCausalLM
from ..padding import pad_along_axis
from .configuration_pllava import PllavaConfig


class PllavaMultiModalProjector(nn.Cell):
    def __init__(self, config: PllavaConfig):
        super().__init__()
        self.use_pooling = config.use_pooling
        self.frame_shape = config.frame_shape
        self.num_frames = config.num_frames
        self.pooling_shape = config.pooling_shape

        self.pooling = nn.AdaptiveAvgPool3d(config.pooling_shape)
        self.linear_1 = nn.Dense(
            config.vision_config["hidden_size"], config.text_config["hidden_size"], has_bias=True, dtype=config.dtype
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Dense(
            config.text_config["hidden_size"], config.text_config["hidden_size"], has_bias=True, dtype=config.dtype
        )

    def convert_Fembeddings2video(self, input, num_videos, frame_shape):
        num_videos_frames, _, embed_dims = input.shape
        num_frames = num_videos_frames // num_videos
        input = ops.reshape(input, (num_videos, num_frames * frame_shape[0] * frame_shape[1], embed_dims))
        input = ops.swapaxes(input, 1, 2)
        input = ops.reshape(input, (num_videos, embed_dims, num_frames, frame_shape[0], frame_shape[1]))
        return input

    def construct(self, image_features, batch_size=None, num_videos=None):
        frame_shape = self.frame_shape
        num_frames = self.num_frames
        hidden_states = image_features

        total_frames, spatial_seqlen, embed_dims = hidden_states.shape
        if total_frames < num_frames and self.use_pooling:
            multiplier = int(num_frames / total_frames) + 1
            hidden_states = hidden_states.repeat_interleave(multiplier, axis=0)[:num_frames]
            total_frames, spatial_seqlen, embed_dims = hidden_states.shape

        assert total_frames % num_frames == 0
        assert frame_shape[0] * frame_shape[1] == spatial_seqlen
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states_videos = self.convert_Fembeddings2video(hidden_states, num_videos * batch_size, frame_shape)
        hidden_states_videos = (self.pooling(hidden_states_videos.astype(ms.float32))).astype(ms.bfloat16)
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
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else self.config.text_config.pad_token_id
        )
        assert (
            self.pad_token_id is not None
        ), "provide the model with pad_token_id, this would be used to arrange new embedings"

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not ops.sum(input_ids[:, -1] == ms.Tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = (input_ids == self.config.image_token_index).astype(ms.int32)
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
        final_embedding = ops.zeros((batch_size, max_embed_dim, embed_dim), dtype=inputs_embeds.dtype)
        final_attention_mask = ops.zeros((batch_size, max_embed_dim), dtype=attention_mask.dtype)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = ops.all(final_embedding == 0, axis=-1)  # .astype(ms.int32)
        image_to_overwrite = (
            image_to_overwrite.int() & (image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None]).int()
        ).bool()

        if image_to_overwrite.sum() != reduce(lambda x, y: x * y, image_features.shape[:-1]):
            raise ValueError(
                f"The inputs provided to the model are wrong. The number of image tokens is "
                f"{ops.sum(special_image_token_mask)} while the number of image given to the model"
                f" is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim)
        final_attention_mask = (final_attention_mask.int() | image_to_overwrite.int()).bool()
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, position_ids

    def construct(
        self,
        input_ids: ms.Tensor = None,
        pixel_values: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_cache_list: Optional[ms.Tensor] = None,
        past_value_cache_list: Optional[ms.Tensor] = None,
        return_key_value_cache: bool = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[ms.Tensor]]:
        if input_ids is not None:
            # cast to type ms.int32
            input_ids = input_ids.astype(ms.int32)
            attention_mask = attention_mask.astype(ms.int32)
            # 1. extract the input embeddings
            no_img_input_ids = ops.where(input_ids != self.config.image_token_index, input_ids, self.pad_token_id)
            inputs_embeds = self.get_input_embeddings()(no_img_input_ids)
        else:
            raise ValueError("input_ids must be provided")

        batch_size = inputs_embeds.shape[0]

        # 2. merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1:
            # Obtain image features from vision_tower
            _, image_output_hidden_states = self.vision_tower(pixel_values)
            vision_feature_layer = self.config.vision_feature_layer
            vision_feature_select_strategy = self.config.vision_feature_select_strategy
            selected_image_feature = image_output_hidden_states[vision_feature_layer]  # (b, img_seqlen, embed_dim)

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            else:
                raise ValueError(f"Unexpected select feature strategy: {vision_feature_select_strategy}")

            num_videos = pixel_values.shape[0] // self.config.num_frames // batch_size
            image_features = self.multi_modal_projector(
                selected_image_feature, batch_size=batch_size, num_videos=num_videos
            )

            inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask
            )
        elif (
            past_key_cache_list is not None
            and past_value_cache_list is not None
            and pixel_values is not None
            and input_ids.shape[1] == 1
        ):
            first_layer_past_key_value = past_key_cache_list[0, :, :, :, 0]
            batch_index, non_attended_tokens = ops.nonzero(
                first_layer_past_key_value.float().sum(-2) == 0, as_tuple=True
            )

            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = ops.ones((attention_mask.shape[0], past_length), dtype=attention_mask.dtype)

            valid_indices = non_attended_tokens < extended_attention_mask.shape[-1]
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]

            if new_batch_index.size > 0 and new_non_attended_tokens.size > 0:
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            attention_mask = ops.concat((extended_attention_mask, attention_mask[:, -target_length:]), axis=1)
            position_ids = ops.sum(attention_mask, dim=1).unsqueeze(-1) - 1

            # by default, language model input method = padding
            attention_mask = pad_along_axis(attention_mask, axis=-1)
            past_key_cache_list = pad_along_axis(past_key_cache_list, axis=-2, shift=-1)
            past_value_cache_list = pad_along_axis(past_value_cache_list, axis=-2, shift=-1)

        self.language_model.set_train(False)
        logits, key_cache_list, value_cache_list = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_cache_list=past_key_cache_list,
            past_value_cache_list=past_value_cache_list,
            return_key_value_cache=return_key_value_cache,
        )

        return logits, key_cache_list, value_cache_list

    def prepare_inputs_for_generation(
        self,
        input_ids: ms.Tensor,
        pixel_values: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        past_key_cache_list: Optional[ms.Tensor] = None,
        past_value_cache_list: Optional[ms.Tensor] = None,
        return_key_value_cache: bool = False,
        **kwargs,
    ) -> Dict[str, Optional[ms.Tensor]]:
        if past_key_cache_list is not None and past_value_cache_list is not None:
            past_length = past_value_cache_list.shape[-2]

            # If attention_mask is longer than input_ids, reduce input_ids:
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                # If we have more input_ids than past_length, discard old tokens
                input_ids = input_ids[:, past_length:]
            elif self.config.image_token_index in input_ids[0]:
                # If image token is present, we only take the last token
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.astype(ms.int32).cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_cache_list is not None and past_value_cache_list is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if pixel_values is not None:
            pixel_values = pixel_values.astype(self.vision_tower.dtype)  # adjust dtype if required

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "past_key_cache_list": past_key_cache_list,
            "past_value_cache_list": past_value_cache_list,
            "return_key_value_cache": return_key_value_cache,
        }

        return model_inputs
