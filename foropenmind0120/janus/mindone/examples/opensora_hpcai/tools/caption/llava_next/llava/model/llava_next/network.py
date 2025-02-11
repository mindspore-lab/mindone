import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor

from ..activation import ACT2FN
from ..clip import CLIPVisionModel
from ..mistral import MistralForCausalLM
from ..padding import pad_along_axis
from .utils import get_anyres_image_grid_shape, image_size_to_num_patches, unpad_image


class LlavaNextMultiModalProjector(nn.Cell):
    def __init__(
        self,
        vision_hidden_size: int = 1024,
        text_hidden_size: int = 4096,
        projector_hidden_act: str = "gelu",
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()

        self.linear_1 = nn.Dense(vision_hidden_size, text_hidden_size, has_bias=True, dtype=dtype)
        self.act = ACT2FN[projector_hidden_act]
        self.linear_2 = nn.Dense(text_hidden_size, text_hidden_size, has_bias=True, dtype=dtype)

    def construct(self, image_features: Tensor) -> Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaNextForConditionalGeneration(nn.Cell):
    def __init__(
        self,
        vision_config: Dict[str, Any],
        text_config: Dict[str, Any],
        image_grid_pinpoints: List[Tuple[int, int]],
        projector_hidden_act: str = "gelu",
        ignore_index: int = -100,
        image_token_index: int = 32000,
        vision_feature_select_strategy: str = "default",
        vision_feature_layer: int = -2,
        attn_implementation: Literal["eager", "flash_attention"] = "eager",
        language_model_input_method: Literal["padding", "dynamic"] = "padding",
        dtype: ms.dtype = ms.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.dtype = dtype

        self.vision_tower = CLIPVisionModel(**vision_config, dtype=dtype)
        self.multi_modal_projector = LlavaNextMultiModalProjector(
            vision_config["hidden_size"],
            text_config["hidden_size"],
            projector_hidden_act=projector_hidden_act,
            dtype=dtype,
        )
        embed_std = 1 / math.sqrt(text_config["hidden_size"])
        self.image_newline = Parameter(Tensor(ops.randn(text_config["hidden_size"]) * embed_std, dtype=dtype))

        self.vocab_size = text_config["vocab_size"]
        self.language_model = MistralForCausalLM(**text_config, attn_implementation=attn_implementation, dtype=dtype)

        self.text_config = text_config
        self.vision_config = vision_config
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.image_grid_pinpoints = image_grid_pinpoints
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.pad_token_id = -1
        self._padding_side = "left"

        self.language_model_input_method = language_model_input_method
        if self.language_model_input_method == "dynamic":
            self._is_language_model_compiled = False

    def get_input_embeddings(self) -> nn.Cell:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Cell) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Cell:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Cell) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder: nn.Cell) -> None:
        self.language_model.set_decoder(decoder)

    def get_decoder(self) -> nn.Cell:
        return self.language_model.get_decoder()

    def _merge_input_ids_with_image_features(
        self,
        image_features: Tensor,
        feature_lens: Tensor,
        inputs_embeds: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # ! in llava 1.6, number of patches is variable
        num_images = feature_lens.shape[0]
        num_image_features, embed_dim = image_features.shape
        if feature_lens.sum() != num_image_features:
            raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
        batch_size = input_ids.shape[0]
        _left_padding = ops.any(attention_mask[:, 0] == 0)
        _right_padding = ops.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                # both side is 1, so cannot tell
                left_padding = self._padding_side == "left"
            else:
                # invalid attention_mask
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        # Whether to turn off right padding
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.image_token_index
        # special_image_token_mask: [bsz, seqlen]
        num_special_image_tokens = ops.sum(special_image_token_mask, dim=-1)
        # num_special_image_tokens: [bsz]
        # Reserve for padding of num_images
        total_num_special_image_tokens = ops.sum(special_image_token_mask)
        if total_num_special_image_tokens != num_images:
            raise ValueError(
                f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."
            )
        # Compute the maximum embed dimension
        # max_image_feature_lens is max_feature_lens per batch
        feature_lens_batch = ops.split(feature_lens, num_special_image_tokens.tolist(), axis=0)
        feature_lens_batch_sum = ops.stack([x.sum() for x in feature_lens_batch])
        embed_sequence_lengths = (
            (attention_mask == 1).to(ms.int32).sum(-1) - num_special_image_tokens + feature_lens_batch_sum
        )
        max_embed_dim = embed_sequence_lengths.max().item()

        batch_indices, non_image_indices = ops.nonzero(
            ops.logical_and(input_ids != self.image_token_index, attention_mask == 1)
        ).unbind(dim=1)
        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        # ! instead of special_image_token_mask * (num_image_patches - 1)
        #   special_image_token_mask * (num_feature_len - 1)
        special_image_token_mask = special_image_token_mask.to(ms.int32)
        special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
        new_token_positions = ops.cumsum((special_image_token_mask + 1), -1) - 1
        if left_padding:
            # shift right token positions so that they are ending at the same number
            # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
            new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = ops.zeros((batch_size, max_embed_dim, embed_dim), dtype=inputs_embeds.dtype)
        final_attention_mask = ops.zeros((batch_size, max_embed_dim), dtype=attention_mask.dtype)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = ops.full((batch_size, max_embed_dim), True, dtype=ms.bool_)
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        embed_indices = ops.arange(max_embed_dim).unsqueeze(0)
        embed_indices = ops.broadcast_to(embed_indices, (batch_size, max_embed_dim))
        embed_seq_lens = embed_sequence_lengths[:, None]

        if left_padding:
            # exclude padding on the left
            val = (max_embed_dim - embed_indices) <= embed_seq_lens
        else:
            # exclude padding on the right
            val = embed_indices < embed_seq_lens
        image_to_overwrite = ops.logical_and(image_to_overwrite, val)

        if image_to_overwrite.sum() != num_image_features:
            raise ValueError(
                f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                f"The number of image tokens is {ops.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. "
                f"This prevents correct indexing and breaks batch generation."
            )
        final_embedding[image_to_overwrite] = image_features.reshape(-1, embed_dim)
        final_attention_mask |= image_to_overwrite
        position_ids = ops.masked_fill(
            ops.cumsum(final_attention_mask.to(ms.int32), -1) - 1, final_attention_mask == 0, Tensor(1, dtype=ms.int32)
        )

        return final_embedding, final_attention_mask, position_ids

    def pack_image_features(
        self, image_features: Tensor, image_sizes: Tensor, image_newline: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.vision_config["image_size"] // self.vision_config["patch_size"]
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.image_grid_pinpoints,
                    self.vision_config["image_size"],
                )
                image_feature = ops.reshape(image_feature, (num_patch_height, num_patch_width, height, width, -1))
                image_feature = ops.transpose(image_feature, (4, 0, 2, 1, 3))
                image_feature = ops.flatten(image_feature, start_dim=1, end_dim=2)
                image_feature = ops.flatten(image_feature, start_dim=2, end_dim=3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = ops.concat(
                        (
                            image_feature,
                            ops.broadcast_to(image_newline[:, None, None], (*image_feature.shape[:-1], 1)).to(
                                image_feature.dtype
                            ),
                        ),
                        axis=-1,
                    )
                image_feature = ops.flatten(image_feature, start_dim=1, end_dim=2)
                image_feature = ops.transpose(image_feature, (1, 0))
                image_feature = ops.concat((base_image_feature, image_feature), axis=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = ops.concat((image_feature, image_newline[None].to(image_feature)), axis=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.shape[0])
        image_features = ops.concat(new_image_features, axis=0)
        feature_lens = Tensor(feature_lens, dtype=ms.int32)
        return image_features, feature_lens

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        image_sizes: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_cache_list: Optional[Tensor] = None,
        past_value_cache_list: Optional[Tensor] = None,
        return_key_value_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # 1. Extract the input embeddings
        # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
        for_inputs_embeds_ids = input_ids.copy()
        for_inputs_embeds_ids[input_ids == self.image_token_index] = 0
        inputs_embeds = self.get_input_embeddings()(for_inputs_embeds_ids)

        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size > 0:
            # ! infer image_num_patches from image_sizes
            image_num_patches = [
                image_size_to_num_patches(
                    image_size=imsize,
                    grid_pinpoints=self.image_grid_pinpoints,
                    patch_size=self.vision_config["image_size"],
                )
                for imsize in image_sizes
            ]
            # figure out if pixel_values is concatenated or stacked
            if len(pixel_values.shape) == 5:
                # stacking when input is (batch_size, num_patches, num_channels, height, width)
                _pixel_values_list = [
                    pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
                ]
                pixel_values = ops.concat(_pixel_values_list, axis=0)
            elif len(pixel_values.shape) != 4:
                # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

            _, hidden_states = self.vision_tower(pixel_values)
            selected_image_feature = hidden_states[self.vision_feature_layer]

            if self.vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature

            image_features = self.multi_modal_projector(selected_image_feature)
            image_features = ops.split(image_features, image_num_patches, axis=0)
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                image_newline=self.image_newline,
            )

            inputs_embeds = inputs_embeds.to(image_features.dtype)
            inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
                image_features,
                feature_lens,
                inputs_embeds,
                input_ids,
                attention_mask,
            )

        elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size == 0:
            # there are no images
            pass

        elif (
            past_key_cache_list is not None
            and past_value_cache_list is not None
            and pixel_values is not None
            and input_ids.shape[1] == 1
        ):
            # Retrieve the first layer to inspect the logits and mask out the hidden states
            # that are set to 0
            first_layer_past_key_value = past_key_cache_list[0, :, :, :, 0]

            # Sum all dimensions of head_dim (-2) to avoid random errors such as:
            # https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            batch_index, non_attended_tokens = ops.nonzero(
                first_layer_past_key_value.to(ms.float32).sum(-2) == 0
            ).unbind(dim=1)

            # Get the target length
            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = ops.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
            )

            # Filter out only the tokens that can be un-attended, this can happen
            # if one uses Llava + Fused modules where the cache on the
            # first iteration is already big enough, or if one passes custom cache
            valid_indices = non_attended_tokens < extended_attention_mask.shape[-1]
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]

            # Zero-out the places where we don't need to attend
            if len(new_batch_index) > 0 and len(new_non_attended_tokens) > 0:
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            attention_mask = ops.concat((extended_attention_mask, attention_mask[:, -target_length:]), axis=1)

            position_ids = ops.sum(attention_mask, dim=1).unsqueeze(-1) - 1

            if self.language_model_input_method == "padding":
                attention_mask = pad_along_axis(attention_mask, axis=-1)
                past_key_cache_list = pad_along_axis(past_key_cache_list, axis=-2, shift=-1)
                past_value_cache_list = pad_along_axis(past_value_cache_list, axis=-2, shift=-1)

        if self.language_model_input_method == "dynamic" and not self._is_language_model_compiled:
            if past_key_cache_list is not None or past_value_cache_list is not None:
                raise ValueError("Dynamic shape compling is not supported with KV caching yet.")
            attention_mask_shape = list(attention_mask.shape)
            position_ids_shape = list(position_ids.shape)
            inputs_embeds_shape = list(inputs_embeds.shape)

            attention_mask_shape[-1] = None
            position_ids_shape[-1] = None
            inputs_embeds_shape[-2] = None

            self.language_model.set_inputs(
                attention_mask=Tensor(shape=attention_mask_shape, dtype=attention_mask.dtype),
                position_ids=Tensor(shape=position_ids_shape, dtype=position_ids.dtype),
                inputs_embeds=Tensor(shape=inputs_embeds_shape, dtype=inputs_embeds.dtype),
            )
            self._is_language_model_compiled = True

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
        input_ids: Tensor,
        pixel_values: Optional[Tensor] = None,
        image_sizes: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_cache_list: Optional[Tensor] = None,
        past_value_cache_list: Optional[Tensor] = None,
        return_key_value_cache: bool = False,
        **kwargs,
    ) -> Dict[str, Optional[Tensor]]:
        if past_key_cache_list is not None and past_value_cache_list is not None:
            past_length = past_value_cache_list.shape[-2]

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
            elif self.image_token_index in input_ids.asnumpy():
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = ops.cumsum(attention_mask.to(ms.int32), -1) - 1
            position_ids = ops.masked_fill(position_ids, attention_mask == 0, Tensor(1, dtype=ms.int32))
            if past_key_cache_list is not None and past_value_cache_list is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        model_inputs = dict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values.to(self.dtype),
                "image_sizes": image_sizes,
                "past_key_cache_list": past_key_cache_list,
                "past_value_cache_list": past_value_cache_list,
                "return_key_value_cache": return_key_value_cache,
            }
        )
        return model_inputs
