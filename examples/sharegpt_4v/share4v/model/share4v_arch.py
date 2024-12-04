from abc import ABC, abstractmethod

from share4v.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector


class Share4VMetaModel:
    def __init__(self, config):
        rope_theta = config["rope_scaling"] if config.get("rope_scaling") else 1000000.0
        attention_dropout = config["attention_dropout"] if config.get("attention_dropout") else 0.0
        dtype = ms.float16 if config.get("dtype") == "float16" else ms.float32

        super(Share4VMetaModel, self).__init__(
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            num_attention_heads=config["num_attention_heads"],
            num_hidden_layers=config["num_hidden_layers"],
            num_key_value_heads=config["num_key_value_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=rope_theta,
            vocab_size=config["vocab_size"],
            attention_dropout=attention_dropout,
            hidden_act=config["hidden_act"],
            pad_token_id=config["pad_token_id"],
            dtype=dtype,
        )

        if config.get("mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        self.config = config

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_mm_projector(self):
        return getattr(self, "mm_projector", None)

    def set_mm_projector_dtype(self, dtype):
        mm_projector = self.get_mm_projector()
        if mm_projector is nn.SequentialCell:
            for cell in mm_projector:
                for param in cell.get_parameters():
                    param.set_dtype(dtype)
        else:
            for param in mm_projector.get_parameters():
                param.set_dtype(dtype)

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # self.config.mm_vision_tower = vision_tower
        self.config["mm_vision_tower"] = vision_tower
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        elif self.get_vision_tower().vision_tower_name != vision_tower:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
                vision_tower.load_model()
            else:
                vision_tower = self.vision_tower
                vision_tower.load_model()

        self.config["use_mm_proj"] = True
        self.config["mm_projector_type"] = getattr(model_args, "mm_projector_type", "linear")
        self.config["mm_hidden_size"] = vision_tower.hidden_size
        self.config["mm_vision_select_layer"] = mm_vision_select_layer
        self.config["mm_vision_select_feature"] = mm_vision_select_feature

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.get_parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            print(f"Load mm_mlp_adapter from {pretrain_mm_mlp_adapter}")
            mm_projector_weights = ms.load_checkpoint(pretrain_mm_mlp_adapter)

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))


class Share4VMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_cache_list, past_value_cache_list, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_cache_list is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = ops.ones(
                    (attention_mask.shape[0], past_value_cache_list[-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                )
            return input_ids, attention_mask, past_key_cache_list, past_value_cache_list, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = ops.cat([image for image in images], axis=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = ops.split(image_features, split_sizes, axis=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = ops.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], axis=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = ops.nonzero(ops.where(cur_input_ids == IMAGE_TOKEN_INDEX, 1, 0)).squeeze(1)
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if self.config.get("tune_mm_mlp_adapter", False) and self.config.get("mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[: image_token_start - 1]).detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start - 1 : image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start + 1 : image_token_start + 2])
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            ops.full((cur_image_features.shape[0],), IGNORE_INDEX, dtype=labels.dtype)
                        )
                        cur_new_labels.append(cur_labels[image_token_start + 1 : image_token_start + 2])
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            ops.full((cur_image_features.shape[0],), IGNORE_INDEX, dtype=labels.dtype)
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                if self.config.get("tune_mm_mlp_adapter", False) and self.config.get("mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = ops.nonzero(ops.where(cur_input_ids == IMAGE_TOKEN_INDEX, 1, 0)).squeeze(1)
            if cur_input_ids.numel() > 0:
                if self.config.get("tune_mm_mlp_adapter", False) and self.config.get("mm_use_im_start_end", False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = ops.cat(cur_new_input_embeds, axis=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = ops.cat(cur_new_labels, axis=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = ops.cat(
                    (
                        cur_new_embed,
                        ops.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype
                        ),
                    ),
                    axis=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = ops.stack(new_input_embeds_align, axis=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = ops.cat(
                        (
                            cur_new_label,
                            ops.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype),
                        ),
                        axis=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = ops.stack(new_labels_align, axis=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = ops.full(
                        (cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype
                    )
                    new_attn_mask_pad_right = ops.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype
                    )
                    cur_new_attention_mask = ops.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), axis=0
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = ops.stack(new_attention_mask, axis=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = ops.stack(new_input_embeds, axis=0)
            if labels is not None:
                new_labels = ops.stack(new_labels, axis=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = ops.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]),
                    True,
                    dtype=attention_mask.dtype,
                )
                attention_mask = ops.cat((new_attn_mask_pad_left, attention_mask), axis=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_cache_list, past_value_cache_list, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(axis=0, keep_dims=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(axis=0, keep_dims=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().get_parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().get_parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = ms.load_checkpoint(model_args.pretrain_mm_mlp_adapter)
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: \
                            {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().get_parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().get_parameters():
                    p.requires_grad = False
