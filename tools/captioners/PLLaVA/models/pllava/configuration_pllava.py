import json
import os

import mindspore as ms


class PllavaConfig:
    model_type = "llava"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        vocab_size=32000,
        pooling_method="avg",
        pooling_shape=(8, 16, 16),
        frame_shape=(24, 24),
        num_frames=1,
        use_pooling=True,
        gradient_checkpointing=False,
        pad_token_id=0,
        dtype=ms.bfloat16,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.vocab_size = vocab_size
        self.use_pooling = use_pooling
        self.gradient_checkpointing = gradient_checkpointing
        self.pad_token_id = pad_token_id
        self.dtype = dtype

        self.pooling_method = pooling_method
        self.pooling_shape = pooling_shape
        self.frame_shape = frame_shape
        self.num_frames = num_frames

        if vision_config is None:
            vision_config = {
                "model_type": "clip_vision_model",
                "intermediate_size": 4096,
                "hidden_size": 1024,
                "patch_size": 14,
                "image_size": 336,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "vocab_size": 32000,
                "projection_dim": 768,
            }
        self.vision_config = vision_config

        if text_config is None:
            text_config = {
                "model_type": "llama",
                "vocab_size": self.vocab_size,
                "gradient_checkpointing": self.gradient_checkpointing,
            }
        self.text_config = text_config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, vision_hidden_size=None, text_hidden_size=None, **kwargs):
        # load config.json from directory
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)  # override any arguments
        if vision_hidden_size is not None:
            config_dict["vision_config"]["hidden_size"] = vision_hidden_size
        if text_hidden_size is not None:
            config_dict["text_config"]["hidden_size"] = text_hidden_size

        return cls(**config_dict)
