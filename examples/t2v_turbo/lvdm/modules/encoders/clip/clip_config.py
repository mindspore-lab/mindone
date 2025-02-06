"""
CLIPConfig class, which consists of CLIPTextConfig and CLIPVisionConfig
"""
from typing import Optional


class CLIPTextConfig:
    def __init__(
        self,
        vocab_size: Optional[int] = 49408,
        hidden_size: Optional[int] = 512,
        intermediate_size: Optional[int] = 2048,
        num_hidden_layers: Optional[int] = 12,
        num_attention_heads: Optional[int] = 8,
        max_position_embeddings: Optional[int] = 77,
        hidden_act: Optional[str] = "quick_gelu",
        initializer_range: Optional[float] = 0.02,
        initializer_factor: Optional[float] = 1.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor


class CLIPVisionConfig:
    def __init__(
        self,
        hidden_size: Optional[int] = 768,
        intermediate_size: Optional[int] = 3072,
        num_hidden_layers: Optional[int] = 12,
        num_attention_heads: Optional[int] = 12,
        image_size: Optional[int] = 224,
        patch_size: Optional[int] = 32,
        hidden_act: Optional[str] = "quick_gelu",
        initializer_range: Optional[float] = 0.02,
        initializer_factor: Optional[float] = 1.0,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor


class CLIPConfig:
    def __init__(
        self,
        text_config: Optional[CLIPTextConfig] = None,
        vision_config: Optional[CLIPVisionConfig] = None,
        projection_dim: Optional[int] = 512,
        logit_scale_init_value: Optional[float] = 2.6592,
        checkpoint_name_or_path: Optional[str] = "clip_vit_b_32",
        dtype: Optional[str] = "float16",
    ):
        if text_config is None:
            text_config = CLIPTextConfig()
            print("text_config is None. Initializing the CLIPTextConfig with default values.")
        elif isinstance(text_config, CLIPTextConfig):
            pass
        else:
            raise TypeError(f"text_config should be a " f"CLIpTextConfig class, but got {type(CLIPTextConfig)}")

        if vision_config is None:
            vision_config = CLIPVisionConfig()
            print("vision_config is None." " Initializing the CLIPTextConfig with default values.")
        elif isinstance(vision_config, CLIPVisionConfig):
            pass
        else:
            raise TypeError("text_config should be a CLIPVisionConfig" f" class, but got {type(CLIPVisionConfig)}")

        self.text_config = text_config
        self.vision_config = vision_config
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.dtype = dtype
