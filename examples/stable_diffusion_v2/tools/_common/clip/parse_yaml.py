"""
Parse config file
"""
import yaml

from ..clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


def parse(path_to_config, path_to_ckpt):
    with open(path_to_config) as f:
        configf = yaml.safe_load(f)
    vconfig = CLIPVisionConfig(**configf["vision_config"])
    tconfig = CLIPTextConfig(**configf["text_config"])
    config = CLIPConfig(
        tconfig, vconfig, configf["projection_dim"], configf["logit_scale_init_value"], path_to_ckpt, configf["dtype"]
    )
    return config
