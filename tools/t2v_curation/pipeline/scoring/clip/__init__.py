"""clip init"""
import os

from .clip import CLIPModel
from .clip_config import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from .clip_processor import CLIPImageProcessor
from .clip_tokenizer import CLIPTokenizer
from .parse_yaml import parse

__all__ = ["CLIPModel", "CLIPConfig", "CLIPVisionConfig", "CLIPImageProcessor", "CLIPTextConfig", "CLIPTokenizer"]

root_dir = os.path.dirname(os.path.abspath(__file__))
support_list = {
    "clip_vit_l_14": os.path.join(root_dir, "configs/clip_vit_l_14.yaml"),
}
