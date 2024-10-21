# Reference to https://github.com/mlfoundations/open_clip

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional

import mindspore as ms

from .model import CLIP

HF_HUB_PREFIX = "hf-hub:"
_MODEL_CONFIG_PATHS = [Path(__file__).parent / "model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            # if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
            if all(a in model_cfg for a in ("embed_dim", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def load_checkpoint(network, weight):
    if weight.endswith(".ckpt"):
        param_dict = ms.load_checkpoint(weight)
        ms.load_param_into_net(network, param_dict)
        logging.info(f'Checkpoint load from "{weight}" success.')
    else:
        raise ValueError("Not support weight format.")


def create_model(
    model_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    jit: bool = False,
    cache_dir: Optional[str] = None,
):
    model_name = model_name.replace("/", "-")  # for callers using old naming with / in ViT names
    pretrained_cfg = {}
    model_cfg = get_model_config(model_name)

    if model_cfg is not None:
        logging.info(f"Loaded {model_name} model config.")
    else:
        logging.error(f"Model config for {model_name} not found. available models {list_models()}.")
        raise RuntimeError(f"Model config for {model_name} not found.")

    # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
    cast_dtype = ms.float32 if precision == "fp32" else ms.float16
    custom_text = model_cfg.pop("custom_text", False)
    assert custom_text is False
    model = CLIP(**model_cfg, cast_dtype=cast_dtype)

    if precision in ("fp16", "bf16", "pure_fp16", "pure_bf16"):
        # manual mixed precision that matches original OpenAI behaviour
        model.to_float(ms.float16)

    if pretrained:
        assert pretrained.endswith(".ckpt"), f"pretrained expect '*.ckpt', but got '{pretrained}'."
        load_checkpoint(model, pretrained)

    if model.visual is not None:
        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = pretrained_cfg.get("mean", None) or OPENAI_DATASET_MEAN
        model.visual.image_std = pretrained_cfg.get("std", None) or OPENAI_DATASET_STD

    @ms.jit
    def jit_func(*args, **kwargs):
        return model(*args, **kwargs)

    return model if not jit else jit_func


if __name__ == "__main__":
    import argparse
    import ast

    from sgm.modules.embedders.open_clip.tokenizer import tokenize

    parser_config = argparse.ArgumentParser(description="Config", add_help=False)
    parser_config.add_argument("--ms_jit", type=ast.literal_eval, default=False)
    args, _ = parser_config.parse_known_args()

    model = create_model(model_name="ViT-H-14-Text", pretrained="")  # "laion2b_s32b_b79k"

    @ms.jit
    def jit_warpper(token):
        return model.token_embedding(token)

    token = tokenize(["a photo of a cat", "a photo of a dog"])
    if not args.ms_jit:
        out = model.token_embedding(token)
    else:
        out = jit_warpper(token)

    print(f"token.shape: {token.shape}")
    print(f"out.shape: {out.shape}")
