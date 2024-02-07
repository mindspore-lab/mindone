import re
from typing import Dict, List, Union

import torch

import mindspore as ms


def append_prefix(content: str, prefix: str) -> str:
    content = prefix + content
    return content


def replace(content: str, old: str, new: str) -> str:
    content = content.replace(old, new)
    return content


def replace_all(content: str) -> str:
    content = replace(content, "vision_model.encoder.layers", "image_encoder.model.visual.transformer.resblocks")
    content = replace(content, "self_attn", "attn")
    content = replace(content, "layer_norm1.weight", "ln_1.gamma")
    content = replace(content, "layer_norm1.bias", "ln_1.beta")
    content = replace(content, "layer_norm2.weight", "ln_2.gamma")
    content = replace(content, "layer_norm2.bias", "ln_2.beta")
    content = replace(content, "fc1", "c_fc")
    content = replace(content, "fc2", "c_proj")
    content = replace(content, "vision_model.embeddings", "image_encoder.model.visual")
    content = replace(content, "patch_embedding", "conv1")
    content = replace(content, "vision_model.pre_layrnorm.weight", "image_encoder.model.visual.ln_pre.gamma")
    content = replace(content, "vision_model.pre_layrnorm.bias", "image_encoder.model.visual.ln_pre.beta")
    content = replace(content, "vision_model.post_layernorm.weight", "image_encoder.model.visual.ln_post.gamma")
    content = replace(content, "vision_model.post_layernorm.bias", "image_encoder.model.visual.ln_post.beta")
    content = replace(content, "position_embedding.weight", "positional_embedding")
    content = replace(content, "visual_projection.weight", "image_encoder.model.visual.proj")
    content = replace(content, "image_proj.norm.weight", "image_proj.norm.gamma")
    content = replace(content, "image_proj.norm.bias", "image_proj.norm.beta")
    content = re.sub(r"(norm[\w]*).weight", r"\1.gamma", content)
    content = re.sub(r"(norm[\w]*).bias", r"\1.beta", content)
    content = re.sub(r"([0-9].[0-9].0).weight", r"\1.gamma", content)
    content = re.sub(r"([0-9].[0-9].0).bias", r"\1.beta", content)
    return content


def create_ip_mapping(ip_mapping: str) -> Dict[str, str]:
    with open(ip_mapping, "r") as f:
        lines = f.readlines()
        ms_names = [x.strip() for x in lines]

    # hard coded num, inspected from torch checkpoint
    nums = range(1, 141, 2)
    torch_names = list()
    for x in nums:
        torch_names.append(f"ip_adapter.{x}.to_k_ip.weight")
        torch_names.append(f"ip_adapter.{x}.to_v_ip.weight")

    mapping = dict(zip(torch_names, ms_names))
    return mapping


def convert_to_ms(tensors: Dict[str, torch.Tensor]) -> List[Dict[str, Union[str, ms.Tensor]]]:
    records = list()
    for k, v in tensors.items():
        record = {"name": k, "data": ms.Tensor(v.numpy(), ms.float32)}
        records.append(record)
    return records
