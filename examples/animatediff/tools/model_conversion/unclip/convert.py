# convert the weight from Pytorch to Mindspore
import argparse
import re
from typing import Optional

import numpy as np
import torch

import mindspore as ms


def valid_name(name: str):
    if "model" not in name:
        return False

    skip_names = [
        "model_ema",
        "logit_scale",
        "embedder.model.transformer",
        "embedder.model.token_embedding",
        "embedder.model.ln_final",
        "embedder.model.positional_embedding",
        "embedder.model.text_projection",
        "cond_stage_model.model.text_projection",
        "cond_stage_model.model.transformer.resblocks.23",
    ]

    if any([x in name for x in skip_names]):
        return False

    return True


def convert_to_fp32(tensor: np.ndarray):
    if tensor.dtype == np.float16:
        return tensor.astype(np.float32)
    return tensor


def change_name(name: str) -> str:
    # GroupNorm
    name = _change_with_pattern(name, r"(norm[0-9]*).weight", r"\1.gamma")
    name = _change_with_pattern(name, r"(norm[0-9]*).bias", r"\1.beta")
    name = _change_with_pattern(name, r"(in_layers.0).weight", r"\1.gamma")
    name = _change_with_pattern(name, r"(in_layers.0).bias", r"\1.beta")
    name = _change_with_pattern(name, r"(out_layers.0).weight", r"\1.gamma")
    name = _change_with_pattern(name, r"(out_layers.0).bias", r"\1.beta")
    name = _change_with_pattern(name, r"(model.diffusion_model.out.0).weight", r"\1.gamma")
    name = _change_with_pattern(name, r"(model.diffusion_model.out.0).bias", r"\1.beta")
    name = _change_with_pattern(name, r"(norm_out).weight", r"\1.gamma")
    name = _change_with_pattern(name, r"(norm_out).bias", r"\1.beta")
    name = _change_with_pattern(name, r"(ln_[0-9]*).weight", r"\1.gamma")
    name = _change_with_pattern(name, r"(ln_[0-9]*).bias", r"\1.beta")
    name = _change_with_pattern(name, r"(ln_[a-z]*).weight", r"\1.gamma")
    name = _change_with_pattern(name, r"(ln_[a-z]*).bias", r"\1.beta")

    # mindone/stablediffusion unet compnent naming diff
    name = _change_with_pattern(name, r"in_proj_weight", r"in_proj.weight")
    name = _change_with_pattern(name, r"in_proj_bias", r"in_proj.bias")
    name = _change_with_pattern(name, r"(token_embedding).weight", r"\1.embedding_table")
    name = _change_with_pattern(name, r"(model.diffusion_model.input_blocks.0.0).weight", r"\1.conv.weight")
    name = _change_with_pattern(name, r"(model.diffusion_model.input_blocks.0.0).bias", r"\1.conv.bias")
    name = _change_with_pattern(
        name, r"(model.diffusion_model.output_blocks.[0-9]*.[0-9]*).conv.weight", r"\1.conv.conv.weight"
    )
    name = _change_with_pattern(
        name, r"(model.diffusion_model.output_blocks.[0-9]*.[0-9]*).conv.bias", r"\1.conv.conv.bias"
    )
    name = _change_with_pattern(name, r"(model.diffusion_model.out.2).weight", r"\1.conv.weight")
    name = _change_with_pattern(name, r"(model.diffusion_model.out.2).bias", r"\1.conv.bias")
    name = _change_with_pattern(name, r"(model.diffusion_model.[\w.]*).op.weight", r"\1.op.conv.weight")
    name = _change_with_pattern(name, r"(model.diffusion_model.[\w.]*).op.bias", r"\1.op.conv.bias")
    name = _change_with_pattern(name, r"(model.diffusion_model.[\w.]*).in_layers.0.gamma", r"\1.in_layers_norm.gamma")
    name = _change_with_pattern(name, r"(model.diffusion_model.[\w.]*).in_layers.0.beta", r"\1.in_layers_norm.beta")
    name = _change_with_pattern(
        name, r"(model.diffusion_model.[\w.]*).in_layers.2.weight", r"\1.in_layers_conv.conv.weight"
    )
    name = _change_with_pattern(
        name, r"(model.diffusion_model.[\w.]*).in_layers.2.bias", r"\1.in_layers_conv.conv.bias"
    )
    name = _change_with_pattern(name, r"(model.diffusion_model.[\w.]*).out_layers.0.gamma", r"\1.out_layers_norm.gamma")
    name = _change_with_pattern(name, r"(model.diffusion_model.[\w.]*).out_layers.0.beta", r"\1.out_layers_norm.beta")
    name = _change_with_pattern(
        name, r"(model.diffusion_model.[\w.]*).out_layers.3.weight", r"\1.out_layers_conv.conv.weight"
    )
    name = _change_with_pattern(
        name, r"(model.diffusion_model.[\w.]*).out_layers.3.bias", r"\1.out_layers_conv.conv.bias"
    )
    name = _change_with_pattern(
        name, r"(model.diffusion_model.[\w.]*).skip_connection.weight", r"\1.skip_connection.conv.weight"
    )
    name = _change_with_pattern(
        name, r"(model.diffusion_model.[\w.]*).skip_connection.bias", r"\1.skip_connection.conv.bias"
    )

    return name


def torch2ms(path: str, out_path: Optional[str] = None) -> None:
    par_dict = torch.load(path)["state_dict"]
    params_list = list()
    for name, parameter in par_dict.items():
        if not valid_name(name):
            continue
        param_dict = dict()
        param_dict["name"] = change_name(name)
        param_dict["data"] = ms.Tensor(convert_to_fp32(parameter.numpy()))

        params_list.append(param_dict)

    if out_path is None:
        out_path = path.replace(".ckpt", "_ms.ckpt")

    ms.save_checkpoint(params_list, out_path)


def _change_with_pattern(name: str, old: str, new: str) -> str:
    new_name = re.sub(old, new, name)
    if new_name != name:
        print(f"{name} -> {new_name}")
    return new_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting Unclip checkpoint from Pytorch to Mindspore")
    parser.add_argument(
        "path", help="Path of the stablediffusion checkpoint, support `sd21-unclip-l.ckpt` and `sd21-unclip-h.ckpt`"
    )
    parser.add_argument("-o", "--out", help="Output path of the converted checkpoint.")
    args = parser.parse_args()

    torch2ms(args.path, args.out)
