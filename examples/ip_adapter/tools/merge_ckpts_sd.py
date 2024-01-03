#!/usr/bin/env python
import argparse
import os

import torch
from safetensors import safe_open
from util import append_prefix, convert_to_ms, create_ip_mapping, merge_qkv, misc_ops, replace, replace_all

import mindspore as ms


def replace_all_vae(content: str) -> str:
    content = replace(content, "mid_block", "mid")
    content = replace(content, "up_blocks", "up")
    content = replace(content, "down_blocks", "down")

    # norm related
    content = replace(content, "conv_norm_out.bias", "norm_out.beta")
    content = replace(content, "conv_norm_out.weight", "norm_out.gamma")
    content = replace(content, "group_norm.bias", "norm.beta")
    content = replace(content, "group_norm.weight", "norm.gamma")
    content = replace(content, "norm1.bias", "norm1.beta")
    content = replace(content, "norm1.weight", "norm1.gamma")
    content = replace(content, "norm2.bias", "norm2.beta")
    content = replace(content, "norm2.weight", "norm2.gamma")

    # name
    content = replace(content, "proj_attn", "proj_out")
    content = replace(content, "attentions.0", "attn_1")
    content = replace(content, "downsamplers.0", "downsample")

    content = replace(content, "mid.resnets.0", "mid.block_1")
    content = replace(content, "mid.resnets.1", "mid.block_2")
    content = replace(content, "down.0.resnets", "down.0.block")
    content = replace(content, "down.1.resnets", "down.1.block")
    content = replace(content, "down.2.resnets", "down.2.block")
    content = replace(content, "down.3.resnets", "down.3.block")

    # reversed name
    content = replace(content, "up.0.resnets", "up.3.block")
    content = replace(content, "up.1.resnets", "up.2.block")
    content = replace(content, "up.2.resnets", "up.1.block")
    content = replace(content, "up.3.resnets", "up.0.block")
    content = replace(content, "0.upsamplers.0", "3.upsample")
    content = replace(content, "1.upsamplers.0", "2.upsample")
    content = replace(content, "2.upsamplers.0", "1.upsample")
    content = replace(content, "3.upsamplers.0", "0.upsample")

    # attention
    content = replace(content, "query", "q")
    content = replace(content, "key", "k")
    content = replace(content, "value", "v")

    # shortcut
    content = replace(content, "conv_shortcut", "nin_shortcut")
    return content


def expand_dim(name: str, value: torch.Tensor) -> torch.Tensor:
    if any([x in name for x in [".k.weight", ".q.weight", ".v.weight", ".proj_out.weight"]]):
        value = value[..., None, None]
    return value


def main():
    parser = argparse.ArgumentParser(description="Merging the checkpoint of sd for training/inference")
    parser.add_argument(
        "--open_clip_vit",
        default="checkpoints/sd_models/IP-Adapter/image_encoder/model.safetensors",
        help="Path of the image encoder checkpoint",
    )
    parser.add_argument(
        "--ip_adapter",
        default="checkpoints/sd_models/IP-Adapter/ip-adapter_sd15.safetensors",
        help="Path of the IP-Adapter checkpoint",
    )
    parser.add_argument(
        "--vae",
        default="checkpoints/sd_models/sd-vae-ft-mse/diffusion_pytorch_model.safetensors",
        help="Path of the finetuned VAE",
    )
    parser.add_argument(
        "--sd", default="checkpoints/sd_models/sd_v1.5-d0ab7146.ckpt", help="Path of the Mindspore SD model"
    )
    parser.add_argument(
        "--out", default="checkpoints/sd_models/merged/sd_v1.5_ip_adapter.ckpt", help="Path of the output checkpoint."
    )
    parser.add_argument(
        "--skip_ip",
        action="store_true",
        help="Whether to skip loading the IP weight, and duplicate the weight of `to_k` and `to_v`.",
    )
    parser.add_argument(
        "--ip_map",
        default="tools/ip_names_sd15.txt",
        help="Path of the IP mapping text file of SD",
    )
    args = parser.parse_args()

    print("Create IP Adapter naming mapping...")
    ip_mapping = create_ip_mapping(args.ip_map)

    # openclip ViT
    print("Converting openclip ViT...")
    tensors = dict()
    with safe_open(args.open_clip_vit, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[append_prefix(replace_all(k), "embedder.")] = f.get_tensor(k)

    merge_qkv(tensors)
    misc_ops(tensors)

    # ip adapter proj
    if args.ip_adapter.endswith("safetensors"):
        print("Converting IP Adatper...")
        ip_tensors = dict()
        with safe_open(args.ip_adapter, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "to_k_ip" not in k and "to_v_ip" not in k:
                    ip_tensors[append_prefix(replace_all(k), "embedder.")] = f.get_tensor(k)
                else:
                    ip_tensors[ip_mapping[k]] = f.get_tensor(k)
    else:
        print("Loading IP Adatper...")
        ip_tensors = ms.load_checkpoint(args.ip_adapter)

    print("Loading SD...")
    ms_tensors = ms.load_checkpoint(args.sd)
    tensors.update(ms_tensors)

    print("Replace VAE by finetuned VAE...")
    vae_tensors = dict()
    with safe_open(args.vae, framework="pt", device="cpu") as f:
        for k in f.keys():
            v = f.get_tensor(k)
            new_k = replace_all_vae(append_prefix(k, "first_stage_model."))
            v = expand_dim(new_k, v)
            vae_tensors[new_k] = v
    tensors.update(vae_tensors)

    if args.skip_ip:
        print("Skip merging IP Adatper and duplicating to_k, to_v values...")
        duplicated_tensors = dict()
        for k in ip_tensors.keys():
            if "to_k_ip" in k or "to_v_ip" in k:
                duplicated_tensors[k] = tensors[k.replace("_ip", "")]
        tensors.update(duplicated_tensors)
    else:
        print("Merging IP Adapter...")
        tensors.update(ip_tensors)

    print("Saving to MS checkpoints...")
    records = convert_to_ms(tensors)

    root = os.path.dirname(args.out)
    os.makedirs(root, exist_ok=True)
    ms.save_checkpoint(records, args.out)


if __name__ == "__main__":
    main()
