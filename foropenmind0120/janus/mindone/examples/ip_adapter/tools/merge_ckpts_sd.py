#!/usr/bin/env python
import argparse
import os
import re

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


def replace_all_controlnet(content: str) -> str:
    # looks ugly, but straightfoward
    content = replace(content, "mid_block", "middle_block")
    content = replace(content, "time_embedding", "time_embed")
    content = replace(content, "controlnet_cond_embedding", "input_hint_block")
    content = replace(content, "controlnet_down_blocks", "zero_convs")
    content = replace(content, "down_blocks", "input_blocks")

    content = replace(content, "time_embed.linear_1", "time_embed.0")
    content = replace(content, "time_embed.linear_2", "time_embed.2")
    content = replace(content, "controlnet.conv_in", "controlnet.input_blocks.0.0.conv")
    content = replace(content, "input_blocks.0.resnets.0", "input_blocks.1.0")
    content = replace(content, "input_blocks.0.attentions.0", "input_blocks.1.1")
    content = replace(content, "input_blocks.0.resnets.1", "input_blocks.2.0")
    content = replace(content, "input_blocks.0.attentions.1", "input_blocks.2.1")
    content = replace(content, "input_blocks.0.downsamplers.0", "input_blocks.3.0.op")
    content = replace(content, "input_blocks.1.resnets.0", "input_blocks.4.0")
    content = replace(content, "input_blocks.1.attentions.0", "input_blocks.4.1")
    content = replace(content, "input_blocks.1.resnets.1", "input_blocks.5.0")
    content = replace(content, "input_blocks.1.attentions.1", "input_blocks.5.1")
    content = replace(content, "input_blocks.1.downsamplers.0", "input_blocks.6.0.op")
    content = replace(content, "input_blocks.2.resnets.0", "input_blocks.7.0")
    content = replace(content, "input_blocks.2.attentions.0", "input_blocks.7.1")
    content = replace(content, "input_blocks.2.resnets.1", "input_blocks.8.0")
    content = replace(content, "input_blocks.2.attentions.1", "input_blocks.8.1")
    content = replace(content, "input_blocks.2.downsamplers.0", "input_blocks.9.0.op")
    content = replace(content, "input_blocks.3.resnets.0", "input_blocks.10.0")
    content = replace(content, "input_blocks.3.attentions.0", "input_blocks.10.1")
    content = replace(content, "input_blocks.3.resnets.1", "input_blocks.11.0")
    content = replace(content, "input_blocks.3.attentions.1", "input_blocks.11.1")
    content = replace(content, "middle_block.resnets.0", "middle_block.0")
    content = replace(content, "middle_block.attentions.0", "middle_block.1")
    content = replace(content, "middle_block.resnets.1", "middle_block.2")

    content = replace(content, "input_blocks.1.0.norm1", "input_blocks.1.0.in_layers_norm")
    content = replace(content, "input_blocks.1.0.norm2", "input_blocks.1.0.out_layers_norm")
    content = replace(content, "input_blocks.2.0.norm1", "input_blocks.2.0.in_layers_norm")
    content = replace(content, "input_blocks.2.0.norm2", "input_blocks.2.0.out_layers_norm")
    content = replace(content, "input_blocks.4.0.norm1", "input_blocks.4.0.in_layers_norm")
    content = replace(content, "input_blocks.4.0.norm2", "input_blocks.4.0.out_layers_norm")
    content = replace(content, "input_blocks.5.0.norm1", "input_blocks.5.0.in_layers_norm")
    content = replace(content, "input_blocks.5.0.norm2", "input_blocks.5.0.out_layers_norm")
    content = replace(content, "input_blocks.7.0.norm1", "input_blocks.7.0.in_layers_norm")
    content = replace(content, "input_blocks.7.0.norm2", "input_blocks.7.0.out_layers_norm")
    content = replace(content, "input_blocks.8.0.norm1", "input_blocks.8.0.in_layers_norm")
    content = replace(content, "input_blocks.8.0.norm2", "input_blocks.8.0.out_layers_norm")
    content = replace(content, "input_blocks.10.0.norm1", "input_blocks.10.0.in_layers_norm")
    content = replace(content, "input_blocks.10.0.norm2", "input_blocks.10.0.out_layers_norm")
    content = replace(content, "input_blocks.11.0.norm1", "input_blocks.11.0.in_layers_norm")
    content = replace(content, "input_blocks.11.0.norm2", "input_blocks.11.0.out_layers_norm")
    content = replace(content, "middle_block.0.norm1", "middle_block.0.in_layers_norm")
    content = replace(content, "middle_block.0.norm2", "middle_block.0.out_layers_norm")
    content = replace(content, "middle_block.2.norm1", "middle_block.2.in_layers_norm")
    content = replace(content, "middle_block.2.norm2", "middle_block.2.out_layers_norm")

    content = replace(content, "conv1", "in_layers_conv.conv")
    content = replace(content, "conv2", "out_layers_conv.conv")
    content = replace(content, "time_emb_proj", "emb_layers.1")
    content = replace(content, "conv_shortcut", "skip_connection.conv")
    content = replace(content, "controlnet_middle_block", "middle_block_out.conv")

    content = replace(content, "input_hint_block.conv_in", "input_hint_block.0.conv")
    content = replace(content, "input_hint_block.blocks.0", "input_hint_block.2.conv")
    content = replace(content, "input_hint_block.blocks.1", "input_hint_block.4.conv")
    content = replace(content, "input_hint_block.blocks.2", "input_hint_block.6.conv")
    content = replace(content, "input_hint_block.blocks.3", "input_hint_block.8.conv")
    content = replace(content, "input_hint_block.blocks.4", "input_hint_block.10.conv")
    content = replace(content, "input_hint_block.blocks.5", "input_hint_block.12.conv")
    content = replace(content, "input_hint_block.conv_out", "input_hint_block.14.conv")

    content = re.sub(r"(zero_convs.[0-9]*).weight", r"\1.conv.weight", content)
    content = re.sub(r"(zero_convs.[0-9]*).bias", r"\1.conv.bias", content)
    content = re.sub(r"(norm[0-9]*).weight", r"\1.gamma", content)
    content = re.sub(r"(norm[0-9]*).bias", r"\1.beta", content)

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
    parser.add_argument("--controlnet", help="Path of the controlnet")
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

    if args.controlnet:
        print("Converting Controlnet...")
        controlnet_tensors = dict()
        with safe_open(args.controlnet, framework="pt", device="cpu") as f:
            for k in f.keys():
                v = f.get_tensor(k)
                new_k = replace_all_controlnet(append_prefix(k, "model.diffusion_model.controlnet."))
                controlnet_tensors[new_k] = v
        tensors.update(controlnet_tensors)

    print("Saving to MS checkpoint...")
    records = convert_to_ms(tensors)

    root = os.path.dirname(args.out)
    os.makedirs(root, exist_ok=True)
    ms.save_checkpoint(records, args.out)
    print(f"Merged MS checkpoint is saved at `{args.out}`.")


if __name__ == "__main__":
    main()
