#!/usr/bin/env python
import argparse
import os
import re

from safetensors import safe_open
from util import append_prefix, convert_to_ms, create_ip_mapping, merge_qkv, misc_ops, replace, replace_all

import mindspore as ms


def replace_all_controlnet(content: str) -> str:
    # looks ugly, but straightfoward
    content = replace(content, "mid_block", "middle_block")
    content = replace(content, "time_embedding", "time_embed")
    content = replace(content, "controlnet_cond_embedding", "input_hint_block")
    content = replace(content, "controlnet_down_blocks", "zero_convs")
    content = replace(content, "down_blocks", "input_blocks")

    content = replace(content, "time_embed.linear_1", "time_embed.0")
    content = replace(content, "time_embed.linear_2", "time_embed.2")
    content = replace(content, "controlnet.conv_in", "controlnet.input_blocks.0.0")
    content = replace(content, "input_blocks.0.resnets.0", "input_blocks.1.0")
    content = replace(content, "input_blocks.0.attentions.0", "input_blocks.1.1")
    content = replace(content, "input_blocks.0.resnets.1", "input_blocks.2.0")
    content = replace(content, "input_blocks.0.attentions.1", "input_blocks.2.1")
    content = replace(content, "input_blocks.0.downsamplers.0", "input_blocks.3.0")
    content = replace(content, "input_blocks.1.resnets.0", "input_blocks.4.0")
    content = replace(content, "input_blocks.1.attentions.0", "input_blocks.4.1")
    content = replace(content, "input_blocks.1.resnets.1", "input_blocks.5.0")
    content = replace(content, "input_blocks.1.attentions.1", "input_blocks.5.1")
    content = replace(content, "input_blocks.1.downsamplers.0", "input_blocks.6.0")
    content = replace(content, "input_blocks.2.resnets.0", "input_blocks.7.0")
    content = replace(content, "input_blocks.2.attentions.0", "input_blocks.7.1")
    content = replace(content, "input_blocks.2.resnets.1", "input_blocks.8.0")
    content = replace(content, "input_blocks.2.attentions.1", "input_blocks.8.1")
    content = replace(content, "input_blocks.2.downsamplers.0", "input_blocks.9.0")
    content = replace(content, "input_blocks.3.resnets.0", "input_blocks.10.0")
    content = replace(content, "input_blocks.3.attentions.0", "input_blocks.10.1")
    content = replace(content, "input_blocks.3.resnets.1", "input_blocks.11.0")
    content = replace(content, "input_blocks.3.attentions.1", "input_blocks.11.1")
    content = replace(content, "middle_block.resnets.0", "middle_block.0")
    content = replace(content, "middle_block.attentions.0", "middle_block.1")
    content = replace(content, "middle_block.resnets.1", "middle_block.2")

    content = replace(content, "conv1", "in_layers.0")
    content = replace(content, "conv2", "out_layers.0")
    content = replace(content, "time_emb_proj", "emb_layers.1")
    content = replace(content, "conv_shortcut", "skip_connection")
    content = replace(content, "controlnet_middle_block", "middle_block_out.0")

    content = replace(content, "input_hint_block.conv_in", "input_hint_block.0")
    content = replace(content, "input_hint_block.blocks.0", "input_hint_block.2")
    content = replace(content, "input_hint_block.blocks.1", "input_hint_block.4")
    content = replace(content, "input_hint_block.blocks.2", "input_hint_block.6")
    content = replace(content, "input_hint_block.blocks.3", "input_hint_block.8")
    content = replace(content, "input_hint_block.blocks.4", "input_hint_block.10")
    content = replace(content, "input_hint_block.blocks.5", "input_hint_block.12")
    content = replace(content, "input_hint_block.conv_out", "input_hint_block.14")

    content = re.sub(r"(zero_convs.[0-9]*).weight", r"\1.0.weight", content)
    content = re.sub(r"(zero_convs.[0-9]*).bias", r"\1.0.bias", content)
    content = re.sub(r"(norm[0-9]*).weight", r"\1.gamma", content)
    content = re.sub(r"(norm[0-9]*).bias", r"\1.beta", content)

    content = replace(content, "in_layers.0.bias", "in_layers.2.bias")
    content = replace(content, "in_layers.0.weight", "in_layers.2.weight")
    content = replace(content, "out_layers.0.bias", "out_layers.3.bias")
    content = replace(content, "out_layers.0.weight", "out_layers.3.weight")

    content = re.sub(r"(input_blocks.[0-9]*.0).norm1.beta", r"\1.in_layers.0.beta", content)
    content = re.sub(r"(input_blocks.[0-9]*.0).norm1.gamma", r"\1.in_layers.0.gamma", content)
    content = re.sub(r"(input_blocks.[0-9]*.0).norm2.beta", r"\1.out_layers.0.beta", content)
    content = re.sub(r"(input_blocks.[0-9]*.0).norm2.gamma", r"\1.out_layers.0.gamma", content)
    content = re.sub(r"(middle_block.[0-9]*).norm1.beta", r"\1.in_layers.0.beta", content)
    content = re.sub(r"(middle_block.[0-9]*).norm1.gamma", r"\1.in_layers.0.gamma", content)
    content = re.sub(r"(middle_block.[0-9]*).norm2.beta", r"\1.out_layers.0.beta", content)
    content = re.sub(r"(middle_block.[0-9]*).norm2.gamma", r"\1.out_layers.0.gamma", content)

    content = replace(content, "conv.bias", "op.bias")
    content = replace(content, "conv.weight", "op.weight")

    content = replace(content, "add_embedding.linear_1", "label_emb.0.0")
    content = replace(content, "add_embedding.linear_2", "label_emb.0.2")
    return content


def main():
    parser = argparse.ArgumentParser(description="Merging the checkpoint of SDXL for training/inference")
    parser.add_argument(
        "--open_clip_vit",
        default="checkpoints/sdxl_models/IP-Adapter/image_encoder/model.safetensors",
        help="Path of the image encoder checkpoint",
    )
    parser.add_argument(
        "--ip_adapter",
        default="checkpoints/sdxl_models/IP-Adapter/ip-adapter_sdxl.safetensors",
        help="Path of the IP-Adapter checkpoint",
    )
    parser.add_argument(
        "--sd_xl_base",
        default="checkpoints/sdxl_models/sd_xl_base_1.0_ms.ckpt",
        help="Path of the Mindspore SD-XL Base",
    )
    parser.add_argument(
        "--out",
        default="checkpoints/sdxl_models/merged/sd_xl_base_1.0_ms_ip_adapter.ckpt",
        help="Path of the output checkpoint.",
    )
    parser.add_argument(
        "--skip_ip",
        action="store_true",
        help="Whether to skip loading the IP weight, and duplicate the weight of `to_k` and `to_v`.",
    )
    parser.add_argument(
        "--ip_map",
        default="tools/ip_names_sdxl.txt",
        help="Path of the IP mapping text file of SDXL",
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
            tensors[append_prefix(replace_all(k), "conditioner.embedders.2.")] = f.get_tensor(k)

    merge_qkv(tensors)
    misc_ops(tensors)

    # ip adapter proj
    if args.ip_adapter.endswith("safetensors"):
        print("Converting IP Adatper...")
        ip_tensors = dict()
        with safe_open(args.ip_adapter, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "to_k_ip" not in k and "to_v_ip" not in k:
                    ip_tensors[append_prefix(replace_all(k), "conditioner.embedders.2.")] = f.get_tensor(k)
                else:
                    ip_tensors[ip_mapping[k]] = f.get_tensor(k)
    else:
        print("Loading IP Adatper...")
        ip_tensors = ms.load_checkpoint(args.ip_adapter)

    print("Loading SD XL base...")
    ms_tensors = ms.load_checkpoint(args.sd_xl_base)
    tensors.update(ms_tensors)

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
