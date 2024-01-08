#!/usr/bin/env python
import argparse
import os

from safetensors import safe_open
from util import append_prefix, convert_to_ms, create_ip_mapping, merge_qkv, misc_ops, replace_all

import mindspore as ms


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

    print("Saving to MS checkpoints...")
    records = convert_to_ms(tensors)

    root = os.path.dirname(args.out)
    os.makedirs(root, exist_ok=True)
    ms.save_checkpoint(records, args.out)


if __name__ == "__main__":
    main()
