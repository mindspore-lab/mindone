"""
kohya sdxl lora model weight conversion (torch safetensors <-> mindspore ckpt)

"""
import argparse

import torch
from safetensors.torch import load_file, save_file

import mindspore as ms


def convert_ckpt_to_safetensors(args):
    weights_sd_safetensors = {}
    weights_sd = ms.load_checkpoint(args.ckpt_path)

    # turn the ms key : lora_unet.xxx.xx.x.weight
    # to the torch key : lora_unet_xxx_xx_x_weight
    def _replaced(key, postfix):
        return key[: -len(postfix)].replace(".", "_") + postfix

    for key, value in weights_sd.items():
        if key.endswith(".lora_down.weight"):
            postfix = ".lora_down.weight"
        elif key.endswith(".lora_up.weight"):
            postfix = ".lora_up.weight"
        elif key.endswith(".alpha"):
            postfix = ".alpha"
        else:
            continue
        key = _replaced(key, postfix)
        weights_sd_safetensors[key] = torch.from_numpy(value.numpy())

    save_file(weights_sd_safetensors, args.safetensor_path)
    print(f"lora ckpt is converted and saved as {args.safetensor_path}")


def convert_safetensors_to_ckpt(args):
    weights_sd_safetensors = load_file(args.safetensor_path)
    weights_sd = []
    # turn the torch key : lora_unet_xxx_xx_x_weight
    # to ms key : lora_unet.xxx.xx.x.weight
    for key, value in weights_sd_safetensors.items():
        if key.startswith("lora_unet_"):
            key.replace("lora_unet_", "lora_unet.")
        elif key.startswith("lora_te1_"):
            key.replace("lora_te1_", "lora_te1.")
        elif key.startswith("lora_te2_"):
            key.replace("lora_te2_", "lora_te2.")
        elif key.startswith("lora_te_"):
            key.replace("lora_te_", "lora_te.")
        else:
            continue
        tmp_l = key.split(".")
        tmp_l[1] = tmp_l.replace("_", ".")
        key = ".".join(tmp_l)
        weights_sd.append({"name": key, "data": ms.Tensor(value.numpy())})
    ms.save_checkpoint(weights_sd, args.ckpt_path)
    print(f"lora safetensors is converted and saved as {args.ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kohya sdxl lora model weight conversion")
    parser.add_argument(
        "--safetensor_path",
        type=str,
        help="path to sdxl lora weight from torch kohya (xxx.safetensors)",
    )
    parser.add_argument("--ckpt_path", type=str, help="path to sdxl lora weight from mindone kohya (xxx.ckpt)")
    parser.add_argument(
        "--convert_type",
        type=str,
        choices=["ms_to_st", "st_to_ms"],
        help="mindspore ckpt to torch safetensor or torch safetensor to mindspore ckpt",
    )

    args, _ = parser.parse_known_args()
    if args.convert_type == "ms_to_st":
        convert_ckpt_to_safetensors(args)
    elif args.convert_type == "st_to_ms":
        convert_safetensors_to_ckpt(args)
