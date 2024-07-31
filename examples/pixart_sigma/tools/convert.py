#!/usr/bin/env python
import argparse
import os

import torch
from tqdm import tqdm

import mindspore as ms


def _load_torch_ckpt(ckpt_file):
    source_data = torch.load(ckpt_file, map_location="cpu")
    if "state_dict" in source_data:
        source_data = source_data["state_dict"]
    return source_data


def _load_huggingface_safetensor(ckpt_file):
    from safetensors import safe_open

    db_state_dict = {}
    with safe_open(ckpt_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            db_state_dict[key] = f.get_tensor(key)
    return db_state_dict


LOAD_PYTORCH_FUNCS = {"others": _load_torch_ckpt, "safetensors": _load_huggingface_safetensor}


def load_torch_ckpt(ckpt_path):
    extension = ckpt_path.split(".")[-1]
    if extension not in LOAD_PYTORCH_FUNCS.keys():
        extension = "others"
    torch_params = LOAD_PYTORCH_FUNCS[extension](ckpt_path)
    return torch_params


def convert_pt_name_to_ms(content: str) -> str:
    return content


def torch_to_ms_weight(source_fp, target_fp):
    source_data = load_torch_ckpt(source_fp)
    if "ema" in source_data:
        source_data = source_data["ema"]
    if "state_dict" in source_data:
        source_data = source_data["state_dict"]
    target_data = []
    for _name_pt in tqdm(source_data, total=len(source_data)):
        _name_ms = convert_pt_name_to_ms(_name_pt)
        _source_data = source_data[_name_pt].cpu().detach().numpy()
        target_data.append({"name": _name_ms, "data": ms.Tensor(_source_data)})
    ms.save_checkpoint(target_data, target_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the PixArt-Sigma checkpoint.")

    parser.add_argument("-s", "--source", required=True, help="file path of the checkpoint (.pth / .safetensors)")
    parser.add_argument(
        "-t",
        "--target",
        help="output file path. If it is None, then the converted file will be saved in the input directory.",
    )

    args = parser.parse_args()

    if args.target is None:
        filename, suffix = os.path.splitext(args.source)
        target_path = filename + ".ckpt"
    else:
        target_path = args.target

    if os.path.exists(target_path):
        print(f"Warnings: {target_path} will be overwritten!")

    torch_to_ms_weight(args.source, target_path)
    print(f"Converted weight saved to {target_path}")
