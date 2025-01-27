"""
A script to convert pytorch safetensors to mindspore compatible safetensors:

Because some weights and variables in networks cannot be auto-converted, e.g. BatchNorm3d.bn2d.weight vs BatchNorm3d.gamma

To run this script, you should have installed both pytorch and mindspore.

Usage:
python convert_model.py --safetensor_path pytorch_model.safetensors --ms_safetensor_path model.savetensors

The converted model `model.savetensors` will be saved in the same directory as this file belonging to.
"""

import argparse
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def convert_safetensors(args):
    with safe_open(args.safetensor_path, framework="np") as f:
        metadata = f.metadata()

    weights_safetensors = load_file(args.safetensor_path)
    weights_ms_safetensors = {}

    # For BatchNorm3d:
    # turn torch key : X.time_res_stack.X.norm*.weight/bias/running_mean/running_var
    # to ms key : X.time_res_stack.X.norm*.bn2d.gamma/beta/moving_mean/moving_variance
    for key, value in weights_safetensors.items():
        if (".time_res_stack" in key) and (".norm" in key):
            origin_key = key
            if key.endswith("norm1.weight") or key.endswith("norm2.weight"):
                key = key.replace("weight", "bn2d.gamma")
            elif key.endswith("norm1.bias") or key.endswith("norm2.bias"):
                key = key.replace("bias", "bn2d.beta")
            elif key.endswith("norm1.running_mean") or key.endswith("norm2.running_mean"):
                key = key.replace("running_mean", "bn2d.moving_mean")
            elif key.endswith("norm1.running_var") or key.endswith("norm2.running_var"):
                key = key.replace("running_var", "bn2d.moving_variance")
            print(f"{origin_key} -> {key}")

        weights_ms_safetensors[key] = torch.from_numpy(value.numpy())

    save_file_dir = os.path.join(os.path.dirname(args.safetensor_path), args.ms_safetensor_path)
    save_file(weights_ms_safetensors, save_file_dir, metadata=metadata)
    print(f"Safetensors is converted and saved as {save_file_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emu2 model weight conversion")
    parser.add_argument(
        "--safetensor_path",
        type=str,
        help="path to Emu3 weight from torch (model.safetensors)",
    )
    parser.add_argument(
        "--ms_safetensor_path", type=str, help="path to sdxl lora weight from mindone kohya (xxx.safetensors)"
    )

    args, _ = parser.parse_known_args()
    print("Converting...")
    convert_safetensors(args)
