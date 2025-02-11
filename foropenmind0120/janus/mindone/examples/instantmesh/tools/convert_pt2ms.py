"""convert pt ckpt: model.safetensors to ms ckpt"""

import argparse

import numpy as np
import torch
from safetensors import safe_open

import mindspore as ms


def convert(pt_ckpt, target_fp, pick_ema=True):
    if pt_ckpt.endswith(".ckpt") or pt_ckpt.endswith(".pt") or pt_ckpt.endswith(".pth"):
        state_dict = torch.load(pt_ckpt, torch.device("cpu"))
        if "ema" in state_dict and pick_ema:
            print("WARNING: find EMA weights in source checkpoint. Will pick it!")
            state_dict = state_dict["ema"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    else:
        state_dict = {}
        with safe_open(pt_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    target_data = []
    for k in state_dict:
        print(k)
        if "." not in k:
            # only for GroupNorm
            ms_name = k.replace("weight", "gamma").replace("bias", "beta")
        else:
            if "norm" in k:
                ms_name = k.replace(".weight", ".gamma").replace(".bias", ".beta")
            else:
                ms_name = k

        val = state_dict[k].detach().numpy().astype(np.float32)
        target_data.append({"name": ms_name, "data": ms.Tensor(val, dtype=ms.float32)})

    ms.save_checkpoint(target_data, target_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="YOUR_PT_CKPT_PATH", help="path to torch checkpoint path")
    parser.add_argument("--trgt", type=str, help="target file path to save the converted checkpoint")
    args = parser.parse_args()

    convert(args.src, args.trgt)
