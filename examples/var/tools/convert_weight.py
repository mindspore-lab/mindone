import argparse

import numpy as np
import torch

import mindspore as ms

"""
Usage: python tools/convert_weight.py \
            --model_name "var" \
            --src /path/to/pt/var-d16.pt \
            --target /path/to/ms/var-d16.ckpt

"""


def convert_pt2ms(pt_param, ms_param, pt_weight, save_fp):
    """#gap PT weights not exists in MS weights, but it's reasonable."""
    with open(pt_param, "r") as f:
        f_pt_param = f.readlines()
    with open(ms_param, "r") as f:
        f_ms_param = f.readlines()
    pt_weight = torch.load(pt_weight, map_location=torch.device("cpu"))
    if "state_dict" in pt_weight:
        pt_weight = pt_weight["state_dict"]

    ms_weight = []

    assert len(f_pt_param) == len(f_ms_param)

    for i in range(len(f_pt_param)):
        line_pt = f_pt_param[i]
        pt_name, pt_shape = line_pt.split(":")[0], line_pt.split(":")[1]
        line_ms = f_ms_param[i]
        ms_name, ms_shape = line_ms.split(":")[0], line_ms.split(":")[1]
        if pt_shape == ms_shape:
            ms_val = pt_weight[pt_name].cpu().detach().numpy()
        else:  # deal with nn.Conv1d. Ref: https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_diff/Conv1d.html
            print(f"INFO: Manually expand one dim to the param '{line_pt}'. Supposed to be nn.Conv1d.")
            ms_val = pt_weight[pt_name].cpu().detach().numpy()[..., None].astype(np.float32)
        ms_weight.append({"name": ms_name, "data": ms.Tensor(ms_val)})

    ms.save_checkpoint(ms_weight, save_fp)
    print(f"INFO: PT to MS weight conversion finished. Saved at {save_fp}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["vae", "var"],
        default="vae",
        help="path to torch checkpoint param text",
    )
    parser.add_argument("--src", type=str, help="path to the torch checkpoint")
    parser.add_argument("--target", type=str, help="path to save the converted MindSpore checkpoint")
    args = parser.parse_args()
    if args.model_name == "vae":
        src_param = "./tools/pt_vae.txt"
        target_param = "./tools/ms_vae.txt"
    else:
        src_param = "./tools/pt_var.txt"
        target_param = "./tools/ms_var.txt"

    convert_pt2ms(src_param, target_param, args.src, args.target)
