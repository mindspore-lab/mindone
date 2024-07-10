import argparse

import torch
import mindspore as ms
import numpy as np
from tqdm import tqdm

"""
Usage: python convert_weight.py \
            --src_param ./pt_param_1024.txt \
            --target_param ./ms_param_1024.txt \
            --src_ckpt /path/to/pt_model_1024.ckpt \
            --target_ckpt /path/to/ms_model_1024.ckpt

Note: There are three resolutions: 256, 512, 1024. Please make sure the resolution is correct.
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
    gap = len(f_pt_param) - len(f_ms_param)
    print(f"INFO: The difference of number of params between PT and MS is {gap}.")
    # assert len(f_ms_param) + gap == len(f_pt_param)

    for idx, line_pt in tqdm(enumerate(f_pt_param)):
        if idx < gap:
            continue
        pt_name, pt_shape, pt_dtype = line_pt.split(":")[0], line_pt.split(":")[1], line_pt.split(":")[2]
        line_ms = f_ms_param[idx - gap]
        ms_name, ms_shape, ms_dtype = line_ms.split(":")[0], line_ms.split(":")[1], line_ms.split(":")[2]
        # assert pt_shape == ms_shape and pt_dtype == ms_dtype, f"PT: {line_pt}. MS: {line_ms}."
        assert pt_dtype == ms_dtype, f"PT: {line_pt}. MS: {line_ms}."
        if pt_shape == ms_shape:
            ms_val = pt_weight[pt_name].cpu().detach().numpy().astype(np.float32)
        else:  # deal with nn.Conv1d. Ref: https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_diff/Conv1d.html
            print(f"INFO: Manually expand one dim to the param '{line_pt}'. Supposed to be nn.Conv1d.")
            ms_val = pt_weight[pt_name].cpu().detach().numpy()[..., None].astype(np.float32)
        ms_weight.append({"name": ms_name, "data": ms.Tensor(ms_val, dtype=ms.float32)})

    ms.save_checkpoint(ms_weight, save_fp)
    print(f"INFO: PT to MS weight conversion finished. Saved at {save_fp}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_param", type=str, default="./pt_param_1024.txt", help="path to torch checkpoint param text")
    parser.add_argument("--target_param", type=str, default="./ms_param_1024.txt", help="path to mindspore checkpoint param text")
    parser.add_argument("--src_ckpt", type=str, help="path to torch checkpoint path")
    parser.add_argument(
        "--target_ckpt", type=str, help="target file path to save the converted checkpoint"
    )
    args = parser.parse_args()

    convert_pt2ms(args.src_param, args.target_param, args.src_ckpt, args.target_ckpt)
