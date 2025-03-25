import argparse

import torch

import mindspore as ms

"""
Usage: python tools/convert_weight.py \
            --src /path/to/pt/var-d16.pt \
            --target /path/to/ms/var-d16.ckpt

"""


def convert_pt2ms(pt_weight, save_fp):
    pt_weight = torch.load(pt_weight, map_location=torch.device("cpu"))
    if "state_dict" in pt_weight:
        pt_weight = pt_weight["state_dict"]

    ms_weight = []
    for pname in pt_weight:
        ms_name = pname
        ms_val = pt_weight[pname].cpu().detach().numpy()
        ms_weight.append({"name": ms_name, "data": ms.Tensor(ms_val)})

    ms.save_checkpoint(ms_weight, save_fp)
    print(f"INFO: PT to MS weight conversion finished. Saved at {save_fp}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="path to the torch checkpoint")
    parser.add_argument("--target", type=str, help="path to save the converted MindSpore checkpoint")
    args = parser.parse_args()

    convert_pt2ms(args.src, args.target)
