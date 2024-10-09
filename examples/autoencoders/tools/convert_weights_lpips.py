"""
Usage:
    python tools/convert_weights_lpips.py --src path/taming/modules/lpips/vgg.pth
"""
import argparse

import torch

import mindspore as ms


def convert(pt_ckpt, save_fp):
    pt_sd = torch.load(pt_ckpt, map_location=torch.device("cpu"))
    pt_pnames = list(pt_sd.keys())
    target_data = []
    for pt_pname in pt_pnames:
        target_data.append({"name": pt_pname, "data": ms.Tensor(pt_sd[pt_pname].detach().numpy())})

    ms.save_checkpoint(target_data, save_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None, help="path to torch checkpoint path")
    parser.add_argument(
        "--target", type=str, default="models/lpips_vgg.ckpt", help="target file path to save the converted checkpoint"
    )
    args = parser.parse_args()

    convert(args.src, args.target)
