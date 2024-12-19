import argparse
import os

import numpy as np
from convert_vae import load_torch_ckpt

import mindspore as ms


def get_shape_from_str(shape):
    shape = shape.replace("(", "").replace(")", "").split(",")
    shape = [int(s) for s in shape if len(s) > 0]

    return shape


def convert(source_fp, target_fp, from_vae2d=False):
    # read param mapping files
    with open("tools/ms_pnames_vae1.2.txt") as file_ms:
        lines_ms = list(file_ms.readlines())
    with open("tools/pt_pnames_vae1.2.txt") as file_pt:
        lines_pt = list(file_pt.readlines())

    if from_vae2d:
        lines_ms = [line for line in lines_ms if line.startswith("spatial_vae")]
        lines_pt = [line for line in lines_pt if line.startswith("spatial_vae")]

    assert len(lines_ms) == len(lines_pt)

    # convert and save
    sd_pt = load_torch_ckpt(source_fp)  # state dict
    num_params_pt = len(list(sd_pt.keys()))
    print("Total params in pt ckpt: ", num_params_pt)
    target_data = []
    for i in range(len(lines_pt)):
        name_pt, shape_pt = lines_pt[i].strip().split("#")
        shape_pt = get_shape_from_str(shape_pt)
        name_ms, shape_ms = lines_ms[i].strip().split("#")
        shape_ms = get_shape_from_str(shape_ms)
        assert np.prod(shape_pt) == np.prod(
            shape_ms
        ), f"Mismatch param: PT: {name_pt}, {shape_pt} vs MS: {name_ms}, {shape_ms}"

        if from_vae2d:
            name_pt = name_pt.replace("spatial_vae.module.", "")

        data = sd_pt[name_pt].cpu().detach().numpy().reshape(shape_ms)

        data = ms.Tensor(input_data=data.astype(np.float32), dtype=ms.float32)
        target_data.append({"name": name_ms, "data": data})  # ms.Tensor(data, dtype=ms.float32)})

    print("Total params converted: ", len(target_data))
    ms.save_checkpoint(target_data, target_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        "-s",
        type=str,
        help="path to vae torch checkpoint",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        help="Filename to save. Specify folder, e.g., ./models, or file path which ends with .ckpt, e.g., ./models/vae.ckpt",
    )
    parser.add_argument("--from_vae2d", action="store_true", help="only convert spatial vae, default: False")

    args = parser.parse_args()

    if not os.path.exists(args.src):
        raise ValueError(f"The provided source file {args.src} does not exist!")

    if not args.target.endswith(".ckpt"):
        os.makedirs(args.target, exist_ok=True)
        target_fp = os.path.join(args.target, os.path.basename(args.src).split(".")[0] + ".ckpt")
    else:
        target_fp = args.target

    if os.path.exists(target_fp):
        print(f"Warnings: {target_fp} will be overwritten!")

    convert(args.src, target_fp, args.from_vae2d)
    print(f"Converted weight saved to {target_fp}")
