import argparse

import numpy as np
import torch

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


def convert(pt_ckpt, target_fp):
    source_data = load_torch_ckpt(pt_ckpt)
    if "ema" in source_data:
        source_data = source_data["ema"]
    if "state_dict" in source_data:
        source_data = source_data["state_dict"]
    if "model" in source_data:
        source_data = source_data["model"]
    target_data = []
    for k in source_data:
        print(k)
        if "." not in k:
            # only for GroupNorm
            ms_name = k.replace("weight", "gamma").replace("bias", "beta")
        else:
            if "norm" in k:
                ms_name = k.replace(".weight", ".gamma").replace(".bias", ".beta")
            else:
                ms_name = k
        # import pdb
        # pdb.set_trace()
        val = source_data[k].detach().numpy().astype(np.float32)
        # print(type(val), val.dtype, val.shape)
        target_data.append({"name": ms_name, "data": ms.Tensor(val, dtype=ms.float32)})

    ms.save_checkpoint(target_data, target_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None, help="path to torch checkpoint path")
    parser.add_argument(
        "--target", type=str, default="models/vae3d.ckpt", help="target file path to save the converted checkpoint"
    )
    args = parser.parse_args()

    convert(args.src, args.target)
