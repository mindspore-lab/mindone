import argparse
import os

import torch

import mindspore as ms


def _load_torch_ckpt(ckpt_file):
    source_data = torch.load(ckpt_file, map_location="cpu")
    if ["state_dict"] in source_data:
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


abs_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sdv2_dir = os.path.join(abs_path, "../../../stable_diffusion_v2/")

with open(os.path.join(sdv2_dir, "tools/model_conversion/ms_names_v2.txt")) as file_ms:
    lines_ms = list(file_ms.readlines())
    lines_ms = [line for line in lines_ms if line.startswith("first_stage_model")]
with open(os.path.join(sdv2_dir, "tools/model_conversion/diffusers_vae_v2.txt")) as file_pt:
    lines_pt_vae = list(file_pt.readlines())
assert len(lines_ms) == len(lines_pt_vae)


def torch_to_ms_weight(source_fp, target_fp):
    source_data = load_torch_ckpt(source_fp)
    target_data = []
    assert len(lines_pt_vae) == len(
        source_data
    ), f"Loaded pt VAE checkpoint has a wrong number of parameters! Expect to have {len(lines_pt_vae)} params, but got {len(source_data)}"
    for i in range(len(lines_pt_vae)):
        line_pt_vae, line_ms_vae = lines_pt_vae[i], lines_ms[i]
        _name_pt, _, _ = line_pt_vae.strip().split("#")
        _name_ms, shape, _ = line_ms_vae.strip().split("#")
        shape = shape.replace("(", "").replace(")", "").split(",")
        shape = [int(s) for s in shape if len(s) > 0]
        _name_ms = _name_ms[len("first_stage_model.") :]
        _source_data = source_data[_name_pt].cpu().detach().numpy().reshape(shape)
        target_data.append({"name": _name_ms, "data": ms.Tensor(_source_data)})
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

    torch_to_ms_weight(args.src, target_fp)
    print(f"Converted weight saved to {target_fp}")
