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


def load_torch_ckpt(ckpt_path_list):
    total_params = {}
    for ckpt_path in ckpt_path_list:
        extension = ckpt_path.split(".")[-1]
        if extension not in LOAD_PYTORCH_FUNCS.keys():
            extension = "others"
        torch_params = LOAD_PYTORCH_FUNCS[extension](ckpt_path)
        total_params.update(torch_params)
    return total_params


def convert_pt_name_to_ms(content: str) -> str:
    # embedding table name conversion
    content = content.replace("shared.weight", "shared.embedding_table")
    content = content.replace("encoder.embed_tokens.weight", "encoder.embed_tokens.embedding_table")
    content = content.replace(
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.embedding_table",
    )
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        "-s",
        nargs="+",
        default=[],
        help="a list of paths to source torch checkpoints, which ends with .pt or .bin",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        help="Filename to save. Specify folder, e.g., ./models, or file path which ends with .ckpt, e.g., ./models/t5-v1_1-xxl.ckpt",
    )

    args = parser.parse_args()

    if not args.target.endswith(".ckpt"):
        os.makedirs(args.target, exist_ok=True)
        target_fp = os.path.join(args.target, os.path.basename(args.src).split(".")[0] + ".ckpt")
    else:
        target_fp = args.target

    if os.path.exists(target_fp):
        print(f"Warnings: {target_fp} will be overwritten!")

    torch_to_ms_weight(args.src, target_fp)
    print(f"Converted weight saved to {target_fp}")
