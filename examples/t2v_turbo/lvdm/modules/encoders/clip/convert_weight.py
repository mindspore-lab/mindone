"""
Convert checkpoint from torch/huggingface
"""
import argparse

import torch

import mindspore as ms
from mindspore import Parameter, Tensor


def convert_weight(torch_path="ViT-B-32.pt", mindspore_path="clip_vit_b_32.ckpt"):
    r"""Convert Weight
    Convert clip_vit_b_32 weights from pytorch to mindspore
    pytorch and GPU required.

    Args:
        torch_path: The path to ViT-B-32.pt.
        mindspore_path： The save path for clip_vit_b_32.ckpt.
    """

    param_dict = torch.load(torch_path)

    new_dict = []
    for name, param in param_dict.items():
        if (
            "ln_pre.weight" in name
            or "ln_1.weight" in name
            or "ln_2.weight" in name
            or "ln_post.weight" in name
            or "ln_final.weight" in name
        ):
            new_name = name.replace("weight", "gamma")
        elif (
            "ln_pre.bias" in name
            or "ln_1.bias" in name
            or "ln_2.bias" in name
            or "ln_post.bias" in name
            or "ln_final.bias" in name
        ):
            new_name = name.replace("bias", "beta")
        elif "in_proj_weight" in name:
            new_name = name.replace("in_proj_weight", "in_proj.weight")
        elif "in_proj_bias" in name:
            new_name = name.replace("in_proj_bias", "in_proj.bias")
        elif "token_embedding.weight" in name:
            new_name = name.replace("token_embedding.weight", "token_embedding.embedding_table")
        else:
            new_name = name

        new_dict.append({"name": new_name, "data": Parameter(Tensor(param.detach().numpy()), name=new_name)})
    ms.save_checkpoint(new_dict, mindspore_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clip weight convert script")
    parser.add_argument(
        "--torch_path", type=str, default="ViT-B-32.pt", required=True, help="The torch checkpoint path."
    )
    parser.add_argument(
        "--mindspore_path",
        type=str,
        required=True,
        default="clip_vit_b_32.ckpt",
        help="The output mindspore checkpoint path.",
    )
    opt = parser.parse_args()

    convert_weight(opt.torch_path, opt.mindspore_path)
