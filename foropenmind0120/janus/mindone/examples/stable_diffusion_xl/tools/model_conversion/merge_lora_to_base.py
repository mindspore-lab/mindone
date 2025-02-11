import argparse

import mindspore as ms
from mindspore import Tensor, ops


def merge_lora_to_base(args):
    scale = args.lora_alpha / args.lora_rank

    lora_dict = ms.load_checkpoint(args.weight_lora)
    lora_keys = list(lora_dict)
    base_dict = ms.load_checkpoint(args.weight_base)
    base_keys = list(base_dict)

    merge_keys = []
    for lora_key in lora_keys:
        if "lora_A." in lora_key:
            merge_keys.append(lora_key.replace("lora_A.", ""))

    for i in range(len(merge_keys)):
        lora_A_key = lora_keys[2 * i]
        lora_B_key = lora_keys[2 * i + 1]
        merge_key = merge_keys[i]
        merge_weight = base_dict[merge_key] + ops.matmul(lora_dict[lora_B_key], lora_dict[lora_A_key]) * scale

        assert base_dict[merge_key].asnumpy().shape == merge_weight.asnumpy().shape
        base_dict[merge_key].set_data(merge_weight)

    new_ckpt = []
    for key in base_keys:
        new_ckpt.append({"name": key, "data": Tensor(base_dict[key].data.asnumpy())})
    ms.save_checkpoint(new_ckpt, args.weight_merged)

    print("Weights merging done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sdxl-base: merge lora weight to pretrained base weight")
    parser.add_argument("--weight_lora", type=str, default="./runs/SDXL-base-1.0_1000_lora.ckpt")
    parser.add_argument("--weight_base", type=str, default="./checkpoints/sd_xl_base_1.0_ms.ckpt")
    parser.add_argument("--weight_merged", type=str, default="./checkpoints/sd_xl_base_finetuned_ms.ckpt")
    parser.add_argument("--lora_config_alpha", type=int, default=4, help="consist with finetuned settings")
    parser.add_argument("--lora_config_rank", type=int, default=4, help="consist with finetuned settings")
    args, _ = parser.parse_known_args()
    merge_lora_to_base(args)
