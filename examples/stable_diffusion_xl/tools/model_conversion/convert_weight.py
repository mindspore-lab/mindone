import argparse

import torch
from safetensors.torch import load_file as load_safetensors

import mindspore as ms
from mindspore import Tensor


def pt_to_ms(args):
    sd = load_safetensors(args.weight_safetensors)
    with open(args.key_torch) as f:
        key_torch = f.readlines()
        key_torch = [s.split(":")[0] for s in key_torch]
    with open(args.key_ms) as f:
        key_ms = f.readlines()
        key_ms = [s.split(":")[0] for s in key_ms]
    assert len(key_torch) == len(key_ms)

    new_ckpt = []
    for i in range(len(key_torch)):
        k_t, k_ms = key_torch[i], key_ms[i]

        assert k_t in sd, f"Keys '{k_t}' not found in {args.key_torch}"
        new_ckpt.append({"name": k_ms, "data": Tensor(sd[k_t].numpy(), ms.float32)})

    ms.save_checkpoint(new_ckpt, args.weight_ms)
    print(f"Convert '{args.weight_safetensors}' to '{args.weight_ms}' Done.")


def ms_to_pt(args):
    sd = ms.load_checkpoint(args.weight_ms)
    with open(args.key_torch) as f:
        key_torch = f.readlines()
        key_torch = [s.split(":")[0] for s in key_torch]
    with open(args.key_ms) as f:
        key_ms = f.readlines()
        key_ms = [s.split(":")[0] for s in key_ms]
    assert len(key_torch) == len(key_ms)

    new_ckpt = {}
    for i in range(len(key_ms)):
        k_t, k_ms = key_torch[i], key_ms[i]

        assert k_ms in sd, f"Keys '{k_ms}' not found in {args.key_ms}"
        new_ckpt[k_t] = torch.from_numpy(sd[k_ms].data.asnumpy())

    ms.save_checkpoint(new_ckpt, args.weight_safetensors)
    print(f"Convert '{args.weight_ms}' to '{args.weight_safetensors}' Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="weight conversion of sd-xl")
    parser.add_argument("--task", type=str, default="pt_to_ms", help="pt_to_ms or ms_to_pt")
    parser.add_argument("--weight_safetensors", type=str, default="./checkpoints/sd_xl_base_1.0.safetensors")
    parser.add_argument("--weight_ms", type=str, default="./checkpoints/sd_xl_base_1.0_ms.ckpt")
    parser.add_argument("--key_torch", type=str, default="./tools/model_conversion/torch_key_base.yaml")
    parser.add_argument("--key_ms", type=str, default="./tools/model_conversion/mindspore_key_base.yaml")
    args, _ = parser.parse_known_args()

    if args.task == "pt_to_ms":
        pt_to_ms(args)
    elif args.task == "ms_to_pt":
        ms_to_pt(args)
    else:
        raise NotImplementedError
