import argparse

from safetensors.torch import load_file as load_safetensors

import mindspore as ms


def convert_weight(args, param_dtype):
    with open(args.mapping_filepath, "r") as f:
        lines = f.readlines()
    mapping_ms2torch = dict()
    for line in lines:
        line = line.strip()
        mapping_ms2torch[line.split(":")[0]] = line.split(":")[1]

    weight_ms = ms.load_checkpoint(args.weight_ms_sdxl)
    weight_torch = load_safetensors(args.weight_torch_controlnet)

    for key_ms, key_torch in mapping_ms2torch.items():
        weight_ms[key_ms] = ms.Parameter(ms.Tensor(weight_torch[key_torch].numpy(), param_dtype), name=key_ms)

    # adapt for mindspore <= 2.1 version
    new_ckpt = []
    for k, v in weight_ms.items():
        new_ckpt.append({"name": k, "data": v})

    ms.save_checkpoint(new_ckpt, args.output_ms_ckpt_path)
    print(f"INFO: SDXL with ControlNet checkpoint is converted and saved as {args.output_ms_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="controlnet + sdxl model weight conversion")
    parser.add_argument(
        "--mapping_filepath",
        type=str,
        default="./controlnet_ms2torch_mapping.yaml",
        help="path to param name mapping file for controlnet weight conversion",
    )
    parser.add_argument(
        "--weight_torch_controlnet",
        type=str,
        help="path to controlnet weight from torch (diffusion_pytorch_model.safetensors)",
    )
    parser.add_argument(
        "--weight_ms_sdxl", type=str, help="path to sdxl base 1.0 weight from mindone (sd_xl_base_1.0_ms.ckpt)"
    )
    parser.add_argument(
        "--output_ms_ckpt_path",
        type=str,
        help="path to controlnet+sdxl weight from mindone (sd_xl_base_1.0_controlnet_canny_ms.ckpt)",
    )

    args, _ = parser.parse_known_args()
    convert_weight(args, param_dtype=ms.float32)
