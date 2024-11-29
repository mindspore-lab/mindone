import argparse
import os
import mindspore as ms

def convert_weight(source_fp, target_fp):
    source_data = ms.load_checkpoint(source_fp)
    target_data = []
    for name, param in source_data.items():
        if 'ln_' in name or 'norm' in name:
            name = name.replace("beta", "bias").replace('gamma', 'weight')
        if "model.diffusion_model.out.0.beta" in name or 'model.diffusion_model.out.0.gamma' in name:
            name = name.replace("beta", "bias").replace('gamma', 'weight')
        target_data.append({"name": name, "data": param})
    ms.save_checkpoint(target_data, target_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        "-s",
        type=str,
        help="path to sd checkpoint",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        help="Filename to save. Specify folder, e.g., ./models, or file path which ends with .ckpt, e.g., ./models/sd.ckpt",
    )

    args = parser.parse_args()

    if not os.path.exists(args.source):
        raise ValueError(f"The provided source file {args.source} does not exist!")

    if not args.target.endswith(".ckpt"):
        os.makedirs(args.target, exist_ok=True)
        target_fp = os.path.join(args.target, os.path.basename(args.source).split(".")[0] + ".ckpt")
    else:
        target_fp = args.target

    if os.path.exists(target_fp):
        print(f"Warnings: {target_fp} will be overwritten!")

    convert_weight(args.source, target_fp)
    print(f"Converted weight saved to {target_fp}")
