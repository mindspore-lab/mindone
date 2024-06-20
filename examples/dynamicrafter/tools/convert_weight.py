import argparse

import torch


def convert(pt_weight, ms_weight):
    pt_weight = torch.load(pt_weight)["state_dict"]
    with open("pt_weight.txt", "w") as f:
        for k, v in pt_weight.items():
            # if "norm" in k:
            #     k = k.replace(".weight", ".gamma").replace(".bias", ".beta")

            f.write(k + ":" + str(tuple(v.shape)) + ":" + str(v.dtype).split(".")[1] + "\n")
    print(f"Num of params of pt weight: {len(pt_weight)}")       


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="path to torch checkpoint path")
    parser.add_argument(
        "--target", type=str, help="target file path to save the converted checkpoint"
    )
    args = parser.parse_args()

    convert(args.src, args.target)
