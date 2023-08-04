import argparse

import numpy as np
import torch

import mindspore as ms

MINDSPORE = "ms"
PYTORCH = "pt"
STABLE_DIFFUSION_V1 = "sdv1"
STABLE_DIFFUSION_V2 = "sdv2"
CONTROLNET_V2 = "controlnet"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--source_version",
    "-sv",
    type=str,
    choices=[MINDSPORE, PYTORCH],
    help="set to mindspore if converting from mindspore to pytorch",
    default=PYTORCH,
)
parser.add_argument(
    "--source",
    "-s",
    type=str,
    help="path to source checkpoint, ms to torch if ends with .ckpt, torch to ms if ends with .pt",
)
parser.add_argument(
    "--target",
    "-t",
    type=str,
    help="where to save",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    choices=[STABLE_DIFFUSION_V1, STABLE_DIFFUSION_V2, CONTROLNET_V2],
    help="version of stable diffusion",
    default=STABLE_DIFFUSION_V2,
)
args = parser.parse_args()


def PYTORCH_MINDSPORE_STABLE_DIFFUSION_V2():
    with open("tools/ms_names_v2.txt") as file_ms:
        lines_ms = file_ms.readlines()
    with open("tools/pt_names_v2.txt") as file_pt:
        lines_pt = file_pt.readlines()

    source_data = torch.load(args.source, map_location="cpu")["state_dict"]
    target_data = []
    for line_ms, line_pt in zip(lines_ms, lines_pt):
        _name_pt, _, _ = line_pt.strip().split("#")
        _name_ms, _, _ = line_ms.strip().split("#")
        _source_data = source_data[_name_pt].cpu().detach().numpy()
        target_data.append({"name": _name_ms, "data": ms.Tensor(_source_data)})
    ms.save_checkpoint(target_data, args.target)


def MINDSPORE_PYTORCH_STABLE_DIFFUSION_V2():
    with open("tools/ms_names_v2.txt") as file_ms:
        lines_ms = file_ms.readlines()
    with open("tools/pt_names_v2.txt") as file_pt:
        lines_pt = file_pt.readlines()

    source_data = ms.load_checkpoint(args.source)
    target_data = {}
    for line_ms, line_pt in zip(lines_ms, lines_pt):
        _name_pt, _, _ = line_pt.strip().split("#")
        _name_ms, _, _ = line_ms.strip().split("#")
        _source_data = source_data[_name_ms].asnumpy()
        target_data[_name_pt] = torch.tensor(_source_data)
    torch.save(target_data, args.target)


def PYTORCH_MINDSPORE_STABLE_DIFFUSION_V1():
    with open("tools/ms_names_v1.txt") as file_ms:
        lines_ms = file_ms.readlines()
    with open("tools/pt_names_v1.txt") as file_pt:
        lines_pt = file_pt.readlines()

    source_data = torch.load(args.source, map_location="cpu")["state_dict"]
    target_data = []
    i = j = 0
    while i < len(lines_ms):
        line_ms = lines_ms[i]
        _name_ms, _, _ = line_ms.strip().split("#")
        if "attn.attn.in_proj" not in line_ms:
            line_pt = lines_pt[j]
            _name_pt, _, _ = line_pt.strip().split("#")
            target_data.append({"name": _name_ms, "data": ms.Tensor(source_data[_name_pt].cpu().detach().numpy())})
            i += 1
            j += 1
        else:
            w, b = [], []
            for k in range(6):
                line_pt = lines_pt[j]
                _name_pt, _, _ = line_pt.strip().split("#")
                j += 1
                if "weight" in _name_pt:
                    w.append(source_data[_name_pt].cpu().detach().numpy())
                else:
                    b.append(source_data[_name_pt].cpu().detach().numpy())
            target_data.append({"name": _name_ms, "data": ms.Tensor(np.concatenate([b[1], b[0], b[2]]))})
            i += 1
            line_ms = lines_ms[i]
            _name_ms, _, _ = line_ms.strip().split("#")
            target_data.append({"name": _name_ms, "data": ms.Tensor(np.concatenate([w[1], w[0], w[2]]))})
            i += 1

    ms.save_checkpoint(target_data, args.target)


SUPPORTED_CONVERSIONS = {
    (PYTORCH, MINDSPORE, STABLE_DIFFUSION_V1, PYTORCH_MINDSPORE_STABLE_DIFFUSION_V1),
    (PYTORCH, MINDSPORE, STABLE_DIFFUSION_V2, PYTORCH_MINDSPORE_STABLE_DIFFUSION_V2),
    (MINDSPORE, PYTORCH, STABLE_DIFFUSION_V1, None),
    (MINDSPORE, PYTORCH, STABLE_DIFFUSION_V2, MINDSPORE_PYTORCH_STABLE_DIFFUSION_V2),
}


def main():
    MODEL = args.model
    if args.source_version == MINDSPORE:
        SOURCE = MINDSPORE
        TARGET = PYTORCH
    elif args.source_version == PYTORCH:
        SOURCE = PYTORCH
        TARGET = MINDSPORE
    else:
        raise NotImplementedError(f"{args.source} should end with .pt or .ckpt")

    for src, tgt, model, convert_func in SUPPORTED_CONVERSIONS:
        if any([src != SOURCE, tgt != TARGET, model != MODEL, convert_func is None]):
            continue
        convert_func()
        print(f"converted {model} from {src} to {tgt}")
        return
    print("... nothing was converted")


if __name__ == "__main__":
    main()
