import argparse
import difflib
import os

import numpy as np
import torch

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))

MINDSPORE = "ms"
PYTORCH = "pt"
STABLE_DIFFUSION_V1 = "sdv1"
STABLE_DIFFUSION_V2 = "sdv2"
CONTROLNET_V2 = "controlnet"
DIFFUSERS_V1 = "diffusersv1"
DIFFUSERS_V2 = "diffusersv2"


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
    help="Filename to save. Specify folder if model is diffuser, , e.g., ./stable-diffusion-2-base",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    choices=[STABLE_DIFFUSION_V1, STABLE_DIFFUSION_V2, CONTROLNET_V2, DIFFUSERS_V2],
    help="version of stable diffusion",
    default=STABLE_DIFFUSION_V2,
)
args = parser.parse_args()


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


def PYTORCH_MINDSPORE_STABLE_DIFFUSION_V2():
    with open(os.path.join(__dir__, "ms_names_v2.txt")) as file_ms:
        lines_ms = file_ms.readlines()
    with open(os.path.join(__dir__, "pt_names_v2.txt")) as file_pt:
        lines_pt = file_pt.readlines()

    verify_name = False

    source_data = load_torch_ckpt(args.source)
    target_data = []
    if verify_name:
        pt_param_names = [line_pt.strip().split("#")[0] for line_pt in lines_pt]
        print("Num params in pt: ", len(pt_param_names))

    for line_ms, line_pt in zip(lines_ms, lines_pt):
        _name_pt, _, _ = line_pt.strip().split("#")
        _name_ms, _, _ = line_ms.strip().split("#")
        if verify_name:
            poss = difflib.get_close_matches(_name_ms, pt_param_names, n=3, cutoff=0.6)
            if poss[0] != _name_pt:
                print(
                    f"ms param {_name_ms}, got closes match in pt: {poss[0]}, but assined with {_name_pt} from pt_names_v2.txt"
                )
        _source_data = source_data[_name_pt].cpu().detach().numpy()
        target_data.append({"name": _name_ms, "data": ms.Tensor(_source_data)})
    ms.save_checkpoint(target_data, args.target)


def MINDSPORE_PYTORCH_STABLE_DIFFUSION_V2():
    with open(os.path.join(__dir__, "ms_names_v2.txt")) as file_ms:
        lines_ms = file_ms.readlines()
    with open(os.path.join(__dir__, "pt_names_v2.txt")) as file_pt:
        lines_pt = file_pt.readlines()

    source_data = ms.load_checkpoint(args.source)
    target_data = {}
    for line_ms, line_pt in zip(lines_ms, lines_pt):
        _name_pt, _, _ = line_pt.strip().split("#")
        _name_ms, _, _ = line_ms.strip().split("#")
        _source_data = source_data[_name_ms].asnumpy()
        target_data[_name_pt] = torch.tensor(_source_data)
    torch.save(target_data, args.target)


def _load_v1_and_split_qkv(source_data, lines_ms, lines_pt):
    target_data = {}
    i = j = 0
    while i < len(lines_ms):
        line_ms = lines_ms[i]
        _name_ms, _, _ = line_ms.strip().split("#")
        if "attn.attn.in_proj" not in line_ms:
            line_pt = lines_pt[j]
            _name_pt, _, _ = line_pt.strip().split("#")
            target_data[_name_pt] = torch.tensor(source_data[_name_ms].asnumpy())
            i += 1
            j += 1
        else:
            b = np.split(source_data[_name_ms].asnumpy(), 3)
            i += 1
            line_ms = lines_ms[i]
            _name_ms, _, _ = line_ms.strip().split("#")
            w = np.split(source_data[_name_ms].asnumpy(), 3)
            i1 = {1: 1, 3: 0, 5: 2}
            i2 = {0: 1, 2: 0, 4: 2}
            for k in range(6):
                line_pt = lines_pt[j]
                _name_pt, _, _ = line_pt.strip().split("#")
                j += 1
                if "weight" in _name_pt:
                    target_data[_name_pt] = torch.tensor(w[i1[k]])
                else:
                    target_data[_name_pt] = torch.tensor(b[i2[k]])
            i += 1
    return target_data


def MINDSPORE_PYTORCH_DIFFUSERS_V2():
    with open("tools/model_conversion/ms_names_v2.txt") as file_ms:
        lines_ms = list(file_ms.readlines())
    with open("tools/model_conversion/diffusers_vae_v2.txt") as file_pt:
        lines_pt_vae = list(file_pt.readlines())
    with open("tools/model_conversion/diffusers_clip_v2.txt") as file_pt:
        lines_pt_clip = list(file_pt.readlines())
    with open("tools/model_conversion/diffusers_unet_v2.txt") as file_pt:
        lines_pt_unet = list(file_pt.readlines())

    source_data = ms.load_checkpoint(args.source)
    target_vae, target_clip, target_unet = {}, {}, {}
    i = j = 0
    for line_ms in lines_ms:
        if "model.diffusion_model" in line_ms:
            line_pt = lines_pt_unet[i]
            i += 1
            _name_pt, _, _ = line_pt.strip().split("#")
            _name_ms, _, _ = line_ms.strip().split("#")
            _source_data = source_data[_name_ms].asnumpy()
            target_unet[_name_pt] = torch.tensor(_source_data)
        elif "first_stage_model" in line_ms:
            line_pt = lines_pt_vae[j]
            j += 1
            _name_pt, shape, _ = line_pt.strip().split("#")
            _name_ms, _, _ = line_ms.strip().split("#")
            shape = shape.replace("torch.Size([", "").replace("])", "").split(", ")
            shape = [int(s) for s in shape]
            _source_data = source_data[_name_ms].asnumpy().reshape(shape)
            target_vae[_name_pt] = torch.tensor(_source_data)
    os.makedirs(os.path.join(args.target, "unet"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "vae"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "text_encoder"), exist_ok=True)
    torch.save(target_unet, os.path.join(args.target, "unet", "diffusion_pytorch_model.bin"))
    torch.save(target_vae, os.path.join(args.target, "vae", "diffusion_pytorch_model.bin"))
    lines_ms = [line_ms for line_ms in lines_ms if "cond_stage_model" in line_ms]
    target_clip = _load_v1_and_split_qkv(source_data, lines_ms, lines_pt_clip)
    torch.save(target_clip, os.path.join(args.target, "text_encoder", "pytorch_model.bin"))


def np2ms_tensor(inp, force_fp32=True):
    ms_dtype = None
    if inp.dtype == np.float16 and force_fp32:
        ms_dtype = ms.float32
    out = ms.Tensor(inp, dtype=ms_dtype)
    return out


def _load_v1_and_merge_qkv(source_data, lines_ms, lines_pt):
    # dtype = ms.float32 if force_fp32 else None
    target_data = []
    i = j = 0
    while i < len(lines_ms):
        line_ms = lines_ms[i]
        _name_ms, _, _ = line_ms.strip().split("#")
        if "attn.attn.in_proj" not in line_ms:
            line_pt = lines_pt[j]
            _name_pt, _, _ = line_pt.strip().split("#")
            target_data.append({"name": _name_ms, "data": np2ms_tensor(source_data[_name_pt].cpu().detach().numpy())})
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
            target_data.append({"name": _name_ms, "data": np2ms_tensor(np.concatenate([b[1], b[0], b[2]]))})
            i += 1
            line_ms = lines_ms[i]
            _name_ms, _, _ = line_ms.strip().split("#")
            target_data.append({"name": _name_ms, "data": np2ms_tensor(np.concatenate([w[1], w[0], w[2]]))})
            i += 1
    return target_data


def PYTORCH_MINDSPORE_STABLE_DIFFUSION_V1():
    with open("tools/model_conversion/ms_names_v1.txt") as file_ms:
        lines_ms = file_ms.readlines()
    with open("tools/model_conversion/pt_names_v1.txt") as file_pt:
        lines_pt = file_pt.readlines()
    source_data = load_torch_ckpt(args.source)
    target_data = _load_v1_and_merge_qkv(source_data, lines_ms, lines_pt)
    ms.save_checkpoint(target_data, args.target)


def MINDSPORE_PYTORCH_DIFFUSERS_V1():
    raise NotImplementedError


SUPPORTED_CONVERSIONS = {
    (PYTORCH, MINDSPORE, STABLE_DIFFUSION_V1, PYTORCH_MINDSPORE_STABLE_DIFFUSION_V1),
    (PYTORCH, MINDSPORE, STABLE_DIFFUSION_V2, PYTORCH_MINDSPORE_STABLE_DIFFUSION_V2),
    (MINDSPORE, PYTORCH, STABLE_DIFFUSION_V1, None),
    (MINDSPORE, PYTORCH, STABLE_DIFFUSION_V2, MINDSPORE_PYTORCH_STABLE_DIFFUSION_V2),
    (MINDSPORE, PYTORCH, DIFFUSERS_V2, MINDSPORE_PYTORCH_DIFFUSERS_V2),
    (MINDSPORE, PYTORCH, DIFFUSERS_V1, MINDSPORE_PYTORCH_DIFFUSERS_V1),
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
    print("warning: you should always backup your old weights")
    for src, tgt, model, convert_func in SUPPORTED_CONVERSIONS:
        if any([src != SOURCE, tgt != TARGET, model != MODEL, convert_func is None]):
            continue
        convert_func()
        print(f"converted {model} from {src} to {tgt}")
        return
    print("... nothing was converted")


if __name__ == "__main__":
    main()
