import argparse
import os.path as osp

import torch
from safetensors.torch import load_file, save_file


def get_map(key_file, value_file):
    key_map = {}
    with open(key_file, "r") as key_file, open(value_file, "r") as value_file:
        keys = key_file.readlines()
        values = value_file.readlines()

        for key, value in zip(keys, values):  # key是sd
            key = key.strip()
            value = value.strip()
            if key not in key_map:
                key_map[key] = value
            else:
                if isinstance(key_map[key], str):
                    key_map[key] = [key_map[key]]
                key_map[key].append(value)
    return key_map


def convert(state_dict, module_map):
    mapping = module_map
    new_state_dict = {}
    for sd_name, hf_name in mapping.items():
        if sd_name in state_dict:
            new_state_dict[hf_name] = state_dict[sd_name]
    return new_state_dict


def vae_convert(new_state_dict, module_map):
    mapping = {value: key for key, value in module_map.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for hf_name, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in mapping[hf_name]:  # mapping[hf_name] 表示对应的torch的key
                print(f"Reshaping {hf_name} for diffusers format")
                new_state_dict[hf_name] = reshape_weight_for_hf(v)
    return new_state_dict


def text2_convert(state_dict, module_map):
    mapping = module_map
    new_state_dict = {}
    for sd_name, hf_name in mapping.items():
        if "attn.in_proj" in sd_name:
            chunks = torch.chunk(state_dict[sd_name], 3, dim=0)
            new_state_dict[hf_name[0]] = chunks[0]
            new_state_dict[hf_name[1]] = chunks[1]
            new_state_dict[hf_name[2]] = chunks[2]
        elif sd_name in state_dict:
            new_state_dict[hf_name] = state_dict[sd_name]
    return new_state_dict


def reshape_weight_for_hf(w):
    return w.reshape(w.shape[:-2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--output_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--use_safetensors", action="store_true", help="Save weights use safetensors, default is ckpt.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    args = parser.parse_args()

    key_file = "diffuser_yaml/torch_key_unet.yaml"
    value_file = "diffuser_yaml/hf_key_unet.yaml"
    unet_map = get_map("diffuser_yaml/torch_key_unet.yaml", "diffuser_yaml/hf_key_unet.yaml")
    vae_map = get_map("diffuser_yaml/torch_key_vae.yaml", "diffuser_yaml/hf_key_vae.yaml")
    text1_map = get_map("diffuser_yaml/torch_key_text1.yaml", "diffuser_yaml/hf_key_text1.yaml")
    text2_map = get_map("diffuser_yaml/torch_key_text2.yaml", "diffuser_yaml/hf_key_text2.yaml")

    if osp.exists(args.model_path):
        if args.model_path.endswith(".pth"):
            state_dict = torch.load(args.model_path, map_location="cpu")
        else:
            state_dict = load_file(args.model_path, device="cpu")
        print("load model from model_path ", args.model_path)
    else:
        print("model_path is not valid, please double-check it!")

    unet = convert(state_dict, unet_map)
    vae = vae_convert(convert(state_dict, vae_map), vae_map)
    text1 = convert(state_dict, text1_map)
    text2 = text2_convert(state_dict, text2_map)

    if args.half:
        unet = {k: v.half() for k, v in unet.items()}
        vae = {k: v.half() for k, v in vae.items()}
        text1 = {k: v.half() for k, v in text1.items()}
        text2 = {k: v.half() for k, v in text2.items()}

    if args.use_safetensors:
        save_file(unet, args.output_path + "/unet.safetensors")
        save_file(vae, args.output_path + "/vae.safetensors")
        save_file(text1, args.output_path + "/text1.safetensors", metadata={"format": "pt"})
        save_file(text2, args.output_path + "/text2.safetensors", metadata={"format": "pt"})
    else:
        torch.save({"state_dict": unet}, args.output_path + "/unet.pth")
        torch.save({"state_dict": vae}, args.output_path + "/vae.pth")
        torch.save({"state_dict": text1}, args.output_path + "/text1.pth")
        torch.save({"state_dict": text2}, args.output_path + "/text2.pth")
