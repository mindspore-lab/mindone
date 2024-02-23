import argparse

import torch
from safetensors.torch import  save_file


def get_map(key_file,value_file):
    key_map = {}
    with open(key_file, 'r') as key_file, open(value_file, 'r') as value_file:
        keys = key_file.readlines()
        values = value_file.readlines()

        for key, value in zip(keys, values):
            key = key.strip()
            value = value.strip()
            key_map[key] = value
    return key_map



def convert(state_dict,module_map):
    mapping = module_map
    new_state_dict = {hf_name: state_dict[sd_name] for sd_name,hf_name in mapping.items()}
    return new_state_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--use_safetensors", action="store_true", help="Save weights use safetensors, default is ckpt.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    args = parser.parse_args()

    key_file = 'torch_key_unet.yaml'
    value_file = 'hf_key_unet.yaml'
    unet_map = get_map('torch_key_unet.yaml', 'hf_key_unet.yaml')
    vae_map = get_map('torch_key_vae.yaml', 'hf_key_vae.yaml')
    text1_map = get_map('torch_key_text1.yaml', 'hf_key_text1.yaml')
    text2_map = get_map('torch_key_text2.yaml', 'hf_key_text2.yaml')

    unet = convert(args.model_path,unet_map)
    vae = convert(args.model_path,vae_map)
    text1 = convert(args.model_path,text1_map)
    text2 = convert(args.model_path,text2_map)

    if args.half:
        unet = {k: v.half() for k, v in unet.items()}
        vae = {k: v.half() for k, v in vae.items()}
        text1 = {k: v.half() for k, v in text1.items()}
        text2 = {k: v.half() for k, v in text2.items()}

    if args.use_safetensors:
        save_file(unet, args.output_path+"/unet.safetensors")
        save_file(vae, args.output_path + "/vae.safetensors")
        save_file(text1, args.output_path + "/text1.safetensors")
        save_file(text2, args.output_path + "/text2.safetensors")
    else:
        torch.save({"state_dict": unet}, args.output_path+"/unet.pth")
        torch.save({"state_dict": vae}, args.output_path + "/vae.pth")
        torch.save({"state_dict": text1}, args.output_path + "/text1.pth")
        torch.save({"state_dict": text2}, args.output_path + "/text2.pth")
