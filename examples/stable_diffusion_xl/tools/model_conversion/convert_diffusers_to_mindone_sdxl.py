# Script step 1:
# for converting a HF Diffusers saved pipeline to a Stable Diffusion checkpoint.
# *Only* converts the UNet, VAE, and Text Encoder.
# Does not convert optimizer state or any other thing.

# Script step 2:
# convert Stable Diffusion checkpoint(torch) to MindOne Stable Diffusion checkpoint(mindspore)
# *Only* converts the UNet, VAE, and Text Encoder as above

# Script step 3：
# insert Stable Diffusion checkpoint(mindspore) to sd_xl_base_1.0_ms.ckpt, then get a new mindspore checkpoint


import argparse
import copy
import os.path as osp
import re

import torch
from safetensors.torch import load_file, save_file

import mindspore as ms
from mindspore import Tensor

# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    # the following are for sdxl
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(3):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i > 0:
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(4):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        if i < 2:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))
unet_conversion_map_layer.append(("output_blocks.2.2.conv.", "output_blocks.2.1.conv."))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))
for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))


def convert_unet_state_dict(unet_state_dict):
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {sd_name: unet_state_dict[hf_name] for hf_name, sd_name in mapping.items()}
    return new_state_dict


# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
    # down_blocks have two resnets
    for j in range(2):
        hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
        sd_down_prefix = f"encoder.down.{i}.block.{j}."
        vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
        sd_downsample_prefix = f"down.{i}.downsample."
        vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"up.{3-i}.upsample."
        vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

    # up_blocks have three resnets
    # also, up blocks in hf are numbered in reverse from sd
    for j in range(3):
        hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
        sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
        vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{i}."
    sd_mid_res_prefix = f"mid.block_{i+1}."
    vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))


vae_conversion_map_attn = [
    # (stable-diffusion, HF Diffusers)
    ("norm.", "group_norm."),
    # the following are for SDXL
    ("q.", "to_q."),
    ("k.", "to_k."),
    ("v.", "to_v."),
    ("proj_out.", "to_out.0."),
]


def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    return w.reshape(*w.shape, 1, 1)


def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                print(f"Reshaping {k} for SD format")
                new_state_dict[k] = reshape_weight_for_sd(v)
    return new_state_dict


# =========================#
# Text Encoder Conversion #
# =========================#


textenc_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("transformer.resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "text_model.final_layer_norm."),
    ("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
    ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
]
protected = {re.escape(x[1]): x[0] for x in textenc_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))

# Ordering is from https://github.com/pytorch/pytorch/blob/master/test/cpp/api/modules.cpp
code2idx = {"q": 0, "k": 1, "v": 2}


def convert_openclip_text_enc_state_dict(text_enc_dict):
    new_state_dict = {}
    capture_qkv_weight = {}
    capture_qkv_bias = {}
    for k, v in text_enc_dict.items():
        if (
            k.endswith(".self_attn.q_proj.weight")
            or k.endswith(".self_attn.k_proj.weight")
            or k.endswith(".self_attn.v_proj.weight")
        ):
            k_pre = k[: -len(".q_proj.weight")]
            k_code = k[-len("q_proj.weight")]
            if k_pre not in capture_qkv_weight:
                capture_qkv_weight[k_pre] = [None, None, None]
            capture_qkv_weight[k_pre][code2idx[k_code]] = v
            continue

        if (
            k.endswith(".self_attn.q_proj.bias")
            or k.endswith(".self_attn.k_proj.bias")
            or k.endswith(".self_attn.v_proj.bias")
        ):
            k_pre = k[: -len(".q_proj.bias")]
            k_code = k[-len("q_proj.bias")]
            if k_pre not in capture_qkv_bias:
                capture_qkv_bias[k_pre] = [None, None, None]
            capture_qkv_bias[k_pre][code2idx[k_code]] = v
            continue

        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k)
        new_state_dict[relabelled_key] = v

    for k_pre, tensors in capture_qkv_weight.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_weight"] = torch.cat(tensors)

    for k_pre, tensors in capture_qkv_bias.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_bias"] = torch.cat(tensors)

    return new_state_dict


def convert_openai_text_enc_state_dict(text_enc_dict):
    return text_enc_dict


def convert_weight(state_dict, msname):
    key_torch = list(state_dict["state_dict"].keys())
    key_ms = copy.deepcopy(key_torch)
    for i in range(len(key_torch)):
        if ("norm" in key_torch[i]) or ("ln_" in key_torch[i]) or ("model.diffusion_model.out.0." in key_torch[i]):
            if "weight" in key_torch[i]:
                key_ms[i] = key_ms[i][:-6] + "gamma"
            if "bias" in key_torch[i]:
                key_ms[i] = key_ms[i][:-4] + "beta"
        pattern1 = r"model\.diffusion_model\.(input_blocks|middle_block|output_blocks)\.[0-8]\.0\.(in_layers|out_layers)\.0\.(weight|bias)"
        pattern2 = r"model\.diffusion_model\.middle_block\.[02]\.(in_layers|out_layers)\.0\.(weight|bias)"
        if re.match(pattern1, key_torch[i]) or re.match(pattern2, key_torch[i]):
            if "weight" in key_torch[i]:
                key_ms[i] = key_ms[i][:-6] + "gamma"
            if "bias" in key_torch[i]:
                key_ms[i] = key_ms[i][:-4] + "beta"
        if "embedding.weight" in key_torch[i]:
            key_ms[i] = key_ms[i][:-6] + "embedding_table"
        if "conditioner.embedders.1.model.text_projection.weight" in key_torch[i]:
            key_ms[i] = "conditioner.embedders.1.model.text_projection"
    newckpt = []
    ms_key = []
    for i in range(len(key_torch)):
        kt, kms = key_torch[i], key_ms[i]
        vms = Tensor(state_dict["state_dict"][kt].numpy(), ms.float32)
        if "conditioner.embedders.1.model.text_projection" == kms:
            print(
                f"[Attention] {kms} is called differently in MindONE and Diffusers, so it is transposed in converting"
            )
            vms = ms.ops.transpose(vms, (1, 0))
        newckpt.append({"name": kms, "data": vms})
        ms_key.append(kms)
    ms.save_checkpoint(newckpt, msname)
    print("convert Stable Diffusion checkpoint(torch) to MindOne Stable Diffusion checkpoint(mindspore) success!")
    return ms_key


def merge_weight(part_ckpt, base_ckpt):
    part = ms.load_checkpoint(part_ckpt)
    al = ms.load_checkpoint(base_ckpt)
    partkey = list(part.keys())
    alkey = list(al.keys())
    newckpt = []
    for i in range(len(al)):
        key = alkey[i]
        if key in partkey:
            newckpt.append({"name": key, "data": part[key]})
        else:
            newckpt.append({"name": key, "data": al[key]})
    ms.save_checkpoint(newckpt, args.output_path)
    print("insert Stable Diffusion checkpoint(mindspore) to sd_xl_base_1.0_ms.ckpt success!")


def compare_missing_key(key_list, reverse=False, ms_key_base_path="mindspore_key_base.yaml"):
    # get the sdxl base ckpt's key
    base_key = []
    with open(ms_key_base_path, "r", encoding="utf-8") as file:
        for line in file:
            key = line.split(":", 1)[0].strip()
            base_key.append(key)
    base_key = set(base_key)
    key_list = set(key_list)
    if not reverse:
        difference = base_key - key_list
    else:
        difference = key_list - base_key
    return list(difference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    parser.add_argument("--use_safetensors", action="store_true", help="Save weights use safetensors, default is ckpt.")
    parser.add_argument(
        "--unet_path",
        default="./unet/diffusion_pytorch_model.fp16.safetensors",
        type=str,
        help="Path to the unet model.",
    )
    parser.add_argument(
        "--vae_path", default="./vae/diffusion_pytorch_model.fp16.safetensors", type=str, help="Path to the vae model."
    )
    parser.add_argument(
        "--text_encoder_path",
        default="./text_encoder/model.fp16.safetensors",
        type=str,
        help="Path to the text_encoder model.",
    )
    parser.add_argument(
        "--text_encoder_2_path",
        default="./text_encoder_2/model.fp16.safetensors",
        type=str,
        help="Path to the text_encoder_2 model.",
    )
    parser.add_argument(
        "--ms_key_base_path",
        default="./mindspore_key_base.yaml",
        type=str,
        help="Path to the mindspore_key_base.yaml.",
    )
    parser.add_argument("--sdxl_base_ckpt", default=None, type=str, help="Path to the sd_xl_base model.")
    parser.add_argument(
        "--save_mindspore",
        default=True,
        type=bool,
        help="Save mindspore weights, default is mindspore.",
    )
    parser.add_argument(
        "--save_torch",
        default=False,
        type=bool,
        help="Save torch weights using .pth format, default is mindspore.",
    )
    parser.add_argument(
        "--save_safetensor",
        default=False,
        type=bool,
        help="Save torch weights using .safetensor format, default is mindspore.",
    )

    args = parser.parse_args()

    assert args.output_path is not None, "Must provide a checkpoint path!"

    unet_state_dict = {}
    vae_state_dict = {}
    text_enc_dict = {}
    text_enc_2_dict = {}

    if osp.exists(args.unet_path):
        if args.unet_path.endswith(".bin"):
            unet_state_dict = torch.load(args.unet_path, map_location="cpu")
        else:
            unet_state_dict = load_file(args.unet_path, device="cpu")
        print("load unet from unet_path: ", args.unet_path)
    else:
        print("unet_path is not valid, please double-check it!")

    if osp.exists(args.vae_path):
        if args.vae_path.endswith(".bin"):
            vae_state_dict = torch.load(args.vae_path, map_location="cpu")
        else:
            vae_state_dict = load_file(args.vae_path, device="cpu")
        print("load vae from vae_path ", args.vae_path)
    else:
        print("vae_path is not valid, please double-check it!")

    if osp.exists(args.text_encoder_path):
        if args.text_encoder_path.endswith(".bin"):
            text_enc_dict = torch.load(args.text_encoder_path, map_location="cpu")
        else:
            text_enc_dict = load_file(args.text_encoder_path, device="cpu")
        print("load text encoder from text_enc_path ", args.text_encoder_path)
    else:
        print("text_encoder_path is not valid, please double-check it!")

    if osp.exists(args.text_encoder_2_path):
        if args.text_encoder_2_path.endswith(".bin"):
            text_enc_2_dict = torch.load(args.text_encoder_2_path, map_location="cpu")
        else:
            text_enc_2_dict = load_file(args.text_encoder_2_path, device="cpu")
        print("load text encoder 2 from text_enc_2_path ", args.text_encoder_2_path)
    else:
        print("text_encoder_2_path is not valid, please double-check it!")

    # Convert the UNet model
    if unet_state_dict != {}:
        unet_state_dict = convert_unet_state_dict(unet_state_dict)
        unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}
    # Convert the VAE model
    if vae_state_dict != {}:
        vae_state_dict = convert_vae_state_dict(vae_state_dict)
        vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}
    if text_enc_dict != {}:
        text_enc_dict = convert_openai_text_enc_state_dict(text_enc_dict)
        text_enc_dict = {"conditioner.embedders.0.transformer." + k: v for k, v in text_enc_dict.items()}
    if text_enc_2_dict != {}:
        text_enc_2_dict = convert_openclip_text_enc_state_dict(text_enc_2_dict)
        text_enc_2_dict = {"conditioner.embedders.1.model." + k: v for k, v in text_enc_2_dict.items()}

    # Put together new checkpoint
    state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict, **text_enc_2_dict}

    if args.half:
        state_dict = {k: v.half() for k, v in state_dict.items()}

    if args.output_path.endswith(".ckpt"):
        save_safetensor_path = args.output_path[:-5] + ".safetensors"
        save_torch_path = args.output_path[:-5] + ".pth"
        save_mindspore_path = args.output_path
    elif args.output_path.endswith(".pth"):
        save_safetensor_path = args.output_path[:-4] + ".safetensors"
        save_torch_path = args.output_path
        save_mindspore_path = args.output_path[:-4] + ".ckpt"
    elif args.output_path.endswith(".safetensors"):
        save_safetensor_path = args.output_path
        save_torch_path = args.output_path[:-12] + ".pth"
        save_mindspore_path = args.output_path[:-12] + ".ckpt"
    else:
        save_safetensor_path = args.output_path + ".safetensors"
        save_torch_path = args.output_path + ".pth"
        save_mindspore_path = args.output_path + ".ckpt"
    state_dict = {"state_dict": state_dict}
    if args.save_safetensor:
        save_file(state_dict, save_safetensor_path)
    if args.save_torch:
        torch.save(state_dict, save_torch_path)
    if args.save_mindspore:
        # Convert the torch ckpt to mindspore ckpt and return mindspore key list
        key_list = convert_weight(state_dict, save_mindspore_path)
        exist_ms_key_base_path = osp.exists(args.ms_key_base_path)

        if exist_ms_key_base_path:
            with open(args.ms_key_base_path, "r") as file:
                line_count = len(file.readlines())
        else:
            line_count = 2514

        if not args.sdxl_base_ckpt:
            print("you did not supply the 'sdxl_base_ckpt' argument; hence, the insertion operation was not executed.")

        # If you have obtained all the keys, you do not need to run the insertion operation
        elif (
            len(key_list) == line_count
            and exist_ms_key_base_path
            and len(compare_missing_key(key_list, ms_key_base_path=args.ms_key_base_path)) == 0
        ):
            print("You have obtained all the keys, hence, the insertion operation was not executed.")

        else:
            print("The MindSpore checkpoint (.ckpt) file that we have obtained contains ", str(len(key_list)), "keys.")
            print("The sdxl base checkpoint contains ", str(line_count), "keys.")
            if exist_ms_key_base_path:
                missing_key_list = compare_missing_key(key_list, ms_key_base_path=args.ms_key_base_path)
                print("There are ", len(missing_key_list), " Missing Parameters.")
                print(
                    "Missing Parameters indicate parameters that are included in the base ckpt but not included in the current ckpt."
                )
                print("The first 20 missing parameters are: ", str(missing_key_list[:20]))
                more_key_list = compare_missing_key(key_list, True, args.ms_key_base_path)
                print("There are ", len(more_key_list), " More Parameters.")
                print(
                    "More Parameters indicate parameters that are included in the current ckpt but not included in the base ckpt."
                )
                print("The first 20 more parameters are: ", str(more_key_list[:20]))

            print(
                "......Begin to Integrate the retrieved MindSpore checkpoint into the sdxl base model checkpoint......"
            )
            merge_weight(args.output_path, args.sdxl_base_ckpt)
