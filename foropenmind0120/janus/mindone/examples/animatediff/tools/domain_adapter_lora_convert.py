import argparse
import os

import torch

import mindspore as ms


# some name mapping rules, which are found from param name printing observation (case by case)
def _param_info_parser(param_info_txt):
    pnames = []
    pshapes = []
    ptypes = []
    with open(param_info_txt) as fp:
        for line in fp:
            pname, pshape, ptype = line.strip().split("#")
            pnames.append(pname)
            if "torch.Size" in pshape:
                pshape = pshape.split("[")[-1].split("]")[0].split(",")
            else:
                pshape = pshape.split("(")[-1].split(")")[0].split(",")
            pshape = [x for x in pshape if len(x) > 0]
            if len(pshape) > 0:
                pshape = [int(x) for x in pshape]
            else:
                raise ValueError("Incorrect shape")
            pshapes.append(pshape)
            ptypes.append(ptype)
    return pnames, pshapes, ptypes


def parse_pnames_and_pshapes(pnames, pshapes, filter_func):
    new_pnames, new_pshapes = [], []
    for i, name in enumerate(pnames):
        if filter_func is None or (filter_func is not None and filter_func(name)):
            new_pnames.append(name)
            new_pshapes.append(pshapes[i])
    return new_pnames, new_pshapes


def get_pt_ms_info(pt_pinfo_path, ms_pinfo_path, pt_filter_func=None, ms_filter_func=None):
    pt_pnames, pt_pshapes, _ = _param_info_parser(pt_pinfo_path)
    ms_pnames, ms_pshapes, _ = _param_info_parser(ms_pinfo_path)
    # filter pt pnames pshapes to exclude up_blocks
    pt_pnames, pt_pshapes = parse_pnames_and_pshapes(pt_pnames, pt_pshapes, pt_filter_func)
    # filter ms pnames pshapes to exclude output_blocks (up_blocks)
    ms_pnames, ms_pshapes = parse_pnames_and_pshapes(ms_pnames, ms_pshapes, ms_filter_func)

    assert len(ms_pnames) == len(pt_pnames), "param names parse error! check the param files."
    return pt_pnames, pt_pshapes, ms_pnames, ms_pshapes


def convert_pt_ms_state_dict(
    lora_ckpt_path,
    pt_pinfo_path,
    ms_pinfo_path,
    output_dir="./",
):
    """
    e.g.
    pt lora param: down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.processor.to_q_lora.down.weight
            {attn_layer}.processor.{to_q/k/v/out}_lora.{down/up}.weight
            = {attn_layer}{lora_postfix}
    pt attn param: down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.to_out.0.weight
            {attn_layer}.{to_q/k/v/out.0}.weight

    ms attn param : model.diffusion_model.input_blocks.1.2.temporal_transformer.transformer_blocks.0.attention_blocks.1.to_out.0.weight
            {ms_attn_layer}.{to_q/k/v/out.0}.weight
    ms lora param: model.diffusion_model.input_blocks.1.2.temporal_transformer.transformer_blocks.0.attention_blocks.0.processor.to_out_lora.down.weight
            = {ms_attn_layer}{lora_postfix}
            = {ms_attn_layer}.processor.{to_q/k/v/out}_lora.{down/up}.weight

    prev ms sdv2 lora naming: to_q.lora_down.weight

    Each motion moduel has 2 attentin layers
    """
    pt_lora_dict = torch.load(lora_ckpt_path, map_location=torch.device("cpu"))
    pt_lora_dict = pt_lora_dict["state_dict"] if "state_dict" in pt_lora_dict else pt_lora_dict
    pt_lora_dict.pop("animatediff_config", "")
    pt_lora_pnames = list(pt_lora_dict.keys())

    pt_pnames, pt_pshapes, _ = _param_info_parser(pt_pinfo_path)
    ms_pnames, ms_pshapes, _ = _param_info_parser(ms_pinfo_path)

    # filter pt names to only include lora pnames
    def filter_lora_params(name):
        if any([x in name for x in ["to_q", "to_k", "to_v", "to_out"]]) and ".weight" in name:
            return True
        else:
            return False

    pt_pnames, pt_pshapes, ms_pnames, ms_pshapes = get_pt_ms_info(
        pt_pinfo_path,
        ms_pinfo_path,
        pt_filter_func=filter_lora_params,
        ms_filter_func=lambda x: filter_lora_params(x) and "model.diffusion_model." in x,
    )

    assert (
        len(ms_pnames) == len(pt_pnames) and len(ms_pnames) == len(pt_lora_pnames) // 2
    ), "param names parse error! check the param files."

    print("Converting ", lora_ckpt_path)
    target_data = []
    for pt_lora_pname in pt_lora_pnames:
        # find the ms param prefix for current lora layer

        # 1. get attention dense layer weight name
        pt_attn_pname = (
            pt_lora_pname.replace("processor.", "").replace("_lora", "").replace("down.", "").replace("up.", "")
        )
        pt_attn_pname = pt_attn_pname.replace("to_out.", "to_out.0.")
        lora_postfix = pt_lora_pname[pt_lora_pname.find(".processor.") :]
        qkvo = pt_lora_pname.split(".")[-3].replace("_lora", "")  # to_q, to_k, to_v, to_out
        tar_idx = pt_pnames.index(pt_attn_pname)

        ms_attn_pname = ms_pnames[tar_idx]
        ms_attn_layer = ms_attn_pname[: ms_attn_pname.find("." + qkvo)]
        ms_lora_pname = ms_attn_layer + lora_postfix

        print("PT Lora: ", pt_lora_pname)
        print("MS Lora: ", ms_lora_pname)

        # check param order and shape
        assert (
            pt_pshapes[tar_idx] == ms_pshapes[tar_idx]
        ), f"pt and ms param shape mismatch, please check order correspondance. pt: {pt_attn_pname}, ms: {ms_attn_pname} "

        target_data.append({"name": ms_lora_pname, "data": ms.Tensor(pt_lora_dict[pt_lora_pname].detach().numpy())})

    ms_path = os.path.join(output_dir, os.path.basename(lora_ckpt_path))
    if ms_path == lora_ckpt_path:
        raise ValueError
    else:
        ms.save_checkpoint(target_data, ms_path)

    print("ms ckpt saved in", ms_path)
    return ms_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None, help="path to torch lora path")
    parser.add_argument(
        "--target",
        type=str,
        default="../models/domain_adapter_lora",
        help="target dir to save the converted checkpoint",
    )
    parser.add_argument("--pt_params_naming", type=str, default="diffuser", help="the naming method of torch params")
    args = parser.parse_args()

    src_pt_ckpt = args.src
    if args.pt_params_naming == "diffuser":
        torch_names_txt = "../../stable_diffusion_v2/tools/model_conversion/diffusers_unet_v2.txt"
    else:
        raise ValueError(f"Torch naming method {args.pt_params_naming} is not supported.")

    ms_names_txt = "../../stable_diffusion_v2/tools/model_conversion/ms_names_v2.txt"

    output_dir = args.target

    ms_path = convert_pt_ms_state_dict(src_pt_ckpt, torch_names_txt, ms_names_txt, output_dir=output_dir)
