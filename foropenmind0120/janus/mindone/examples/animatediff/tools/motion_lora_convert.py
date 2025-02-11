import argparse
import os

import torch

import mindspore as ms


# convert based on order correspondance
def convert_pt_ms_state_dict(
    lora_ckpt_path,
    pt_pinfo_path,
    ms_pinfo_path,
    add_ldm_prefix=False,
    ldm_prefix="model.diffusion_model.",
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
    pt_sd = torch.load(lora_ckpt_path, map_location=torch.device("cpu"))["state_dict"]
    pt_lora_pnames = list(pt_sd.keys())

    # some name mapping rules, which are found from param name printing observation (case by case)
    def _param_info_parser(param_info_txt):
        pnames = []
        pshapes = []
        ptypes = []
        with open(param_info_txt) as fp:
            for line in fp:
                pname, pshape, ptype = line.strip().split("#")
                pnames.append(pname)
                pshapes.append(pshape)
                ptypes.append(ptype)
        return pnames, pshapes, ptypes

    pt_pnames, pt_pshapes, pt_ptypes = _param_info_parser(pt_pinfo_path)
    ms_pnames, ms_pshapes, ms_ptypes = _param_info_parser(ms_pinfo_path)

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

        target_data.append({"name": ms_lora_pname, "data": ms.Tensor(pt_sd[pt_lora_pname].detach().numpy())})

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
        "--target", type=str, default="../models/motion_lora", help="target dir to save the converted checkpoint"
    )
    args = parser.parse_args()

    src_pt_ckpt = args.src
    torch_names_txt = "./torch_mm_params.txt"
    ms_names_txt = "./ms_mm_params.txt"

    output_dir = args.target

    ms_path = convert_pt_ms_state_dict(src_pt_ckpt, torch_names_txt, ms_names_txt, output_dir=output_dir)
