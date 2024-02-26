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


# convert based on order correspondance
def convert_pt_ms_state_dict(
    ckpt_path,
    pt_unet_pinfo_path,
    ms_unet_pinfo_path,
    pt_mm_pinfo_path,
    ms_mm_pinfo_path,
    output_dir="./",
):
    """ """
    pt_state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    pt_state_dict = pt_state_dict["state_dict"] if "state_dict" in pt_state_dict else pt_state_dict
    pt_state_dict.pop("animatediff_config", "")
    pt_dict_pnames = list(pt_state_dict.keys())

    # filter pt names to only include the input blocks and middle blocks (down blocks and middle blocks)
    def filter_params(name):
        if ".output_blocks." not in name and "up_blocks." not in name:
            return True
        else:
            return False

    pt_unet_pnames, pt_unet_pshapes, ms_unet_pnames, ms_unet_pshapes = get_pt_ms_info(
        pt_unet_pinfo_path,
        ms_unet_pinfo_path,
        pt_filter_func=filter_params,
        ms_filter_func=lambda x: filter_params(x) and "model.diffusion_model." in x,
    )
    pt_mm_pnames, pt_mm_pshapes, ms_mm_pnames, ms_mm_pshapes = get_pt_ms_info(
        pt_mm_pinfo_path,
        ms_mm_pinfo_path,
        pt_filter_func=filter_params,
        ms_filter_func=filter_params,
    )
    print("Converting ", ckpt_path)
    target_data = []
    for pt_dict_pname in pt_dict_pnames:
        if "controlnet" not in pt_dict_pname and "temporal_transformer" not in pt_dict_pname:
            # unet original param
            tar_idx = pt_unet_pnames.index(pt_dict_pname)
            ms_dict_pname = ms_unet_pnames[tar_idx]
            # check param order and shape
            assert (
                pt_unet_pshapes[tar_idx] == ms_unet_pshapes[tar_idx]
            ), f"pt and ms param shape mismatch, please check order correspondance. pt: {pt_dict_pname}, ms: {ms_dict_pname} "
            ms_dict_pname = (
                "model.diffusion_model.controlnet." + ms_dict_pname[len("model.diffusion_model.") :]
            )  # remove "model.diffusion_model."
        elif "controlnet" in pt_dict_pname:
            # controlnet params are all conv layers
            # change conv layer name
            ms_dict_pname = pt_dict_pname.replace(".weight", ".conv.weight").replace(".bias", ".conv.bias")
            # change blocks names
            ms_dict_pname = ms_dict_pname.replace("controlnet_down_blocks", "controlnet_input_blocks").replace(
                "controlnet_mid_block", "controlnet_middle_block"
            )
            ms_dict_pname = "model.diffusion_model.controlnet." + ms_dict_pname
        elif "temporal_transformer" in pt_dict_pname:
            # motion module params
            tar_idx = pt_mm_pnames.index(pt_dict_pname)
            ms_dict_pname = ms_mm_pnames[tar_idx]
            # check param order and shape
            assert (
                pt_mm_pshapes[tar_idx] == ms_mm_pshapes[tar_idx]
            ), f"pt and ms param shape mismatch, please check order correspondance. pt: {pt_dict_pname}, ms: {ms_dict_pname} "
            ms_dict_pname = (
                "model.diffusion_model.controlnet." + ms_dict_pname[len("model.diffusion_model.") :]
            )  # remove "model.diffusion_model."
        else:
            raise ValueError(f"incorrect pname {pt_dict_pname}")
        print("PT Param Name: ", pt_dict_pname)
        print("MS Param Name: ", ms_dict_pname)
        target_data.append({"name": ms_dict_pname, "data": ms.Tensor(pt_state_dict[pt_dict_pname].detach().numpy())})

    ms_path = os.path.join(output_dir, os.path.basename(ckpt_path))
    if ms_path == ckpt_path:
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
        pt_unet_names_txt = "../../stable_diffusion_v2/tools/model_conversion/diffusers_unet_v2.txt"
        pt_mm_names_txt = "torch_mm_params.txt"
    else:
        raise ValueError(f"Torch naming method {args.pt_params_naming} is not supported.")

    ms_unet_names_txt = "../../stable_diffusion_v2/tools/model_conversion/ms_names_v2.txt"
    ms_mm_names_txt = "ms_mm_params.txt"

    output_dir = args.target

    ms_path = convert_pt_ms_state_dict(
        src_pt_ckpt, pt_unet_names_txt, ms_unet_names_txt, pt_mm_names_txt, ms_mm_names_txt, output_dir=output_dir
    )
