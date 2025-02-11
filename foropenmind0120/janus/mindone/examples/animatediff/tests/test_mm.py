import json
import sys

import numpy as np
import torch

import mindspore as ms

sys.path.append(".")
from ldm.modules.diffusionmodules.motion_module import VanillaTemporalModule
from ldm.modules.diffusionmodules.unet3d import rearrange_in, rearrange_out


def load_pt_rt_states(mm_idx, folder):
    # read args, inputs, output stored in torch real-time run
    args_fp = folder + f"mm{mm_idx}_args.json"
    inps_fp = folder + f"mm{mm_idx}_inputs.npz"
    outs_fp = folder + f"mm{mm_idx}_outputs.npz"

    with open(args_fp, "r") as fp:
        args = json.load(fp)

    inps = np.load(inps_fp, allow_pickle=True)
    inps_dict = {}
    for name in inps:
        if inps[name].ndim > 0:
            inps_dict[name] = inps[name]
            # print(name, type(inps[name]), inps[name].ndim)
        else:
            inps_dict[name] = None

    output = np.load(outs_fp, allow_pickle=True)["output"]

    return args, inps_dict, output


def get_mm_param_prefix_indices(
    pt_mm_path, prefix_order=["down_blocks", "mid_block", "up_blocks"], num_params_per_mm=588 // 21
):
    pt_mm_sd = torch.load(pt_mm_path)

    mm_prefix = []
    mm_prefix_shape = []
    mm_param_start_idx = []
    for tar_prefix in prefix_order:
        mm_param_idx = 0
        for idx, pname in enumerate(pt_mm_sd):
            # print(f"{pname}#{tuple(pt_mm_sd[pname].size())}#{pt_mm_sd[pname].dtype}")
            if mm_param_idx % num_params_per_mm == 0:
                if pname.startswith(tar_prefix):
                    # print(pname)
                    end_char_idx = pname.find("motion_modules.") + len("motion_modules.0")
                    prefix = pname[:end_char_idx]
                    mm_prefix.append(prefix)
                    mm_prefix_shape.append(tuple(pt_mm_sd[pname].shape))
                    mm_param_start_idx.append(idx)
            mm_param_idx += 1

    return mm_prefix, mm_prefix_shape, mm_param_start_idx


def ms_extract_mm_params_by_index(
    mm_ckpt_path, ms_pinfo_path, start_idx=0, num_params=588 // 21, remove_prefix=True, zero_initialize=True
):
    # ms_pinfo_path, the param names have been mapped to PT in order
    mm_sd = ms.load_checkpoint(mm_ckpt_path)  # TODO: change to use pnames.txt
    # all_mm_pnames = list(mm_sd.keys())  # order not mapped to pt mm

    param_info = []
    ordered_pnames = []
    with open(ms_pinfo_path) as fp:
        for line in fp:
            pname, pshape, ptype = line.strip().split("#")
            param_info.append((pname, pshape, ptype))
            ordered_pnames.append(pname)

    new_sd = {}
    for i in range(start_idx, start_idx + num_params):
        pname = ordered_pnames[i]
        if remove_prefix:
            # print('D--: ', pname)
            # ------------------------------ ATTENTION !!!!!!!!!!!!!!!!!!!!!!! ------------------
            # when zero_initialize=True, temporal_transformer prefix is somehow removed in proj_out layer in the mm Cell
            if zero_initialize and "proj_out." in pname:
                new_pname = pname[pname.find("proj_out.") :]
            else:
                new_pname = pname[pname.find("temporal_transformer.") :]
        else:
            new_pname = pname
        new_sd[new_pname] = mm_sd[pname]

    return new_sd


def measure_error(src, tar, eps=1e-7):
    src = np.array(src)
    tar = np.array(tar)
    ae = np.abs(src - tar)
    mean_ae = ae.mean()
    max_ae = ae.max()

    re = np.abs((src - tar) / (np.abs(tar) + eps))
    max_re = np.max(re)
    mean_re = np.mean(re)

    return mean_re, max_re, mean_ae, max_ae


def test():
    ms.set_context(mode=0)
    # 1. load mm args, inputs, outputs stored from real-time running
    # 2. to load the correct weights, need to map mm index to param indices and param prefix (e.g. down_blocks.0.motion_module.1)

    pt_inspect_folder = "pt_mm_inspect/"
    pt_mm_path = "/home/mindocr/yx/AnimateDiff/models/Motion_Module/animatediff/mm_sd_v15_v2.ckpt"

    ms_mm_path = "models/Motion_Module/animatediff/mm_sd_v15_v2.ckpt"
    ms_pinfo_path = "tools/ms_mm_params.txt"

    num_mm = 21
    num_params_per_mm = 28

    mm_prefix, mm_prefix_shape, mm_param_start_idx = get_mm_param_prefix_indices(
        pt_mm_path, num_params_per_mm=num_params_per_mm
    )

    # mm_idx = 0
    for mm_idx in range(num_mm):
        args, inps_dict, output = load_pt_rt_states(mm_idx, folder=pt_inspect_folder)
        # print(inps_dict.keys())

        assert args["in_channels"] == inps_dict["input_tensor"].shape[1]
        in_ch = args["in_channels"]
        ftr_hw = inps_dict["input_tensor"].shape[-1]
        print(
            f"Idx: {mm_idx}, c: {in_ch}, hw: {ftr_hw}, param_prefix: {mm_prefix[mm_idx]}",
            "param_start_idx: {mm_param_start_idx[mm_idx]}, param_shape: {mm_prefix_shape[mm_idx]}",
        )

        # create mm
        cur_mm = VanillaTemporalModule(**args)
        cur_mm.set_train(False)

        # load mm weights
        cur_state_dict = ms_extract_mm_params_by_index(
            ms_mm_path, ms_pinfo_path, mm_param_start_idx[mm_idx], num_params=num_params_per_mm
        )
        pnl, nnl = ms.load_param_into_net(cur_mm, cur_state_dict)
        print("pnl", pnl)
        print("nnl", nnl)

        # infer
        f = inps_dict["input_tensor"].shape[2]
        # print("D-- Frames: ", f)
        for inp in inps_dict:
            if inps_dict[inp] is not None:
                inps_dict[inp] = ms.Tensor(inps_dict[inp])

                # for input_tensor in PT is (b, c, f, h, w), rearrange to (b*f c h w)
                if inp == "input_tensor":
                    inps_dict[inp] = rearrange_in(inps_dict[inp])
            else:
                inps_dict[inp] = None

        inps_dict["video_length"] = f

        cur_output = cur_mm(**inps_dict)
        # reshape (b*f c h w) -> (b c f h w)
        cur_output = rearrange_out(cur_output, f=f)
        cur_output = cur_output.asnumpy()

        mean_re, max_re, mean_ae, max_ae = measure_error(cur_output, output)
        print(f"mm {mm_idx}\t{mean_re}\t{max_re}\t{mean_ae}\t{max_ae}")


if __name__ == "__main__":
    test()
