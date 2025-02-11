import torch
from fire import Fire

import mindspore as ms


def get_ms_param_list(ms_ckpt, save_fp="ms_lora_params.txt"):
    # input: ms checkoint path
    sd = ms.load_checkpoint(ms_ckpt)

    with open(save_fp, "w") as fp:
        for pname in sd:
            # print("{}#{}#{}".format(pname, sd[pname].shape, sd[pname].dtype))
            fp.write("{}#{}#{}\n".format(pname, sd[pname].shape, sd[pname].dtype))
    print("Get ms checkpoint param list in ", save_fp)


def get_pt_param_list(pt_ckpt, save_fp="pt_lora_params.txt"):
    from safetensors import safe_open

    sd = {}
    if pt_ckpt.endswith(".safetensors"):
        with safe_open(pt_ckpt, framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    else:
        sd = torch.load(pt_ckpt, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]

    with open(save_fp, "w") as fp:
        for k in sd:
            # print("{}#{}#{}".format(k, sd[k].shape, sd[k].dtype))
            # print("{}#{}".format(k, tuple(sd[k].shape)))
            fp.write("{}#{}#{}\n".format(k, sd[k].shape, sd[k].dtype))

    print("Get PT checkpoint param list in ", save_fp)


def get_ordered_param_list(ori_pt_lora_param_list, save=True):
    # re-order pt param list
    txt_fp = ori_pt_lora_param_list
    pl = []
    with open(txt_fp) as fp:
        for line in fp:
            pl.append(line.strip())
    num_params = len(pl)

    # k, out, q, v -> q, k, v, out
    ordered_pl = ["tbc"] * num_params
    for i in range(num_params // 8):
        ordered_pl[i * 8 + 0] = pl[i * 8 + 2 * 2 + 0]
        ordered_pl[i * 8 + 1] = pl[i * 8 + 2 * 2 + 1]

        ordered_pl[i * 8 + 2] = pl[i * 8 + 0 * 2 + 0]
        ordered_pl[i * 8 + 3] = pl[i * 8 + 0 * 2 + 1]

        ordered_pl[i * 8 + 4] = pl[i * 8 + 3 * 2 + 0]
        ordered_pl[i * 8 + 5] = pl[i * 8 + 3 * 2 + 1]

        ordered_pl[i * 8 + 6] = pl[i * 8 + 1 * 2 + 0]
        ordered_pl[i * 8 + 7] = pl[i * 8 + 1 * 2 + 1]

    if save:
        with open(txt_fp.replace(".txt", "_ordered.txt"), "w") as fp:
            for line in ordered_pl:
                fp.write(line + "\n")

    return ordered_pl


def convert(ms_ckpt, ms_plist="ms_lora_params.txt", pt_plist="pt_lora_params_ordered.txt", pt_ckpt_save_path=None):
    with open(ms_plist) as fp_ms:
        ms_pnames = fp_ms.readlines()
    with open(pt_plist) as fp_pt:
        pt_pnames = fp_pt.readlines()

    ms_state_dict = ms.load_checkpoint(ms_ckpt)
    pt_state_dict = {}

    for line_ms, line_pt in zip(ms_pnames, pt_pnames):
        ms_pname = line_ms.split("#")[0]
        pt_pname = line_pt.split("#")[0]

        # todo: check shape
        ms_pval = ms_state_dict[ms_pname].asnumpy()
        pt_state_dict[pt_pname] = torch.tensor(ms_pval)

    if pt_ckpt_save_path is None:
        pt_ckpt_save_path = ms_ckpt.replace(".ckpt", "_pt.ckpt")

    torch.save(pt_state_dict, pt_ckpt_save_path)
    print("Done. Checkpoint saved in ", pt_ckpt_save_path)


if __name__ == "__main__":
    # Fire(get_ordered_param_list)
    Fire(convert)
