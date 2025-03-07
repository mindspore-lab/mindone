import torch


def extract(ckpt_path, mm_double_blocks_depth=None, mm_single_blocks_depth=None):
    state_dict = torch.load(ckpt_path)
    load_key = "module"
    sd = state_dict[load_key]
    pnames = list(sd.keys())

    extract_ckpt = (mm_double_blocks_depth is not None) and (mm_single_blocks_depth is not None)
    for pname in pnames:
        print("{}\t{}\t{}".format(pname, tuple(sd[pname].shape), sd[pname].dtype))

        if extract_ckpt:
            if pname.startswith("double_blocks"):
                idx = int(pname.split(".")[1])
                if idx >= mm_double_blocks_depth:
                    sd.pop(pname)
            if pname.startswith("single_blocks"):
                idx = int(pname.split(".")[1])
                if idx >= mm_single_blocks_depth:
                    sd.pop(pname)

    if extract_ckpt:
        torch.save(state_dict, f"ckpts/transformer_depth{mm_double_blocks_depth}.pt")


if __name__ == "__main__":
    extract("ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt", 1, 1)
    # extract("ckpts/dit_small.pt")
