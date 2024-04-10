import argparse

import mindspore as ms

# path = "../stable_diffusion_v2/models/sd_v1.5-d0ab7146.ckpt"


def convert(path, save_fp):
    ignore_keys = list()
    remove_prefix = ["first_stage_model.", "autoencoder."]

    sd = ms.load_checkpoint(path)
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]

    for pname in keys:
        is_vae_param = False
        for pf in remove_prefix:
            if pname.startswith(pf):
                sd[pname.replace(pf, "")] = sd.pop(pname)
                is_vae_param = True
        if not is_vae_param:
            sd.pop(pname)

    ms.save_checkpoint(sd, save_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None, help="path to ms sd checkpoint")
    parser.add_argument(
        "--target",
        type=str,
        default="models/vae_sd.ckpt",
        help="target file path to save the extracted vae checkpoint",
    )
    args = parser.parse_args()

    convert(args.src, args.target)
