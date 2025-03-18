import argparse
import os

import torch
from safetensors.torch import load_file, save_file


def load_torch_ckpt(ckpt_file):
    # copied from modeling_wfvae.py init_from_ckpt
    def init_from_ckpt(path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        print("init from " + path)
        if "ema_state_dict" in sd and len(sd["ema_state_dict"]) > 0 and os.environ.get("NOT_USE_EMA_MODEL", 0) == 0:
            print("Load from ema model!")
            sd = sd["ema_state_dict"]
            sd = {key.replace("module.", ""): value for key, value in sd.items()}
        elif "state_dict" in sd:
            print("Load from normal model!")
            if "gen_model" in sd["state_dict"]:
                sd = sd["state_dict"]["gen_model"]
            else:
                sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        return sd

    return init_from_ckpt(ckpt_file)


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%, \n - {sf_filename} {sf_size} \n - {pt_filename} {pt_size}
         """
        )


def convert_file(
    pt_filename: str,
    sf_filename: str,
):
    loaded = load_torch_ckpt(pt_filename)

    # to_removes = _remove_duplicate_names(loaded, discard_names=discard_names)
    metadata = {"format": "pt"}
    # Force tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata=metadata)
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None, help="path to torch checkpoint path, e.g., merged.ckpt")
    parser.add_argument(
        "--target",
        type=str,
        help="The path to save the converted `safetensors` file (e.g., model.safetensors).",
    )
    args = parser.parse_args()

    convert_file(args.src, args.target)
    print(f"converted checkpoint saved to {args.target}")
