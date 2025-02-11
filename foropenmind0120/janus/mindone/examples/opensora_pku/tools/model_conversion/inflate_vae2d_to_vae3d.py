# init Causal VAE from vae 2d
import argparse
import json
import os
import sys

sys.path.append(".")
mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
from opensora.models.causalvideovae.model.causal_vae.modeling_causalvae import CausalVAEModel
from opensora.utils.ms_utils import init_env

import mindspore as ms
from mindspore import mint


def inflate(args):
    vae_ckpt = args.src
    save_fp = args.target
    assert os.path.exists(args.model_config), f"{args.model_config} does not exist!"
    model_config = json.load(open(args.model_config, "r"))
    ae = CausalVAEModel.from_config(model_config)
    vae2d_sd = ms.load_checkpoint(vae_ckpt)

    vae_2d_keys = list(vae2d_sd.keys())
    vae_3d_keys = list(ae.parameters_dict().keys())

    # 3d -> 2d
    map_dict = {
        "conv.weight": "weight",
        "conv.bias": "bias",
        "downsample.weight": "downsample.conv.weight",
        "downsample.bias": "downsample.conv.bias",
    }

    new_state_dict = {}

    for key_3d in vae_3d_keys:
        if key_3d.startswith("loss"):
            continue

        # param name mapping from vae-3d to vae-2d
        key_2d = key_3d
        # if not 'attn' in key_2d:
        for kw in map_dict:
            key_2d = key_2d.replace(kw, map_dict[kw])

        if key_2d not in vae_2d_keys:
            if "time_upsample" in key_2d or "time_downsample" in key_2d:
                continue
            else:
                print(f"Key {key_2d} ({key_3d}) not found in 2D VAE")

        # set vae 3d state dict
        shape_3d = ae.parameters_dict()[key_3d].shape
        shape_2d = vae2d_sd[key_2d].shape
        if "bias" in key_2d:
            assert shape_3d == shape_2d, f"Shape mismatch for key {key_3d} ({key_2d})"
            new_state_dict[key_3d] = vae2d_sd[key_2d]
        elif "norm" in key_2d:
            assert shape_3d == shape_2d, f"Shape mismatch for key {key_3d} ({key_2d})"
            new_state_dict[key_3d] = vae2d_sd[key_2d]
        elif "conv" in key_2d or "nin_shortcut" in key_2d:
            if shape_3d[:2] != shape_2d[:2]:
                print(key_2d, shape_3d, shape_2d)
            if len(shape_3d) > len(shape_2d):
                w = vae2d_sd[key_2d]
                new_w = mint.zeros(shape_3d, dtype=w.dtype)
                # tail initialization
                new_w[:, :, -1, :, :] = w  # cin, cout, t, h, w

                new_w = ms.Parameter(new_w, name=key_3d)

                new_state_dict[key_3d] = new_w
            else:
                new_state_dict[key_3d] = vae2d_sd[key_2d]
        elif "attn_1" in key_2d:
            new_val = vae2d_sd[key_2d].expand_dims(axis=2)
            new_param = ms.Parameter(new_val, name=key_3d)
            new_state_dict[key_3d] = new_param
        else:
            raise NotImplementedError(f"Key {key_3d} ({key_2d}) not implemented")

    ms.save_checkpoint(new_state_dict, save_fp)
    print(f"The inflated checkpoint is saved in {save_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None, help="path to mindspore vae 2d checkpoint")
    parser.add_argument(
        "--target",
        type=str,
        default="models/causal_vae_488_init.ckpt",
        help="target file path to save the inflated checkpoint",
    )
    parser.add_argument("--model_config", type=str, default="scripts/causalvae/release.json")
    parser.add_argument("--device", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--jit_level", type=str, default="O1", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    args = parser.parse_args()
    init_env(mode=0, device_target=args.device, jit_level=args.jit_level)

    inflate(args)
