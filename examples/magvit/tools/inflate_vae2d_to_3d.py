import argparse
import os
import sys

import mindspore as ms
from mindspore import context

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from videogvt.config.vqgan3d_ucf101_config import get_config
from videogvt.models.vqvae import build_model

context.set_context(mode=1, device_target="Ascend", device_id=7)


def inflate(vae2d_ckpt, save_fp):
    model_config = get_config()
    dtype = ms.float32
    vae3d = build_model("vqvae-3d", model_config, is_training=False, dtype=dtype)
    vae2d = ms.load_checkpoint(vae2d_ckpt)

    vae_2d_keys = list(vae2d.keys())
    vae_3d_keys = list(vae3d.parameters_dict().keys())

    # 3d -> 2d
    map_dict = {
        "conv.weight": "weight",
        "conv.bias": "bias",
    }

    new_state_dict = {}

    for key_3d in vae_3d_keys:
        if key_3d.startswith("loss"):
            continue

        # param name mapping from vae-3d to vae-2d
        key_2d = "vqvae." + key_3d
        # if not 'attn' in key_2d:
        for kw in map_dict:
            key_2d = key_2d.replace(kw, map_dict[kw])

        assert key_2d in vae_2d_keys, f"Key {key_2d} ({key_3d}) not found in 2D VAE"

        # set vae 3d state dict
        shape_3d = vae3d.parameters_dict()[key_3d].shape
        shape_2d = vae2d[key_2d].shape
        if "bias" in key_2d:
            assert shape_3d == shape_2d, f"Shape mismatch for key {key_3d} ({key_2d})"
            new_state_dict[key_3d] = vae2d[key_2d]
        elif "norm" in key_2d:
            assert shape_3d == shape_2d, f"Shape mismatch for key {key_3d} ({key_2d})"
            new_state_dict[key_3d] = vae2d[key_2d]
        elif "conv" in key_2d or "nin_shortcut" in key_2d:
            if shape_3d[:2] != shape_2d[:2]:
                print(key_2d, shape_3d, shape_2d)

                weights = vae2d[key_2d]
                shape_3d_new = tuple([shape_3d[0] // 2]) + shape_3d[1:]
                new_w = ms.ops.zeros(shape_3d_new, dtype=weights.dtype)
                new_w[:, :, -1, :, :] = weights
                new_w = new_w.repeat(2, 0)

                new_w = ms.Parameter(new_w, name=key_3d)
                new_state_dict[key_3d] = new_w

            else:
                w = vae2d[key_2d]
                new_w = ms.ops.zeros(shape_3d, dtype=w.dtype)
                # tail initialization
                new_w[:, :, -1, :, :] = w  # cin, cout, t, h, w

                new_w = ms.Parameter(new_w, name=key_3d)
                new_state_dict[key_3d] = new_w

        elif "attn_1" in key_2d:
            new_val = vae2d[key_2d].expand_dims(axis=2)
            new_param = ms.Parameter(new_val, name=key_3d)
            new_state_dict[key_3d] = new_param
        else:
            raise NotImplementedError(f"Key {key_3d} ({key_2d}) not implemented")

    ms.save_checkpoint(new_state_dict, save_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None, help="path to mindspore vae 2d checkpoint")
    parser.add_argument(
        "--target",
        type=str,
        default="models/causal_vae_488_init.ckpt",
        help="target file path to save the inflated checkpoint",
    )
    args = parser.parse_args()

    inflate(args.src, args.target)
