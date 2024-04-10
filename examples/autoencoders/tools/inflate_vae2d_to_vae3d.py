# init Causal VAE from vae 2d
import sys

sys.path.append(".")
import argparse

from ae.models.causal_vae_3d import CausalVAEModel

import mindspore as ms


def inflate(vae_ckpt, save_fp):
    args = dict(
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=4,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla3D",  # diff 3d
        time_compress=2,  # diff 3d
        split_time_upsample=True,
    )
    ae = CausalVAEModel(ddconfig=args, embed_dim=4)
    vae2d_sd = ms.load_checkpoint(vae_ckpt)

    vae_2d_keys = list(vae2d_sd.keys())
    vae_3d_keys = list(ae.parameters_dict().keys())

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
        key_2d = key_3d
        # if not 'attn' in key_2d:
        for kw in map_dict:
            key_2d = key_2d.replace(kw, map_dict[kw])

        assert key_2d in vae_2d_keys, f"Key {key_2d} ({key_3d}) not found in 2D VAE"

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
            w = vae2d_sd[key_2d]
            new_w = ms.ops.zeros(shape_3d, dtype=w.dtype)
            # tail initialization
            new_w[:, :, -1, :, :] = w  # cin, cout, t, h, w

            new_w = ms.Parameter(new_w, name=key_3d)

            new_state_dict[key_3d] = new_w
        elif "attn_1" in key_2d:
            new_val = vae2d_sd[key_2d].expand_dims(axis=2)
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
