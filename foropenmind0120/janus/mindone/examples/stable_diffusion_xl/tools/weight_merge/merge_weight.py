import argparse

import mindspore as ms


def get_parser():
    parser = argparse.ArgumentParser(description="sd-xl weight merge")
    parser.add_argument("--base_weight", type=str, default="checkpoints/sd_xl_base_1.0_ms.ckpt")
    parser.add_argument("--additional_weights", type=str, default=None, help="e.g. unet.ckpt,vae.ckpt")
    parser.add_argument("--save_path", type=str, default="./merged_weight.ckpt")
    return parser


def merge(args):
    assert args.additional_weights is not None

    base_weight = args.base_weight
    additional_weights = args.additional_weights.split(",")

    assert base_weight.endswith(".ckpt")
    sd_dict = ms.load_checkpoint(base_weight)
    sd_dict = fix_ckpt_prefix(sd_dict)

    for weight in additional_weights:
        assert weight.endswith(".ckpt")
        _sd_dict = ms.load_checkpoint(weight)
        _sd_dict = fix_ckpt_prefix(_sd_dict)

        # update
        sd_dict.update(_sd_dict)

    new_ckpt = [{"name": k, "data": sd_dict[k]} for k in sd_dict]
    ms.save_checkpoint(new_ckpt, args.save_path)
    print(f"Merged `{args.base_weight}` and `{args.additional_weights}` to '{args.save_path}' Done.")


def fix_ckpt_prefix(ckpt_dict):
    # FIXME: parameter auto-prefix name bug on mindspore 2.2.10
    _new_ckpt_dict = {}
    for k in ckpt_dict:
        if "._backbone" in k:
            _index = k.find("._backbone")
            new_k = k[:_index] + k[_index + len("._backbone") :]
        else:
            new_k = k[:]
        _new_ckpt_dict[new_k] = ckpt_dict[k]
    ckpt_dict = _new_ckpt_dict
    return ckpt_dict


if __name__ == "__main__":
    parser = get_parser()
    args, _ = parser.parse_known_args()
    merge(args)
