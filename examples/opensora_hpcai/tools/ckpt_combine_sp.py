"""Merge the checkpoint from sequence parallel training
"""
import argparse
import os

import mindspore as ms
import mindspore.ops as ops
from mindspore import Parameter


def main():
    parser = argparse.ArgumentParser(description="Merge the saving slices from sequence parallel training.")
    parser.add_argument("--src", default="output/ckpt", help="Root path of the saving slices.")
    parser.add_argument("--dest", default="output/ckpt_full", help="Path of the merged ckeckpoint.")
    args = parser.parse_args()

    ms.set_context(device_target="CPU")

    print("Combining the distribued checkpoint...")
    stretegy_file = os.path.join(os.path.dirname(args.src), "src_strategy.ckpt")
    ms.transform_checkpoints(args.src, args.dest, "full_", stretegy_file, None)

    output_path = os.path.join(args.dest, "rank_0", "full_0.ckpt")
    assert os.path.isfile(output_path)

    # for compatibility of ckpt loading without sequence parallel, do some post-processing.
    print("Doing post-processing...")
    param_dict = ms.load_checkpoint(output_path)
    new_param_dict = dict()
    for k, v in param_dict.items():
        if ".q_linear." in k and ".cross_attn." in k:
            k_linear = param_dict[k.replace(".q_linear.", ".k_linear.")]
            v_linear = param_dict[k.replace(".q_linear.", ".v_linear.")]
            new_param_dict[k] = v
            kv_linear = ops.concat([k_linear, v_linear])
            new_param_dict[k.replace(".q_linear.", ".kv_linear.")] = Parameter(kv_linear)
        elif ".q_linear." in k:
            k_linear = param_dict[k.replace(".q_linear.", ".k_linear.")]
            v_linear = param_dict[k.replace(".q_linear.", ".v_linear.")]
            qkv = ops.concat([v, k_linear, v_linear])
            new_param_dict[k.replace(".q_linear.", ".qkv.")] = Parameter(qkv)
        elif ".k_linear." in k or ".v_linear." in k:
            continue
        elif ".scale_shift_table" in k and len(v.shape) == 3:
            new_param_dict[k] = Parameter(v[0])
        elif ".scale_shift_table_temporal" in k and len(v.shape) == 3:
            new_param_dict[k] = Parameter(v[0])
        else:
            new_param_dict[k] = v

    ms.save_checkpoint(new_param_dict, output_path)
    print(f"Merged checkpoint is saved as `{output_path}`.")


if __name__ == "__main__":
    main()
