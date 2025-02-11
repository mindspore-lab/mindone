"""Merge the checkpoint from parallel training
    Example usage:
        python tools/ckpt/combine_ckpt.py --src output_dir/ckpt --dest output_dir/ckpt --strategy_ckpt output_dir/src_strategy.ckpt
"""
import argparse
import os

import mindspore as ms


def main():
    parser = argparse.ArgumentParser(description="Merge the saving slices from sequence parallel training.")
    parser.add_argument("--src", default="output/ckpt", help="Root path of the saving slices.")
    parser.add_argument("--dest", default="output/ckpt_full", help="Path of the merged ckeckpoint.")
    parser.add_argument(
        "--strategy_ckpt",
        default=None,
        help="The source strategy ckpt file path. If not provided, will search under `src` dir.",
    )
    args = parser.parse_args()

    strategy_file = (
        os.path.join(os.path.dirname(args.src), "src_strategy.ckpt")
        if args.strategy_ckpt is None
        else args.strategy_ckpt
    )
    assert os.path.exists(strategy_file), f"{strategy_file} does not exist!"
    ms.transform_checkpoints(args.src, args.dest, "full_", strategy_file, None)

    output_path = os.path.join(args.dest, "rank_0", "full_0.ckpt")
    assert os.path.isfile(output_path)
    print(f"Merged checkpoint is saved as `{output_path}`.")


if __name__ == "__main__":
    main()
