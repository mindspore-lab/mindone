"""Merge the checkpoint from sequence parallel training
"""
import argparse
import os

import mindspore as ms


def main():
    parser = argparse.ArgumentParser(description="Merge the saving slices from sequence parallel training.")
    parser.add_argument("--src", default="output/ckpt", help="Root path of the saving slices.")
    parser.add_argument("--dest", default="output/ckpt_full", help="Path of the merged ckeckpoint.")
    args = parser.parse_args()

    stretegy_file = os.path.join(os.path.dirname(args.src), "src_strategy.ckpt")
    ms.transform_checkpoints(args.src, args.dest, "full_", stretegy_file, None)

    output_path = os.path.join(args.dest, "rank_0", "full_0.ckpt")
    assert os.path.isfile(output_path)
    print(f"Merged checkpoint is saved as `{output_path}`.")


if __name__ == "__main__":
    main()
