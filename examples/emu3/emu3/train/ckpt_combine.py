"""
Offline checkpoint combine (based on zero stage = 3)

Usage:
EXP_NAME=T2I-SFT
python emu3/train/ckpt_combine.py \
    --checkpoint_dir outputs/parallel_logs/${EXP_NAME} \
        --ckpt_name emu3-e50.ckpt \
        --params_info_dir params_info \
        --group_size 8
"""
import argparse
import os

from mindone.trainers.zero import convert_checkpoints


def do_ckpt_combine_offline(checkpoint_dir, ckpt_name, params_info_dir, group_size):
    src_checkpoint = checkpoint_dir + "/rank_{}/ckpt/" + ckpt_name
    src_param_split_info_json = params_info_dir + "/params_split_info_{}.json"
    src_checkpoint_dir = (checkpoint_dir + "/rank_{}/ckpt/").format(f"all_{group_size}")
    os.makedirs(src_checkpoint_dir, exist_ok=True)

    convert_checkpoints(src_checkpoint, src_param_split_info_json, group_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, help="model path")
    parser.add_argument("--ckpt_name", type=str, default=None, help="model ckpt name, e.g. emu3-e50.ckpt")
    parser.add_argument(
        "--params_info_dir", type=str, default="params_info", help="model ckpt name, e.g. emu3-e50.ckpt"
    )
    parser.add_argument("--group_size", type=int, default=8, help="group size")
    args = parser.parse_args()

    do_ckpt_combine_offline(args.checkpoint_dir, args.ckpt_name, args.params_info_dir, args.group_size)
