"""
To test in distributed mode:

export RANK_SIZE=2
export RANK_ID=0
python tests/test_loader.py

export RANK_ID=1
python tests/test_loader.py

"""

import argparse
import ast
import sys
import time

from tqdm import tqdm

sys.path.insert(0, "./")

import os

from gm.data.loader import create_loader
from omegaconf import OmegaConf

import mindspore as ms

os.environ["WIDS_VERBOSE"] = "1"


def get_parser():
    parser = argparse.ArgumentParser(description="Dataset Check")
    parser.add_argument("--config", type=str, default="configs/training/sd_xl_base_finetune_910b_wds.yaml")
    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--per_batch_size", type=int, default=None)
    parser.add_argument("--num_parallel_workers", type=int, default=None)
    parser.add_argument("--python_multiprocessing", type=ast.literal_eval, default=None)
    parser.add_argument(
        "--dataset_load_tokenizer", type=ast.literal_eval, default=False, help="create dataset with tokenizer"
    )

    parser.add_argument("--scan_step", type=int, default=-1)
    parser.add_argument("--verbose", type=ast.literal_eval, default=False)
    parser.add_argument("--is_print_detail", type=ast.literal_eval, default=False)

    return parser


def scan(args):
    if args.dataset_load_tokenizer:
        raise NotImplementedError

    config = OmegaConf.load(args.config)
    config.data["total_step"] = 1  # scan with one epoch

    # reset param
    per_batch_size, num_parallel_workers, python_multiprocessing = (
        config.data.pop("per_batch_size"),
        config.data.pop("num_parallel_workers"),
        config.data.pop("python_multiprocessing"),
    )
    per_batch_size = per_batch_size if args.per_batch_size is None else args.per_batch_size
    num_parallel_workers = num_parallel_workers if args.num_parallel_workers is None else args.num_parallel_workers
    python_multiprocessing = (
        python_multiprocessing if args.python_multiprocessing is None else args.python_multiprocessing
    )

    dataloader = create_loader(
        data_path=args.data_path,
        rank=args.rank,
        rank_size=args.rank_size,
        tokenizer=None,
        token_nums=None,
        per_batch_size=per_batch_size,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
        **config.data,
    )

    ms.set_context(mode=0)

    num_batches = dataloader.get_dataset_size()
    print(f"Rank: {args.rank}/{args.rank_size}, num batches: {num_batches}")
    start = time.time()
    iterator = dataloader.create_dict_iterator()
    tot = 0

    cur_step = 0
    for batch in tqdm(iterator):
        if args.scan_step > 0 and cur_step >= args.scan_step:
            break
        dur = time.time() - start
        tot += dur
        cur_step += 1
        if args.is_print_detail:
            for k in batch:
                data = batch[k]
                print(
                    f"Batch {cur_step}/{num_batches}, sample: {k}/{len(batch)}, txt: {data['txt']}, txt type: {type(data['txt'])}"
                )
        start = time.time()

    print(f"Scan total dataset done, dataset path: {args.data_path}, time cost: {tot / 60:.2f} mins")


if __name__ == "__main__":
    rank_id = int(os.environ.get("RANK_ID", 0))
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    parser = get_parser()
    args, _ = parser.parse_known_args()
    args.rank = rank_id
    args.rank_size = rank_size

    scan(args)
