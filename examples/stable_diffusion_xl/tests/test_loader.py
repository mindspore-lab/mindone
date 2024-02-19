"""
To test in distributed mode:

export RANK_SIZE=2
export RANK_ID=0
python tests/test_loader.py

export RANK_ID=1
python tests/test_loader.py

"""

import sys
import time

import numpy as np

sys.path.insert(0, "./")

import os

from gm.data.dataset_wds import T2I_Webdataset, T2I_Webdataset_RndAcs
from gm.data.loader import create_loader
from omegaconf import OmegaConf

import mindspore as ms

os.environ["WIDS_VERBOSE"] = "1"


def test_src_dataset(target="T2I_Webdataset"):
    data_path = "datasets/telecom_ori"
    # data_path = "datasets/custom"
    # shardlist_desc = "datasets/custom/data_info.json"

    transforms = [
        {"target": "gm.data.mappers.Resize", "params": {"size": 1024, "interpolation": 3}},
        {"target": "gm.data.mappers.Rescaler", "params": {"isfloat": False}},
        {"target": "gm.data.mappers.AddOriginalImageSizeAsTupleAndCropToSquare"},
    ]
    if target == "T2I_Webdataset":
        dataset = T2I_Webdataset(
            data_path=data_path, target_size=1024, transforms=transforms, caption_key="text_english"
        )
    elif target == "T2I_Webdataset_RndAcs":
        dataset = T2I_Webdataset_RndAcs(
            data_path=data_path,
            # shardlist_desc=args.shardlist_desc,
            target_size=1024,
            transforms=transforms,
            caption_key="text_english",
        )
    else:
        raise ValueError("Unknown dataset target")

    dataset_size = len(dataset)
    print(f"dataset size: {dataset_size}")

    s_time = time.time()
    tot_time = 0
    n_read = len(dataset)
    visited = []
    for i, data in enumerate(dataset):
        if i > n_read:
            break
        tot_time += time.time() - s_time
        # print(f"{i}/{dataset_size}, image shape: {data.pop('image')}, {data}")
        print(f"{i+1}/{dataset_size}, time cost: {(time.time()-s_time) * 1000} ms")
        print(data["txt"], type(data["txt"]))
        codes = [ord(c) for c in str(data["txt"])]
        sum_code = (sum(codes), np.std(codes))
        print(sum_code)
        if sum_code not in visited:
            visited.append(sum_code)
        s_time = time.time()
    print("Num unique samples: ", len(visited))
    print("Total read time: ", tot_time)


def test_loader(rank=0, rank_size=1):
    # data_path = "datasets/custom"
    # data_path = "datasets/telecom_ori"
    # data_path = "datasets/multi_tars"
    # data_path = "datasets/multi_tars"
    data_path = "datasets/multi_tars_with_errors"
    # shardlist_desc = "datasets/custom/data_info.json"

    config_file = "configs/training/sd_xl_base_finetune_910b_wds.yaml"
    config = OmegaConf.load(config_file)

    num_steps = 100
    config.data["total_step"] = num_steps

    dataloader = create_loader(
        data_path=data_path,
        rank=rank,
        rank_size=rank_size,
        tokenizer=None,
        token_nums=None,
        **config.data,
    )

    ms.set_context(mode=0)

    num_batches = dataloader.get_dataset_size()
    print("num batches: ", num_batches)
    start = time.time()
    iterator = dataloader.create_dict_iterator()
    tot = 0
    run_steps = 0
    verbose = 1

    visited = []
    for i, batch in enumerate(iterator):
        if i >= num_steps:
            break
        dur = time.time() - start
        tot += dur
        run_steps += 1
        for k in batch:
            # print(f"{i+1}/{num_steps}, time cost: {dur * 1000} ms")
            if verbose:
                data = batch[k]
                print(data["txt"], type(data["txt"]))
                codes = [ord(c) for c in str(data["txt"])]
                sum_code = (sum(codes), np.std(codes))
                print(f"{i+1}/{num_steps}: ", sum_code)
                if sum_code not in visited:
                    visited.append(sum_code)
                else:
                    print("Replicated sample!!")
        start = time.time()
    print("Num unique samples: ", len(visited))

    mean = tot / run_steps
    print("Avg batch loading time: ", mean)


if __name__ == "__main__":
    rank_id = int(os.environ.get("RANK_ID", 0))
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    # test_src_dataset(target="T2I_Webdataset")
    # test_src_dataset(target="T2I_Webdataset_RndAcs")
    test_loader(rank_id, rank_size)
