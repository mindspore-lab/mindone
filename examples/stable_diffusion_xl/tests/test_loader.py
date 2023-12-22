import time

from gm.data.loader import create_loader
from omegaconf import OmegaConf

import mindspore as ms


def test_loader():
    data_path = "datasets/custom"
    # shardlist_desc = "datasets/custom/data_info.json"

    rank = 0
    rank_size = 1

    config_file = "configs/training/sd_xl_base_finetune_910b_wds.yaml"
    config = OmegaConf.load(config_file)

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
    num_steps = config.data["total_step"]
    run_steps = 0
    for i, batch in enumerate(iterator):
        if i >= num_steps:
            break
        dur = time.time() - start
        tot += dur
        run_steps += 1
        for k in batch:
            print(f"{i+1}/{num_steps}, time cost: {dur * 1000} ms")
        start = time.time()

    mean = tot / run_steps
    print("Avg batch loading time: ", mean)


if __name__ == "__main__":
    test_loader()
