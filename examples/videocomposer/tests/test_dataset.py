import os
import sys
import time

from configs.train_base import cfg
from vc.config import Config
from vc.data.dataset_train import build_dataset

import mindspore as ms

# from vc.data.transforms import create_transforms

# from mindspore import dataset as ds

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../stable_diffusion_v2/")))
from tools._common.clip import CLIPTokenizer


def test_dataset():
    args_for_update = Config(load=True).cfg_dict  # config args from CLI (arg parser) and yaml files

    # update base config
    for k, v in args_for_update.items():
        cfg[k] = v
    cfg.num_parallel_workers = 8

    print(cfg)

    # cfg.root_dir = "datasets/webvid5"
    # cfg.force_start_from_first_frame = True
    # cfg.misc_random_interpolation = False
    # cfg.max_frames = 16
    # cfg.batch_size = 1

    tokenizer = CLIPTokenizer(os.path.join(__dir__, "../model_weights/bpe_simple_vocab_16e6.txt.gz"))
    dl = build_dataset(cfg, 1, 0, tokenizer, record_data_stat=True)
    num_batches = dl.get_dataset_size()

    ms.set_context(mode=0)

    num_tries = num_batches
    start = time.time()
    warmup = 0
    warmup_steps = 3
    iterator = dl.create_dict_iterator()
    for i, batch in enumerate(iterator):
        print(f"{i}/{num_batches}")
        for k in batch:
            print(k, batch[k].shape)  # , batch[k].min(), batch[k].max())
            # if k in ["cap_tokens", "feature_framerate"]:
            #    print(batch[k])
        if i == warmup_steps - 1:
            warmup = time.time() - start
    tot_time = time.time() - start - warmup

    mean = tot_time / (num_tries - warmup_steps)
    print("Avg batch loading time: ", mean)


if __name__ == "__main__":
    test_dataset()
