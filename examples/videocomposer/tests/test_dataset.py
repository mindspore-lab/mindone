import sys
import time
import os

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
    dl = build_dataset(cfg, 1, 0, tokenizer)
    dl.get_dataset_size()

    ms.set_context(mode=0)

    #num_tries = 4 * 100
    num_tries = 5
    start = time.time()
    times = []
    iterator = dl.create_dict_iterator()
    for i, batch in enumerate(iterator):
        # print(batch)
        for k in batch:
            print(k, batch[k].shape) #, batch[k].min(), batch[k].max())
            #if k in ["cap_tokens", "feature_framerate"]:
            #    print(batch[k])
        times.append(time.time() - start)
        if i >= num_tries:
            break
        start = time.time()

    WU = 2
    tot = sum(times[WU:])  # skip warmup
    mean = tot / (num_tries - WU)
    print("Avg batch loading time: ", mean)


if __name__ == "__main__":
    test_dataset()
