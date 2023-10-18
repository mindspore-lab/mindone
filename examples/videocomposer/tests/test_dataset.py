import sys
import time

from configs.train_base import cfg
from vc.data.dataset_train import build_dataset

import mindspore as ms

# from vc.data.transforms import create_transforms

# from mindspore import dataset as ds

sys.path.append("../stable_diffusion_v2/")
from tools._common.clip import CLIPTokenizer


def test_dataset():
    # data_dir = "./demo_video"
    # cfg.max_frames = 16
    # cfg.batch_size = 1

    tokenizer = CLIPTokenizer("./model_weights/bpe_simple_vocab_16e6.txt.gz")
    dl = build_dataset(cfg, 1, 0, tokenizer)
    dl.get_dataset_size()

    ms.set_context(mode=0)

    num_tries = 4 * 100
    start = time.time()
    times = []
    iterator = dl.create_dict_iterator()
    for i, batch in enumerate(iterator):
        for k in batch:
            # print(k, batch[k].shape, batch[k].min(), batch[k].max())
            if k in ["cap_tokens", "feature_framerate"]:
                print(batch[k])
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
