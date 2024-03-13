import os
import sys
import time

from omegaconf import OmegaConf
from tqdm import tqdm

import mindspore as ms

sys.path.append("./")
from ad.data.dataset import TextVideoDataset, check_sanity, create_dataloader

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.utils.config import instantiate_from_config

csv_path = "../videocomposer/datasets/webvid5/video_caption.csv"
video_folder = "../videocomposer/datasets/webvid5"
video_column = "video"
caption_column = "caption"

# csv_path = "./datasets/webvid_overfit/video_caption.csv"
# video_folder = "./datasets/webvid_overfit"
cfg_path = "configs/stable_diffusion/v1-train-mmv2.yaml"


def test_src_dataset(backend="al", is_image=False, use_tokenizer=False):
    use_tool_clip = False

    if use_tokenizer:
        if not use_tool_clip:
            cfg = OmegaConf.load(cfg_path)
            text_encoder_config = cfg.model.params.cond_stage_config
            text_encoder = instantiate_from_config(text_encoder_config)
            tokenizer = text_encoder.tokenize
        else:
            from functools import partial

            from examples.stable_diffusion_v2.tools._common.clip import CLIPTokenizer

            tokenizer_raw = CLIPTokenizer(os.path.join(__dir__, "ad/models/clip/bpe_simple_vocab_16e6.txt.gz"))
            tokenizer = partial(tokenizer_raw, padding="max_length", max_length=77)

    else:
        tokenizer = None

    ds = TextVideoDataset(
        csv_path,
        video_folder,
        sample_size=256,
        sample_stride=1,
        sample_n_frames=16,
        is_image=is_image,
        transform_backend=backend,  # pt, al
        tokenizer=tokenizer,
        video_column=video_column,
        caption_column=caption_column,
    )
    num_samples = len(ds)
    steps = 100
    start = time.time()
    tot = 0
    for i in tqdm(range(steps)):
        video, caption = ds.__getitem__(i % num_samples)

        dur = time.time() - start
        tot += dur

        if i < 3:
            print("D--: ", video.shape, video.dtype, video.min(), video.max())
            print(f"{i+1}/{steps}, time cost: {dur * 1000} ms")
            check_sanity(video, f"tmp_{i}.gif")
            print(type(caption), caption)

        start = time.time()

    mean = tot / steps
    print("Avg sample loading time: ", mean)


def test_loader(image_finetune=False):
    cfg = OmegaConf.load(cfg_path)
    text_encoder_config = cfg.model.params.cond_stage_config
    text_encoder = instantiate_from_config(text_encoder_config)
    tokenizer = text_encoder.tokenize

    # data_config = cfg.train_data
    data_config = dict(
        video_folder=video_folder,
        csv_path=csv_path,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
        batch_size=4,
        shuffle=True,
        num_parallel_workers=12,
        max_rowsize=64,
        video_column=video_column,
        caption_column=caption_column,
        train_data_type="video_file",
        disable_flip=True,
        random_drop_text=False,
        random_drop_text_ratio=0.0,
    )
    dl = create_dataloader(data_config, tokenizer=tokenizer, is_image=image_finetune, device_num=1, rank_id=0)

    num_batches = dl.get_dataset_size()
    ms.set_context(mode=0)

    steps = 50
    iterator = dl.create_dict_iterator(100)
    tot = 0

    progress_bar = tqdm(range(steps))
    progress_bar.set_description("Steps")

    start = time.time()
    for epoch in range(steps // num_batches):
        for i, batch in enumerate(iterator):
            dur = time.time() - start
            tot += dur

            if epoch * num_batches + i < 3:
                for k in batch:
                    print(k, batch[k].shape, batch[k].dtype)  # , batch[k].min(), batch[k].max())
                print(f"time cost: {dur * 1000} ms")
                check_sanity(batch["video"][0].asnumpy(), f"tmp_{i}.gif")

            progress_bar.update(1)
            if i + 1 > steps:  # in case the data size is too large
                break
            start = time.time()

    mean = tot / steps
    print("Avg batch loading time: ", mean)


if __name__ == "__main__":
    # test_src_dataset('al', False, False)
    test_loader(False)
