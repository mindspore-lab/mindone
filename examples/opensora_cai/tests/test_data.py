import os
import sys
import time

from tqdm import tqdm

import mindspore as ms

sys.path.append("./")
from opensora.datasets.t2v_dataset import TextVideoDataset, create_dataloader

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

csv_path = "../videocomposer/datasets/webvid5/video_caption.csv"
video_folder = "../videocomposer/datasets/webvid5"
video_column = "video"
caption_column = "caption"


def test_src_dataset(backend="al", is_image=False, use_tokenizer=False):
    ds = TextVideoDataset(
        csv_path,
        video_folder,
        sample_size=256,
        sample_stride=1,
        sample_n_frames=16,
        is_image=is_image,
        transform_backend=backend,  # pt, al
        tokenizer=None,
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
            print(type(caption), caption)

        start = time.time()

    mean = tot / steps
    print("Avg sample loading time: ", mean)


def test_loader(image_finetune=False):
    ds_config = dict(
        csv_path="../videocomposer/datasets/webvid5/video_caption.csv",
        tokenizer=None,
        video_folder="../videocomposer/datasets/webvid5",
        text_emb_folder="../videocomposer/datasets/webvid5",
        return_text_emb=True,
    )

    dl = create_dataloader(
        ds_config,
        ds_name="text_video",
        batch_size=2,
    )

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
                # check_sanity(batch["video"][0].asnumpy(), f"tmp_{i}.gif")

            progress_bar.update(1)
            if i + 1 > steps:  # in case the data size is too large
                break
            start = time.time()

    mean = tot / steps
    print("Avg batch loading time: ", mean)


if __name__ == "__main__":
    # test_src_dataset('al', False, False)
    test_loader(False)
