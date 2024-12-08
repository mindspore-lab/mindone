import mindspore as ms

from .image_dataset import ImageDataset
from .video_dataset import VideoDataset


def create_dataloader(
    ds_config,
    batch_size,
    ds_name="image",
    num_parallel_workers=12,
    max_rowsize=32,
    shuffle=True,
    device_num=1,
    rank_id=0,
    drop_remainder=True,
):
    """
    Args:
        ds_config, dataset config, args for ImageDataset or VideoDataset
        ds_name: dataset name, image or video
    """
    if ds_name == "image":
        dataset = ImageDataset(**ds_config)
    elif ds_name == "video":
        dataset = VideoDataset(**ds_config)
    print("Total number of samples: ", len(dataset))

    # Larger value leads to more memory consumption. Default: 16
    # prefetch_size = config.get("prefetch_size", 16)
    # ms.dataset.config.set_prefetch_size(prefetch_size)

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=[ds_name],
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=max_rowsize,
    )

    dl = dataloader.batch(
        batch_size,
        drop_remainder=drop_remainder,
    )

    return dl


if __name__ == "__main__":
    import math
    import time

    from tqdm import tqdm

    ds_config = dict(
        csv_path="/home/mindocr/yx/datasets/chinese_art_blip/train/metadata.csv",
        data_folder="/home/mindocr/yx/datasets/chinese_art_blip/train",
    )

    # test loader
    dl = create_dataloader(ds_config, 4)

    num_batches = dl.get_dataset_size()
    # ms.set_context(mode=0)
    print(num_batches)

    steps = 50
    iterator = dl.create_dict_iterator(100)  # create 100 repeats
    tot = 0

    progress_bar = tqdm(range(steps))
    progress_bar.set_description("Steps")

    start = time.time()
    for epoch in range(math.ceil(steps / num_batches)):
        for i, batch in enumerate(iterator):
            print("epoch", epoch, "step", i)
            dur = time.time() - start
            tot += dur

            if epoch * num_batches + i < 2:
                for k in batch:
                    print(k, batch[k].shape, batch[k].dtype)  # , batch[k].min(), batch[k].max())
                print(f"time cost: {dur * 1000} ms")

            progress_bar.update(1)
            if i + 1 > steps:  # in case the data size is too large
                break
            start = time.time()

    mean = tot / steps
    print("Avg batch loading time: ", mean)
