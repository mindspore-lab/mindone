import copy
import csv
import glob
import logging
import os

import albumentations
import cv2
import numpy as np
from PIL import Image

import mindspore as ms

logger = logging.getLogger()


def create_image_transforms(
    size=384, crop_size=256, interpolation="bicubic", backend="al", random_crop=False, flip=False
):
    # expect rgb image in range 0-255, shape (h w c)
    from albumentations import CenterCrop, HorizontalFlip, RandomCrop, SmallestMaxSize

    mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
    transforms = [
        SmallestMaxSize(max_size=size, interpolation=mapping[interpolation]),
        CenterCrop(crop_size, crop_size) if not random_crop else RandomCrop(crop_size, crop_size),
    ]
    if flip:
        transforms += [HorizontalFlip(p=0.5)]

    pixel_transforms = albumentations.Compose(transforms)

    return pixel_transforms


def get_image_path_list(folder):
    # TODO: find recursively
    fmts = ["jpg", "png", "jpeg", "JPEG"]
    out = []
    for fmt in fmts:
        out += glob.glob(os.path.join(folder, f"*.{fmt}"))
    return sorted(out)


class ImageDataset:
    def __init__(
        self,
        csv_path=None,
        image_folder=None,
        size=384,
        crop_size=256,
        random_crop=False,
        flip=False,
        image_column="file_name",
    ):
        if csv_path is not None:
            with open(csv_path, "r") as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
            self.read_from_csv = True
        else:
            self.dataset = get_image_path_list(image_folder)
            self.read_from_csv = False
        self.length = len(self.dataset)

        logger.info(f"Num data samples: {self.length}")

        self.image_folder = image_folder

        self.pixel_transforms = create_image_transforms(
            size,
            crop_size,
            random_crop=random_crop,
            flip=flip,
        )
        self.image_column = image_column

        # prepare replacement data
        # max_attempts = 100
        # self.prev_ok_sample = self.get_replace_data(max_attempts)
        # self.require_update_prev = False

    def get_replace_data(self, max_attempts=100):
        replace_data = None
        attempts = min(max_attempts, self.length)
        for idx in range(attempts):
            try:
                pixel_values, caption = self.read_sample(idx)
                replace_data = copy.deepcopy((pixel_values, caption))
                break
            except Exception as e:
                print("\tError msg: {}".format(e), flush=True)

        assert replace_data is not None, f"Fail to preload sample in {attempts} attempts."

        return replace_data

    def read_sample(self, idx):
        if self.read_from_csv:
            image_dict = self.dataset[idx]
            # first column is image path
            image_fn = image_dict[list(image_dict.keys())[0]]
            image_path = os.path.join(self.image_folder, image_fn)
        else:
            image_path = self.dataset[idx]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        return image

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # try:
        image = self.read_sample(idx)
        """
        if (self.prev_ok_sample is None) or (self.require_update_prev):
            self.prev_ok_sample = copy.deepcopy(image)
            self.require_update_prev = False
        except Exception as e:
            logger.warning(f"Fail to get sample of idx {idx}. The corrupted video will be replaced.")
            print("\tError msg: {}".format(e), flush=True)
            assert self.prev_ok_sample is not None
            image = self.prev_ok_sample  # unless the first sample is already not ok
            self.require_update_prev = True

            if idx >= self.length:
                raise IndexError  # needed for checking the end of dataset iteration
        """

        # import pdb
        # pdb.set_trace()
        trans_image = self.pixel_transforms(image=image)["image"]

        out_image = (trans_image / 127.5 - 1.0).astype(np.float32)
        out_image = out_image.transpose((2, 0, 1))  # h w c -> c h w

        return out_image


def create_dataloader(
    ds_config,
    batch_size,
    num_parallel_workers=12,
    max_rowsize=32,
    shuffle=True,
    device_num=1,
    rank_id=0,
    drop_remainder=True,
):
    dataset = ImageDataset(
        **ds_config,
    )
    print("Total number of samples: ", len(dataset))

    # Larger value leads to more memory consumption. Default: 16
    # prefetch_size = config.get("prefetch_size", 16)
    # ms.dataset.config.set_prefetch_size(prefetch_size)

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=["image"],
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

    # TODO: add repeat for for-loop trainer?

    return dl


def check_sanity(x, save_fp="./tmp.png"):
    # reverse normalization and visulaize the transformed video
    if len(x.shape) == 4:
        print("only save the first image")
        x = x[0]
    x = np.transpose(x, (1, 2, 0))

    x = (x + 1.0) / 2.0  # -1,1 -> 0,1
    x = (x * 255).astype(np.uint8)

    if isinstance(x, ms.Tensor):
        x = x.asnumpy()
    Image.fromarray(x).save(save_fp)


if __name__ == "__main__":
    import math
    import time

    from tqdm import tqdm

    ds_config = dict(
        csv_path="/home/mindocr/yx/datasets/chinese_art_blip/train/metadata.csv",
        image_folder="/home/mindocr/yx/datasets/chinese_art_blip/train",
    )
    # test source dataset
    ds = ImageDataset(**ds_config)
    sample = ds.__getitem__(0)
    print(sample.shape)

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
                    check_sanity(batch[k])
                print(f"time cost: {dur * 1000} ms")

            progress_bar.update(1)
            if i + 1 > steps:  # in case the data size is too large
                break
            start = time.time()

    mean = tot / steps
    print("Avg batch loading time: ", mean)
